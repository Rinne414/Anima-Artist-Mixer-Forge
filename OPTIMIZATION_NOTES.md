# Optimization Notes

This document lists optimization directions identified but not yet implemented in the Anima Artist Cross-Attn node, providing a technical roadmap for future contributors. **The code here is not part of the main plugin** — it serves as implementation reference only.

## Current performance baseline

In `output_avg + interpolate` mode, each layer runs `N + 1` cross-attention forwards (N artists + base). This is mathematically necessary:

```
sum_i (w_i * softmax(Q @ K_i^T / √d) @ V_i)
```

Each softmax must be computed independently over its own K, V. The implemented batched parallel optimization combines N serial forwards into one large-batch forward, but doesn't reduce total computation.

Measured timing (30 steps, single-artist baseline 36s):

| Artists | Time | Per-artist increment |
|---|---|---|
| 1 | 36s | - |
| 4 | 50s | 4.7s |
| 8 | 60s | 3.4s |

Cross-attention is roughly 20% of total time at 1 artist; up to ~50% at 8 artists. Even completely freeing cross-attention only brings the 8-artist case from 60s down to ~30s.

**The real lever is reducing the number of cross-attention calls themselves**, not constant-factor optimizations on each call. The remainder of this doc is sorted by that priority.

---

## 1. Attention-output deferred cache

**Expected payoff**: 1.5-2x speedup  
**Complexity**: medium  
**Risk**: medium (engineering approximation, no theoretical guarantee)

### Principle

Observation: different stages of diffusion sampling do different things.

```
High sigma (early steps)  → overall composition, style direction
Mid sigma (middle steps)  → shape refinement, layout
Low sigma (late steps)    → texture detail, high-frequency refinement
```

Empirical pattern: **artist style signal is mostly absorbed in early steps**. Later, the directional component of artist contribution to cross-attention output changes only slightly.

Treating "artist injection" as a per-layer bias added to the base cross-attention output:

```
inject_out = artist_total - base_out  # per-layer artist increment
```

If we accumulate `inject_out` over the first K steps and use the average as a cache for the rest:

```
# First K steps: full computation
for step in range(K):
    base_out = cross_attn_base(x)
    artist_total = sum_i (w_i * cross_attn(x, K_i, V_i))
    inject_contrib_sum += (artist_total - base_out).detach()

cache = inject_contrib_sum / K

# Remaining steps: reuse cache
for step in range(K, total_steps):
    base_out = cross_attn_base(x)
    return base_out + cache
```

Late-step `N + 1` forwards collapse to 1 forward.

### Code sketch

```python
class _DelayedCacheWrapper(nn.Module):
    """Experimental: early-warmup + late-cache injection.

    Notes for users:
    - Requires sigma capture (depends on current sigma in transformer_options or shared_state)
    - Cache is established per patch; reset on re-patching
    - Only applicable to output_avg + interpolate/concat_with_base style modes
    """

    def __init__(self, original, shared_state, layer_idx, warmup_sigma_threshold):
        super().__init__()
        self.original = original
        self._st = shared_state
        self._idx = layer_idx
        self._threshold = warmup_sigma_threshold  # sigma > threshold = early stage

        self._contrib_sum = None    # accumulated artist increment
        self._contrib_count = 0
        self._cache = None          # static contribution used in late stage

    def forward(self, x, context=None, rope_emb=None, transformer_options={}):
        st = self._st
        sigma = st.get("current_sigma")

        # 1) No sigma info: degrade to full path
        if sigma is None:
            return self._full_path(x, context, rope_emb, transformer_options)

        is_warmup = sigma > self._threshold

        if is_warmup:
            # 2) Early stage: full base + artist computation, accumulate contrib
            base_out = self.original(x, context, rope_emb=rope_emb,
                                     transformer_options=transformer_options)
            artist_total = self._compute_artist_total(
                x, context, rope_emb, transformer_options
            )
            with torch.no_grad():
                contrib = (artist_total - base_out).detach()
                if self._contrib_sum is None:
                    self._contrib_sum = contrib
                else:
                    self._contrib_sum = self._contrib_sum + contrib
                self._contrib_count += 1
            return artist_total

        # 3) Late stage: use cache
        if self._cache is None:
            if self._contrib_count == 0:
                # No warmup data, degrade
                return self._full_path(x, context, rope_emb, transformer_options)
            self._cache = self._contrib_sum / self._contrib_count

        base_out = self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)
        return base_out + self._cache

    def _compute_artist_total(self, x, context, rope_emb, t_opts):
        """Reuse main plugin's _fwd_output_avg logic (omitted; copy in when integrating)"""
        raise NotImplementedError

    def _full_path(self, x, context, rope_emb, t_opts):
        """Degrade to full output_avg path"""
        raise NotImplementedError
```

### Risk points

1. **Cache shape depends on x.shape**: x.shape is usually constant during sampling, but if a workflow includes dynamic resolution switching, cache shape will mismatch. Need shape validation
2. **CFG batch row order changes**: ComfyUI's cond/uncond order is typically stable, but if a sampler implementation varies cond_or_uncond between steps, cache row alignment breaks. Need to accumulate per cond_or_uncond marker
3. **Cross-workflow contamination**: cache must be cleared on each re-run. Under the current architecture, wrappers are re-instantiated when `add_object_patch` re-mounts, which clears automatically
4. **Style-switch artifacts**: the warmup → cached transition step may show a visible "jump". Consider linear blending instead of hard switching

### Cache invalidation boundaries

```python
# Detect shape changes
if self._cache is not None and self._cache.shape != base_out.shape:
    self._cache = None
    self._contrib_sum = None
    self._contrib_count = 0
    return self._full_path(x, context, rope_emb, transformer_options)
```

### Suggested user-facing parameter

Add to `AnimaArtistOptions`:

```python
"warmup_sigma": ("FLOAT", {
    "default": 0.0,  # 0.0 = optimization off
    "min": 0.0, "max": 14.0, "step": 0.1,
    "tooltip": "Experimental. Steps with sigma > this threshold are warmup (full computation), "
              "the rest use cache. 0 = off. Suggest starting at 7.0 (~30% of sampling)."
}),
```

---

## 2. K/V projection cross-step cache

**Expected payoff**: ~1.1x speedup  
**Complexity**: medium  
**Risk**: low

### Principle

Inside cross-attention:

```
Q = q_proj(x)         # x changes every step, must recompute
K = k_proj(context)   # context is the artist embedding, constant during sampling
V = v_proj(context)   # ditto
```

`k_proj` and `v_proj` are linear layers. The same artist's context stays the same throughout sampling (artist embedding is computed once via LLMAdapter and cached in `state["individuals"]`), so K and V can be reused across steps.

Approximate component cost in `predict2.Attention`:

| Component | Share | Cacheable |
|---|---|---|
| q_proj + q_norm + apply_rotary on Q | ~15% | No |
| k_proj + k_norm | ~10% | **Yes** |
| v_proj | ~7% | **Yes** |
| apply_rotary on K | ~3% | Maybe (depends on rope_emb constancy) |
| Q @ K^T + softmax | ~40% | No |
| @ V | ~15% | No |
| o_proj | ~10% | No |

Total cacheable ~17%, leading to ~1.2x speedup on cross-attention itself, and ~1.05-1.1x on full image.

### Implementation path

Bypass `predict2.Attention.forward` and write a simplified version that caches only K/V:

```python
def _cached_cross_attn_forward(attn, x, context, rope_emb, t_opts,
                               kv_cache, cache_key):
    """Replicates predict2.Attention.forward, but K/V goes through cache.

    cache_key: tuple (layer_idx, artist_idx, cache_version)
    kv_cache: shared dict, key -> (k_normed, v)
    """
    from einops import rearrange
    from comfy.ldm.cosmos.predict2 import (
        torch_attention_op, apply_rotary_pos_emb,
    )

    n_heads = attn.heads
    head_dim = attn.dim_head

    # Q: compute every step
    q = attn.q_proj(x)
    q_shape = (*x.shape[:-1], n_heads, head_dim)
    q = q.view(q_shape)
    q = attn.q_norm(q)

    # K, V: cached
    if cache_key in kv_cache:
        k, v = kv_cache[cache_key]
    else:
        k = attn.k_proj(context).view(*context.shape[:-1], n_heads, head_dim)
        k = attn.k_norm(k)
        v = attn.v_proj(context).view(*context.shape[:-1], n_heads, head_dim)
        kv_cache[cache_key] = (k.detach(), v.detach())

    # rope on Q (rope on K usually compatible with cache since rope_emb_context
    # is constant during sampling)
    if rope_emb is not None:
        q = apply_rotary_pos_emb(q, rope_emb)

    out = torch_attention_op(q, k, v, transformer_options=t_opts)
    return attn.o_proj(out)
```

### Cache key design

```python
# At patch() time
state["kv_cache"] = {}
state["cache_version"] = id(state["individuals"])  # individuals change -> all cache invalid

# In wrapper call
cache_key = (self._idx, artist_idx, state["cache_version"])
```

### Risk points

1. **predict2 upgrade compatibility**: this approach copies internal logic from `Attention.forward`. If ComfyUI updates predict2 (changing norm order, attention op, RoPE variants, etc.), this needs to be kept in sync. Annotate which predict2 version is targeted
2. **rope_emb assumption**: assumes rope on K can be cached, but only if rope_emb is constant during sampling. Validate `id(rope_emb)` doesn't change across calls; invalidate otherwise

### Why not prioritized

1. Payoff too small (~1.1x)
2. Maintenance cost high (must track predict2 versions)
3. Compared to direction 1 (1.5-2x), the ratio is clearly worse

Listed here for completeness only.

---

## 3. Similar-artist merging

**Expected payoff**: N → M (merged count) linear speedup  
**Complexity**: high  
**Risk**: medium

### Principle

If a user supplies 8 artists where 3 are highly similar (cosine similarity > 0.95 in their LLMAdapter embeddings), merging those 3 into 1 brings the 8-artist case down to a 6-artist case.

### Implementation idea

```python
def _maybe_merge_similar_artists(individuals, weights, threshold=0.95):
    """Merge highly-similar artists based on LLMAdapter embedding cosine similarity.

    Returns (merged_individuals, merged_weights, merge_log).
    """
    n = len(individuals)
    if n <= 1:
        return individuals, weights, []

    # Mean-pool each artist's (1, 512, 1024) along token dim → (1024,) summary vector
    summaries = [a.mean(dim=1).flatten() for a in individuals]
    summaries = torch.stack(summaries, dim=0)  # (N, 1024)
    summaries_norm = torch.nn.functional.normalize(summaries, dim=-1)
    sim_matrix = summaries_norm @ summaries_norm.T  # (N, N)

    # Union-find to identify similarity clusters
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j].item() > threshold:
                union(i, j)

    # Merge per cluster
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged_individuals = []
    merged_weights = []
    merge_log = []
    for root, members in clusters.items():
        if len(members) == 1:
            merged_individuals.append(individuals[members[0]])
            merged_weights.append(weights[members[0]])
        else:
            ws = [weights[i] for i in members]
            total_w = sum(ws)
            merged = sum(individuals[i] * ws[idx] for idx, i in enumerate(members)) / total_w
            merged_individuals.append(merged)
            merged_weights.append(total_w)
            merge_log.append(f"merged artists {members} into one (sum weight={total_w:.3f})")

    return merged_individuals, merged_weights, merge_log
```

### Risk points

1. **Cosine similarity ≠ visual similarity**: similarity in LLMAdapter output space doesn't necessarily correspond to "similar painting style". May merge artists who shouldn't be merged, or fail to merge those who should
2. **Merging breaks output_avg's mathematical purity**: output_avg requires each artist's softmax to be independent. Merging similar artists discards their fine-grained differences
3. **Threshold is hard to tune**: too strict → never triggers; too loose → false merges

### Suggested user interface

```python
"merge_similar_threshold": ("FLOAT", {
    "default": 0.0,
    "min": 0.0, "max": 1.0, "step": 0.01,
    "tooltip": "Experimental. 0 = off. > 0 auto-merges artists with cosine similarity above threshold. Try starting at 0.95."
}),
```

---

## 4. Adaptive injection schedule

**Expected payoff**: unknown, research level  
**Complexity**: high (needs training or offline experiments)  
**Risk**: high

### Principle

Different artists may peak in contribution at different sigma stages:
- Strong-style artists may need only steps 1-5 to set tone
- Detail-focused artists are useful only at step 20+

Ideally each artist would have its own "optimal injection range".

### Two implementation paths

**Path A: Offline experiments**

Run grid search over common artist combinations, recording per-artist "style strength vs computational cost" curves across different sigma ranges. Build a lookup table.

```python
# pseudo-code
ARTIST_OPTIMAL_RANGE = {
    "artist_a": (0.0, 0.3),  # early-only
    "artist_b": (0.0, 1.0),  # full range
    "artist_c": (0.5, 1.0),  # late-only
    ...
}
```

Implementation cost: build a benchmark with dozens to hundreds of artists, run an evaluation pipeline, requires subjective scoring or automated metrics.

**Path B: Learned injector**

Train a lightweight neural network:

```
input: (artist embedding, current sigma, current layer idx)
output: per-artist injection weight in [0, 1]
```

Requires labeled training data (e.g. "schedule X produced better outputs than schedule Y"). Significant work.

### Why not done

Engineering scope exceeds single-plugin scale. Listed for future contributors to know the theoretical ceiling.

---

## 5. Other directions considered but not recommended

### Cross-layer base_out sharing

Not feasible: each layer's base_out depends on that layer's x; different layers have different x and can't share.

### Cross-cond/uncond base_out sharing

Not feasible: cond and uncond have different K/V (cond is the main prompt, uncond is the negative), so outputs differ inherently.

### Use flash-attn or similar specialized ops

ComfyUI's `optimized_attention` is already a wrapper around SDPA / xformers / flash-attn, automatically choosing the fastest available. No headroom here.

### Async / pipelining

PyTorch already does this by default. Manual control yields negligible gain.

---

## General cache invalidation principles

Any caching strategy needs to consider these invalidation conditions:

```python
def should_invalidate_cache(state, current_call_info):
    # 1) Artist conditioning changed
    if state.get("cache_version") != id(state["individuals"]):
        return True
    # 2) Input shape changed (resolution switch)
    if cache_shape_mismatch(state["cache"], current_call_info["x_shape"]):
        return True
    # 3) cond_or_uncond order changed (rare, defensive)
    if state.get("cache_cou") != current_call_info["cou"]:
        return True
    return False
```

`add_object_patch` re-instantiates wrappers on each patch, so "workflow re-run" handles itself. But within the same wrapper instance across multiple calls, you must manage invalidation manually.

---

## Debugging recommendations

Caching makes debugging harder because visual errors aren't always immediately obvious. Suggestions:

1. **A/B comparison**: same seed, run once with optimization off and once with it on. Output should be near-identical (allow ~1% diff for fp16 accumulation order)
2. **Layer-wise output diff**: in the wrapper, add a debug mode that dumps key intermediate tensors to disk. Compare diffs before/after optimization
3. **Numerical monitoring**: track L2 norm of cached outputs across steps to detect numerical drift

---

## Roadmap suggestion

By payoff-vs-cost priority:

1. (Implemented) Batched parallel forward
2. (Implemented) `start_percent / end_percent` as practical defaults
3. (Recommended next) Direction 1: attention-output deferred cache
4. (When time allows) Direction 2: K/V projection cross-step cache
5. (Experimental) Direction 3: similar-artist merging
6. (Research) Direction 4: adaptive injection schedule

Issues / PRs welcome.
