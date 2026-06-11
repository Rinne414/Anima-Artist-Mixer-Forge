# Changelog

## v26.0.0 (2026-06-11)

### Restructure
- Split the 2,600-line `nodes.py` into the `anima_mixer` package
  (`constants`, `parsing`, `math_utils`, `options`, `chain_tools`,
  `patching`, `wrapper`, `anchor`, `recipe`, `nodes_core`, `nodes_ui`).
  `nodes.py` remains as a backward-compatibility shim.
- All log messages, error messages, and tooltips are now in English.
- Added `pyproject.toml` with Comfy Registry metadata and a GitHub Actions
  CI workflow (ruff + unittest on Python 3.10/3.12).

### Fixes
- **CFG mask (HIGH)**: when ComfyUI batches several latents per cond entry,
  `cond_or_uncond` markers are now expanded over the row chunks instead of
  falling back to injecting into every row (which silently styled the
  uncond pass and weakened CFG for batch sizes > 1).
- **Anchor cache key**: replaced `id(context)` with a content-based
  fingerprint (shape + dtype + value checksum). A freed tensor's id can be
  reused, which could silently re-hit a stale anchor cache.
- **Anchor pre-run no longer re-runs every step**: the cache check now fires
  only at the start of a sampling run (sigma jump). Previously the cache key
  contained the *current* step's sigma, so it missed on every step and the
  anchor pre-pass silently re-ran each step, making `anchor_q` far more
  expensive than documented.
- **Anchor pre-run hardening**: the private `dm._forward` API is now used
  only when present, with a public-call fallback.
- **`enabled=False` early return**: the patch node now returns the
  unpatched model immediately instead of installing wrappers that check a
  flag on every forward.
- **concat + static_capture**: the combined path now uses the same K-step
  temporal averaging as `output_avg` instead of a first-step-only snapshot.
- **Inspector**: accepts an optional MODEL input to read the real block
  count instead of assuming 28.

### Features
- **Timing fade** — `%start-end~fade` adds smoothstep ramps at the edges of
  per-artist timing windows, removing hard style pops at window boundaries:
  `wlop%0.0-0.45~0.1`.
- **Negative weights (style subtraction)** — `::artist::-0.5` pushes a
  style away instead of adding it. Weight range is now [-4, 4].
- **`embed_avg` combine mode** — mixes in LLMAdapter embedding space with a
  single extra forward per layer regardless of artist count. The fastest
  mode; trades some per-artist fidelity.
- **`max_batch_artists` option** — caps how many artists share one batched
  forward, bounding peak VRAM with many artists at high resolution.
- **`low_vram_cache` option** — stores static-capture and anchor caches in
  system RAM instead of VRAM.
- **Recipe nodes** — `AnimaArtistRecipeSave` / `AnimaArtistRecipeLoad`
  serialize a full mixer setup (chain + modes + options) to a shareable
  JSON string.
- **Layer probe** — `AnimaArtistProbe` + `AnimaArtistProbeReport` measure
  each artist's per-layer style influence during a sampling run and suggest
  `@layers` routes, replacing guesswork with measurement.

### Tests
- Real-torch test suite: low-rank determinism, perpendicular projection,
  fusion math, CFG mask expansion, timing fade factors, embed averaging,
  chunking, anchor fingerprints, recipe round-trips (57 tests).

---

## v25.2
- Per-artist sampling timing (`%start-end`), `compatibility_safe` preset,
  Inspector block maps, runtime warnings for suspicious cross-attention /
  model-wrapper conflicts, and UX helper nodes (Starter, Chain Builder,
  Chain Preview) for building chains before CLIP encoding.

## v25.1
- Per-artist layer routing (`@layers`): different artists can inject into
  different DiT block ranges from the same chain.

## v25
- Fixed `::name::weight` explicit weights not actually disabling
  `normalize_weights` on the patch path.
- Added `AnimaArtistPreset` (balanced / strong_style / stable_seed /
  fast_preview / identity_guard) and `AnimaArtistInspector`.
- `lowrank_avg` switched to a deterministic Gram eigendecomposition
  (no randomized SVD approximation).
- Anchor cache key gained the first timestep/sigma to reduce stale reuse
  after sampling-condition changes.

## v24
- New `::name::weight` chain syntax: a linear weight applied at the
  cross-attention injection layer (vs. the non-linear CLIP-side parentheses
  syntax). Any explicit `::weight` bypasses `normalize_weights`.

## v23
- `strength` upper bound raised from 1.0 to 4.0; values above 1.0 enter
  CFG-style extrapolation `out = base + strength * (artist - base)`.

## v22
- Anchor-Q tuning: `anchor_seeds_count` (multi-seed anchor averaging),
  `anchor_user_blend` (anchor/user Q blend), and
  `anchor_deep_layer_threshold` (shallow-anchor / deep-user split).

## v21
- Anchor-Q: replace the user-seed hidden state with a fixed-seed anchor's
  hidden state as the artist attention Q, decoupling style mixing from the
  user seed.

## v20
- `static_capture_k` made configurable (default 6, was hardcoded 3).

## v18-v19
- Static capture: freeze artist attention outputs after the first K steps
  (cross-seed stabilization + 30-50% speedup).

## v17
- Re-added `base_preserve` fusion mode (perpendicular-only artist deltas);
  EMA stabilizer (`artist_ema_alpha`); `lowrank_avg` combine mode.
