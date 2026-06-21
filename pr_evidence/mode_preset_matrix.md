# Mode and preset function matrix

## Core modes

| Name | Type | Function | Use when | Cost shape |
|---|---|---|---|---|
| `output_avg` | combine mode | Runs each artist condition separately, then averages cross-attention outputs. | Default quality path; best regression-safe behavior. | More forwards, roughly scales with artist count. |
| `concat` | combine mode | Concatenates artist conditioning into one longer context. | Fast preview or compatibility with other patchers. | Fastest. |
| `lowrank_avg` | combine mode | Averages artist deltas after deterministic low-rank projection. | You want more cross-seed stability and accept more style compromise. | Similar to `output_avg` plus SVD. |
| `interpolate` | fusion mode | Blends base and artist output with `strength`. | Default smooth control. | No special overhead. |
| `concat_with_base` | fusion mode | Sends base and artist context together to attention. | Compatibility-safe path with `concat`. | Fast. |
| `base_preserve` | fusion mode | Adds only style direction less aligned with base content. | Face/content preservation, identity-sensitive prompts. | Similar to output mix path. |

## Presets

| Preset | Function | Internal route | Best use |
|---|---|---|---|
| `prompt_passthrough` | Direct prompt/no-mixer parity with positive artist weights preserved as prompt weights. | No attention patch; returns direct prompt conditioning and the unpatched model. | Reviewer comparisons, control images, users who want no mixer but still want `1.2::tag::` convenience. |
| `balanced` | Original-style default. Keeps the mixer predictable and close to the old behavior. | `output_avg + interpolate`, strength `1.0`, EMA off, norm lock off. | First run, regression comparison, normal artist mixing. |
| `strong_style` | Stronger visible style without changing the core path. | `output_avg + interpolate`, strength about `1.65`, light EMA, end percent `0.92`. | Artist effect is too weak. |
| `stable_seed` | Reduces seed-to-seed style swings without freezing style capture. | `output_avg + interpolate`, mixed delta cap ratio `0.75`, strength `1.0`, auto layer `9-20`. | Batch generations where style consistency matters. |
| `drift_soft` | Softer low-drift preset. | `output_avg + interpolate`, strength `0.85`, light EMA `0.12`, auto layer `9-20`. | Simple portraits/fullbody prompts, especially with several artists. |
| `face_lock` | Preserves face/content while applying style. | `output_avg + base_preserve`, token norm lock, mixed delta cap, strength `0.90`, auto layer `9-20`. | Close-up, face focus, detailed eyes. |
| `scene_lock` | Preserves wider scene/background structure. | `output_avg + base_preserve`, light EMA `0.10`, strength `0.85`, auto layer `9-15`. | Wide shots, scenery, background-heavy prompts. |
| `drift_auto` | Runtime router for prompt/artist-count cases. | Chooses `drift_soft`, `stable_seed`, `face_lock`, or `scene_lock` by `base_prompt` and artist count. | Users who want lower drift without picking a specific route. |
| `anchor_lock` | Strong fixed-anchor stabilizer. | `output_avg + interpolate`, single anchor Q, user-Q blend `0.35`, strength `0.90`, auto layer `9-15`. | More consistency, accepting more content constraint. |
| `fast_preview` | Quick preview path. | `concat + concat_with_base`, end percent `0.82`. | Prompt iteration before final settings. |
| `identity_guard` | Conservative content/identity protection. | `output_avg + base_preserve`, EMA `0.12`, norm lock, mixed delta cap ratio `0.90`, strength `0.85`. | Character/identity-sensitive prompts. |
| `compatibility_safe` | Minimal patch interaction surface. | `concat + concat_with_base`, stabilizers disabled. | Regional prompting, Forge Couple-style routing, or other attention/model patch nodes. |

## Feature controls

| Feature | Function | Example |
|---|---|---|
| Explicit positive weight | Scales an artist direction in attention-output space. | `1.2::@yuchi \(salmon-1000\)::` |
| Negative weight | Subtracts an artist direction. | `::@artist::-0.5` |
| Layer routing | Restricts an artist to selected DiT blocks. | `@artist@9-20`, `@artist@33%-67%` |
| Timing routing | Restricts an artist to a sampling-progress window. | `@artist%0.0-0.45` |
| Timing fade | Smoothly ramps a timing window at the edges. | `@artist%0.0-0.45~0.1` |
| `max_batch_artists` | Caps batched artist width for peak VRAM. | expert option |
| `low_vram_cache` | Stores static/anchor caches in system RAM. | expert option |
| `stabilizer_end_percent` | Lets EMA/static/anchor stabilizers yield after an early sampling window. | `0.5` keeps early style lock, restores late-step motion |
| Recipe save/load | Serializes a whole mixer setup into JSON. | `AnimaArtistRecipeSave` / `AnimaArtistRecipeLoad` |
| Probe | Measures per-layer artist influence without changing the image. | `AnimaArtistProbe` / `AnimaArtistProbeReport` |

`prompt_passthrough` is intentionally not a mixer. It supports positive artist weights by converting them into normal prompt weights like `(@artist:1.2)`. Negative style subtraction, per-artist layer routes, and per-artist timing routes require a mixer preset such as `balanced`.

## Evidence files

| File | Purpose |
|---|---|
| `compare-pr4-reported-issues.png` | Visual answer for the two reported reviewer issues. |
| `fresh-multi-artist-matrix.png` | Fresh 1/2/4 artist prompt vs balanced vs drift_auto result sheet. |
| `fresh-multi-artist-matrix.metrics.json` | Machine-readable timings and metrics for the fresh multi-artist matrix. |
| `mode-preset-value-summary.png` | Visual summary of preset functions and timing results. |
| `pr4_reported_issues.metrics.json` | Machine-readable metrics behind the reported-issue evidence. |
| `compare-1024-32-single-yuchi.png` | Current/original/direct-prompt comparison for single yuchi, 1024px, 32 steps. |
| `compare-1024-32-preset-two-sampler-yuchi.png` | Same-model two-KSampler success evidence. |
| `compare-1024-16-single-double-multi.png` | Broader single/double/multi artist comparison. |
