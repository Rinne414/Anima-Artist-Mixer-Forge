# Anima-Artist-Mixer

A ComfyUI custom node that enables **multi-artist mixing** for the Anima model by hooking into its cross-attention layers.

![surtr](docs/images/ComfyUI_01092_.png)
## What it does

Anima uses an LLM as its text encoder. When multiple artist tags are stacked in a single prompt, the LLM's contextualization causes them to interfere with each other, producing a conditioning that resembles neither artist clearly. This node encodes each artist independently and mixes them at the model's cross-attention layer, sidestepping the interference at the prompt-encoding stage.

The bundled `AnimaArtistPack` node provides a one-shot experience: write your artist list (separated by commas or newlines) in one text box, your main prompt in another, and the node handles splitting, encoding, and packaging automatically.

The current release (v26) adds negative artist weights (style subtraction), smoothstep timing fades (`%0.0-0.45~0.1`), stronger style-drift reduction via token-level base-energy norm locking (`match_base_norm`, `norm_lock_mode`, `norm_lock_scope`), the content-safer `stable_seed` static-capture preset, prompt-aware `drift_auto` routing plus scene-tuned low-drift presets (`drift_soft`, `face_lock`, `scene_lock`), a stronger legacy `anchor_lock` preset, a per-layer style probe that measures where each artist actually lives in the model, shareable JSON recipes, VRAM controls (`max_batch_artists`, `low_vram_cache`), a CFG correctness fix for batch sizes > 1, and a full package restructure with tests and CI. Earlier releases added one-click presets, UX helper nodes, an in-UI inspector, deterministic low-rank mixing, layered cross-seed stabilizers, CFG-style strength extrapolation, the linear injection-layer weight syntax `::name::weight`, per-artist layer/timing routing, and a compatibility-safe preset. See [CHANGELOG.md](CHANGELOG.md).

## Quick links

- [Full documentation](docs/USAGE.md) — usage, parameters, modes, stabilizers, performance tips
- [Changelog](CHANGELOG.md) — version history
- [Issues](../../issues) — bug reports, feature requests
- [Discussions](../../discussions) — usage questions, results sharing

## Installation

Clone or download into your ComfyUI `custom_nodes` directory:

```
ComfyUI/custom_nodes/<this-plugin-folder>/
```

Restart ComfyUI. No extra dependencies.

## Requirements

- **Anima model only** — depends on Anima's built-in `LLMAdapter` (`preprocess_text_embeds`)
- Use the **same CLIP loader** that Anima's own text-encoding workflow uses (the one whose tokens carry `t5xxl_ids`)
- Inference only

## Quick start
![workflow](docs/images/workflow.png)

Open [`workflow/Shift testing.before-basic-simplify.json`](<workflow/Shift testing.before-basic-simplify.json>)
for a complete importable example. It keeps the real generation/output chain
and shows the recommended `AnimaArtistPreset(preset = drift_auto)` wiring
without requiring users to build the workflow from scratch.

- Top text box of `AnimaArtistPack`: your artist chain (comma or newline separated)
- Fastest first run: use `AnimaArtistStarter`, fill `artist_table`, then follow its in-UI guide
- Use `AnimaArtistChainBuilder` when you do not want to hand-write `::weight`, `@layers`, and `%timing`
- Builder's three visible rows are only shortcuts; use its `artist_table` field for larger chains
- Use `AnimaArtistChainPreview` to validate a chain before paying the CLIP encoding cost
- Bottom text box: the main prompt (no need to repeat artist names here)
- Wire `AnimaArtistCrossAttn`'s `base_prompt` output directly to KSampler's positive input
- For a sane first run, connect `AnimaArtistPreset` with `preset = balanced`
- If the workflow also uses regional prompting, Forge Couple-style routing, or other attention patchers, start with `preset = compatibility_safe`
- When a workflow behaves strangely, connect `AnimaArtistInspector` and read the effective weights / warnings directly in ComfyUI

For full parameter explanations and recommended combinations, see [docs/USAGE.md](docs/USAGE.md).

## Recommended defaults

For most users, start with:

```
AnimaArtistStarter:
recipe    = balanced
layout    = layer_scheduled

or:

AnimaArtistPreset:
preset    = balanced
intensity = 1.0
```

Manual equivalent:

```
combine_mode = output_avg
fusion_mode  = interpolate
strength     = 1.0
artist_ema_alpha = 0.25
```

To weight individual artists within the chain, use either of two syntaxes (they can coexist and stack):

```
wlop, ::sakimichan::1.2, (krenz:0.7), ::pixiv_style::-0.4
```

- `(name:1.2)` — CLIP-side weighting (same as SD/A1111), non-linear, applied at text encoding
- `::name::1.2` — injection-side weighting (v24+), linear and predictable, applied at cross-attention output
- `::name::-0.4` — **negative weight (v26+)**: subtracts that artist's style direction instead of adding it (style subtraction); range is [-4, 4]
- In v25+, any valid `::weight` automatically disables normalization at runtime so explicit weights stay absolute
- Per-artist layer routing is supported with `@layers`: `wlop@0-8, krenz@9-18, hiten@19-27`
- Per-artist sampling timing is supported with `%start-end`: `wlop@0-8%0.0-0.45, krenz@9-18%0.45-0.85`
- Timing windows can fade in/out smoothly with `~fade` (v26+): `wlop%0.0-0.45~0.1` ramps the artist's weight with a smoothstep over a 0.1-progress-wide edge instead of switching on/off abruptly
- Anima artist tags that start with `@` are safe: `@wlop` remains the artist name; only a final numeric suffix like `@0-8` is treated as layer routing

## Compatibility notes

This node wraps Anima cross-attention. Other nodes that also patch attention, regional prompts, Forge Couple-style routing, or model forward wrappers can change the same execution path. If the artist effect disappears or becomes very weak, use `AnimaArtistPreset(preset = compatibility_safe)` first. It forces the tolerant `concat + concat_with_base` path and disables cache-heavy stabilizers. Use `AnimaArtistInspector` to confirm parsed artists, weights, layer routes, timing routes, block map, and effective normalize state.

## Cross-seed stability

In multi-artist setups, the same prompt with different seeds tends to produce noticeably different style mixes — sometimes one artist dominates, other times another, even at equal weights. This is structural to how cross-attention interacts with seed-driven hidden state.

v26 keeps norm locking on by default and provides four optional stabilizers via `AnimaArtistOptions`, ordered from light to heavy:

| Stabilizer | Strength | Notes |
|---|---|---|
| `match_base_norm` + `norm_lock_mode=token` + `norm_lock_scope=per_artist` | default | Per-artist token RMS lock; reduces seed-specific style-strength spikes before artists are mixed |
| `artist_ema_alpha` | light | Temporal EMA across sampling steps |
| `combine_mode = lowrank_avg` + `lowrank_k` | medium | Deterministic low-rank constraint on multi-artist deltas |
| `artist_static_capture` + `static_capture_k` | heavy | Freeze artist attention after K warmup steps; `stable_seed` uses K=4 with auto layers 9-20 and norm lock disabled. Advanced `static_capture_mode` values include `output`, `delta`, `blend`, and `blend_perp`; `output` remains the measured default. |
| `contribution_balance` | optional | Delta-strength equalizer for artist dominance flips; default off because static capture was more reliable in live A/B |
| `mixed_delta_cap` | optional | Caps the final mixed artist delta against base attention energy before fusion; default off while it is evaluated as a live A/B candidate |
| `artist_anchor_q` | heaviest | Replace user-seed Q with a fixed-seed anchor's Q; `anchor_lock` keeps the legacy 4-anchor path with auto layers 9-25 and user-Q handoff at L16 |
| `anchor_base_norm_ref` | optional | Anchor the norm reference too when testing `anchor_q + match_base_norm`; useful for A/B, not the measured default |

Recommended progression: keep the default token/per-artist norm lock on, then use `stable_seed` for content-safer cross-seed work. For lower drift, try `drift_auto` first; it routes 4+ artist wide / background-heavy scenes to `face_lock`, smaller explicit wide / background-heavy scenes to `scene_lock`, 4+ artist simple fullbody prompts to `drift_soft`, 4+ artist close-ups to `stable_seed` plus `mixed_delta_cap_ratio=0.75`, 4+ artist street / urban prompts to `compatibility_safe`, other 4+ artist portrait / broad-subject prompts to the internal `compatibility_safe_9_15` route, smaller close-up face prompts to `face_lock`, and the remaining portrait / broad-subject and plain street/fullbody prompts to `drift_soft` from the `base_prompt` and artist count. If a known artist set still shows large seed-to-seed swings, A/B `mixed_delta_cap` at ratios around `0.75-1.0` before changing presets globally. Use the manual variants when you already know the prompt type, and use `anchor_lock` only when you explicitly want the stronger anchor-Q lock. See [docs/USAGE.md](docs/USAGE.md) for detailed mechanics and tuning.

## Style amplification

`strength` accepts values in `[0, 4]`:

- `0 ~ 1` — interpolation between base and artist (`strength=1` = pure artist replacement)
- `1 ~ 4` — CFG-style extrapolation: `out = base + strength * (artist - base)`, amplifying the artist's deviation from base for stronger style

`1.5 ~ 2.5` is a common range for "stronger style without breaking content"; pushing past `3` tends to oversaturate.

## Performance notes

Generation time scales with artist count. Per the math of `output_avg`, each layer runs `N + 1` cross-attention forwards (N artists + base). Approximate measured cost (varies by GPU):

| Configuration | Relative time |
|---|---|
| 1 artist | 1.0x |
| 4 artists | ~1.4x |
| 8 artists | ~1.7x |
| 5 artists + `artist_static_capture` (K=6) | ~1.1x |
| 5 artists + `artist_anchor_q` (cached) | ~1.05x |

**Strongly recommended**: connect `AnimaArtistOptions` and limit either the layer range (`start_block / end_block`) or the sampling-step range (`start_percent / end_percent`). Both can dramatically reduce generation time with minimal quality loss, and stack with the cache-based stabilizers above. With many artists at high resolution, set `max_batch_artists` (2-8) to bound peak VRAM and `low_vram_cache` to keep stabilizer caches in system RAM. See the docs for details.

## Measuring where styles live (v26)

Instead of guessing `@layers` routes, wire `AnimaArtistProbe` between your model loader and the sampler, run one generation, and read `AnimaArtistProbeReport` (connect any post-sampler output as its trigger). The report shows each artist's per-layer style influence (`||artist_out − base_out|| / ||base_out||`) as a bar chart and suggests a concrete `artist@lo-hi` route per artist. The probe pass does not alter the generated image.

## Sharing recipes (v26)

`AnimaArtistRecipeSave` packs the artist chain plus the full effective configuration (combine/fusion/strength/advanced options) into one JSON string; `AnimaArtistRecipeLoad` turns it back into `artist_chain` + a `preset` payload you can wire straight into `AnimaArtistCrossAttn`. Paste-friendly for sharing exact mixes with other users.

## Important caveat

This node **cannot achieve the near-lossless artist mixing that SDXL does**. Anima's text encoder is non-linear, so any mixing strategy introduces some distortion. What this node does is make that distortion controllable. Style-similar artists mix well; style-divergent artists may "regress to the mean" into a compromise look — `lowrank_avg` accepts more of this regression in exchange for cross-seed stability.

## Development

The implementation lives in the `anima_mixer/` package (`nodes.py` is a compatibility shim). Run the test suite with:

```
python -m unittest discover -s tests -v
```

CI (ruff + unittest on Python 3.10/3.12) runs on every push and PR.

## Acknowledgements

Special thanks to **汐浮尘/utowo** for co-development, testing, and design contributions. The `AnimaArtistPack` split-and-encode design comes from their improvement.

## License

MIT License. See [LICENSE](LICENSE) for the full text.
