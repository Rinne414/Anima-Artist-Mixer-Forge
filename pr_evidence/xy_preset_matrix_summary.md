# XY preset matrix evidence

- Source workflow: `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\source_Anima.Style.Mixer.json`
- Server: `http://127.0.0.1:8190`
- Size: `1024x1024`
- Steps / CFG / sampler / scheduler: `32` / `5.0` / `er_sde` / `beta`
- Seed: `1098716302142360`
- Metrics: `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix.metrics.json`
- Grid part 1: `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_part1.png`
- Grid part 2: `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_part2.png`

Red column labels in the PNG grids mark Anima preset columns.

Model substitutions:
- `UNETLoader`: `anima-base-v1.0.safetensors` -> `Anima\anime\anima_baseV10.safetensors`

Preset summaries:

| preset | combine | fusion | strength | layer filter | stabilizer |
|---|---|---|---:|---|---|
| prompt_passthrough | output_avg | interpolate | 0.00 | all | direct prompt |
| balanced | output_avg | interpolate | 1.00 | all | - |
| strong_style | output_avg | interpolate | 1.65 | all | EMA 0.20 |
| stable_seed | output_avg | interpolate | 1.00 | 9-20 | delta cap 0.75 |
| drift_auto | output_avg | interpolate | 0.85 | 9-20 | EMA 0.12 |
| drift_soft | output_avg | interpolate | 0.85 | 9-20 | EMA 0.12 |
| face_lock | output_avg | base_preserve | 0.90 | 9-20 | norm lock, delta cap 1.00 |
| scene_lock | output_avg | base_preserve | 0.85 | 9-15 | EMA 0.10 |
| anchor_lock | output_avg | interpolate | 0.90 | 9-15 | anchor Q |
| fast_preview | concat | concat_with_base | 1.00 | all | - |
| identity_guard | output_avg | base_preserve | 0.85 | all | EMA 0.12, norm lock, delta cap 0.90 |
| compatibility_safe | concat | concat_with_base | 1.00 | all | compatibility |

| row | column | seconds | image |
|---|---|---:|---|
| 1 artist | no mixer | 40.3 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__no_mixer.png` |
| 1 artist | prompt_passthrough | 35.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__prompt_passthrough.png` |
| 1 artist | original mixer | 41.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__original_mixer.png` |
| 1 artist | balanced | 35.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__balanced.png` |
| 1 artist | strong_style | 33.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__strong_style.png` |
| 1 artist | stable_seed | 35.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__stable_seed.png` |
| 1 artist | drift_auto | 35.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__drift_auto.png` |
| 1 artist | drift_soft | 35.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__drift_soft.png` |
| 1 artist | face_lock | 37.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__face_lock.png` |
| 1 artist | scene_lock | 31.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__scene_lock.png` |
| 1 artist | anchor_lock | 32.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__anchor_lock.png` |
| 1 artist | fast_preview | 32.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__fast_preview.png` |
| 1 artist | identity_guard | 36.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__identity_guard.png` |
| 1 artist | compatibility_safe | 27.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\1_artists__compatibility_safe.png` |
| 2 artists | no mixer | 31.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__no_mixer.png` |
| 2 artists | prompt_passthrough | 33.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__prompt_passthrough.png` |
| 2 artists | original mixer | 44.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__original_mixer.png` |
| 2 artists | balanced | 43.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__balanced.png` |
| 2 artists | strong_style | 42.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__strong_style.png` |
| 2 artists | stable_seed | 38.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__stable_seed.png` |
| 2 artists | drift_auto | 37.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__drift_auto.png` |
| 2 artists | drift_soft | 37.3 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__drift_soft.png` |
| 2 artists | face_lock | 35.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__face_lock.png` |
| 2 artists | scene_lock | 28.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__scene_lock.png` |
| 2 artists | anchor_lock | 29.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__anchor_lock.png` |
| 2 artists | fast_preview | 29.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__fast_preview.png` |
| 2 artists | identity_guard | 39.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__identity_guard.png` |
| 2 artists | compatibility_safe | 28.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\2_artists__compatibility_safe.png` |
| 4 artists | no mixer | 26.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__no_mixer.png` |
| 4 artists | prompt_passthrough | 27.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__prompt_passthrough.png` |
| 4 artists | original mixer | 41.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__original_mixer.png` |
| 4 artists | balanced | 42.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__balanced.png` |
| 4 artists | strong_style | 39.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__strong_style.png` |
| 4 artists | stable_seed | 41.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__stable_seed.png` |
| 4 artists | drift_auto | 41.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__drift_auto.png` |
| 4 artists | drift_soft | 38.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__drift_soft.png` |
| 4 artists | face_lock | 35.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__face_lock.png` |
| 4 artists | scene_lock | 32.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__scene_lock.png` |
| 4 artists | anchor_lock | 32.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__anchor_lock.png` |
| 4 artists | fast_preview | 29.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__fast_preview.png` |
| 4 artists | identity_guard | 49.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__identity_guard.png` |
| 4 artists | compatibility_safe | 29.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\4_artists__compatibility_safe.png` |
| 10 artists | no mixer | 27.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__no_mixer.png` |
| 10 artists | prompt_passthrough | 27.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__prompt_passthrough.png` |
| 10 artists | original mixer | 64.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__original_mixer.png` |
| 10 artists | balanced | 64.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__balanced.png` |
| 10 artists | strong_style | 62.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__strong_style.png` |
| 10 artists | stable_seed | 43.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__stable_seed.png` |
| 10 artists | drift_auto | 44.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__drift_auto.png` |
| 10 artists | drift_soft | 52.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__drift_soft.png` |
| 10 artists | face_lock | 61.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__face_lock.png` |
| 10 artists | scene_lock | 45.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__scene_lock.png` |
| 10 artists | anchor_lock | 45.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__anchor_lock.png` |
| 10 artists | fast_preview | 33.1 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__fast_preview.png` |
| 10 artists | identity_guard | 76.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__identity_guard.png` |
| 10 artists | compatibility_safe | 41.2 | `L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence\xy_preset_matrix_cells\10_artists__compatibility_safe.png` |
