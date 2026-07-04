"""Live ComfyUI smoke test for Anima-Artist-Mixer v26.

Submits a matrix of real sampling workflows against a running ComfyUI and
checks that each completes without node errors and produces an image. This is
a manual integration harness, not a unit test (it needs a GPU, the Anima
model, and a live server) — run it directly:

    python tests/live_comfy_smoke.py
"""

import json
import os
import time
import urllib.error
import urllib.request

SERVER = os.environ.get("ANIMA_SMOKE_SERVER", "http://127.0.0.1:8188")

# --- base pipeline models (discovered on the target machine) ----------------
UNET = "Anima\\anime\\anima_baseV10.safetensors"
CLIP = "qwen_3_06b_base.safetensors"
VAE = "qwen_image_vae.safetensors"

BASE_PROMPT = (
    "1girl, solo, masterpiece, best quality, upper body portrait, face visible, "
    "wearing a white blouse and navy jacket, looking at viewer, simple background"
)
NEG_PROMPT = (
    "nsfw, nude, naked, bare chest, cleavage, nipples, cropped head, "
    "head out of frame, lowres, worst quality, bad anatomy"
)
ARTISTS = "@uof, @kieed, @ciloranko"

WIDTH, HEIGHT, STEPS, CFG, SEED = 512, 512, 8, 5.0, 42


def _post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER + path, data=data, headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=30))


def _get(path):
    return json.load(urllib.request.urlopen(SERVER + path, timeout=30))


def default_opts(**over):
    o = {
        "start_block": 0, "end_block": -1,
        "start_percent": 0.0, "end_percent": 1.0,
        "normalize_weights": True, "artist_ema_alpha": 0.0,
        "lowrank_k": 1, "artist_static_capture": False,
        "static_capture_k": 6, "static_capture_mode": "output",
        "static_capture_blend_alpha": 0.25,
        "artist_anchor_q": False,
        "anchor_seeds_count": 1, "anchor_user_blend": 0.0,
        "anchor_deep_layer_threshold": -1, "anchor_refresh_each_step": False,
        "stabilizer_end_percent": 1.0,
        "layer_filter": "",
        "compatibility_mode": False, "max_batch_artists": 0,
        "low_vram_cache": False, "match_base_norm": True,
        "anchor_base_norm_ref": False,
        "norm_lock_mode": "token", "norm_lock_scope": "per_artist",
        "contribution_balance": False, "contribution_balance_alpha": 1.0,
        "mixed_delta_cap": False, "mixed_delta_cap_ratio": 1.0,
    }
    o.update(over)
    return o


def base_loaders():
    """Nodes shared by every graph: loaders, latent, negative encode."""
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "stable_diffusion",
                         "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG_PROMPT, "clip": ["2", 0]}},
        "8": {"class_type": "EmptyLatentImage",
              "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "anima_test"}},
    }


def sampler(batch_size=1):
    return {
        "class_type": "KSampler",
        "inputs": {
            "model": ["6", 0], "positive": ["6", 1], "negative": ["7", 0],
            "latent_image": ["8", 0], "seed": SEED, "steps": STEPS,
            "cfg": CFG, "sampler_name": "er_sde", "scheduler": "beta",
            "denoise": 1.0,
        },
    }


def build_sampling_graph(chain, combine="output_avg", fusion="interpolate",
                         strength=1.0, opts=None, use_preset=None,
                         batch_size=1):
    g = base_loaders()
    g["8"]["inputs"]["batch_size"] = batch_size
    g["4"] = {"class_type": "AnimaArtistPack",
              "inputs": {"clip": ["2", 0], "artist_chain": chain,
                         "base_prompt": BASE_PROMPT}}
    cross_inputs = {
        "model": ["1", 0], "artist_pack": ["4", 0],
        "combine_mode": combine, "fusion_mode": fusion,
        "strength": strength, "enabled": True, "apply_to_uncond": False,
    }
    if use_preset is not None:
        g["5"] = {"class_type": "AnimaArtistPreset",
                  "inputs": {"preset": use_preset, "intensity": 1.0,
                             "normalize_weights": True, "layer_mode": "auto",
                             "custom_layer_filter": ""}}
        cross_inputs["preset"] = ["5", 0]
    else:
        g["5"] = {"class_type": "AnimaArtistOptions",
                  "inputs": opts or default_opts()}
        cross_inputs["advanced_options"] = ["5", 0]
    g["6"] = {"class_type": "AnimaArtistCrossAttn", "inputs": cross_inputs}
    g["9"] = sampler(batch_size)
    return g


def build_probe_graph(chain):
    g = base_loaders()
    g["4"] = {"class_type": "AnimaArtistPack",
              "inputs": {"clip": ["2", 0], "artist_chain": chain,
                         "base_prompt": BASE_PROMPT}}
    g["6"] = {"class_type": "AnimaArtistProbe",
              "inputs": {"model": ["1", 0], "artist_pack": ["4", 0],
                         "probe_steps": STEPS}}
    g["9"] = sampler()
    g["12"] = {"class_type": "AnimaArtistProbeReport",
               "inputs": {"probe_id": ["6", 2], "trigger": ["10", 0]}}
    return g


def build_recipe_graph():
    """RecipeSave -> RecipeLoad -> Pack/CrossAttn round-trip through real nodes."""
    g = base_loaders()
    g["20"] = {"class_type": "AnimaArtistRecipeSave",
               "inputs": {"artist_chain": ARTISTS, "combine_mode": "output_avg",
                          "fusion_mode": "interpolate", "strength": 1.2,
                          "notes": "live smoke"}}
    g["21"] = {"class_type": "AnimaArtistRecipeLoad",
               "inputs": {"recipe_json": ["20", 0]}}
    g["4"] = {"class_type": "AnimaArtistPack",
              "inputs": {"clip": ["2", 0], "artist_chain": ["21", 0],
                         "base_prompt": BASE_PROMPT}}
    g["6"] = {"class_type": "AnimaArtistCrossAttn",
              "inputs": {"model": ["1", 0], "artist_pack": ["4", 0],
                         "combine_mode": "output_avg", "fusion_mode": "interpolate",
                         "strength": 1.2, "enabled": True,
                         "apply_to_uncond": False,
                         "advanced_options": ["21", 2]}}
    g["9"] = sampler()
    return g


def submit_and_wait(name, graph, timeout=240):
    t0 = time.time()
    try:
        resp = _post("/prompt", {"prompt": graph})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:600]
        return name, "SUBMIT_FAIL", body
    if resp.get("node_errors"):
        return name, "NODE_ERRORS", json.dumps(resp["node_errors"])[:600]
    pid = resp["prompt_id"]
    while time.time() - t0 < timeout:
        hist = _get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                detail = ""
                for m in msgs:
                    if m[0] in ("execution_error", "execution_interrupted"):
                        detail = json.dumps(m[1])[:600]
                return name, "EXEC_ERROR", detail or json.dumps(msgs)[:600]
            imgs = []
            for out in entry.get("outputs", {}).values():
                for img in out.get("images", []):
                    imgs.append(img["filename"])
            texts = []
            for out in entry.get("outputs", {}).values():
                if "text" in out:
                    texts.append(str(out["text"])[:200])
            dt = time.time() - t0
            tag = "OK" if imgs or texts else "NO_OUTPUT"
            return name, tag, f"{dt:.1f}s imgs={imgs} text={texts}"
        time.sleep(2)
    return name, "TIMEOUT", f">{timeout}s"


TESTS = [
    ("01 baseline output_avg",
     lambda: build_sampling_graph(ARTISTS)),
    ("02 fade timing",
     lambda: build_sampling_graph("@uof%0.0-0.5~0.1, @kieed%0.4-1.0~0.1")),
    ("03 negative weight",
     lambda: build_sampling_graph("::@uof::1, ::@kieed::-0.5")),
    ("04 output_avg explicit weights",
     lambda: build_sampling_graph("::@uof::1.5, ::@kieed::0.7, @ciloranko")),
    ("05 lowrank_avg",
     lambda: build_sampling_graph(ARTISTS, combine="lowrank_avg",
                                  opts=default_opts(lowrank_k=2))),
    ("06 concat + base_preserve",
     lambda: build_sampling_graph(ARTISTS, combine="concat",
                                  fusion="base_preserve")),
    ("07 preset compatibility_safe",
     lambda: build_sampling_graph(ARTISTS, use_preset="compatibility_safe")),
    ("08 preset stable_seed",
     lambda: build_sampling_graph(ARTISTS, use_preset="stable_seed")),
    ("08a preset drift_auto",
     lambda: build_sampling_graph(ARTISTS, use_preset="drift_auto")),
    ("08b preset drift_soft",
     lambda: build_sampling_graph(ARTISTS, use_preset="drift_soft")),
    ("08c preset face_lock",
     lambda: build_sampling_graph(ARTISTS, use_preset="face_lock")),
    ("08d preset scene_lock",
     lambda: build_sampling_graph(ARTISTS, use_preset="scene_lock")),
    ("08e preset anchor_lock",
     lambda: build_sampling_graph(ARTISTS, use_preset="anchor_lock")),
    ("09 static_capture",
     lambda: build_sampling_graph(
         ARTISTS, opts=default_opts(artist_static_capture=True,
                                    static_capture_k=4))),
    ("10 anchor_q multiseed",
     lambda: build_sampling_graph(
         ARTISTS, opts=default_opts(artist_anchor_q=True,
                                    anchor_seeds_count=2))),
    ("10b anchor norm ref",
     lambda: build_sampling_graph(
         ARTISTS, opts=default_opts(artist_anchor_q=True,
                                    anchor_seeds_count=2,
                                    anchor_base_norm_ref=True))),
    ("10c anchor refresh",
     lambda: build_sampling_graph(
         ARTISTS, opts=default_opts(artist_anchor_q=True,
                                    anchor_refresh_each_step=True,
                                    match_base_norm=False))),
    ("11 layer route @0-8",
     lambda: build_sampling_graph("@uof@0-8, @kieed@9-27")),
    ("12 max_batch + low_vram (4 artists)",
     lambda: build_sampling_graph(
         "@uof, @kieed, @ciloranko, @huanxiang_heitu",
         opts=default_opts(max_batch_artists=2, low_vram_cache=True))),
    ("13 batch_size=2 CFG mask",
     lambda: build_sampling_graph(ARTISTS, batch_size=2)),
    ("14 strength extrapolation 2.0",
     lambda: build_sampling_graph(ARTISTS, strength=2.0)),
    ("15 recipe save/load roundtrip",
     build_recipe_graph),
    ("16 probe + report",
     lambda: build_probe_graph(ARTISTS)),
    ("17 contribution balance",
     lambda: build_sampling_graph(
         ARTISTS, opts=default_opts(contribution_balance=True,
                                    contribution_balance_alpha=1.0))),
]


def main():
    print(f"== Anima-Artist-Mixer live smoke test ({len(TESTS)} cases) ==\n")
    results = []
    for name, builder in TESTS:
        graph = builder()
        res = submit_and_wait(name, graph)
        results.append(res)
        print(f"[{res[1]:>11}] {res[0]:<34} {res[2]}", flush=True)
    print("\n== Summary ==")
    ok = sum(1 for r in results if r[1] == "OK")
    for r in results:
        if r[1] != "OK":
            print(f"  FAIL {r[0]}: {r[1]} -> {r[2]}")
    print(f"\n{ok}/{len(results)} passed")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
