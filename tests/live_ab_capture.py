"""Live A/B capture for the v26.2.0 -> v27 runtime changes.

Renders fixed-seed production-setting images (1536x1024 / 32 steps) for the
presets whose output the v26.2.0 review fixes actually changed
(fast_preview / compatibility_safe via the concat_with_base CFG fix) plus a
balanced reference, and downloads the PNGs so two code versions can be
compared with AnimaArtistImpactMap.

Manual harness (GPU + live server required):

    python tests/live_ab_capture.py --tag old
    ... switch the installed pack version, restart ComfyUI ...
    python tests/live_ab_capture.py --tag new
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

SERVER = os.environ.get("ANIMA_SMOKE_SERVER", "http://127.0.0.1:8188")

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

WIDTH, HEIGHT, STEPS, CFG, SEED = 1536, 1024, 32, 5.0, 20260704
PRESETS = ("fast_preview", "compatibility_safe", "balanced")


def _post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER + path, data=data, headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=30))


def _get(path):
    return json.load(urllib.request.urlopen(SERVER + path, timeout=30))


def build_graph(preset, prefix):
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "stable_diffusion",
                         "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "AnimaArtistPack",
              "inputs": {"clip": ["2", 0], "artist_chain": ARTISTS,
                         "base_prompt": BASE_PROMPT}},
        "5": {"class_type": "AnimaArtistPreset",
              "inputs": {"preset": preset, "intensity": 1.0,
                         "normalize_weights": True, "layer_mode": "auto",
                         "custom_layer_filter": ""}},
        "6": {"class_type": "AnimaArtistPresetApply",
              "inputs": {"model": ["1", 0], "artist_pack": ["4", 0],
                         "preset": ["5", 0], "enabled": True,
                         "apply_to_uncond": False}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG_PROMPT, "clip": ["2", 0]}},
        "8": {"class_type": "EmptyLatentImage",
              "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["6", 0], "positive": ["6", 1],
                         "negative": ["7", 0], "latent_image": ["8", 0],
                         "seed": SEED, "steps": STEPS, "cfg": CFG,
                         "sampler_name": "er_sde", "scheduler": "beta",
                         "denoise": 1.0}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def run_case(preset, tag, out_dir, timeout=600):
    prefix = f"ab_{tag}_{preset}"
    t0 = time.time()
    try:
        resp = _post("/prompt", {"prompt": build_graph(preset, prefix)})
    except urllib.error.HTTPError as e:
        return preset, "SUBMIT_FAIL", e.read().decode("utf-8", "replace")[:400]
    if resp.get("node_errors"):
        return preset, "NODE_ERRORS", json.dumps(resp["node_errors"])[:400]
    pid = resp["prompt_id"]
    while time.time() - t0 < timeout:
        hist = _get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            if entry.get("status", {}).get("status_str") == "error":
                return preset, "EXEC_ERROR", json.dumps(
                    entry["status"].get("messages", []))[:400]
            for out in entry.get("outputs", {}).values():
                for img in out.get("images", []):
                    url = (f"/view?filename={urllib.parse.quote(img['filename'])}"
                           f"&subfolder={urllib.parse.quote(img.get('subfolder', ''))}"
                           f"&type={img.get('type', 'output')}")
                    dest = os.path.join(out_dir, f"{tag}_{preset}.png")
                    with urllib.request.urlopen(SERVER + url, timeout=60) as fh:
                        with open(dest, "wb") as f:
                            f.write(fh.read())
                    return preset, "OK", f"{time.time() - t0:.0f}s -> {dest}"
            return preset, "NO_OUTPUT", ""
        time.sleep(3)
    return preset, "TIMEOUT", f">{timeout}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="old / new — used in filenames")
    ap.add_argument("--out", default=os.path.join("test_results", "ab_v27"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"== A/B capture (tag={args.tag}, seed={SEED}, "
          f"{WIDTH}x{HEIGHT}/{STEPS} steps) ==")
    failures = 0
    for preset in PRESETS:
        name, status, detail = run_case(preset, args.tag, args.out)
        print(f"  {name:20s} {status:12s} {detail}")
        failures += status != "OK"
    print(f"== done, {failures} failure(s) ==")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
