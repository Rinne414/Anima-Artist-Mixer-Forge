from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from PIL import Image, ImageChops, ImageDraw, ImageFont


class ComfyImage(TypedDict):
    filename: str
    subfolder: str
    type: str


class RunResult(TypedDict):
    label: str
    prompt_id: str
    seconds: float
    image: ComfyImage
    image_path: str
    case: str
    variant: str
    artists: str


SERVER = os.environ.get("ANIMA_PR_SERVER", "http://127.0.0.1:8190")
OUTPUT_DIR = Path(os.environ.get("ANIMA_COMFY_OUTPUT", r"I:\ComfyUI-aki-v1.6\ComfyUI\output"))
RESULT_DIR = Path(os.environ.get("ANIMA_PR_RESULT_DIR", r"L:\Antigravitiy code\comfyui\Anima-Artist-Mixer\pr_evidence"))

UNET = "Anima\\anime\\anima_baseV10.safetensors"
CLIP = "qwen_3_06b_base.safetensors"
VAE = "qwen_image_vae.safetensors"
WIDTH = int(os.environ.get("ANIMA_PR_WIDTH", "512"))
HEIGHT = int(os.environ.get("ANIMA_PR_HEIGHT", "512"))
STEPS = int(os.environ.get("ANIMA_PR_STEPS", "8"))
CFG = float(os.environ.get("ANIMA_PR_CFG", "5.0"))
SAMPLER = os.environ.get("ANIMA_PR_SAMPLER", "er_sde")
SCHEDULER = os.environ.get("ANIMA_PR_SCHEDULER", "beta")
SEED = int(os.environ.get("ANIMA_PR_SEED", "42424242"))

BASE_PROMPT = (
    "1girl, solo, upper body portrait, face visible, white blouse, navy jacket, "
    "looking at viewer, simple background, clean linework, detailed eyes"
)
NEG_PROMPT = (
    "nsfw, nude, naked, bare chest, cleavage, nipples, cropped head, "
    "head out of frame, lowres, worst quality, bad anatomy"
)
CASES: list[tuple[str, str]] = [
    ("single_yuchi", r"@yuchi \(salmon-1000\)"),
    ("double_yuchi_uof", r"@yuchi \(salmon-1000\), @uof"),
    ("multi_yuchi_uof_kieed_ciloranko", r"@yuchi \(salmon-1000\), @uof, @kieed, @ciloranko"),
]


def request_json(path: str, payload: dict[str, object] | None, timeout: int) -> dict[str, object]:
    if payload is None:
        with urllib.request.urlopen(SERVER + path, timeout=timeout) as response:
            data = json.load(response)
    else:
        req = urllib.request.Request(
            SERVER + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object from {path}, got {type(data).__name__}")
    return data


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:80]


def base_nodes(prefix: str) -> dict[str, dict[str, object]]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "stable_diffusion", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG_PROMPT, "clip": ["2", 0]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def sampler(model_ref: list[object], positive_ref: list[object], seed: int) -> dict[str, object]:
    return {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": positive_ref,
            "negative": ["7", 0],
            "latent_image": ["8", 0],
            "seed": seed,
            "steps": STEPS,
            "cfg": CFG,
            "sampler_name": SAMPLER,
            "scheduler": SCHEDULER,
            "denoise": 1.0,
        },
    }


def graph_prompt_no_mixer(label: str, artists: str, seed: int) -> dict[str, dict[str, object]]:
    prefix = f"anima_pr_fresh_{safe_name(label)}_prompt"
    graph = base_nodes(prefix)
    graph["4"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": f"{artists}, {BASE_PROMPT}", "clip": ["2", 0]},
    }
    graph["9"] = sampler(["1", 0], ["4", 0], seed)
    return graph


def graph_basic(label: str, artists: str, seed: int, preset: str) -> dict[str, dict[str, object]]:
    prefix = f"anima_pr_fresh_{safe_name(label)}_{preset}"
    graph = base_nodes(prefix)
    graph["4"] = {
        "class_type": "AnimaArtistBasic",
        "inputs": {
            "model": ["1", 0],
            "clip": ["2", 0],
            "artist_chain": artists,
            "base_prompt": BASE_PROMPT,
            "preset": preset,
            "intensity": 1.0,
            "enabled": True,
        },
    }
    graph["9"] = sampler(["4", 0], ["4", 1], seed)
    return graph


def output_image_path(image: ComfyImage) -> Path:
    return OUTPUT_DIR / image.get("subfolder", "") / image["filename"]


def images_from_history(entry: dict[str, object]) -> list[ComfyImage]:
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return []
    images: list[ComfyImage] = []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        output_images = output.get("images")
        if not isinstance(output_images, list):
            continue
        for item in output_images:
            if not isinstance(item, dict):
                continue
            filename = item.get("filename")
            subfolder = item.get("subfolder")
            image_type = item.get("type")
            if isinstance(filename, str) and isinstance(subfolder, str) and isinstance(image_type, str):
                images.append({"filename": filename, "subfolder": subfolder, "type": image_type})
    return images


def submit_and_wait(label: str, graph: dict[str, dict[str, object]], timeout: int) -> tuple[str, float, ComfyImage]:
    start = time.perf_counter()
    response = request_json("/prompt", {"prompt": graph}, 30)
    node_errors = response.get("node_errors")
    if node_errors:
        raise RuntimeError(f"{label} node_errors: {node_errors}")
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str):
        raise TypeError(f"{label} response missing prompt_id: {response}")
    while time.perf_counter() - start < timeout:
        history = request_json(f"/history/{prompt_id}", None, 30)
        entry = history.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status")
            if isinstance(status, dict) and status.get("status_str") == "error":
                raise RuntimeError(f"{label} execution error: {status}")
            images = images_from_history(entry)
            if not images:
                raise RuntimeError(f"{label} produced no image")
            return prompt_id, time.perf_counter() - start, images[0]
        time.sleep(1.0)
    raise TimeoutError(f"{label} timed out after {timeout}s")


def load_image(path: str) -> Image.Image:
    p = Path(path)
    if p.exists():
        return Image.open(p).convert("RGB")
    params = urllib.parse.urlencode({"filename": p.name, "subfolder": "", "type": "output"})
    with urllib.request.urlopen(SERVER + "/view?" + params, timeout=30) as response:
        return Image.open(response).convert("RGB")


def descriptor(img: Image.Image, size: int) -> list[float]:
    small = img.resize((size, size), Image.Resampling.BILINEAR)
    data = getattr(small, "get_flattened_data", small.getdata)()
    pixels = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in data]
    channels = list(zip(*pixels))
    gray = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels]
    features: list[float] = []
    for values in (*channels, gray):
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        features.extend([mean, std])
    for values in (*channels, gray):
        hist = [0.0] * 4
        for value in values:
            idx = min(3, int(value * 4))
            hist[idx] += 1.0 / len(values)
        features.extend(hist)
    return features


def descriptor_distance(left: Image.Image, right: Image.Image) -> float:
    left_desc = descriptor(left, 32)
    right_desc = descriptor(right, 32)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left_desc, right_desc)))


def image_mae(left: Image.Image, right: Image.Image) -> float:
    diff = ImageChops.difference(left, right).convert("L")
    data = getattr(diff, "get_flattened_data", diff.getdata)()
    values = list(data)
    return sum(values) / (len(values) * 255.0)


def build_contact_sheet(runs: list[RunResult], path: Path) -> None:
    font_path = Path(r"C:\Windows\Fonts\segoeui.ttf")
    bold_path = Path(r"C:\Windows\Fonts\segoeuib.ttf")
    normal = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    bold = ImageFont.truetype(str(bold_path), 18) if bold_path.exists() else ImageFont.load_default()
    thumb_w = 300
    thumb_h = 300
    label_h = 58
    cols = 3
    rows = math.ceil(len(runs) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + 46), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "Fresh multi-artist evidence: prompt vs balanced vs drift_auto", font=bold, fill=(20, 24, 32))
    for idx, run in enumerate(runs):
        img = load_image(run["image_path"])
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h) + 46
        canvas = Image.new("RGB", (thumb_w, thumb_h), (240, 243, 247))
        canvas.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        sheet.paste(canvas, (x, y + label_h))
        label = f"{run['case']}\n{run['variant']}  {run['seconds']:.1f}s"
        draw.multiline_text((x + 8, y + 8), label, font=normal, fill=(20, 24, 32), spacing=3)
    sheet.save(path)


def run_matrix() -> dict[str, object]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[RunResult] = []
    for case, artists in CASES:
        variants: list[tuple[str, dict[str, dict[str, object]]]] = [
            ("prompt", graph_prompt_no_mixer(f"{case}_prompt", artists, SEED)),
            ("balanced", graph_basic(f"{case}_balanced", artists, SEED, "balanced")),
            ("drift_auto", graph_basic(f"{case}_drift_auto", artists, SEED, "drift_auto")),
        ]
        for variant, graph in variants:
            label = f"{case}_{variant}"
            print(f"run {label}", flush=True)
            prompt_id, seconds, image = submit_and_wait(label, graph, 900)
            runs.append({
                "label": label,
                "prompt_id": prompt_id,
                "seconds": seconds,
                "image": image,
                "image_path": str(output_image_path(image)),
                "case": case,
                "variant": variant,
                "artists": artists,
            })
    comparisons: dict[str, dict[str, float]] = {}
    for case, _ in CASES:
        by_variant = {run["variant"]: run for run in runs if run["case"] == case}
        prompt_img = load_image(by_variant["prompt"]["image_path"])
        case_metrics: dict[str, float] = {}
        for variant in ("balanced", "drift_auto"):
            img = load_image(by_variant[variant]["image_path"])
            case_metrics[f"{variant}_vs_prompt_descriptor_distance"] = descriptor_distance(img, prompt_img)
            case_metrics[f"{variant}_vs_prompt_mae"] = image_mae(img, prompt_img)
            case_metrics[f"{variant}_seconds"] = by_variant[variant]["seconds"]
        case_metrics["prompt_seconds"] = by_variant["prompt"]["seconds"]
        comparisons[case] = case_metrics
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sheet_path = RESULT_DIR / f"fresh-multi-artist-matrix-{stamp}.png"
    json_path = RESULT_DIR / f"fresh-multi-artist-matrix-{stamp}.metrics.json"
    build_contact_sheet(runs, sheet_path)
    result: dict[str, object] = {
        "captured_at_local": stamp,
        "server": SERVER,
        "settings": {
            "width": WIDTH,
            "height": HEIGHT,
            "steps": STEPS,
            "cfg": CFG,
            "sampler": SAMPLER,
            "scheduler": SCHEDULER,
            "seed": SEED,
            "unet": UNET,
            "clip": CLIP,
            "vae": VAE,
            "base_prompt": BASE_PROMPT,
            "negative_prompt": NEG_PROMPT,
        },
        "runs": runs,
        "comparisons": comparisons,
        "contact_sheet": str(sheet_path),
        "timing_note": "Single run timings include queue, model/cache state, and decode overhead; use them as cost visibility, not proof of speedup.",
        "average_seconds": statistics.mean(run["seconds"] for run in runs),
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "contact_sheet": str(sheet_path)}, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    run_matrix()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
