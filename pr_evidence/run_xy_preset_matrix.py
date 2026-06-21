from __future__ import annotations

import io
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from PIL import Image, ImageDraw, ImageFont


class ImageInfo(TypedDict):
    filename: str
    subfolder: str
    type: str


class CellResult(TypedDict):
    row_key: str
    row_label: str
    artist_count: int
    artist_chain: str
    column_key: str
    column_label: str
    is_preset_column: bool
    prompt_id: str
    seconds: float
    image: ImageInfo
    local_image: str


class ModelSubstitution(TypedDict):
    loader: str
    source_name: str
    resolved_name: str


SERVER = os.environ.get("ANIMA_MATRIX_SERVER", "http://127.0.0.1:8190")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_mixer.options import build_preset_payload  # noqa: E402

RESULT_DIR = ROOT / "pr_evidence"
SOURCE_WORKFLOW = RESULT_DIR / "source_Anima.Style.Mixer.json"
CELL_DIR = RESULT_DIR / "xy_preset_matrix_cells"
METRICS_PATH = RESULT_DIR / "xy_preset_matrix.metrics.json"
SUMMARY_PATH = RESULT_DIR / "xy_preset_matrix_summary.md"
GRID_PATHS = [
    RESULT_DIR / "xy_preset_matrix_part1.png",
    RESULT_DIR / "xy_preset_matrix_part2.png",
]

PRESET_COLUMNS = [
    ("prompt_passthrough", "prompt_passthrough", True),
    ("balanced", "balanced", True),
    ("strong_style", "strong_style", True),
    ("stable_seed", "stable_seed", True),
    ("drift_auto", "drift_auto", True),
    ("drift_soft", "drift_soft", True),
    ("face_lock", "face_lock", True),
    ("scene_lock", "scene_lock", True),
    ("anchor_lock", "anchor_lock", True),
    ("fast_preview", "fast_preview", True),
    ("identity_guard", "identity_guard", True),
    ("compatibility_safe", "compatibility_safe", True),
]

PRESET_LAYER_MODE = "auto"

COLUMNS = [
    ("no_mixer", "no mixer", False),
    ("original_mixer", "original mixer", False),
]
for preset_key, preset_label, is_preset in PRESET_COLUMNS:
    COLUMNS.insert(1 if preset_key == "prompt_passthrough" else len(COLUMNS), (
        preset_key,
        preset_label,
        is_preset,
    ))

ROW_ARTISTS = [
    r"@yuchi \(salmon-1000\)",
    "@momisan",
    "@toosaka asagi",
    "@derauea",
    "@tsukishiro saika",
    "@tonee",
    "@umanosuke",
    "@hanzou",
    "@swordsouls",
    "@kuroi mimei",
]


def request_json(path: str, payload: object | None, timeout: int) -> dict[str, object]:
    if payload is None:
        with urllib.request.urlopen(SERVER + path, timeout=timeout) as response:
            data = json.load(response)
    else:
        request = urllib.request.Request(
            SERVER + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    if not isinstance(data, dict):
        raise TypeError(f"{path} returned {type(data).__name__}, expected object")
    return data


def node_input_choices(object_info: dict[str, object], node_type: str, input_name: str) -> list[str]:
    node_info = object_info.get(node_type)
    if not isinstance(node_info, dict):
        raise ValueError(f"/object_info missing {node_type}")
    raw_input = node_info.get("input")
    if not isinstance(raw_input, dict):
        raise ValueError(f"/object_info {node_type} has no input object")
    raw_required = raw_input.get("required")
    if not isinstance(raw_required, dict):
        raise ValueError(f"/object_info {node_type} has no required input object")
    raw_field = raw_required.get(input_name)
    if not isinstance(raw_field, list) or not raw_field:
        raise ValueError(f"/object_info {node_type}.{input_name} has no choices")
    raw_choices = raw_field[0]
    if not isinstance(raw_choices, list):
        raise TypeError(f"/object_info {node_type}.{input_name} choices are not a list")
    choices: list[str] = []
    for raw_choice in raw_choices:
        if isinstance(raw_choice, str):
            choices.append(raw_choice)
    return choices


def resolve_model_name(loader: str, source_name: str, choices: list[str]) -> tuple[str, ModelSubstitution | None]:
    if source_name in choices:
        return source_name, None
    normalized_source = source_name.replace("\\", "/").split("/")[-1].lower()
    for choice in choices:
        normalized_choice = choice.replace("\\", "/").split("/")[-1].lower()
        if normalized_choice == normalized_source:
            return choice, {"loader": loader, "source_name": source_name, "resolved_name": choice}
    if loader == "UNETLoader" and source_name == "anima-base-v1.0.safetensors":
        replacement = r"Anima\anime\anima_baseV10.safetensors"
        if replacement in choices:
            return replacement, {"loader": loader, "source_name": source_name, "resolved_name": replacement}
    available_preview = ", ".join(choices[:12])
    raise ValueError(
        f"{loader} source model {source_name!r} is not available on {SERVER}. "
        f"Available examples: {available_preview}"
    )


def resolve_source_models(settings: dict[str, object]) -> tuple[dict[str, object], list[ModelSubstitution]]:
    object_info = request_json("/object_info", None, 60)
    resolved = dict(settings)
    substitutions: list[ModelSubstitution] = []
    loader_fields = [
        ("UNETLoader", "unet_name", "unet"),
        ("CLIPLoader", "clip_name", "clip"),
        ("VAELoader", "vae_name", "vae"),
    ]
    for loader, input_name, setting_key in loader_fields:
        source_name = resolved.get(setting_key)
        if not isinstance(source_name, str):
            raise TypeError(f"settings[{setting_key!r}] is not a string")
        choices = node_input_choices(object_info, loader, input_name)
        resolved_name, substitution = resolve_model_name(loader, source_name, choices)
        resolved[setting_key] = resolved_name
        if substitution is not None:
            substitutions.append(substitution)
    return resolved, substitutions


def load_source_workflow(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} is {type(data).__name__}, expected object")
    return data


def nodes_by_type(source: dict[str, object], class_type: str) -> list[dict[str, object]]:
    raw_nodes = source.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("source workflow has no nodes list")
    nodes: list[dict[str, object]] = []
    for node in raw_nodes:
        if isinstance(node, dict) and node.get("type") == class_type:
            nodes.append(node)
    return nodes


def widget_string(node: dict[str, object], index: int, label: str) -> str:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or len(widgets) <= index:
        raise ValueError(f"{label} missing widget index {index}")
    value = widgets[index]
    if not isinstance(value, str):
        raise TypeError(f"{label} widget {index} is {type(value).__name__}, expected string")
    return value


def widget_number(node: dict[str, object], index: int, label: str) -> float:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or len(widgets) <= index:
        raise ValueError(f"{label} missing widget index {index}")
    value = widgets[index]
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"{label} widget {index} is {type(value).__name__}, expected number")


def source_settings(source: dict[str, object]) -> dict[str, object]:
    unet = nodes_by_type(source, "UNETLoader")[0]
    clip = nodes_by_type(source, "CLIPLoader")[0]
    vae = nodes_by_type(source, "VAELoader")[0]
    sampler = nodes_by_type(source, "KSampler")[0]
    latent = nodes_by_type(source, "EmptyLatentImage")[0]
    pack = nodes_by_type(source, "AnimaArtistPack")[0]
    negative = [
        node
        for node in nodes_by_type(source, "CLIPTextEncode")
        if "Negative" in str(node.get("title", ""))
    ][0]
    seed_nodes = nodes_by_type(source, "Seed (rgthree)")
    seed = int(widget_number(seed_nodes[0], 0, "seed")) if seed_nodes else int(widget_number(sampler, 0, "sampler"))
    widgets = sampler.get("widgets_values")
    if not isinstance(widgets, list) or len(widgets) < 7:
        raise ValueError("sampler widgets are incomplete")
    return {
        "unet": widget_string(unet, 0, "unet"),
        "clip": widget_string(clip, 0, "clip"),
        "clip_type": widget_string(clip, 1, "clip"),
        "clip_device": widget_string(clip, 2, "clip"),
        "vae": widget_string(vae, 0, "vae"),
        "width": int(widget_number(latent, 0, "latent")),
        "height": int(widget_number(latent, 1, "latent")),
        "steps": int(widget_number(sampler, 2, "sampler")),
        "cfg": float(widget_number(sampler, 3, "sampler")),
        "sampler_name": str(widgets[4]),
        "scheduler": str(widgets[5]),
        "denoise": float(widget_number(sampler, 6, "sampler")),
        "seed": seed,
        "source_artist_chain": widget_string(pack, 0, "pack"),
        "base_prompt": widget_string(pack, 1, "pack"),
        "negative_prompt": widget_string(negative, 0, "negative"),
    }


def artist_rows() -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for count in [1, 2, 4, 10]:
        rows.append((f"{count}_artists", f"{count} artist" if count == 1 else f"{count} artists", count))
    return rows


def artist_chain_for_count(count: int) -> str:
    return "\n".join(ROW_ARTISTS[:count])


def preset_summaries() -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for preset_key, preset_label, _ in PRESET_COLUMNS:
        payload = build_preset_payload(
            preset_key,
            1.0,
            layer_mode=PRESET_LAYER_MODE,
            custom_layer_filter="",
            normalize_weights=True,
            artist_count=0,
        )
        advanced_options = payload.get("advanced_options")
        if not isinstance(advanced_options, dict):
            advanced_options = {}
        summaries.append({
            "key": preset_key,
            "label": preset_label,
            "combine_mode": payload.get("combine_mode"),
            "fusion_mode": payload.get("fusion_mode"),
            "strength": payload.get("strength"),
            "layer_mode": PRESET_LAYER_MODE,
            "layer_filter": advanced_options.get("layer_filter", ""),
            "prompt_passthrough": advanced_options.get("prompt_passthrough", False),
            "artist_static_capture": advanced_options.get("artist_static_capture", False),
            "artist_anchor_q": advanced_options.get("artist_anchor_q", False),
            "artist_ema_alpha": advanced_options.get("artist_ema_alpha", 0.0),
            "match_base_norm": advanced_options.get("match_base_norm", False),
            "mixed_delta_cap": advanced_options.get("mixed_delta_cap", False),
            "mixed_delta_cap_ratio": advanced_options.get("mixed_delta_cap_ratio", 1.0),
            "compatibility_mode": advanced_options.get("compatibility_mode", False),
        })
    return summaries


def direct_prompt(artist_chain: str, base_prompt: str) -> str:
    return f"{artist_chain}\n\n{base_prompt}"


def base_graph(settings: dict[str, object], prefix: str) -> dict[str, dict[str, object]]:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": settings["unet"], "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": settings["clip"],
                "type": settings["clip_type"],
                "device": settings["clip_device"],
            },
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": settings["vae"]}},
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": settings["negative_prompt"], "clip": ["2", 0]},
        },
        "8": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": settings["width"],
                "height": settings["height"],
                "batch_size": 1,
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def sampler_node(settings: dict[str, object], model_ref: list[object], positive_ref: list[object]) -> dict[str, object]:
    return {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": positive_ref,
            "negative": ["7", 0],
            "latent_image": ["8", 0],
            "seed": settings["seed"],
            "steps": settings["steps"],
            "cfg": settings["cfg"],
            "sampler_name": settings["sampler_name"],
            "scheduler": settings["scheduler"],
            "denoise": settings["denoise"],
        },
    }


def graph_no_mixer(settings: dict[str, object], artist_chain: str, prefix: str) -> dict[str, dict[str, object]]:
    graph = base_graph(settings, prefix)
    graph["4"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": direct_prompt(artist_chain, str(settings["base_prompt"])), "clip": ["2", 0]},
    }
    graph["9"] = sampler_node(settings, ["1", 0], ["4", 0])
    return graph


def graph_original_mixer(settings: dict[str, object], artist_chain: str, prefix: str) -> dict[str, dict[str, object]]:
    graph = base_graph(settings, prefix)
    graph["4"] = {
        "class_type": "AnimaArtistPack",
        "inputs": {
            "clip": ["2", 0],
            "artist_chain": artist_chain,
            "base_prompt": settings["base_prompt"],
        },
    }
    graph["5"] = {
        "class_type": "AnimaArtistCrossAttn",
        "inputs": {
            "model": ["1", 0],
            "artist_pack": ["4", 0],
            "combine_mode": "output_avg",
            "fusion_mode": "interpolate",
            "strength": 1.0,
            "enabled": True,
            "apply_to_uncond": False,
        },
    }
    graph["9"] = sampler_node(settings, ["5", 0], ["5", 1])
    return graph


def graph_preset(settings: dict[str, object], artist_chain: str, preset: str, prefix: str) -> dict[str, dict[str, object]]:
    graph = base_graph(settings, prefix)
    graph["4"] = {
        "class_type": "AnimaArtistPack",
        "inputs": {
            "clip": ["2", 0],
            "artist_chain": artist_chain,
            "base_prompt": settings["base_prompt"],
        },
    }
    graph["5"] = {
        "class_type": "AnimaArtistPreset",
        "inputs": {
            "preset": preset,
            "intensity": 1.0,
            "normalize_weights": True,
            "layer_mode": "auto",
            "custom_layer_filter": "",
        },
    }
    graph["6"] = {
        "class_type": "AnimaArtistPresetApply",
        "inputs": {
            "model": ["1", 0],
            "artist_pack": ["4", 0],
            "preset": ["5", 0],
            "enabled": True,
            "apply_to_uncond": False,
        },
    }
    graph["9"] = sampler_node(settings, ["6", 0], ["6", 1])
    return graph


def graph_for_cell(settings: dict[str, object], column_key: str, artist_chain: str, prefix: str) -> dict[str, dict[str, object]]:
    if column_key == "no_mixer":
        return graph_no_mixer(settings, artist_chain, prefix)
    if column_key == "original_mixer":
        return graph_original_mixer(settings, artist_chain, prefix)
    return graph_preset(settings, artist_chain, column_key, prefix)


def images_from_history(entry: dict[str, object]) -> list[ImageInfo]:
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return []
    images: list[ImageInfo] = []
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        raw_images = output.get("images")
        if not isinstance(raw_images, list):
            continue
        for raw_image in raw_images:
            if not isinstance(raw_image, dict):
                continue
            filename = raw_image.get("filename")
            subfolder = raw_image.get("subfolder")
            image_type = raw_image.get("type")
            if isinstance(filename, str) and isinstance(subfolder, str) and isinstance(image_type, str):
                images.append({"filename": filename, "subfolder": subfolder, "type": image_type})
    return images


def view_image(image: ImageInfo) -> Image.Image:
    params = urllib.parse.urlencode({
        "filename": image["filename"],
        "subfolder": image["subfolder"],
        "type": image["type"],
    })
    with urllib.request.urlopen(SERVER + "/view?" + params, timeout=120) as response:
        data = response.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def submit_and_wait(label: str, graph: dict[str, dict[str, object]], timeout: int) -> tuple[str, float, ImageInfo]:
    start = time.perf_counter()
    response = request_json("/prompt", {"prompt": graph}, 60)
    node_errors = response.get("node_errors")
    if node_errors:
        raise RuntimeError(f"{label} node_errors: {json.dumps(node_errors, ensure_ascii=False)}")
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str):
        raise RuntimeError(f"{label} response missing prompt_id: {json.dumps(response, ensure_ascii=False)}")
    while time.perf_counter() - start < timeout:
        history = request_json(f"/history/{prompt_id}", None, 60)
        entry = history.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status")
            if isinstance(status, dict) and status.get("status_str") == "error":
                raise RuntimeError(f"{label} execution error: {json.dumps(status, ensure_ascii=False)}")
            images = images_from_history(entry)
            if images:
                return prompt_id, time.perf_counter() - start, images[0]
        time.sleep(1.0)
    raise TimeoutError(f"{label} timed out after {timeout}s")


def save_metrics(settings: dict[str, object], substitutions: list[ModelSubstitution], results: list[CellResult], status: str, started_at: str) -> None:
    payload = {
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "server": SERVER,
        "source_workflow": str(SOURCE_WORKFLOW),
        "settings": settings,
        "model_substitutions": substitutions,
        "preset_layer_mode": PRESET_LAYER_MODE,
        "preset_summaries": preset_summaries(),
        "columns": [
            {"key": key, "label": label, "is_preset_column": is_preset}
            for key, label, is_preset in COLUMNS
        ],
        "rows": [
            {
                "key": key,
                "label": label,
                "artist_count": count,
                "artist_chain": artist_chain_for_count(count),
            }
            for key, label, count in artist_rows()
        ],
        "results": results,
        "timing_note": "Seconds are one-run wall time from /prompt submit to history image, including queue/cache/decode overhead.",
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def font(size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\segoeuib.ttf") if bold else Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def open_local_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, text_font: ImageFont.ImageFont) -> list[str]:
    words = text.replace("_", "_ ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=text_font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def build_grid(results: list[CellResult], part_index: int, column_slice: slice, path: Path) -> None:
    columns = COLUMNS[column_slice]
    row_data = artist_rows()
    thumb = 384
    header_h = 154
    cell_label_h = 66
    row_label_w = 230
    cell_w = thumb
    cell_h = thumb + cell_label_h
    width = row_label_w + len(columns) * cell_w
    height = header_h + len(row_data) * cell_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28, True)
    header_font = font(20, True)
    small_font = font(16, False)
    row_font = font(22, True)
    red = (190, 20, 20)
    black = (24, 28, 35)
    gray = (88, 96, 110)
    line = (214, 220, 230)
    title = f"Anima preset matrix part {part_index}: source workflow params, 1024px cells rendered separately"
    draw.text((18, 18), title, font=title_font, fill=black)
    draw.text((18, 60), "Red column names are Anima preset columns. Seconds are one-run wall time.", font=small_font, fill=gray)
    for col_idx, (column_key, column_label, is_preset) in enumerate(columns):
        x = row_label_w + col_idx * cell_w
        draw.rectangle((x, 98, x + cell_w, header_h), fill=(248, 250, 252), outline=line)
        label_lines = wrap_text(draw, column_label, cell_w - 18, header_font)
        y = 106
        for label_line in label_lines[:2]:
            draw.text((x + 10, y), label_line, font=header_font, fill=red if is_preset else black)
            y += 24
        draw.text((x + 10, y + 2), column_key, font=small_font, fill=gray)
    by_key = {(result["row_key"], result["column_key"]): result for result in results}
    for row_idx, (row_key, row_label, count) in enumerate(row_data):
        y = header_h + row_idx * cell_h
        draw.rectangle((0, y, row_label_w, y + cell_h), fill=(248, 250, 252), outline=line)
        draw.text((16, y + 26), row_label, font=row_font, fill=black)
        draw.multiline_text((16, y + 64), artist_chain_for_count(count), font=small_font, fill=gray, spacing=3)
        for col_idx, (column_key, _, _) in enumerate(columns):
            x = row_label_w + col_idx * cell_w
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill="white", outline=line)
            result = by_key[(row_key, column_key)]
            cell_img = open_local_image(result["local_image"])
            cell_img.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (thumb, thumb), (241, 244, 248))
            canvas.paste(cell_img, ((thumb - cell_img.width) // 2, (thumb - cell_img.height) // 2))
            image.paste(canvas, (x, y + cell_label_h))
            draw.text((x + 10, y + 8), f"{result['seconds']:.1f}s", font=header_font, fill=black)
            draw.text((x + 10, y + 34), result["column_label"], font=small_font, fill=red if result["is_preset_column"] else gray)
    image.save(path)


def write_summary(settings: dict[str, object], substitutions: list[ModelSubstitution], results: list[CellResult]) -> None:
    lines = [
        "# XY preset matrix evidence",
        "",
        f"- Source workflow: `{SOURCE_WORKFLOW}`",
        f"- Server: `{SERVER}`",
        f"- Size: `{settings['width']}x{settings['height']}`",
        f"- Steps / CFG / sampler / scheduler: `{settings['steps']}` / `{settings['cfg']}` / `{settings['sampler_name']}` / `{settings['scheduler']}`",
        f"- Seed: `{settings['seed']}`",
        f"- Metrics: `{METRICS_PATH}`",
        f"- Grid part 1: `{GRID_PATHS[0]}`",
        f"- Grid part 2: `{GRID_PATHS[1]}`",
        "",
        "Red column labels in the PNG grids mark Anima preset columns.",
        "",
        "Model substitutions:",
    ]
    if substitutions:
        for substitution in substitutions:
            lines.append(
                f"- `{substitution['loader']}`: `{substitution['source_name']}` -> "
                f"`{substitution['resolved_name']}`"
            )
    else:
        lines.append("- None")
    lines.extend([
        "",
        "Preset summaries:",
        "",
        "| preset | combine | fusion | strength | layer filter | stabilizer |",
        "|---|---|---|---:|---|---|",
    ])
    for summary in preset_summaries():
        stabilizers: list[str] = []
        if summary["prompt_passthrough"]:
            stabilizers.append("direct prompt")
        if summary["artist_static_capture"]:
            stabilizers.append("static capture")
        if summary["artist_anchor_q"]:
            stabilizers.append("anchor Q")
        if float(summary["artist_ema_alpha"]) > 0.0:
            stabilizers.append(f"EMA {float(summary['artist_ema_alpha']):.2f}")
        if summary["match_base_norm"]:
            stabilizers.append("norm lock")
        if summary["mixed_delta_cap"]:
            stabilizers.append(f"delta cap {float(summary['mixed_delta_cap_ratio']):.2f}")
        if summary["compatibility_mode"]:
            stabilizers.append("compatibility")
        stabilizer_text = ", ".join(stabilizers) if stabilizers else "-"
        lines.append(
            f"| {summary['label']} | {summary['combine_mode']} | "
            f"{summary['fusion_mode']} | {float(summary['strength']):.2f} | "
            f"{summary['layer_filter'] or 'all'} | {stabilizer_text} |"
        )
    lines.extend([
        "",
        "| row | column | seconds | image |",
        "|---|---|---:|---|",
    ])
    for result in results:
        lines.append(
            f"| {result['row_label']} | {result['column_label']} | "
            f"{result['seconds']:.1f} | `{result['local_image']}` |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    source = load_source_workflow(SOURCE_WORKFLOW)
    settings, substitutions = resolve_source_models(source_settings(source))
    started_at = datetime.now().isoformat(timespec="seconds")
    if CELL_DIR.exists():
        shutil.rmtree(CELL_DIR)
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    results: list[CellResult] = []
    save_metrics(settings, substitutions, results, "running", started_at)
    total = len(artist_rows()) * len(COLUMNS)
    current = 0
    for row_key, row_label, count in artist_rows():
        artist_chain = artist_chain_for_count(count)
        for column_key, column_label, is_preset in COLUMNS:
            current += 1
            prefix = f"xy_preset_matrix_{row_key}_{column_key}"
            print(f"[{current}/{total}] {row_label} / {column_label}", flush=True)
            graph = graph_for_cell(settings, column_key, artist_chain, prefix)
            prompt_id, seconds, image_info = submit_and_wait(prefix, graph, 1800)
            local_path = CELL_DIR / f"{row_key}__{column_key}.png"
            view_image(image_info).save(local_path)
            result: CellResult = {
                "row_key": row_key,
                "row_label": row_label,
                "artist_count": count,
                "artist_chain": artist_chain,
                "column_key": column_key,
                "column_label": column_label,
                "is_preset_column": is_preset,
                "prompt_id": prompt_id,
                "seconds": seconds,
                "image": image_info,
                "local_image": str(local_path),
            }
            results.append(result)
            save_metrics(settings, substitutions, results, "running", started_at)
    build_grid(results, 1, slice(0, 7), GRID_PATHS[0])
    build_grid(results, 2, slice(7, 14), GRID_PATHS[1])
    write_summary(settings, substitutions, results)
    save_metrics(settings, substitutions, results, "complete", started_at)
    print(json.dumps({
        "metrics": str(METRICS_PATH),
        "summary": str(SUMMARY_PATH),
        "grids": [str(path) for path in GRID_PATHS],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
