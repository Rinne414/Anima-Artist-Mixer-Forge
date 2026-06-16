"""Live A/B drift check for Anima-Artist-Mixer.

Runs the same multi-artist workflow across several seeds against a live
ComfyUI server and compares simple image descriptor variance between an
unstabilized config and the default stabilizers. This is intentionally a
manual integration harness: it needs a GPU, Anima model files, Pillow, and a
running ComfyUI server.
"""

import json
import math
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Pillow is required for live_drift_ab.py") from e

from live_comfy_smoke import (  # noqa: E402
    ARTISTS,
    BASE_PROMPT,
    CFG,
    HEIGHT,
    NEG_PROMPT,
    STEPS,
    UNET,
    CLIP,
    VAE,
    WIDTH,
    default_opts,
)


SERVER = os.environ.get("ANIMA_SMOKE_SERVER", "http://127.0.0.1:8188")
OUTPUT_DIR = os.environ.get("ANIMA_COMFY_OUTPUT", r"I:\ComfyUI-aki-v1.6\ComfyUI\output")
SEEDS = [int(s) for s in os.environ.get("ANIMA_DRIFT_SEEDS", "11,22,33,44,55").split(",")]
DRIFT_ARTISTS = os.environ.get("ANIMA_DRIFT_ARTISTS", ARTISTS)
DRIFT_BASE_PROMPT = os.environ.get("ANIMA_DRIFT_BASE_PROMPT", BASE_PROMPT)
DRIFT_NEG_PROMPT = os.environ.get("ANIMA_DRIFT_NEG_PROMPT", NEG_PROMPT)
DRIFT_CONFIGS = [
    s.strip() for s in os.environ.get("ANIMA_DRIFT_CONFIGS", "").split(",")
    if s.strip()
]
AGGREGATE_FILES = [
    s.strip() for s in os.environ.get("ANIMA_DRIFT_AGGREGATE_FILES", "").split(";")
    if s.strip()
]
SAVE_JSON = os.environ.get("ANIMA_DRIFT_SAVE_JSON", "").strip()

REGION_BOXES = {
    "full": (0.0, 0.0, 1.0, 1.0),
    "center": (0.32, 0.12, 0.68, 0.96),
    "upper_center": (0.36, 0.06, 0.64, 0.54),
}
FOREGROUND_DESCRIPTOR_WEIGHTS = {
    "upper_center": 0.50,
    "center": 0.45,
    "full": 0.05,
}


def _post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER + path, data=data, headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=30))


def _get(path):
    return json.load(urllib.request.urlopen(SERVER + path, timeout=30))


def _image_path(image):
    params = {
        "filename": image["filename"],
        "subfolder": image.get("subfolder", ""),
        "type": image.get("type", "output"),
    }
    return SERVER + "/view?" + urllib.parse.urlencode(params)


def _load_image(image):
    local = os.path.join(OUTPUT_DIR, image.get("subfolder", ""), image["filename"])
    if os.path.exists(local):
        return Image.open(local).convert("RGB")
    return Image.open(urllib.request.urlopen(_image_path(image), timeout=30)).convert("RGB")


def _graph(seed, opts):
    return _graph_full(
        seed=seed,
        opts=opts,
        combine="output_avg",
        fusion="interpolate",
        strength=1.0,
        use_preset=None,
    )


def _graph_full(seed, opts, combine, fusion, strength, use_preset):
    graph = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "stable_diffusion",
                         "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "AnimaArtistPack",
              "inputs": {"clip": ["2", 0], "artist_chain": DRIFT_ARTISTS,
                         "base_prompt": DRIFT_BASE_PROMPT}},
        "6": {"class_type": "AnimaArtistCrossAttn",
              "inputs": {"model": ["1", 0], "artist_pack": ["4", 0],
                         "combine_mode": combine, "fusion_mode": fusion,
                         "strength": strength, "enabled": True,
                         "apply_to_uncond": False}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": DRIFT_NEG_PROMPT, "clip": ["2", 0]}},
        "8": {"class_type": "EmptyLatentImage",
              "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["6", 0], "positive": ["6", 1],
                         "negative": ["7", 0], "latent_image": ["8", 0],
                         "seed": seed, "steps": STEPS, "cfg": CFG,
                         "sampler_name": "er_sde", "scheduler": "beta",
                         "denoise": 1.0}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "anima_drift"}},
    }
    if use_preset is not None:
        graph["5"] = {"class_type": "AnimaArtistPreset",
                      "inputs": {"preset": use_preset, "intensity": 1.0,
                                 "normalize_weights": True,
                                 "layer_mode": "auto",
                                 "custom_layer_filter": ""}}
        graph["6"]["inputs"]["preset"] = ["5", 0]
    else:
        graph["5"] = {"class_type": "AnimaArtistOptions", "inputs": opts}
        graph["6"]["inputs"]["advanced_options"] = ["5", 0]
    return graph


def _run_graph(label, seed, graph, timeout=240):
    t0 = time.time()
    try:
        resp = _post("/prompt", {"prompt": graph})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"{label} seed={seed} submit failed: {body}") from e
    if resp.get("node_errors"):
        raise RuntimeError(f"{label} seed={seed} node errors: {resp['node_errors']}")
    pid = resp["prompt_id"]
    while time.time() - t0 < timeout:
        hist = _get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"{label} seed={seed} execution error: {entry['status']}")
            for out in entry.get("outputs", {}).values():
                images = out.get("images") or []
                if images:
                    return images[0]
            raise RuntimeError(f"{label} seed={seed} produced no image")
        time.sleep(1)
    raise TimeoutError(f"{label} seed={seed} timed out")


def _resample_filter():
    resampling = getattr(Image, "Resampling", Image)
    return resampling.BILINEAR


def _crop_fraction(img, box):
    w, h = img.size
    left, top, right, bottom = box
    px_box = (
        max(0, min(w - 1, int(round(left * w)))),
        max(0, min(h - 1, int(round(top * h)))),
        max(1, min(w, int(round(right * w)))),
        max(1, min(h, int(round(bottom * h)))),
    )
    if px_box[2] <= px_box[0] or px_box[3] <= px_box[1]:
        return img
    return img.crop(px_box)


def _mean(values):
    return sum(values) / max(1, len(values))


def _std(values, mean=None):
    if not values:
        return 0.0
    if mean is None:
        mean = _mean(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def _histogram(values, bins):
    hist = [0.0] * bins
    if not values:
        return hist
    scale = bins - 1e-6
    for v in values:
        idx = int(max(0.0, min(scale, v * scale)))
        hist[idx] += 1.0
    inv = 1.0 / len(values)
    return [v * inv for v in hist]


def _edge_stats(gray, width):
    if width <= 1 or not gray:
        return [0.0, 0.0]
    diffs = []
    height = len(gray) // width
    for y in range(height):
        row = y * width
        for x in range(width - 1):
            diffs.append(abs(gray[row + x + 1] - gray[row + x]))
    for y in range(height - 1):
        row = y * width
        next_row = (y + 1) * width
        for x in range(width):
            diffs.append(abs(gray[next_row + x] - gray[row + x]))
    return [_mean(diffs), _std(diffs)]


def _grid_means(gray, width, cells=4):
    if width <= 0 or not gray:
        return [0.0] * (cells * cells)
    height = len(gray) // width
    out = []
    for gy in range(cells):
        y0 = int(round(gy * height / cells))
        y1 = int(round((gy + 1) * height / cells))
        for gx in range(cells):
            x0 = int(round(gx * width / cells))
            x1 = int(round((gx + 1) * width / cells))
            vals = [
                gray[y * width + x]
                for y in range(y0, max(y0 + 1, y1))
                for x in range(x0, max(x0 + 1, x1))
            ]
            out.append(_mean(vals))
    return out


def _region_descriptor(img, size=32):
    small = img.resize((size, size), _resample_filter())
    get_pixels = getattr(small, "get_flattened_data", small.getdata)
    pix = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in get_pixels()]
    channels = list(zip(*pix))
    gray = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pix]

    features = []
    for vals in (*channels, gray):
        mean = _mean(vals)
        features.extend([mean, _std(vals, mean)])
    for vals in channels:
        features.extend(_histogram(vals, 4))
    features.extend(_histogram(gray, 8))
    features.extend(_edge_stats(gray, size))
    features.extend(_grid_means(gray, size, cells=4))
    return features


def _weighted_descriptor(descs, weights):
    out = []
    for name, weight in weights.items():
        scale = math.sqrt(float(weight))
        out.extend(v * scale for v in descs[name])
    return out


def _descriptors_for_loaded_image(img):
    descs = {
        name: _region_descriptor(_crop_fraction(img, box))
        for name, box in REGION_BOXES.items()
    }
    descs["foreground"] = _weighted_descriptor(descs, FOREGROUND_DESCRIPTOR_WEIGHTS)
    return descs


def _image_descriptors(image):
    return _descriptors_for_loaded_image(_load_image(image))


def _image_descriptor(image):
    return _image_descriptors(image)["foreground"]


def _descriptor_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _pairwise_distance(descs):
    distances = []
    for i, a in enumerate(descs):
        for b in descs[i + 1:]:
            distances.append(_descriptor_distance(a, b))
    if not distances:
        return 0.0
    return statistics.mean(distances)


def _descriptor_std_sum(descs):
    columns = list(zip(*descs))
    return sum(statistics.pstdev(col) for col in columns)


def _pairwise_metric_distances(descs):
    if not descs:
        return {}
    return {
        name: _pairwise_distance([d[name] for d in descs])
        for name in descs[0]
    }


def _metric_std_sums(descs):
    if not descs:
        return {}
    return {
        name: _descriptor_std_sum([d[name] for d in descs])
        for name in descs[0]
    }


def _run_case(label, spec):
    opts = spec.get("opts", default_opts())
    combine = spec.get("combine", "output_avg")
    fusion = spec.get("fusion", "interpolate")
    strength = spec.get("strength", 1.0)
    use_preset = spec.get("use_preset")
    descs = []
    images = []
    for seed in SEEDS:
        graph = _graph_full(seed, opts, combine, fusion, strength, use_preset)
        image = _run_graph(label, seed, graph)
        images.append(image["filename"])
        descs.append(_image_descriptors(image))
    drift_std = _metric_std_sums(descs)
    pairwise = _pairwise_metric_distances(descs)
    return {
        "images": images,
        "descriptor_std_sum": drift_std["foreground"],
        "pairwise_descriptor_distance": pairwise["foreground"],
        "metric_descriptor_std_sum": drift_std,
        "metric_pairwise_distances": pairwise,
    }


def _stable_seed_opts(**overrides):
    opts = default_opts(
        match_base_norm=False,
        contribution_balance=False,
        artist_static_capture=True,
        static_capture_k=4,
        static_capture_mode="output",
        static_capture_blend_alpha=0.25,
        artist_ema_alpha=0.0,
        artist_anchor_q=False,
        layer_filter="9-20",
    )
    opts.update(overrides)
    return opts


def _anchor_lock_opts(**overrides):
    opts = default_opts(
        match_base_norm=False,
        contribution_balance=False,
        artist_static_capture=False,
        artist_anchor_q=True,
        anchor_seeds_count=4,
        anchor_user_blend=0.0,
        anchor_deep_layer_threshold=16,
        artist_ema_alpha=0.0,
        layer_filter="9-25",
    )
    opts.update(overrides)
    return opts


def _selected_configs(configs):
    if not DRIFT_CONFIGS:
        return configs
    selected = {name: configs[name] for name in DRIFT_CONFIGS if name in configs}
    missing = [name for name in DRIFT_CONFIGS if name not in configs]
    if missing:
        raise SystemExit(
            "Unknown ANIMA_DRIFT_CONFIGS entries: "
            + ", ".join(missing)
            + ". Valid entries: "
            + ", ".join(configs)
        )
    if "unstabilized" not in selected:
        selected = {"unstabilized": configs["unstabilized"], **selected}
    return selected


def _summarize_reductions(reductions, metric_reductions,
                          comparison_metric="foreground"):
    labels = sorted(reductions)
    metrics = sorted(metric_reductions)
    best_by_metric = {}
    for metric in metrics:
        values = metric_reductions.get(metric, {})
        if values:
            best_by_metric[metric] = max(values, key=values.get)

    configs = {}
    for label in labels:
        per_metric = {
            metric: metric_reductions.get(metric, {}).get(label, 0.0)
            for metric in metrics
        }
        negative_metrics = [
            metric for metric, value in per_metric.items()
            if value < 0.0
        ]
        configs[label] = {
            "comparison_reduction": reductions[label],
            "all_positive": not negative_metrics,
            "negative_metrics": negative_metrics,
            "best_metric_count": sum(
                1 for winner in best_by_metric.values() if winner == label
            ),
        }

    best_label = None
    if reductions:
        best_label = max(reductions, key=reductions.get)
    return {
        "comparison_metric": comparison_metric,
        "best_by_comparison_metric": best_label,
        "best_by_metric": best_by_metric,
        "configs": configs,
    }


def aggregate_reduction_summaries(summaries):
    run_count = len(summaries)
    config_stats = {}
    for summary in summaries:
        reductions = summary.get("pairwise_distance_reduction", {})
        best_reduction = max(reductions.values()) if reductions else 0.0
        reduction_summary = summary.get("reduction_summary", {})
        best = reduction_summary.get("best_by_comparison_metric")
        configs = reduction_summary.get("configs", {})
        for label, reduction in reductions.items():
            stats = config_stats.setdefault(label, {
                "reductions": [],
                "regrets": [],
                "negative_reduction_count": 0,
                "winner_count": 0,
                "all_positive_count": 0,
                "negative_metric_counts": {},
            })
            stats["reductions"].append(float(reduction))
            stats["regrets"].append(float(best_reduction) - float(reduction))
            if float(reduction) < 0.0:
                stats["negative_reduction_count"] += 1
            if label == best:
                stats["winner_count"] += 1
            config = configs.get(label, {})
            if config.get("all_positive", False):
                stats["all_positive_count"] += 1
            for metric in config.get("negative_metrics", []):
                counts = stats["negative_metric_counts"]
                counts[metric] = counts.get(metric, 0) + 1

    configs_out = {}
    for label, stats in config_stats.items():
        reductions = stats["reductions"]
        average = _mean(reductions)
        configs_out[label] = {
            "runs": len(reductions),
            "average_reduction": average,
            "average_regret": _mean(stats["regrets"]),
            "max_regret": max(stats["regrets"]) if stats["regrets"] else 0.0,
            "min_reduction": min(reductions) if reductions else 0.0,
            "max_reduction": max(reductions) if reductions else 0.0,
            "complete": len(reductions) == run_count,
            "negative_reduction_count": stats["negative_reduction_count"],
            "winner_count": stats["winner_count"],
            "all_positive_count": stats["all_positive_count"],
            "negative_metric_counts": dict(sorted(
                stats["negative_metric_counts"].items()
            )),
        }

    best_by_available_average = None
    if configs_out:
        best_by_available_average = max(
            configs_out,
            key=lambda label: configs_out[label]["average_reduction"],
        )
    complete_configs = {
        label: stats for label, stats in configs_out.items()
        if stats["complete"]
    }
    best_by_average = None
    if complete_configs:
        best_by_average = max(
            complete_configs,
            key=lambda label: complete_configs[label]["average_reduction"],
        )
    best_by_regret = None
    if complete_configs:
        best_by_regret = min(
            complete_configs,
            key=lambda label: complete_configs[label]["average_regret"],
        )
    return {
        "runs": run_count,
        "best_by_average_reduction": best_by_average,
        "best_by_available_average_reduction": best_by_available_average,
        "best_by_average_regret": best_by_regret,
        "configs": configs_out,
    }


def extract_summary(payload):
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        return payload["summary"]
    return payload


def _aggregate_files(paths):
    summaries = []
    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            summaries.append(extract_summary(json.load(f)))
    return aggregate_reduction_summaries(summaries)


def write_json_result(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_config_registry():
    return {
        "unstabilized": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_ema_alpha=0.0, artist_static_capture=False,
            artist_anchor_q=False,
        )},
        "norm_only": {"opts": default_opts(
            contribution_balance=False, artist_ema_alpha=0.0,
            artist_static_capture=False, artist_anchor_q=False,
        )},
        "anchor_q": {"opts": default_opts(
            contribution_balance=False, artist_anchor_q=True,
            anchor_seeds_count=1, anchor_user_blend=0.0,
            artist_ema_alpha=0.0, artist_static_capture=False,
        )},
        "anchor_q_2seed": {"opts": default_opts(
            contribution_balance=False, artist_anchor_q=True,
            anchor_seeds_count=2, anchor_user_blend=0.0,
            artist_ema_alpha=0.0, artist_static_capture=False,
        )},
        "anchor_q_norm_ref": {"opts": default_opts(
            contribution_balance=False, artist_anchor_q=True,
            anchor_base_norm_ref=True, anchor_seeds_count=1,
            anchor_user_blend=0.0, artist_ema_alpha=0.0,
            artist_static_capture=False,
        )},
        "anchor_q_norm_ref_2seed": {"opts": default_opts(
            contribution_balance=False, artist_anchor_q=True,
            anchor_base_norm_ref=True, anchor_seeds_count=2,
            anchor_user_blend=0.0, artist_ema_alpha=0.0,
            artist_static_capture=False,
        )},
        "anchor_q_no_norm": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_anchor_q=True, anchor_seeds_count=1,
            anchor_user_blend=0.0, artist_ema_alpha=0.0,
            artist_static_capture=False,
        )},
        "anchor_q_no_norm_2seed": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_anchor_q=True, anchor_seeds_count=2,
            anchor_user_blend=0.0, artist_ema_alpha=0.0,
            artist_static_capture=False,
        )},
        "anchor_q_refresh_no_norm": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_anchor_q=True, anchor_refresh_each_step=True,
            anchor_seeds_count=1, anchor_user_blend=0.0,
            artist_ema_alpha=0.0, artist_static_capture=False,
        )},
        "anchor_q_refresh_norm_ref": {"opts": default_opts(
            contribution_balance=False, artist_anchor_q=True,
            anchor_refresh_each_step=True, anchor_base_norm_ref=True,
            anchor_seeds_count=1, anchor_user_blend=0.0,
            artist_ema_alpha=0.0, artist_static_capture=False,
        )},
        "anchor_q_mixed_norm": {"opts": default_opts(
            contribution_balance=False, artist_anchor_q=True,
            anchor_seeds_count=1, anchor_user_blend=0.0,
            artist_ema_alpha=0.0, artist_static_capture=False,
            norm_lock_scope="mixed",
        )},
        "anchor_q_blend25": {"opts": default_opts(
            contribution_balance=False, artist_anchor_q=True,
            anchor_seeds_count=1, anchor_user_blend=0.25,
            artist_ema_alpha=0.0, artist_static_capture=False,
        )},
        "static_k1": {"opts": default_opts(
            contribution_balance=False, artist_static_capture=True,
            static_capture_k=1, artist_ema_alpha=0.0,
            artist_anchor_q=False,
        )},
        "static_k2": {"opts": default_opts(
            contribution_balance=False, artist_static_capture=True,
            static_capture_k=2, artist_ema_alpha=0.0,
            artist_anchor_q=False,
        )},
        "static_k6": {"opts": default_opts(
            contribution_balance=False, artist_static_capture=True,
            static_capture_k=6, artist_ema_alpha=0.0,
            artist_anchor_q=False,
        )},
        "static_k4_no_norm": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_static_capture=True, static_capture_k=4,
            static_capture_mode="output",
            artist_ema_alpha=0.0, artist_anchor_q=False,
        )},
        "static_k4_no_norm_9_25": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_static_capture=True, static_capture_k=4,
            static_capture_mode="output",
            artist_ema_alpha=0.0, artist_anchor_q=False,
            layer_filter="9-25",
        )},
        "static_k4_no_norm_9_15": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_static_capture=True, static_capture_k=4,
            static_capture_mode="output",
            artist_ema_alpha=0.0, artist_anchor_q=False,
            layer_filter="9-15",
        )},
        "static_k4_no_norm_9_18": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_static_capture=True, static_capture_k=4,
            static_capture_mode="output",
            artist_ema_alpha=0.0, artist_anchor_q=False,
            layer_filter="9-18",
        )},
        "static_k4_no_norm_9_20": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_static_capture=True, static_capture_k=4,
            static_capture_mode="output",
            artist_ema_alpha=0.0, artist_anchor_q=False,
            layer_filter="9-20",
        )},
        "static_k4_delta_9_20": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_static_capture=True, static_capture_k=4,
            static_capture_mode="delta",
            artist_ema_alpha=0.0, artist_anchor_q=False,
            layer_filter="9-20",
        )},
        "static_k4_delta_9_21": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_static_capture=True, static_capture_k=4,
            static_capture_mode="delta",
            artist_ema_alpha=0.0, artist_anchor_q=False,
            layer_filter="9-21",
        )},
        "static_k4_no_norm_9_19": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_static_capture=True, static_capture_k=4,
            static_capture_mode="output",
            artist_ema_alpha=0.0, artist_anchor_q=False,
            layer_filter="9-19",
        )},
        "static_k4_no_norm_12_20": {"opts": default_opts(
            match_base_norm=False, contribution_balance=False,
            artist_static_capture=True, static_capture_k=4,
            static_capture_mode="output",
            artist_ema_alpha=0.0, artist_anchor_q=False,
            layer_filter="12-20",
        )},
        "output_avg_concat_base_9_20": {
            "opts": default_opts(
                match_base_norm=False, contribution_balance=False,
                artist_static_capture=False, artist_anchor_q=False,
                artist_ema_alpha=0.0, layer_filter="9-20",
            ),
            "combine": "output_avg",
            "fusion": "concat_with_base",
            "strength": 1.0,
        },
        "output_avg_concat_base_9_25": {
            "opts": default_opts(
                match_base_norm=False, contribution_balance=False,
                artist_static_capture=False, artist_anchor_q=False,
                artist_ema_alpha=0.0, layer_filter="9-25",
            ),
            "combine": "output_avg",
            "fusion": "concat_with_base",
            "strength": 1.0,
        },
        "compatibility_safe_layer_9_15": {
            "opts": default_opts(
                compatibility_mode=True,
                layer_filter="9-15",
            ),
        },
        "compatibility_safe_layer_9_18": {
            "opts": default_opts(
                compatibility_mode=True,
                layer_filter="9-18",
            ),
        },
        "compatibility_safe_layer_9_20": {
            "opts": default_opts(
                compatibility_mode=True,
                layer_filter="9-20",
            ),
        },
        "compatibility_safe_layer_9_25": {
            "opts": default_opts(
                compatibility_mode=True,
                layer_filter="9-25",
            ),
        },
        "compatibility_safe_layer_10_20": {
            "opts": default_opts(
                compatibility_mode=True,
                layer_filter="10-20",
            ),
        },
        "compatibility_safe_layer_12_20": {
            "opts": default_opts(
                compatibility_mode=True,
                layer_filter="12-20",
            ),
        },
        "balanced_preset": {"use_preset": "balanced"},
        "stable_seed_preset": {"use_preset": "stable_seed"},
        "drift_auto_preset": {"use_preset": "drift_auto"},
        "drift_soft_preset": {"use_preset": "drift_soft"},
        "face_lock_preset": {"use_preset": "face_lock"},
        "face_lock_strength085": {
            "opts": _stable_seed_opts(match_base_norm=True),
            "strength": 0.85,
        },
        "face_lock_norm_both": {
            "opts": _stable_seed_opts(match_base_norm=True, norm_lock_scope="both"),
            "strength": 1.0,
        },
        "face_lock_base_preserve": {
            "opts": _stable_seed_opts(match_base_norm=True),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "face_lock_base_preserve_strength085": {
            "opts": _stable_seed_opts(match_base_norm=True),
            "fusion": "base_preserve",
            "strength": 0.85,
        },
        "face_lock_layer_9_15": {
            "opts": _stable_seed_opts(match_base_norm=True, layer_filter="9-15"),
            "strength": 1.0,
        },
        "face_lock_k6": {
            "opts": _stable_seed_opts(match_base_norm=True, static_capture_k=6),
            "strength": 1.0,
        },
        "face_lock_k5_base_preserve": {
            "opts": _stable_seed_opts(match_base_norm=True, static_capture_k=5),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "face_lock_k6_base_preserve": {
            "opts": _stable_seed_opts(match_base_norm=True, static_capture_k=6),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "face_lock_k6_no_norm_base_preserve": {
            "opts": _stable_seed_opts(match_base_norm=False, static_capture_k=6),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "stable_seed_delta_cap075": {
            "opts": _stable_seed_opts(
                mixed_delta_cap=True,
                mixed_delta_cap_ratio=0.75,
            ),
            "strength": 1.0,
        },
        "stable_seed_delta_cap100": {
            "opts": _stable_seed_opts(
                mixed_delta_cap=True,
                mixed_delta_cap_ratio=1.0,
            ),
            "strength": 1.0,
        },
        "face_lock_delta_cap075": {
            "opts": _stable_seed_opts(
                match_base_norm=True,
                mixed_delta_cap=True,
                mixed_delta_cap_ratio=0.75,
            ),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "face_lock_delta_cap100": {
            "opts": _stable_seed_opts(
                match_base_norm=True,
                mixed_delta_cap=True,
                mixed_delta_cap_ratio=1.0,
            ),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "scene_lock_preset": {"use_preset": "scene_lock"},
        "scene_lock_strength085": {
            "opts": _stable_seed_opts(),
            "fusion": "base_preserve",
            "strength": 0.85,
        },
        "scene_lock_norm": {
            "opts": _stable_seed_opts(match_base_norm=True),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "scene_lock_blend_perp15": {
            "opts": _stable_seed_opts(
                static_capture_mode="blend_perp",
                static_capture_blend_alpha=0.15,
            ),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "scene_lock_layer_9_15": {
            "opts": _stable_seed_opts(layer_filter="9-15"),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "stable_seed_manual": {
            "opts": _stable_seed_opts(),
            "strength": 1.0,
        },
        "stable_seed_layer_9_18": {
            "opts": _stable_seed_opts(layer_filter="9-18"),
            "strength": 1.0,
        },
        "stable_seed_layer_9_19": {
            "opts": _stable_seed_opts(layer_filter="9-19"),
            "strength": 1.0,
        },
        "stable_seed_layer_8_20": {
            "opts": _stable_seed_opts(layer_filter="8-20"),
            "strength": 1.0,
        },
        "stable_seed_layer_10_20": {
            "opts": _stable_seed_opts(layer_filter="10-20"),
            "strength": 1.0,
        },
        "stable_seed_layer_9_21": {
            "opts": _stable_seed_opts(layer_filter="9-21"),
            "strength": 1.0,
        },
        "stable_seed_layer_10_21": {
            "opts": _stable_seed_opts(layer_filter="10-21"),
            "strength": 1.0,
        },
        "stable_seed_delta": {
            "opts": _stable_seed_opts(static_capture_mode="delta"),
            "strength": 1.0,
        },
        "stable_seed_delta_9_21": {
            "opts": _stable_seed_opts(static_capture_mode="delta", layer_filter="9-21"),
            "strength": 1.0,
        },
        "stable_seed_blend15": {
            "opts": _stable_seed_opts(
                static_capture_mode="blend",
                static_capture_blend_alpha=0.15,
            ),
            "strength": 1.0,
        },
        "stable_seed_blend25": {
            "opts": _stable_seed_opts(
                static_capture_mode="blend",
                static_capture_blend_alpha=0.25,
            ),
            "strength": 1.0,
        },
        "stable_seed_blend35": {
            "opts": _stable_seed_opts(
                static_capture_mode="blend",
                static_capture_blend_alpha=0.35,
            ),
            "strength": 1.0,
        },
        "stable_seed_blend_perp15": {
            "opts": _stable_seed_opts(
                static_capture_mode="blend_perp",
                static_capture_blend_alpha=0.15,
            ),
            "strength": 1.0,
        },
        "stable_seed_blend_perp25": {
            "opts": _stable_seed_opts(
                static_capture_mode="blend_perp",
                static_capture_blend_alpha=0.25,
            ),
            "strength": 1.0,
        },
        "stable_seed_blend_perp35": {
            "opts": _stable_seed_opts(
                static_capture_mode="blend_perp",
                static_capture_blend_alpha=0.35,
            ),
            "strength": 1.0,
        },
        "stable_seed_k2": {
            "opts": _stable_seed_opts(static_capture_k=2),
            "strength": 1.0,
        },
        "stable_seed_k3": {
            "opts": _stable_seed_opts(static_capture_k=3),
            "strength": 1.0,
        },
        "stable_seed_k5": {
            "opts": _stable_seed_opts(static_capture_k=5),
            "strength": 1.0,
        },
        "stable_seed_k6": {
            "opts": _stable_seed_opts(static_capture_k=6),
            "strength": 1.0,
        },
        "stable_seed_end92": {
            "opts": _stable_seed_opts(end_percent=0.92),
            "strength": 1.0,
        },
        "stable_seed_end90": {
            "opts": _stable_seed_opts(end_percent=0.90),
            "strength": 1.0,
        },
        "stable_seed_end85": {
            "opts": _stable_seed_opts(end_percent=0.85),
            "strength": 1.0,
        },
        "stable_seed_start05_end90": {
            "opts": _stable_seed_opts(start_percent=0.05, end_percent=0.90),
            "strength": 1.0,
        },
        "stable_seed_start10_end90": {
            "opts": _stable_seed_opts(start_percent=0.10, end_percent=0.90),
            "strength": 1.0,
        },
        "anchor_lock_preset": {"use_preset": "anchor_lock"},
        "anchor_lock_manual": {
            "opts": _anchor_lock_opts(),
            "strength": 1.2,
        },
        "anchor_lock_1anchor": {
            "opts": _anchor_lock_opts(anchor_seeds_count=1),
            "strength": 1.2,
        },
        "anchor_lock_2anchor": {
            "opts": _anchor_lock_opts(anchor_seeds_count=2),
            "strength": 1.2,
        },
        "stable_seed_strength1": {
            "opts": _stable_seed_opts(),
            "strength": 1.0,
        },
        "stable_seed_strength085": {
            "opts": _stable_seed_opts(),
            "strength": 0.85,
        },
        "stable_seed_strength08": {
            "opts": _stable_seed_opts(),
            "strength": 0.8,
        },
        "stable_seed_strength065": {
            "opts": _stable_seed_opts(),
            "strength": 0.65,
        },
        "stable_seed_strength05": {
            "opts": _stable_seed_opts(),
            "strength": 0.5,
        },
        "drift_soft_end92": {
            "opts": _stable_seed_opts(end_percent=0.92),
            "strength": 0.85,
        },
        "drift_soft_end90": {
            "opts": _stable_seed_opts(end_percent=0.90),
            "strength": 0.85,
        },
        "drift_soft_end85": {
            "opts": _stable_seed_opts(end_percent=0.85),
            "strength": 0.85,
        },
        "drift_soft_start05_end90": {
            "opts": _stable_seed_opts(start_percent=0.05, end_percent=0.90),
            "strength": 0.85,
        },
        "drift_soft_start10_end90": {
            "opts": _stable_seed_opts(start_percent=0.10, end_percent=0.90),
            "strength": 0.85,
        },
        "stable_seed_norm_strength1": {
            "opts": _stable_seed_opts(match_base_norm=True),
            "strength": 1.0,
        },
        "stable_seed_mid_9_15": {
            "opts": _stable_seed_opts(layer_filter="9-15"),
            "strength": 1.2,
        },
        "stable_seed_mid_9_15_strength1": {
            "opts": _stable_seed_opts(layer_filter="9-15"),
            "strength": 1.0,
        },
        "stable_seed_base_preserve": {
            "opts": _stable_seed_opts(),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "stable_seed_mid_base_preserve": {
            "opts": _stable_seed_opts(layer_filter="9-15"),
            "fusion": "base_preserve",
            "strength": 1.0,
        },
        "anchor_lock_l14": {
            "opts": _anchor_lock_opts(anchor_deep_layer_threshold=14),
            "strength": 1.2,
        },
        "anchor_lock_l17": {
            "opts": _anchor_lock_opts(anchor_deep_layer_threshold=17),
            "strength": 1.2,
        },
        "anchor_lock_all_anchor_layers": {
            "opts": _anchor_lock_opts(
                layer_filter="",
                anchor_deep_layer_threshold=-1,
            ),
            "strength": 1.2,
        },
        "identity_guard_preset": {"use_preset": "identity_guard"},
        "compatibility_safe_preset": {"use_preset": "compatibility_safe"},
    }


def main():
    if AGGREGATE_FILES:
        print(json.dumps(
            _aggregate_files(AGGREGATE_FILES),
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    configs = _selected_configs(build_config_registry())
    results = {label: _run_case(label, spec) for label, spec in configs.items()}
    base = results["unstabilized"]["pairwise_descriptor_distance"]
    reductions = {}
    for label, result in results.items():
        if label == "unstabilized":
            continue
        current = result["pairwise_descriptor_distance"]
        reductions[label] = 0.0 if base <= 1e-8 else (base - current) / base
    base_metrics = results["unstabilized"].get("metric_pairwise_distances", {})
    metric_reductions = {}
    for metric, metric_base in base_metrics.items():
        metric_reductions[metric] = {}
        for label, result in results.items():
            if label == "unstabilized":
                continue
            current = result["metric_pairwise_distances"].get(metric, 0.0)
            metric_reductions[metric][label] = (
                0.0 if metric_base <= 1e-8 else (metric_base - current) / metric_base
            )
    results["summary"] = {
        "seeds": SEEDS,
        "artists": DRIFT_ARTISTS,
        "base_prompt": DRIFT_BASE_PROMPT,
        "comparison_metric": "foreground",
        "pairwise_distance_reduction": reductions,
        "metric_pairwise_distance_reduction": metric_reductions,
        "reduction_summary": _summarize_reductions(
            reductions, metric_reductions, comparison_metric="foreground",
        ),
        "server": SERVER,
    }
    if SAVE_JSON:
        write_json_result(SAVE_JSON, results)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
