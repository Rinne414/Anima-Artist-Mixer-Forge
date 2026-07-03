"""Recipe serialization: share a full mixer setup as a single JSON string."""

import json

from .constants import (
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    ANCHOR_SEEDS_MAX,
    COMBINE_CHOICES,
    COMBINE_OUTPUT_AVG,
    FUSION_CHOICES,
    FUSION_INTERPOLATE,
    MAX_ARTISTS,
    MIXED_DELTA_CAP_RATIO_MAX,
    PRESET_CHOICES,
    RECIPE_FORMAT,
    RECIPE_VERSION,
    STATIC_CAPTURE_K_MAX,
)
from .options import base_advanced_options
from .parsing import clamp_float

# Authoritative numeric ranges mirror the UI widget min/max in
# nodes_ui.AnimaArtistOptions so a hand-edited or malformed recipe cannot push
# a value past what the node UI would ever produce.
_RECIPE_NUMERIC_RANGES = {
    "start_block": (0, 63),
    "end_block": (-1, 63),
    "start_percent": (0.0, 1.0),
    "end_percent": (0.0, 1.0),
    "artist_ema_alpha": (0.0, 0.95),
    "lowrank_k": (1, MAX_ARTISTS),
    "static_capture_k": (1, STATIC_CAPTURE_K_MAX),
    "static_capture_blend_alpha": (0.0, 1.0),
    "anchor_seeds_count": (1, ANCHOR_SEEDS_MAX),
    "anchor_user_blend": (0.0, 1.0),
    "anchor_deep_layer_threshold": (ANCHOR_LAYER_THRESHOLD_DISABLED, 64),
    "stabilizer_end_percent": (0.0, 1.0),
    "max_batch_artists": (0, MAX_ARTISTS),
    "contribution_balance_alpha": (0.0, 1.0),
    "mixed_delta_cap_ratio": (0.0, MIXED_DELTA_CAP_RATIO_MAX),
}


def _clamp_intensity(value):
    try:
        return clamp_float(value, 0.0, 2.0)
    except (TypeError, ValueError):
        return 1.0


def serialize_recipe(artist_chain, combine_mode, fusion_mode, strength,
                     advanced_options=None, notes="", source_preset=None):
    """Pack a mixer configuration into a stable, shareable JSON string.

    ``source_preset`` is the ANIMA_PRESET payload that produced these options.
    Its name (and knobs) are stored so a deferred preset such as drift_auto can
    re-resolve against the real prompt when the recipe is loaded, instead of
    baking whatever route the empty-prompt save happened to pick.
    """
    adv = base_advanced_options()
    if isinstance(advanced_options, dict):
        for key in adv:
            if key in advanced_options:
                adv[key] = advanced_options[key]
    payload = {
        "format": RECIPE_FORMAT,
        "version": RECIPE_VERSION,
        "artist_chain": str(artist_chain or ""),
        "preset": "",
        "combine_mode": combine_mode if combine_mode in COMBINE_CHOICES else COMBINE_OUTPUT_AVG,
        "fusion_mode": fusion_mode if fusion_mode in FUSION_CHOICES else FUSION_INTERPOLATE,
        "strength": clamp_float(strength, 0.0, 4.0),
        "advanced_options": adv,
        "notes": str(notes or ""),
    }
    if isinstance(source_preset, dict):
        name = str(source_preset.get("preset") or "")
        payload["preset"] = name
        if name:
            payload["preset_intensity"] = _clamp_intensity(source_preset.get("intensity", 1.0))
            payload["preset_layer_mode"] = str(source_preset.get("layer_mode") or "")
            payload["preset_custom_layer_filter"] = str(
                source_preset.get("custom_layer_filter") or ""
            )
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _coerce_bool_value(value, default, key, warnings):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
    elif isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1"):
            return True
        if v in ("false", "0"):
            return False
    warnings.append(
        f"invalid value for advanced option {key!r}: {value!r}; using the default"
    )
    return default


def _coerce_number_value(value, default, key, is_int, bounds, warnings):
    try:
        num = float(value)
    except (TypeError, ValueError):
        warnings.append(
            f"invalid value for advanced option {key!r}: {value!r}; using the default"
        )
        return default
    if num != num:  # NaN
        warnings.append(
            f"invalid value for advanced option {key!r}: NaN; using the default"
        )
        return default
    if bounds is not None:
        lo, hi = bounds
        clamped = max(lo, min(hi, num))
        if clamped != num:
            warnings.append(
                f"advanced option {key!r} value {num!r} out of range "
                f"[{lo}, {hi}]; clamped to {clamped}"
            )
        num = clamped
    return int(round(num)) if is_int else float(num)


def _coerce_advanced_options(raw_adv, warnings):
    adv = base_advanced_options()
    if not isinstance(raw_adv, dict):
        return adv
    for key, default_value in adv.items():
        if key not in raw_adv:
            continue
        value = raw_adv[key]
        # bool must be checked before int (bool is a subclass of int).
        if isinstance(default_value, bool):
            adv[key] = _coerce_bool_value(value, default_value, key, warnings)
        elif isinstance(default_value, int):
            adv[key] = _coerce_number_value(
                value, default_value, key, True,
                _RECIPE_NUMERIC_RANGES.get(key), warnings,
            )
        elif isinstance(default_value, float):
            adv[key] = _coerce_number_value(
                value, default_value, key, False,
                _RECIPE_NUMERIC_RANGES.get(key), warnings,
            )
        elif isinstance(default_value, str):
            adv[key] = str(value)
        else:
            adv[key] = value
    unknown = set(raw_adv) - set(adv)
    if unknown:
        warnings.append(f"ignored unknown advanced options: {sorted(unknown)}")
    return adv


def deserialize_recipe(recipe_json):
    """Parse a recipe JSON string. Returns (payload, warnings).

    Unknown keys are ignored; missing keys fall back to defaults so older
    recipes keep loading after the schema grows. Out-of-range numeric options
    and mistyped booleans are clamped/parsed with a warning rather than passed
    straight through to the mixer.
    """
    warnings = []
    try:
        data = json.loads(str(recipe_json or ""))
    except (TypeError, ValueError) as e:
        raise ValueError(f"[AnimaArtistRecipe] invalid recipe JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError("[AnimaArtistRecipe] recipe JSON must be an object")

    fmt = data.get("format")
    if fmt != RECIPE_FORMAT:
        warnings.append(f"unexpected format marker {fmt!r}; trying to load anyway")
    version = data.get("version")
    if isinstance(version, int) and version > RECIPE_VERSION:
        warnings.append(
            f"recipe version {version} is newer than supported {RECIPE_VERSION}; "
            "some settings may be ignored"
        )

    combine_mode = data.get("combine_mode", COMBINE_OUTPUT_AVG)
    if combine_mode not in COMBINE_CHOICES:
        warnings.append(f"unknown combine_mode {combine_mode!r}; using {COMBINE_OUTPUT_AVG}")
        combine_mode = COMBINE_OUTPUT_AVG
    fusion_mode = data.get("fusion_mode", FUSION_INTERPOLATE)
    if fusion_mode not in FUSION_CHOICES:
        warnings.append(f"unknown fusion_mode {fusion_mode!r}; using {FUSION_INTERPOLATE}")
        fusion_mode = FUSION_INTERPOLATE

    try:
        strength = clamp_float(data.get("strength", 1.0), 0.0, 4.0)
    except (TypeError, ValueError):
        warnings.append("invalid strength; using 1.0")
        strength = 1.0

    preset_name = data.get("preset")
    if not isinstance(preset_name, str):
        preset_name = ""
    if preset_name and preset_name not in PRESET_CHOICES:
        warnings.append(
            f"unknown source preset {preset_name!r}; loading as a baked recipe"
        )
        preset_name = ""

    adv = _coerce_advanced_options(data.get("advanced_options"), warnings)

    payload = {
        "artist_chain": str(data.get("artist_chain", "") or ""),
        "preset": preset_name,
        "preset_intensity": _clamp_intensity(data.get("preset_intensity", 1.0)),
        "preset_layer_mode": str(data.get("preset_layer_mode") or ""),
        "preset_custom_layer_filter": str(data.get("preset_custom_layer_filter") or ""),
        "combine_mode": combine_mode,
        "fusion_mode": fusion_mode,
        "strength": strength,
        "advanced_options": adv,
        "notes": str(data.get("notes", "") or ""),
    }
    return payload, warnings
