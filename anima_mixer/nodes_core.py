"""Core ComfyUI nodes: Pack, CrossAttn patcher, and the layer Probe."""

import logging
import uuid

import torch

from .anchor import make_sigma_capture
from .chain_tools import format_layer_span, lint_parsed_artists
from .constants import (
    COMBINE_CHOICES,
    COMBINE_CONCAT,
    COMBINE_LOWRANK_AVG,
    COMBINE_OUTPUT_AVG,
    FUSION_BASE_PRESERVE,
    FUSION_CHOICES,
    FUSION_CONCAT_WITH_BASE,
    FUSION_INTERPOLATE,
    ANCHOR_LAYER_THRESHOLD_DISABLED,
    ANCHOR_SEEDS_MAX,
    CONTRIB_BALANCE_ALPHA_DEFAULT,
    MIXED_DELTA_CAP_RATIO_DEFAULT,
    MIXED_DELTA_CAP_RATIO_MAX,
    NORM_LOCK_SCOPE_PER_ARTIST,
    NORM_LOCK_TOKEN,
    MAX_ARTISTS,
    PRESET_BALANCED,
    PRESET_RECOMMENDED_CHOICES,
    STATIC_CAPTURE_BLEND_ALPHA_DEFAULT,
    STATIC_CAPTURE_K_DEFAULT,
    STATIC_CAPTURE_K_MAX,
    STATIC_CAPTURE_MODE_OUTPUT,
)
from .options import build_preset_payload, merge_runtime_options
from .parsing import (
    build_passthrough_prompt,
    expand_prompt_weights,
    parse_artist_entries,
    parse_artist_layer_routes,
    parse_artist_timing_routes,
    resolve_artist_layer_routes,
    resolve_artist_timing_routes,
    resolve_target_blocks_from_options,
    split_artist_chain,
)
from .patching import (
    describe_external_cross_attn_patches,
    extract_conditioning,
    make_cross_attn_forward_patch,
    unwrap_cross_attn,
    unwrap_cross_attn_forward,
    validate_model,
)
from .wrapper import CrossAttnWrapper

logger = logging.getLogger(__name__)


class AnyType(str):
    """Wildcard type marker accepted by ComfyUI for 'any' inputs."""

    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")

# Probe results live here between graph executions, keyed by probe_id. Only
# a slim view of the runtime state is stored (the stats containers, shared
# by reference with the live wrappers) — never the full state, which would
# pin the diffusion model and artist tensors in memory across runs.
PROBE_REGISTRY = {}
_PROBE_REGISTRY_LIMIT = 8

# Small FIFO memo so AnimaArtistBasic does not re-encode every artist when only
# preset/intensity change. Keyed by (chain, prompt); a hit additionally
# requires the pack's stored clip to be the SAME object (identity check) —
# id()-based keys are unsafe because CPython reuses a freed clip's address for
# its replacement, which would silently serve conditionings from the old
# text encoder.
_BASIC_PACK_CACHE = {}
_BASIC_PACK_CACHE_LIMIT = 4


def _registry_store(probe_id, state):
    PROBE_REGISTRY[probe_id] = {
        "probe_stats": state["probe_stats"],
        "probe_labels": state["probe_labels"],
        "probe_num_blocks": state["probe_num_blocks"],
        "_probe_seen_sigmas": state["_probe_seen_sigmas"],
    }
    while len(PROBE_REGISTRY) > _PROBE_REGISTRY_LIMIT:
        PROBE_REGISTRY.pop(next(iter(PROBE_REGISTRY)))


class AnimaArtistPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Artist chain. Separate artists with commas or newlines.\n"
                        "Example: wlop, sakimichan, krenz\n"
                        "\n"
                        "Two weight syntaxes (they can coexist and stack):\n"
                        "  1) parentheses (wlop:1.5) - CLIP-side, non-linear\n"
                        "  2) ::weight ::wlop::1.5 - injection-side, linear\n"
                        "\n"
                        "Default weight 1.0; range [-4.0, 4.0]. Negative weights\n"
                        "subtract that artist's style (style subtraction).\n"
                        "::weight stacks with parentheses: ::(wlop:1.1)::0.8\n"
                        "Optional per-artist layer route: wlop@0-8, krenz@33%-67%\n"
                        "Optional per-artist timing: wlop@0-8%0.0-0.45\n"
                        "Use % for sampling timing; @0.0-0.5 is a layer range.\n"
                        "Optional timing fade: wlop%0.0-0.45~0.1 (smoothstep edges)\n"
                        "\n"
                        "When any artist uses ::weight, normalize_weights is\n"
                        "bypassed at runtime (explicit weights stay absolute)."
                    )
                }),
            },
            "optional": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Main prompt (optional). Follows Anima's recommended layout: "
                        "artist first, then a newline, then the main prompt "
                        "('<artist>\\n<base_prompt>'). Leave empty to encode the "
                        "artist names alone."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_PACK",)
    RETURN_NAMES = ("artist_pack",)
    FUNCTION = "pack"
    CATEGORY = "Anima/CrossAttn"

    def pack(self, clip, artist_chain, base_prompt=""):
        raw_artist_chain = str(artist_chain or "").strip()
        parts = split_artist_chain(artist_chain)
        parts, timing_routes = parse_artist_timing_routes(parts)
        parts, layer_routes = parse_artist_layer_routes(parts)
        entries = parse_artist_entries(parts)
        names = [entry[0] for entry in entries]
        parsed_weights = [entry[1] for entry in entries]
        explicit_flags = [entry[2] for entry in entries]
        has_explicit = any(explicit_flags)

        for hint in lint_parsed_artists(names, layer_routes, timing_routes, raw_artist_chain):
            logger.warning("[AnimaArtistPack] %s", hint)

        # v26: Expand weight::target:: syntax in base_prompt
        base = expand_prompt_weights((base_prompt or "").strip())

        try:
            base_tokens = clip.tokenize(base)
            base_conditioning = clip.encode_from_tokens_scheduled(base_tokens)
        except Exception as e:
            raise ValueError(
                f"[AnimaArtistPack] failed to encode base_prompt (text={base!r}): {e}"
            )

        if not names:
            return ({
                "conditionings": [],
                "labels": [],
                "weights": [],
                "layer_routes": [],
                "timing_routes": [],
                "has_explicit_weights": False,
                "raw_artist_chain": raw_artist_chain,
                "base_prompt": base,
                "base_conditioning": base_conditioning,
                "clip": clip,
            },)

        if len(names) > MAX_ARTISTS:
            logger.warning(
                "[AnimaArtistPack] artist count %d exceeds the limit %d; truncating",
                len(names), MAX_ARTISTS,
            )
            names = names[:MAX_ARTISTS]
            parsed_weights = parsed_weights[:MAX_ARTISTS]
            explicit_flags = explicit_flags[:MAX_ARTISTS]
            layer_routes = layer_routes[:MAX_ARTISTS]
            timing_routes = timing_routes[:MAX_ARTISTS]
            # An explicit weight only on a truncated entry must not leak into
            # the surviving chain: it would disable normalization and skip the
            # weight-sum guard for 32 weight-1.0 artists.
            has_explicit = any(explicit_flags)

        conditionings = []
        for name in names:
            text = f"{name}\n{base}" if base else name
            try:
                tokens = clip.tokenize(text)
                cond = clip.encode_from_tokens_scheduled(tokens)
            except Exception as e:
                raise ValueError(
                    f"[AnimaArtistPack] encoding failed (text={text!r}): {e}"
                )
            conditionings.append(cond)

        if has_explicit:
            logger.info(
                "[AnimaArtistPack] %d artists carry ::weight syntax; the linear "
                "injection path will be used",
                sum(1 for flag in explicit_flags if flag),
            )

        return ({
            "conditionings": conditionings,
            "labels": names,
            "weights": parsed_weights,
            "layer_routes": layer_routes,
            "timing_routes": timing_routes,
            "has_explicit_weights": has_explicit,
            "raw_artist_chain": raw_artist_chain,
            "base_prompt": base,
            "base_conditioning": base_conditioning,
            "clip": clip,
        },)


class AnimaArtistBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "artist_chain": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Artists separated by comma or newline.",
                }),
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Main positive prompt. Do not repeat artist names here.",
                }),
                "preset": (PRESET_RECOMMENDED_CHOICES, {
                    "default": PRESET_BALANCED,
                    "tooltip": (
                        "Recommended modes only. Use Anima Artist Preset for "
                        "advanced modes.\n"
                        "balanced: original-compatible default\n"
                        "strong_style: stronger artist style\n"
                        "drift_auto: automatic low-drift route\n"
                        "prompt_passthrough: direct prompt/no mixer"
                    ),
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Preset strength multiplier.",
                }),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "apply"
    CATEGORY = "Anima/CrossAttn"

    def apply(self, model, clip, artist_chain, base_prompt, preset, intensity, enabled):
        cache_key = (artist_chain, base_prompt)
        artist_pack = _BASIC_PACK_CACHE.get(cache_key)
        if artist_pack is None or artist_pack.get("clip") is not clip:
            artist_pack = AnimaArtistPack().pack(clip, artist_chain, base_prompt)[0]
            _BASIC_PACK_CACHE[cache_key] = artist_pack
            while len(_BASIC_PACK_CACHE) > _BASIC_PACK_CACHE_LIMIT:
                _BASIC_PACK_CACHE.pop(next(iter(_BASIC_PACK_CACHE)))
        payload = build_preset_payload(
            preset,
            intensity,
            normalize_weights=True,
            artist_count=len(artist_pack.get("labels") or []),
        )
        return AnimaArtistCrossAttn().patch(
            model,
            artist_pack,
            COMBINE_OUTPUT_AVG,
            FUSION_INTERPOLATE,
            1.0,
            enabled,
            False,
            preset=payload,
        )


# ComfyUI's percent_to_sigma(0.0) returns a huge sentinel (999999999.9)
# meaning "always active". Fine for window-inclusion tests, but useless as a
# finite anchor for fade interpolation.
_SIGMA_SENTINEL = 1e8


def _finite_sigma(ms, percent):
    """percent_to_sigma with a finite stand-in for the percent=0 sentinel.

    The stand-in is the real near-max sigma padded by 0.1% so the very first
    sampling step (at sigma_max) still falls inside the window.
    """
    s = float(ms.percent_to_sigma(percent))
    if s >= _SIGMA_SENTINEL:
        s = float(ms.percent_to_sigma(1e-4)) * 1.001
    return s


def _percent_window_to_sigma_route(ms, timing):
    """Convert a percent-space (start, end, fade) window to sigma space.

    Returns (lo, hi, fade_in_lo, fade_out_hi) for timing_fade_factor.
    Sigma decreases as percent increases, so window start -> hi.
    """
    start, end, fade = timing
    fade_eff = min(float(fade), max(0.0, (end - start) / 2.0))
    hi = _finite_sigma(ms, start)
    lo = _finite_sigma(ms, end)
    fade_in_lo = _finite_sigma(ms, min(start + fade_eff, end))
    fade_out_hi = _finite_sigma(ms, max(end - fade_eff, start))
    lo, hi = min(lo, hi), max(lo, hi)
    fade_in_lo = min(max(fade_in_lo, lo), hi)
    fade_out_hi = min(max(fade_out_hi, lo), hi)
    return (lo, hi, fade_in_lo, fade_out_hi)


def _build_runtime_state(enabled, fusion_mode, combine_mode, strength,
                         apply_to_uncond, raws, ids_list, w_list, user_weights,
                         labels, artist_layer_routes, has_artist_layer_routes,
                         artist_timing_routes, has_artist_timing_routes,
                         normalize_w, has_explicit_weights, preset_name,
                         adv, dm, sigma_range, external_patches):
    return {
        "enabled": bool(enabled),
        "fusion_mode": fusion_mode,
        "combine_mode": combine_mode,
        "strength": float(strength),
        "apply_to_uncond": bool(apply_to_uncond),
        "raws": raws,
        "ids_list": ids_list,
        "w_list": w_list,
        "user_weights": user_weights,
        "labels": labels,
        "artist_layer_routes": artist_layer_routes,
        "has_artist_layer_routes": has_artist_layer_routes,
        "artist_timing_routes": artist_timing_routes,
        "has_artist_timing_routes": has_artist_timing_routes,
        "normalize_weights": normalize_w,
        "has_explicit_weights": has_explicit_weights,
        "preset_name": preset_name,
        "artist_ema_alpha": float(adv.get("artist_ema_alpha", 0.0)),
        "lowrank_k": int(adv.get("lowrank_k", 1)),
        "artist_static_capture": bool(adv.get("artist_static_capture", False)),
        "static_capture_k": int(adv.get("static_capture_k", STATIC_CAPTURE_K_DEFAULT)),
        "static_capture_mode": str(adv.get("static_capture_mode", STATIC_CAPTURE_MODE_OUTPUT)),
        "static_capture_blend_alpha": float(
            adv.get("static_capture_blend_alpha", STATIC_CAPTURE_BLEND_ALPHA_DEFAULT)
        ),
        "artist_anchor_q": bool(adv.get("artist_anchor_q", False)),
        "anchor_seeds_count": int(adv.get("anchor_seeds_count", 1)),
        "anchor_user_blend": float(adv.get("anchor_user_blend", 0.0)),
        "anchor_deep_layer_threshold": int(
            adv.get("anchor_deep_layer_threshold", ANCHOR_LAYER_THRESHOLD_DISABLED)
        ),
        "anchor_refresh_each_step": bool(adv.get("anchor_refresh_each_step", False)),
        "max_batch_artists": int(adv.get("max_batch_artists", 0) or 0),
        "low_vram_cache": bool(adv.get("low_vram_cache", False)),
        "match_base_norm": bool(adv.get("match_base_norm", False)),
        "anchor_base_norm_ref": bool(adv.get("anchor_base_norm_ref", False)),
        "norm_lock_mode": str(adv.get("norm_lock_mode", NORM_LOCK_TOKEN)),
        "norm_lock_scope": str(adv.get("norm_lock_scope", NORM_LOCK_SCOPE_PER_ARTIST)),
        "contribution_balance": bool(adv.get("contribution_balance", False)),
        "contribution_balance_alpha": float(
            adv.get("contribution_balance_alpha", CONTRIB_BALANCE_ALPHA_DEFAULT)
        ),
        "mixed_delta_cap": bool(adv.get("mixed_delta_cap", False)),
        "mixed_delta_cap_ratio": float(
            adv.get("mixed_delta_cap_ratio", MIXED_DELTA_CAP_RATIO_DEFAULT)
        ),
        "individuals": None,
        "real_lens": None,
        "dm_ref": dm,
        "sigma_range": sigma_range,
        "stabilizer_min_sigma": adv.get("stabilizer_min_sigma"),
        "current_sigma": None,
        "external_cross_attn_patches": external_patches,
        "_disabled_layers": set(),
        "_run_last_sigma": None,
        "_ema_cache": {},
        "_ema_last_sigma": None,
        "_static_cache": {},
        "_static_last_sigma": None,
        "_ctx_fp_memo": {},
        "_anchor_cache": {},
        "_anchor_base_cache": {},
        "_anchor_cache_key": None,
        "_anchor_last_sigma": None,
        "_in_anchor_run": False,
        "_anchor_failed": False,
    }


class AnimaArtistCrossAttn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "combine_mode": (
                    COMBINE_CHOICES,
                    {
                        "default": COMBINE_OUTPUT_AVG,
                        "tooltip": (
                            "concat: concatenate artist tokens as K/V\n"
                            "output_avg: weighted average of per-artist attention outputs\n"
                            "lowrank_avg: deterministic low-rank constraint on artist deltas"
                        ),
                    },
                ),
                "fusion_mode": (
                    FUSION_CHOICES,
                    {
                        "default": FUSION_INTERPOLATE,
                        "tooltip": (
                            "interpolate: out = lerp(base, artist, strength)\n"
                            "concat_with_base: KV=[base; artist] single forward\n"
                            "base_preserve: strip the artist component parallel to base\n"
                            "  out = base + strength * proj_perp(artist - base)\n"
                            "  base content direction stays untouched\n"
                            "  compatible with lowrank_avg / EMA"
                        ),
                    },
                ),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": (
                        "Artist injection strength.\n"
                        "0.0-1.0: interpolation lerp(base, artist, strength)\n"
                        "1.0-4.0: extrapolation base + strength * (artist - base)\n"
                        "  Amplifies the artist's deviation; decoupled from artist count.\n"
                        "  Recommended 1.5-2.5; above 3 tends to oversaturate."
                    ),
                }),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS",),
                "preset": ("ANIMA_PRESET", {
                    "tooltip": (
                        "Compatibility input for older workflows. For preset "
                        "workflows, prefer Anima Artist Apply Preset so the "
                        "manual combine/fusion/strength widgets are not shown."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "patch"
    CATEGORY = "Anima/CrossAttn"

    def patch(self, model, artist_pack, combine_mode, fusion_mode,
              strength, enabled, apply_to_uncond, advanced_options=None, preset=None):
        base_prompt = ""
        artist_count = 0
        if isinstance(artist_pack, dict):
            base_prompt = str(artist_pack.get("base_prompt", "") or "")
            artist_count = len(artist_pack.get("labels") or [])
        combine_mode, fusion_mode, strength, adv, preset_name = merge_runtime_options(
            combine_mode, fusion_mode, strength, advanced_options, preset,
            base_prompt=base_prompt,
            artist_count=artist_count,
        )

        if not isinstance(artist_pack, dict):
            raise ValueError(
                "[AnimaCrossAttn] artist_pack has the wrong type; connect the "
                "output of an AnimaArtistPack node"
            )

        base_cond_out = artist_pack.get("base_conditioning")
        if base_cond_out is None:
            raise ValueError(
                "[AnimaCrossAttn] artist_pack is missing base_conditioning. "
                "Restart ComfyUI so AnimaArtistPack reloads at the current version"
            )

        conditionings = artist_pack.get("conditionings") or []
        if not enabled or not conditionings:
            # Nothing to inject: hand back the unpatched model with zero overhead.
            return (model, base_cond_out)

        labels = artist_pack.get("labels") or []
        parsed_weights = artist_pack.get("weights")
        if isinstance(parsed_weights, (list, tuple)) and len(parsed_weights) == len(labels):
            user_weights = [float(w) for w in parsed_weights]
        else:
            user_weights = [1.0] * len(labels)

        if adv.get("prompt_passthrough", False):
            clip = artist_pack.get("clip")
            if clip is None:
                raise ValueError(
                    "[AnimaCrossAttn] prompt_passthrough requires an artist_pack "
                    "built by the current AnimaArtistPack node."
                )
            if any(str(route or "").strip() for route in artist_pack.get("layer_routes") or []):
                raise ValueError(
                    "[AnimaCrossAttn] prompt_passthrough does not support layer routes; "
                    "use balanced or another mixer preset for @layer routing."
                )
            if any(str(route or "").strip() for route in artist_pack.get("timing_routes") or []):
                raise ValueError(
                    "[AnimaCrossAttn] prompt_passthrough does not support timing routes; "
                    "use balanced or another mixer preset for %timing routing."
                )
            prompt_text = build_passthrough_prompt(
                artist_pack.get("raw_artist_chain", ""),
                labels,
                user_weights,
                artist_pack.get("has_explicit_weights", False),
                base_prompt,
            )
            try:
                tokens = clip.tokenize(prompt_text)
                direct_conditioning = clip.encode_from_tokens_scheduled(tokens)
            except Exception as e:
                raise ValueError(
                    "[AnimaCrossAttn] prompt_passthrough failed to encode direct "
                    f"prompt (text={prompt_text!r}): {e}"
                )
            return (model, direct_conditioning)

        start_percent = float(adv.get("start_percent", 0.0))
        end_percent = float(adv.get("end_percent", 1.0))
        normalize_w = bool(adv.get("normalize_weights", True))
        artist_ema_alpha = float(adv.get("artist_ema_alpha", 0.0))
        artist_static_capture = bool(adv.get("artist_static_capture", False))
        static_capture_k = int(adv.get("static_capture_k", STATIC_CAPTURE_K_DEFAULT))
        adv["static_capture_k"] = max(1, min(static_capture_k, STATIC_CAPTURE_K_MAX))
        static_blend_alpha = float(
            adv.get("static_capture_blend_alpha", STATIC_CAPTURE_BLEND_ALPHA_DEFAULT)
        )
        adv["static_capture_blend_alpha"] = max(0.0, min(1.0, static_blend_alpha))
        artist_anchor_q = bool(adv.get("artist_anchor_q", False))
        anchor_seeds_count = int(adv.get("anchor_seeds_count", 1))
        adv["anchor_seeds_count"] = max(1, min(anchor_seeds_count, ANCHOR_SEEDS_MAX))
        anchor_user_blend = float(adv.get("anchor_user_blend", 0.0))
        adv["anchor_user_blend"] = max(0.0, min(1.0, anchor_user_blend))
        contribution_balance_alpha = float(
            adv.get("contribution_balance_alpha", CONTRIB_BALANCE_ALPHA_DEFAULT)
        )
        adv["contribution_balance_alpha"] = max(0.0, min(1.0, contribution_balance_alpha))
        mixed_delta_cap_ratio = float(
            adv.get("mixed_delta_cap_ratio", MIXED_DELTA_CAP_RATIO_DEFAULT)
        )
        adv["mixed_delta_cap_ratio"] = max(
            0.0, min(MIXED_DELTA_CAP_RATIO_MAX, mixed_delta_cap_ratio)
        )
        stabilizer_end_percent = float(adv.get("stabilizer_end_percent", 1.0))
        adv["stabilizer_end_percent"] = max(0.0, min(1.0, stabilizer_end_percent))

        use_sigma_range = (start_percent > 0.0) or (end_percent < 1.0)
        use_stabilizer_window = adv["stabilizer_end_percent"] < 1.0
        need_sigma_capture = (
            use_sigma_range or (artist_ema_alpha > 0.0)
            or artist_static_capture or artist_anchor_q
            or use_stabilizer_window
        )

        # Mutual-exclusion checks.
        if artist_static_capture and artist_ema_alpha > 0.0:
            logger.info(
                "[AnimaCrossAttn] artist_ema_alpha=%.2f is ignored while "
                "artist_static_capture is on (artist outputs are already static).",
                artist_ema_alpha,
            )
        if artist_static_capture and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning(
                "[AnimaCrossAttn] artist_static_capture is incompatible with "
                "fusion=concat_with_base (x changes every step); static capture "
                "is ignored for this run."
            )
        if artist_anchor_q and artist_static_capture:
            logger.warning(
                "[AnimaCrossAttn] artist_anchor_q and artist_static_capture are "
                "mutually exclusive; static_capture is disabled (anchor_q wins)."
            )
            adv["artist_static_capture"] = False
        if artist_anchor_q and fusion_mode == FUSION_CONCAT_WITH_BASE:
            logger.warning(
                "[AnimaCrossAttn] artist_anchor_q is incompatible with "
                "fusion=concat_with_base; anchor_q is disabled for this run."
            )
            adv["artist_anchor_q"] = False
        if not adv.get("artist_anchor_q", False):
            adv["anchor_base_norm_ref"] = False
            adv["anchor_refresh_each_step"] = False
        layer_route_texts = artist_pack.get("layer_routes") or []
        timing_route_texts = artist_pack.get("timing_routes") or []

        raws, ids_list, w_list = [], [], []
        for idx, c in enumerate(conditionings):
            raw, ids, w = extract_conditioning(c)
            if raw is None:
                label = labels[idx] if idx < len(labels) else f"#{idx}"
                raise ValueError(
                    f"[AnimaCrossAttn] artist[{label}] conditioning is empty. "
                    "Do the CLIP and model match?"
                )
            raws.append(raw)
            ids_list.append(ids)
            w_list.append(w)

        n = len(raws)
        has_explicit_weights = bool(artist_pack.get("has_explicit_weights", False))
        if len(user_weights) != n:
            user_weights = [1.0] * n
            has_explicit_weights = False

        if has_explicit_weights and normalize_w:
            normalize_w = False
            logger.info(
                "[AnimaCrossAttn] explicit ::weight detected; "
                "normalize_weights is bypassed."
            )

        if any(w < 0.0 for w in user_weights):
            logger.info(
                "[AnimaCrossAttn] negative artist weights detected; those "
                "artists subtract their style direction (style subtraction)."
            )
            if combine_mode == COMBINE_CONCAT:
                logger.warning(
                    "[AnimaCrossAttn] combine=concat treats negative weights as "
                    "sign-flipped K/V tokens, not style subtraction. Delta-space "
                    "style subtraction only exists in output_avg/lowrank_avg."
                )

        if fusion_mode == FUSION_BASE_PRESERVE and float(strength) < 0.3:
            logger.info(
                "[AnimaCrossAttn] fusion=base_preserve at strength=%.2f (<0.3) "
                "is very subtle; consider strength >= 0.7.", float(strength),
            )

        if float(strength) > 1.0:
            logger.info(
                "[AnimaCrossAttn] strength=%.2f > 1.0 enters extrapolation: "
                "out = base + %.2f * (artist - base). %s.",
                float(strength), float(strength),
                "Recommended range 1.5-2.5" if float(strength) <= 3.0
                else "Current value is high and may oversaturate",
            )

        if not normalize_w and n > 1 and combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG):
            effective_weight_sum = sum(abs(w) for w in user_weights)
            if effective_weight_sum >= 4.0 and not has_explicit_weights:
                raise ValueError(
                    f"[AnimaCrossAttn] normalize_weights=False with {n} artists "
                    f"(effective weight sum {effective_weight_sum:.2f}) will "
                    f"visibly amplify the cross-attention output under "
                    f"combine={combine_mode} and almost always break the image.\n"
                    f"Pick one:\n"
                    f"  1) set normalize_weights=True in AnimaArtistOptions (recommended)\n"
                    f"  2) lower the linear strength via ::name::0.25 in the chain\n"
                    f"  3) switch combine_mode to concat (no weighted sum)"
                )
            elif effective_weight_sum > 1.5:
                logger.warning(
                    "[AnimaCrossAttn] normalize_weights=False and effective weight "
                    "sum %.2f (artists=%d, combine=%s); the output may be too "
                    "strong. Lower ::weight values, enable normalize, or use concat.",
                    effective_weight_sum, n, combine_mode,
                )

        try:
            dm = model.get_model_object("diffusion_model")
        except Exception:
            dm = model.model.diffusion_model

        ok, num_blocks, ctx_dim, msg = validate_model(dm)
        if not ok:
            raise ValueError(f"[AnimaCrossAttn] unsupported model: {msg}")
        if not hasattr(dm, "preprocess_text_embeds"):
            raise ValueError(
                "[AnimaCrossAttn] this is not an Anima model "
                "(missing preprocess_text_embeds)"
            )

        artist_layer_routes, has_artist_layer_routes = resolve_artist_layer_routes(
            layer_route_texts, num_blocks,
        )
        if len(artist_layer_routes) < n:
            artist_layer_routes.extend([None] * (n - len(artist_layer_routes)))
        elif len(artist_layer_routes) > n:
            artist_layer_routes = artist_layer_routes[:n]

        artist_timing_percent_routes, has_artist_timing_routes = resolve_artist_timing_routes(
            timing_route_texts,
        )
        if len(artist_timing_percent_routes) < n:
            artist_timing_percent_routes.extend([None] * (n - len(artist_timing_percent_routes)))
        elif len(artist_timing_percent_routes) > n:
            artist_timing_percent_routes = artist_timing_percent_routes[:n]

        target_blocks = resolve_target_blocks_from_options(adv, num_blocks, strict=True)
        external_patches = describe_external_cross_attn_patches(dm, target_blocks)
        if external_patches and not adv.get("compatibility_mode", False):
            logger.warning(
                "[AnimaCrossAttn] possible external cross-attn wrappers detected: %s. "
                "If the artist effect weakens or disappears, try the "
                "compatibility_safe preset.",
                "; ".join(external_patches[:8]),
            )

        sigma_range = None
        if use_sigma_range:
            try:
                ms = model.get_model_object("model_sampling")
                s_at_start = float(ms.percent_to_sigma(start_percent))
                s_at_end = float(ms.percent_to_sigma(end_percent))
                lo, hi = sorted([s_at_end, s_at_start])
                sigma_range = (lo, hi)
            except Exception as e:
                logger.warning(
                    "[AnimaCrossAttn] failed to resolve the sigma range: %s. "
                    "Step-range control is disabled.", e,
                )
                sigma_range = None

        adv["stabilizer_min_sigma"] = None
        if use_stabilizer_window:
            try:
                ms = model.get_model_object("model_sampling")
                adv["stabilizer_min_sigma"] = float(
                    ms.percent_to_sigma(adv["stabilizer_end_percent"])
                )
            except Exception as e:
                logger.warning(
                    "[AnimaCrossAttn] failed to resolve stabilizer_end_percent "
                    "%.3f: %s. Stabilizers stay active for the whole sampling pass.",
                    adv["stabilizer_end_percent"],
                    e,
                )
                adv["stabilizer_min_sigma"] = None

        artist_timing_routes = [None] * n
        if has_artist_timing_routes:
            try:
                ms = model.get_model_object("model_sampling")
                artist_timing_routes = []
                for timing in artist_timing_percent_routes:
                    if timing is None:
                        artist_timing_routes.append(None)
                        continue
                    artist_timing_routes.append(
                        _percent_window_to_sigma_route(ms, timing)
                    )
            except Exception as e:
                logger.warning(
                    "[AnimaCrossAttn] failed to resolve per-artist timing sigma "
                    "ranges: %s. Timing routes are treated as always active.", e,
                )
                artist_timing_routes = [None] * n
                has_artist_timing_routes = False

        need_sigma_capture = need_sigma_capture or has_artist_timing_routes

        m = model.clone()

        state = _build_runtime_state(
            enabled, fusion_mode, combine_mode, strength, apply_to_uncond,
            raws, ids_list, w_list, user_weights, labels,
            artist_layer_routes, has_artist_layer_routes,
            artist_timing_routes, has_artist_timing_routes,
            normalize_w, has_explicit_weights, preset_name,
            adv, dm, sigma_range, external_patches,
        )

        # The sigma-capture wrapper is always installed: it is the single
        # run-start reset point every configuration relies on (disabled-layer
        # set, EMA/static caches, warning latches). The chaining warning stays
        # gated to sigma-dependent features so plain configs chain silently.
        prev = m.model_options.get("model_function_wrapper")
        if need_sigma_capture and prev is not None and not adv.get("compatibility_mode", False):
            logger.warning(
                "[AnimaCrossAttn] another model_function_wrapper is already "
                "installed; this node chains to it. If timing routes or "
                "stabilizers stop working, enable compatibility_safe or "
                "simplify the other wrapper nodes."
            )
        m.set_model_unet_function_wrapper(make_sigma_capture(state, prev))

        # If a previous Anima mixer node already patched some of these blocks
        # (dual-mixer chain), warn once: object patches replace, so the later
        # node wins on the overlap.
        existing_patches = getattr(m, "object_patches", None) or {}
        overlapped = [
            i for i in target_blocks
            if f"diffusion_model.blocks.{i}.cross_attn.forward" in existing_patches
        ]
        if overlapped:
            logger.warning(
                "[AnimaCrossAttn] another Anima Artist mixer node already patches "
                "blocks %d-%d on this model; the later node wins on overlapping "
                "blocks.", min(overlapped), max(overlapped),
            )

        for i in target_blocks:
            ca = dm.blocks[i].cross_attn
            current_forward = getattr(ca, 'forward', None)

            # Check if already patched by this code (handles multiple sampler workflows)
            if hasattr(current_forward, '_anima_artist_mixer_forward_patch'):
                # Already patched - unwrap to get the true original
                inner = unwrap_cross_attn_forward(ca)
            else:
                # Not yet patched - proceed normally
                inner = unwrap_cross_attn_forward(unwrap_cross_attn(ca))

            wrapper = CrossAttnWrapper(inner, state, i)
            m.add_object_patch(
                f"diffusion_model.blocks.{i}.cross_attn.forward",
                make_cross_attn_forward_patch(wrapper),
            )

        return (m, base_cond_out)


class AnimaArtistPresetApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "preset": ("ANIMA_PRESET", {
                    "tooltip": (
                        "Preset payload from Anima Artist Preset, Starter, or "
                        "Recipe Load. It owns combine/fusion/strength."
                    ),
                }),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to_uncond": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Default False. Applying style to uncond usually breaks CFG.",
                }),
            },
            "optional": {
                "advanced_options": ("ANIMA_OPTS", {
                    "tooltip": (
                        "Optional explicit option override. Leave disconnected "
                        "when the preset alone is enough."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "base_prompt")
    FUNCTION = "apply"
    CATEGORY = "Anima/CrossAttn"

    def apply(self, model, artist_pack, preset, enabled, apply_to_uncond,
              advanced_options=None):
        return AnimaArtistCrossAttn().patch(
            model,
            artist_pack,
            COMBINE_OUTPUT_AVG,
            FUSION_INTERPOLATE,
            1.0,
            enabled,
            apply_to_uncond,
            advanced_options=advanced_options,
            preset=preset,
        )


class AnimaArtistProbe:
    """Measure per-artist, per-layer style influence during one sampling run.

    Patches the model in probe mode: every layer records the relative delta
    norm ``||artist_out - base_out|| / ||base_out||`` for each artist while
    the image generates from the base prompt only (injection is NOT applied,
    so the measurement reflects the unmixed trajectory). Read the results
    with AnimaArtistProbeReport after the sampler has run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artist_pack": ("ANIMA_PACK",),
                "probe_steps": ("INT", {
                    "default": 6, "min": 1, "max": 24, "step": 1,
                    "tooltip": (
                        "Number of sampling steps to measure. Early steps carry "
                        "the most style signal; 4-8 is usually enough."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "base_prompt", "probe_id")
    FUNCTION = "probe"
    CATEGORY = "Anima/CrossAttn"

    def probe(self, model, artist_pack, probe_steps=6):
        if not isinstance(artist_pack, dict):
            raise ValueError(
                "[AnimaArtistProbe] artist_pack has the wrong type; connect the "
                "output of an AnimaArtistPack node"
            )
        base_cond_out = artist_pack.get("base_conditioning")
        if base_cond_out is None:
            raise ValueError(
                "[AnimaArtistProbe] artist_pack is missing base_conditioning."
            )
        conditionings = artist_pack.get("conditionings") or []
        labels = list(artist_pack.get("labels") or [])
        if not conditionings:
            raise ValueError("[AnimaArtistProbe] artist_pack has no artists to probe.")

        raws, ids_list, w_list = [], [], []
        for idx, c in enumerate(conditionings):
            raw, ids, w = extract_conditioning(c)
            if raw is None:
                label = labels[idx] if idx < len(labels) else f"#{idx}"
                raise ValueError(
                    f"[AnimaArtistProbe] artist[{label}] conditioning is empty."
                )
            raws.append(raw)
            ids_list.append(ids)
            w_list.append(w)

        try:
            dm = model.get_model_object("diffusion_model")
        except Exception:
            dm = model.model.diffusion_model

        ok, num_blocks, _, msg = validate_model(dm)
        if not ok:
            raise ValueError(f"[AnimaArtistProbe] unsupported model: {msg}")
        if not hasattr(dm, "preprocess_text_embeds"):
            raise ValueError(
                "[AnimaArtistProbe] this is not an Anima model "
                "(missing preprocess_text_embeds)"
            )

        probe_id = uuid.uuid4().hex[:12]
        n = len(raws)

        state = _build_runtime_state(
            True, FUSION_INTERPOLATE, COMBINE_OUTPUT_AVG, 1.0, False,
            raws, ids_list, w_list, [1.0] * n, labels,
            [None] * n, False, [None] * n, False,
            True, False, None,
            {"static_capture_k": STATIC_CAPTURE_K_DEFAULT},
            dm, None, [],
        )
        state["probe_steps"] = max(1, int(probe_steps))
        state["probe_stats"] = {}      # layer_idx -> [ [sum, count], ... ] per artist
        state["probe_labels"] = labels
        state["probe_num_blocks"] = num_blocks
        state["_probe_seen_sigmas"] = set()
        # Fallback step budget for the degenerate case where the sigma-capture
        # hook was overridden and current_sigma is never set.
        state["_probe_forward_count"] = 0

        m = model.clone()
        prev = m.model_options.get("model_function_wrapper")
        m.set_model_unet_function_wrapper(make_sigma_capture(state, prev))
        for i in range(num_blocks):
            inner = unwrap_cross_attn_forward(unwrap_cross_attn(dm.blocks[i].cross_attn))
            wrapper = _ProbeCrossAttnWrapper(inner, state, i)
            m.add_object_patch(
                f"diffusion_model.blocks.{i}.cross_attn.forward",
                make_cross_attn_forward_patch(wrapper),
            )

        _registry_store(probe_id, state)
        logger.info(
            "[AnimaArtistProbe] probe %s armed for %d artists x %d layers "
            "(first %d steps)", probe_id, n, num_blocks, state["probe_steps"],
        )
        return (m, base_cond_out, probe_id)


class _ProbeCrossAttnWrapper(CrossAttnWrapper):
    """Measurement-only wrapper: never alters the output."""

    def _dispatch(self, x, context, rope_emb, transformer_options):
        st = self._st
        base_out = self.original(x, context, rope_emb=rope_emb,
                                 transformer_options=transformer_options)

        stats = st.setdefault("probe_stats", {})
        layer_stats = stats.get(self._idx)
        n = len(st["raws"])
        if layer_stats is None:
            layer_stats = [[0.0, 0] for _ in range(n)]
            stats[self._idx] = layer_stats
        # Measure only the first probe_steps distinct sigmas; afterwards the
        # forward is a plain pass-through (and the seen-set stops growing so
        # the report's step count stays accurate). When the sigma is missing
        # (capture hook overridden), fall back to a raw forward counter so the
        # budget is still enforced instead of measuring forever.
        budget = int(st.get("probe_steps", 6))
        cur = st.get("current_sigma")
        if cur is not None:
            seen = st.setdefault("_probe_seen_sigmas", set())
            cur_key = round(float(cur), 4)
            if cur_key not in seen:
                if len(seen) >= budget:
                    return base_out
                seen.add(cur_key)
        else:
            count = int(st.get("_probe_forward_count", 0))
            if count >= budget:
                return base_out
            st["_probe_forward_count"] = count + 1

        # Restrict the delta measurement to the cond rows: under CFG the uncond
        # rows carry the unstyled trajectory and would understate influence.
        from .patching import build_artists, resolve_mask
        cou = transformer_options.get("cond_or_uncond") \
            if isinstance(transformer_options, dict) else None
        mask = resolve_mask(cou, context.shape[0], False, {})
        row_mask = torch.tensor(mask, device=base_out.device, dtype=torch.bool)

        individuals, _ = build_artists(st, context)
        outs = self._collect_artist_outputs(
            x, context, rope_emb, transformer_options, individuals,
            FUSION_INTERPOLATE,
        )
        base_sel = base_out.detach().to(torch.float32)[row_mask]
        base_norm = float(base_sel.norm().item())
        if base_norm <= 1e-8:
            return base_out
        for i, out_i in enumerate(outs):
            delta = out_i.detach().to(torch.float32)[row_mask] - base_sel
            rel = float(delta.norm().item()) / base_norm
            layer_stats[i][0] += rel
            layer_stats[i][1] += 1
        return base_out


def _suggest_layer_range(scores, top_fraction=0.35):
    """Suggest a contiguous layer range covering the strongest layers."""
    if not scores:
        return None
    indexed = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = max(1, int(round(len(scores) * top_fraction)))
    top = sorted(indexed[:keep])
    return top[0], top[-1]


class AnimaArtistProbeReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "probe_id": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
                "trigger": (ANY_TYPE, {
                    "tooltip": (
                        "Connect any post-sampler output (e.g. the decoded IMAGE) "
                        "so this report runs after sampling finished."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "report"
    CATEGORY = "Anima/CrossAttn"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, probe_id, trigger=None):
        return float("nan")  # always re-run; registry contents change

    def report(self, probe_id, trigger=None):
        state = PROBE_REGISTRY.get(str(probe_id or "").strip())
        if state is None:
            text = (
                "Anima Artist Probe Report\n\n"
                f"status: NO DATA\nprobe_id: {probe_id!r}\n\n"
                "No measurements found. Wire AnimaArtistProbe's model output "
                "through the sampler, connect its probe_id here, and connect "
                "a post-sampler output (e.g. IMAGE) to trigger."
            )
            return {"ui": {"text": [text]}, "result": (text,)}

        stats = state.get("probe_stats") or {}
        labels = list(state.get("probe_labels") or [])
        num_blocks = int(state.get("probe_num_blocks", 0))
        if not stats:
            text = (
                "Anima Artist Probe Report\n\n"
                "status: EMPTY\n\n"
                "The probe is armed but no samples were recorded yet. "
                "Run the sampler first (connect trigger to a post-sampler output)."
            )
            return {"ui": {"text": [text]}, "result": (text,)}

        n = len(labels)
        # scores[artist][layer] = mean relative delta
        scores = [[0.0] * num_blocks for _ in range(n)]
        sample_counts = [0] * n
        for layer_idx, layer_stats in stats.items():
            for i, (total, count) in enumerate(layer_stats):
                if i < n and 0 <= layer_idx < num_blocks and count > 0:
                    scores[i][layer_idx] = total / count
                    sample_counts[i] = max(sample_counts[i], count)

        lines = [
            "Anima Artist Probe Report",
            "",
            "status: OK",
            f"artists: {n}",
            f"layers: {num_blocks}",
            f"measured steps: {len(state.get('_probe_seen_sigmas') or [])}",
            "",
            "relative style influence per layer "
            "(||artist_out - base_out|| / ||base_out||):",
        ]
        for i, label in enumerate(labels):
            row = scores[i]
            peak = max(row) if row else 0.0
            lines.append("")
            lines.append(
                f"artist {i + 1}: {label} "
                f"(peak {peak:.3f}, {sample_counts[i]} samples)"
            )
            # Compact bar chart, 8 layers per line.
            for start in range(0, num_blocks, 8):
                seg = row[start:start + 8]
                cells = []
                for j, v in enumerate(seg):
                    bar_len = 0 if peak <= 0 else int(round(6 * v / peak))
                    cells.append(f"L{start + j:>2}:{'#' * bar_len:<6}")
                lines.append("  " + " ".join(cells))
            suggestion = _suggest_layer_range(row)
            if suggestion is not None:
                lo, hi = suggestion
                lines.append(
                    f"  suggested route: {label}@{lo}-{hi}  "
                    f"({format_layer_span(lo, hi)} carries the strongest signal)"
                )
        lines.extend([
            "",
            "how to use:",
            "  - copy the suggested @routes into your artist chain",
            "  - artists with flat profiles mix well at all layers",
            "  - artists with sharp peaks benefit most from layer routing",
        ])
        text = "\n".join(lines)
        return {"ui": {"text": [text]}, "result": (text,)}
