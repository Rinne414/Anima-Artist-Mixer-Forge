"""Parsing / formatting / preset helper tests for the anima_mixer package."""

import json
import os
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from anima_mixer import chain_tools, constants, options, parsing, patching, recipe  # noqa: E402
from anima_mixer.nodes_core import (  # noqa: E402
    AnimaArtistBasic,
    AnimaArtistCrossAttn,
    AnimaArtistPack,
    AnimaArtistPresetApply,
)
from anima_mixer.nodes_ui import (  # noqa: E402
    AnimaArtistChainBuilder,
    AnimaArtistInspector,
    AnimaArtistOptions,
    AnimaArtistPreset,
    AnimaArtistRecipeLoad,
    AnimaArtistRecipeSave,
    AnimaArtistSimpleOptions,
    AnimaArtistStarter,
)


class ArtistRoutingHelpersTest(unittest.TestCase):
    def test_artist_timing_layers_and_weights_parse_together(self):
        parts = parsing.split_artist_chain(
            "::@wlop::1.2@0,2,4%0.0-0.45, "
            "::(krenz:1.1)::0.8@9，18%0.45-0.85, "
            "@hiten"
        )

        parts, timings = parsing.parse_artist_timing_routes(parts)
        parts, layers = parsing.parse_artist_layer_routes(parts)
        names, weights, explicit = parsing.parse_artist_weights(parts)

        self.assertTrue(explicit)
        self.assertEqual(names, ["@wlop", "(krenz:1.1)", "@hiten"])
        self.assertEqual(weights, [1.2, 0.8, 1.0])
        self.assertEqual(layers, ["0,2,4", "9，18", ""])
        self.assertEqual(timings, ["0.0-0.45", "0.45-0.85", ""])

        resolved_layers, has_layers = parsing.resolve_artist_layer_routes(layers, 28)
        self.assertTrue(has_layers)
        self.assertEqual(resolved_layers[0], {0, 2, 4})
        self.assertEqual(resolved_layers[1], {9, 18})

        resolved, has_timings = parsing.resolve_artist_timing_routes(timings)
        self.assertTrue(has_timings)
        self.assertEqual(resolved[0], (0.0, 0.45, 0.0))
        self.assertEqual(resolved[1], (0.45, 0.85, 0.0))
        self.assertIsNone(resolved[2])

    def test_artist_layer_percent_windows_resolve_to_blocks(self):
        parts = parsing.split_artist_chain(
            "@background@0%-33%, "
            "@character@0.33-0.67, "
            "@clothing@67%-100%"
        )

        parts, timing_routes = parsing.parse_artist_timing_routes(parts)
        parts, layer_routes = parsing.parse_artist_layer_routes(parts)
        names, weights, explicit = parsing.parse_artist_weights(parts)

        self.assertFalse(explicit)
        self.assertEqual(names, ["@background", "@character", "@clothing"])
        self.assertEqual(weights, [1.0, 1.0, 1.0])
        self.assertEqual(layer_routes, ["0%-33%", "0.33-0.67", "67%-100%"])
        self.assertEqual(timing_routes, ["", "", ""])

        resolved_layers, has_layers = parsing.resolve_artist_layer_routes(layer_routes, 28)
        self.assertTrue(has_layers)
        self.assertEqual(resolved_layers[0], set(range(0, 9)))
        self.assertEqual(resolved_layers[1], set(range(9, 19)))
        self.assertEqual(resolved_layers[2], set(range(19, 28)))

    def test_artist_layer_percent_window_combines_with_sampling_timing(self):
        parts = parsing.split_artist_chain(
            "@character@0.33-0.67%0.20-0.85~0.05"
        )

        parts, timing_routes = parsing.parse_artist_timing_routes(parts)
        parts, layer_routes = parsing.parse_artist_layer_routes(parts)

        self.assertEqual(parts, ["@character"])
        self.assertEqual(layer_routes, ["0.33-0.67"])
        self.assertEqual(timing_routes, ["0.20-0.85~0.05"])

        resolved_layers, _ = parsing.resolve_artist_layer_routes(layer_routes, 28)
        resolved_timing, _ = parsing.resolve_artist_timing_routes(timing_routes)
        self.assertEqual(resolved_layers[0], set(range(9, 19)))
        self.assertEqual(resolved_timing[0], (0.2, 0.85, 0.05))

    def test_negative_weight_parses_as_style_subtraction(self):
        names, weights, explicit = parsing.parse_artist_weights(["::wlop::-0.5", "krenz"])

        self.assertTrue(explicit)
        self.assertEqual(names, ["wlop", "krenz"])
        self.assertEqual(weights, [-0.5, 1.0])

    def test_escaped_yuchi_artist_tag_is_kept_intact(self):
        parts = parsing.split_artist_chain(r"@yuchi \(salmon-1000\)")
        parts, timing_routes = parsing.parse_artist_timing_routes(parts)
        parts, layer_routes = parsing.parse_artist_layer_routes(parts)
        names, weights, explicit = parsing.parse_artist_weights(parts)

        self.assertEqual(names, [r"@yuchi \(salmon-1000\)"])
        self.assertEqual(weights, [1.0])
        self.assertFalse(explicit)
        self.assertEqual(layer_routes, [""])
        self.assertEqual(timing_routes, [""])

    def test_weight_clamps_to_range(self):
        _, weights, _ = parsing.parse_artist_weights(["::a::9.5", "::b::-9.5"])
        self.assertEqual(weights, [constants.WEIGHT_MAX, constants.WEIGHT_MIN])

    def test_timing_fade_syntax_parses(self):
        self.assertEqual(parsing.parse_timing_filter("0.0-0.45~0.1"), (0.0, 0.45, 0.1))
        self.assertEqual(parsing.parse_timing_filter("0.0-0.45"), (0.0, 0.45, 0.0))
        self.assertIsNone(parsing.parse_timing_filter("0.0-0.45~x"))
        self.assertIsNone(parsing.parse_timing_filter("0.0-0.45~0.1~0.2"))
        self.assertIsNone(parsing.parse_timing_filter("0.5-0.5~0.1"))

    def test_timing_route_with_fade_attaches_to_artist(self):
        clean, timing = parsing.parse_artist_timing_route("wlop%0.0-0.45~0.1")
        self.assertEqual(clean, "wlop")
        self.assertEqual(timing, "0.0-0.45~0.1")

    def test_invalid_timing_suffix_stays_in_artist_text(self):
        clean, timing = parsing.parse_artist_timing_route("artist%0.5-0.5")
        self.assertEqual(clean, "artist%0.5-0.5")
        self.assertEqual(timing, "")

    def test_compatibility_safe_preset_overrides_risky_settings(self):
        payload = options.build_preset_payload(constants.PRESET_COMPATIBILITY_SAFE)
        self.assertEqual(payload["combine_mode"], constants.COMBINE_CONCAT)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_CONCAT_WITH_BASE)
        self.assertTrue(payload["advanced_options"]["compatibility_mode"])

        combine_mode, fusion_mode, _, adv, _ = options.merge_runtime_options(
            constants.COMBINE_OUTPUT_AVG,
            constants.FUSION_INTERPOLATE,
            1.0,
            {
                "compatibility_mode": True,
                "artist_ema_alpha": 0.5,
                "artist_static_capture": True,
                "artist_anchor_q": True,
            },
            None,
        )

        self.assertEqual(combine_mode, constants.COMBINE_CONCAT)
        self.assertEqual(fusion_mode, constants.FUSION_CONCAT_WITH_BASE)
        self.assertEqual(adv["artist_ema_alpha"], 0.0)
        self.assertFalse(adv["artist_static_capture"])
        self.assertFalse(adv["artist_anchor_q"])

        combine_mode, fusion_mode, _, adv, _ = options.merge_runtime_options(
            constants.COMBINE_OUTPUT_AVG,
            constants.FUSION_INTERPOLATE,
            1.0,
            {"compatibility_mode": False, "artist_static_capture": True},
            payload,
        )
        self.assertEqual(combine_mode, constants.COMBINE_CONCAT)
        self.assertEqual(fusion_mode, constants.FUSION_CONCAT_WITH_BASE)
        self.assertTrue(adv["compatibility_mode"])
        self.assertFalse(adv["artist_static_capture"])

    def test_prompt_passthrough_is_available_as_preset(self):
        payload = options.build_preset_payload(constants.PRESET_PROMPT_PASSTHROUGH)

        self.assertIn(constants.PRESET_PROMPT_PASSTHROUGH, constants.PRESET_CHOICES)
        self.assertTrue(payload["advanced_options"]["prompt_passthrough"])
        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_INTERPOLATE)

    def test_artist_chain_prefix_weight_accepts_prompt_style_suffix(self):
        names, weights, explicit = parsing.parse_artist_weights(["1.2::wlop::"])

        self.assertTrue(explicit)
        self.assertEqual(names, ["wlop"])
        self.assertEqual(weights, [1.2])

    def test_build_direct_prompt_keeps_prompt_side_weighting(self):
        text = parsing.build_direct_artist_prompt(
            ["@yuchi \\(salmon-1000\\)", "@uof"],
            [1.2, 0.8],
            "1girl, solo",
        )

        self.assertEqual(
            text,
            "(@yuchi \\(salmon-1000\\):1.2), (@uof:0.8), 1girl, solo",
        )

    def test_build_direct_prompt_without_base_uses_artist_list(self):
        text = parsing.build_direct_artist_prompt(["wlop", "krenz"], [1.0, 1.0], "")

        self.assertEqual(text, "wlop, krenz")

    def test_build_direct_prompt_rejects_negative_weight(self):
        with self.assertRaisesRegex(ValueError, "negative artist weight"):
            parsing.build_direct_artist_prompt(["bad"], [-0.5], "1girl")

    def test_build_direct_prompt_omits_zero_weight(self):
        text = parsing.build_direct_artist_prompt(["wlop", "krenz"], [0.0, 1.0], "1girl")

        self.assertEqual(text, "krenz, 1girl")

    def test_base_advanced_options_includes_new_keys(self):
        adv = options.base_advanced_options()
        self.assertIn("max_batch_artists", adv)
        self.assertIn("low_vram_cache", adv)
        self.assertIn("anchor_base_norm_ref", adv)
        self.assertIn("anchor_refresh_each_step", adv)
        self.assertIn("norm_lock_mode", adv)
        self.assertIn("norm_lock_scope", adv)
        self.assertIn("contribution_balance", adv)
        self.assertIn("contribution_balance_alpha", adv)
        self.assertIn("mixed_delta_cap", adv)
        self.assertIn("mixed_delta_cap_ratio", adv)
        self.assertIn("static_capture_mode", adv)
        self.assertIn("stabilizer_end_percent", adv)
        self.assertIn(constants.STATIC_CAPTURE_MODE_BLEND_PERP, constants.STATIC_CAPTURE_MODE_CHOICES)
        self.assertEqual(adv["max_batch_artists"], 0)
        self.assertFalse(adv["low_vram_cache"])
        self.assertFalse(adv["anchor_base_norm_ref"])
        self.assertFalse(adv["anchor_refresh_each_step"])
        self.assertEqual(adv["norm_lock_mode"], constants.NORM_LOCK_TOKEN)
        self.assertEqual(adv["norm_lock_scope"], constants.NORM_LOCK_SCOPE_PER_ARTIST)
        self.assertFalse(adv["contribution_balance"])
        self.assertFalse(adv["mixed_delta_cap"])
        self.assertFalse(adv["match_base_norm"])
        self.assertFalse(adv["prompt_passthrough"])
        self.assertAlmostEqual(adv["stabilizer_end_percent"], 1.0)
        self.assertAlmostEqual(
            adv["mixed_delta_cap_ratio"],
            constants.MIXED_DELTA_CAP_RATIO_DEFAULT,
        )
        self.assertEqual(adv["static_capture_mode"], constants.STATIC_CAPTURE_MODE_OUTPUT)
        self.assertAlmostEqual(
            adv["contribution_balance_alpha"],
            constants.CONTRIB_BALANCE_ALPHA_DEFAULT,
        )

    def test_simple_options_keeps_default_surface_small(self):
        inputs = AnimaArtistSimpleOptions.INPUT_TYPES()
        visible_count = sum(len(group) for group in inputs.values())

        self.assertLessEqual(visible_count, 6)
        self.assertIn("layer_mode", inputs["required"])
        self.assertNotIn("artist_anchor_q", inputs["required"])
        self.assertNotIn("match_base_norm", inputs.get("optional", {}))

    def test_simple_options_builds_original_style_payload(self):
        adv = AnimaArtistSimpleOptions().build(
            True,
            constants.LAYER_MODE_STYLE_CORE,
            0.1,
            0.9,
            "",
            False,
        )[0]

        self.assertTrue(adv["normalize_weights"])
        self.assertEqual(adv["layer_filter"], "0-18")
        self.assertAlmostEqual(adv["start_percent"], 0.1)
        self.assertAlmostEqual(adv["end_percent"], 0.9)
        self.assertFalse(adv["compatibility_mode"])
        self.assertNotIn("artist_ema_alpha", adv)
        self.assertNotIn("artist_static_capture", adv)
        self.assertNotIn("artist_anchor_q", adv)
        self.assertNotIn("match_base_norm", adv)

    def test_simple_options_preserves_preset_style_guards_when_merged(self):
        simple = AnimaArtistSimpleOptions().build(
            True,
            constants.LAYER_MODE_STYLE_CORE,
            0.1,
            0.9,
            "",
            False,
        )[0]
        preset = options.build_preset_payload(constants.PRESET_FACE_LOCK)

        _, _, _, adv, preset_name = options.merge_runtime_options(
            constants.COMBINE_OUTPUT_AVG,
            constants.FUSION_INTERPOLATE,
            1.0,
            advanced_options=simple,
            preset=preset,
            base_prompt="close-up portrait",
            artist_count=1,
        )

        self.assertEqual(preset_name, constants.PRESET_FACE_LOCK)
        self.assertFalse(adv["artist_static_capture"])
        self.assertTrue(adv["match_base_norm"])
        self.assertEqual(adv["layer_filter"], "0-18")
        self.assertAlmostEqual(adv["start_percent"], 0.1)
        self.assertAlmostEqual(adv["end_percent"], 0.9)

    def test_simple_options_preserves_drift_auto_style_route(self):
        simple = AnimaArtistSimpleOptions().build(
            True,
            constants.LAYER_MODE_AUTO,
            0.0,
            1.0,
            "",
            False,
        )[0]
        preset = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        combine_mode, fusion_mode, strength, adv, preset_name = options.merge_runtime_options(
            constants.COMBINE_OUTPUT_AVG,
            constants.FUSION_INTERPOLATE,
            1.0,
            advanced_options=simple,
            preset=preset,
            base_prompt="portrait, upper body",
            artist_count=4,
        )

        self.assertEqual(preset_name, constants.PRESET_DRIFT_AUTO)
        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_DRIFT_SOFT)
        self.assertFalse(adv["compatibility_mode"])
        self.assertEqual(combine_mode, constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(fusion_mode, constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(strength, 0.85)

    def test_advanced_options_node_still_exposes_expert_controls(self):
        inputs = AnimaArtistOptions.INPUT_TYPES()
        required = inputs["required"]
        optional = inputs["optional"]

        self.assertIn("artist_anchor_q", required)
        self.assertIn("static_capture_mode", required)
        self.assertIn("stabilizer_end_percent", required)
        self.assertIn("match_base_norm", optional)
        self.assertIn("mixed_delta_cap", optional)

    def test_stable_seed_preset_uses_delta_cap_path(self):
        payload = options.build_preset_payload(constants.PRESET_STABLE_SEED)

        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(payload["strength"], 1.0)
        self.assertFalse(payload["advanced_options"]["artist_static_capture"])
        self.assertFalse(payload["advanced_options"]["artist_anchor_q"])
        self.assertEqual(payload["advanced_options"]["anchor_seeds_count"], 1)
        self.assertEqual(
            payload["advanced_options"]["anchor_deep_layer_threshold"],
            constants.ANCHOR_LAYER_THRESHOLD_DISABLED,
        )
        self.assertFalse(payload["advanced_options"]["match_base_norm"])
        self.assertFalse(payload["advanced_options"]["anchor_base_norm_ref"])
        self.assertFalse(payload["advanced_options"]["contribution_balance"])
        self.assertTrue(payload["advanced_options"]["mixed_delta_cap"])
        self.assertAlmostEqual(payload["advanced_options"]["mixed_delta_cap_ratio"], 0.75)
        self.assertEqual(payload["advanced_options"]["layer_filter"], "9-20")

    def test_balanced_single_artist_keeps_original_output_avg_path(self):
        payload = options.build_preset_payload(
            constants.PRESET_BALANCED,
            artist_count=1,
        )

        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_INTERPOLATE)
        self.assertFalse(payload["advanced_options"]["compatibility_mode"])
        self.assertAlmostEqual(payload["advanced_options"]["artist_ema_alpha"], 0.0)
        self.assertFalse(payload["advanced_options"]["match_base_norm"])

    def test_balanced_multi_artist_keeps_output_avg_path(self):
        payload = options.build_preset_payload(
            constants.PRESET_BALANCED,
            artist_count=2,
        )

        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_INTERPOLATE)
        self.assertFalse(payload["advanced_options"]["compatibility_mode"])
        self.assertAlmostEqual(payload["advanced_options"]["artist_ema_alpha"], 0.0)
        self.assertFalse(payload["advanced_options"]["match_base_norm"])

    def test_drift_soft_preset_uses_soft_ema_path(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_SOFT)

        self.assertIn(constants.PRESET_DRIFT_SOFT, constants.PRESET_CHOICES)
        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(payload["strength"], 0.85)
        self.assertFalse(payload["advanced_options"]["artist_static_capture"])
        self.assertAlmostEqual(payload["advanced_options"]["artist_ema_alpha"], 0.12)
        self.assertEqual(
            payload["advanced_options"]["static_capture_mode"],
            constants.STATIC_CAPTURE_MODE_OUTPUT,
        )
        self.assertFalse(payload["advanced_options"]["artist_anchor_q"])
        self.assertFalse(payload["advanced_options"]["match_base_norm"])
        self.assertFalse(payload["advanced_options"]["contribution_balance"])
        self.assertEqual(payload["advanced_options"]["layer_filter"], "9-20")

    def test_face_lock_preset_uses_base_preserve_norm_delta_cap_path(self):
        payload = options.build_preset_payload(constants.PRESET_FACE_LOCK)

        self.assertIn(constants.PRESET_FACE_LOCK, constants.PRESET_CHOICES)
        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_BASE_PRESERVE)
        self.assertAlmostEqual(payload["strength"], 0.9)
        self.assertFalse(payload["advanced_options"]["artist_static_capture"])
        self.assertFalse(payload["advanced_options"]["artist_anchor_q"])
        self.assertTrue(payload["advanced_options"]["match_base_norm"])
        self.assertEqual(payload["advanced_options"]["norm_lock_mode"], constants.NORM_LOCK_TOKEN)
        self.assertEqual(
            payload["advanced_options"]["norm_lock_scope"],
            constants.NORM_LOCK_SCOPE_PER_ARTIST,
        )
        self.assertFalse(payload["advanced_options"]["contribution_balance"])
        self.assertTrue(payload["advanced_options"]["mixed_delta_cap"])
        self.assertAlmostEqual(payload["advanced_options"]["mixed_delta_cap_ratio"], 1.0)
        self.assertEqual(payload["advanced_options"]["layer_filter"], "9-20")

    def test_scene_lock_preset_uses_base_preserve_ema_path(self):
        payload = options.build_preset_payload(constants.PRESET_SCENE_LOCK)

        self.assertIn(constants.PRESET_SCENE_LOCK, constants.PRESET_CHOICES)
        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_BASE_PRESERVE)
        self.assertAlmostEqual(payload["strength"], 0.85)
        self.assertFalse(payload["advanced_options"]["artist_static_capture"])
        self.assertAlmostEqual(payload["advanced_options"]["artist_ema_alpha"], 0.10)
        self.assertFalse(payload["advanced_options"]["artist_anchor_q"])
        self.assertFalse(payload["advanced_options"]["match_base_norm"])
        self.assertFalse(payload["advanced_options"]["contribution_balance"])
        self.assertEqual(payload["advanced_options"]["layer_filter"], "9-15")

    def test_identity_guard_preset_uses_light_base_preserve_path(self):
        payload = options.build_preset_payload(constants.PRESET_IDENTITY_GUARD)

        self.assertIn(constants.PRESET_IDENTITY_GUARD, constants.PRESET_CHOICES)
        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_BASE_PRESERVE)
        self.assertAlmostEqual(payload["strength"], 0.85)
        self.assertFalse(payload["advanced_options"]["artist_static_capture"])
        self.assertFalse(payload["advanced_options"]["artist_anchor_q"])
        self.assertAlmostEqual(payload["advanced_options"]["artist_ema_alpha"], 0.12)
        self.assertTrue(payload["advanced_options"]["match_base_norm"])
        self.assertTrue(payload["advanced_options"]["mixed_delta_cap"])
        self.assertAlmostEqual(payload["advanced_options"]["mixed_delta_cap_ratio"], 0.9)

    def test_drift_auto_routes_upper_body_portrait_to_drift_soft(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        combine_mode, fusion_mode, strength, adv, preset_name = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, masterpiece, best quality, upper body portrait, "
                "face visible, looking at viewer, simple background"
            ),
        )

        self.assertEqual(preset_name, constants.PRESET_DRIFT_AUTO)
        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_DRIFT_SOFT)
        self.assertEqual(combine_mode, constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(fusion_mode, constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(strength, 0.85)
        self.assertFalse(adv["artist_static_capture"])
        self.assertAlmostEqual(adv["artist_ema_alpha"], 0.12)
        self.assertFalse(adv["match_base_norm"])
        self.assertEqual(adv["layer_filter"], "9-20")

    def test_drift_auto_routes_many_artist_portrait_to_style_mixer(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        combine_mode, fusion_mode, strength, adv, preset_name = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, masterpiece, best quality, upper body portrait, "
                "face visible, looking at viewer, simple background"
            ),
            artist_count=4,
        )

        self.assertEqual(preset_name, constants.PRESET_DRIFT_AUTO)
        self.assertEqual(
            adv["drift_auto_resolved_preset"],
            constants.PRESET_DRIFT_SOFT,
        )
        self.assertIn("4+ artists", adv["drift_auto_reason"])
        self.assertEqual(combine_mode, constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(fusion_mode, constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(strength, 0.85)
        self.assertFalse(adv["compatibility_mode"])
        self.assertEqual(adv["layer_filter"], "9-20")
        self.assertFalse(adv["artist_static_capture"])
        self.assertFalse(adv["artist_anchor_q"])

    def test_drift_auto_routes_closeup_face_to_face_lock(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        _, fusion_mode, strength, adv, preset_name = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, close-up portrait, face visible, detailed eyes, "
                "looking at viewer, simple background"
            ),
        )

        self.assertEqual(preset_name, constants.PRESET_DRIFT_AUTO)
        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_FACE_LOCK)
        self.assertEqual(fusion_mode, constants.FUSION_BASE_PRESERVE)
        self.assertAlmostEqual(strength, 0.9)
        self.assertTrue(adv["match_base_norm"])
        self.assertEqual(adv["norm_lock_mode"], constants.NORM_LOCK_TOKEN)
        self.assertEqual(adv["norm_lock_scope"], constants.NORM_LOCK_SCOPE_PER_ARTIST)

    def test_drift_auto_routes_many_artist_closeup_to_stable_seed(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        _, fusion_mode, strength, adv, preset_name = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, close-up portrait, face visible, detailed eyes, "
                "looking at viewer, simple background"
            ),
            artist_count=4,
        )

        self.assertEqual(preset_name, constants.PRESET_DRIFT_AUTO)
        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_STABLE_SEED)
        self.assertIn("4+ artists", adv["drift_auto_reason"])
        self.assertEqual(fusion_mode, constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(strength, 1.0)
        self.assertFalse(adv["match_base_norm"])
        self.assertTrue(adv["mixed_delta_cap"])
        self.assertAlmostEqual(adv["mixed_delta_cap_ratio"], 0.75)

    def test_drift_auto_many_artist_plain_portrait_uses_style_mixer(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        combine_mode, fusion_mode, strength, adv, preset_name = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, upper body portrait, detailed hair, soft light, "
                "looking at viewer"
            ),
            artist_count=4,
        )

        self.assertEqual(preset_name, constants.PRESET_DRIFT_AUTO)
        self.assertEqual(
            adv["drift_auto_resolved_preset"],
            constants.PRESET_DRIFT_SOFT,
        )
        self.assertEqual(combine_mode, constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(fusion_mode, constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(strength, 0.85)
        self.assertFalse(adv["compatibility_mode"])
        self.assertEqual(adv["layer_filter"], "9-20")
        self.assertFalse(adv["mixed_delta_cap"])

    def test_drift_auto_preset_wins_over_its_preview_advanced_options(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        _, fusion_mode, strength, adv, preset_name = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            payload["advanced_options"],
            payload,
            base_prompt=(
                "1girl, solo, close-up portrait, face visible, detailed eyes, "
                "looking at viewer, simple background"
            ),
        )

        self.assertEqual(preset_name, constants.PRESET_DRIFT_AUTO)
        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_FACE_LOCK)
        self.assertEqual(fusion_mode, constants.FUSION_BASE_PRESERVE)
        self.assertAlmostEqual(strength, 0.9)
        self.assertTrue(adv["match_base_norm"])

    def test_drift_auto_routes_plain_street_scene_to_drift_soft(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        combine_mode, fusion_mode, strength, adv, preset_name = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, standing on a city street, wearing a white blouse, "
                "daylight, detailed background"
            ),
        )

        self.assertEqual(preset_name, constants.PRESET_DRIFT_AUTO)
        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_DRIFT_SOFT)
        self.assertEqual(combine_mode, constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(fusion_mode, constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(strength, 0.85)
        self.assertFalse(adv["artist_static_capture"])
        self.assertAlmostEqual(adv["artist_ema_alpha"], 0.12)
        self.assertFalse(adv["match_base_norm"])

    def test_drift_auto_routes_many_artist_street_scene_to_style_mixer(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        combine_mode, fusion_mode, strength, adv, preset_name = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, street scene, walking, casual outfit, "
                "urban background, daylight, looking at viewer"
            ),
            artist_count=4,
        )

        self.assertEqual(preset_name, constants.PRESET_DRIFT_AUTO)
        self.assertEqual(
            adv["drift_auto_resolved_preset"],
            constants.PRESET_DRIFT_SOFT,
        )
        self.assertIn("4+ artists", adv["drift_auto_reason"])
        self.assertEqual(combine_mode, constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(fusion_mode, constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(strength, 0.85)
        self.assertFalse(adv["compatibility_mode"])

    def test_drift_auto_does_not_treat_plain_walking_as_street_scene(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        _, _, _, adv, _ = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, walking pose, casual outfit, simple background, "
                "looking at viewer"
            ),
            artist_count=4,
        )

        self.assertEqual(
            adv["drift_auto_resolved_preset"],
            constants.PRESET_DRIFT_SOFT,
        )
        self.assertEqual(adv["layer_filter"], "9-20")
        self.assertIn("default portrait", adv["drift_auto_reason"])

    def test_drift_auto_routes_fullbody_tag_with_simple_background_to_drift_soft(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        _, fusion_mode, _, adv, _ = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, fullbody, standing pose, white blouse, navy skirt, "
                "simple background"
            ),
        )

        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_DRIFT_SOFT)
        self.assertEqual(fusion_mode, constants.FUSION_INTERPOLATE)

    def test_drift_auto_routes_many_artist_simple_fullbody_to_drift_soft(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        _, fusion_mode, strength, adv, _ = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, fullbody, standing pose, white blouse, navy skirt, "
                "simple background"
            ),
            artist_count=4,
        )

        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_DRIFT_SOFT)
        self.assertIn("simple fullbody", adv["drift_auto_reason"])
        self.assertEqual(fusion_mode, constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(strength, 0.85)
        self.assertFalse(adv["match_base_norm"])
        self.assertFalse(adv["mixed_delta_cap"])

    def test_drift_auto_routes_wide_background_scene_to_scene_lock(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        _, fusion_mode, _, adv, _ = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, wide shot, full body, small figure, cityscape, "
                "detailed background, daylight"
            ),
        )

        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_SCENE_LOCK)
        self.assertEqual(fusion_mode, constants.FUSION_BASE_PRESERVE)

    def test_drift_auto_routes_many_artist_wide_background_to_scene_lock(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        _, fusion_mode, strength, adv, _ = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, wide shot, full body, small figure, cityscape, "
                "detailed background, daylight"
            ),
            artist_count=4,
        )

        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_SCENE_LOCK)
        self.assertIn("4+ artists wide", adv["drift_auto_reason"])
        self.assertEqual(fusion_mode, constants.FUSION_BASE_PRESERVE)
        self.assertAlmostEqual(strength, 0.85)
        self.assertFalse(adv["match_base_norm"])
        self.assertFalse(adv["mixed_delta_cap"])
        self.assertEqual(adv["layer_filter"], "9-15")

    def test_drift_auto_does_not_treat_streetwear_as_street_scene(self):
        payload = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)

        _, _, _, adv, _ = options.merge_runtime_options(
            constants.COMBINE_CONCAT,
            constants.FUSION_CONCAT_WITH_BASE,
            2.0,
            None,
            payload,
            base_prompt=(
                "1girl, solo, streetwear fashion portrait, upper body, "
                "plain studio background"
            ),
        )

        self.assertEqual(adv["drift_auto_resolved_preset"], constants.PRESET_DRIFT_SOFT)

    def test_anchor_lock_preset_keeps_anchor_path(self):
        payload = options.build_preset_payload(constants.PRESET_ANCHOR_LOCK)

        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(payload["strength"], 0.9)
        self.assertFalse(payload["advanced_options"]["artist_static_capture"])
        self.assertTrue(payload["advanced_options"]["artist_anchor_q"])
        self.assertEqual(payload["advanced_options"]["anchor_seeds_count"], 1)
        self.assertAlmostEqual(payload["advanced_options"]["anchor_user_blend"], 0.35)
        self.assertEqual(payload["advanced_options"]["anchor_deep_layer_threshold"], 12)
        self.assertFalse(payload["advanced_options"]["match_base_norm"])
        self.assertFalse(payload["advanced_options"]["anchor_base_norm_ref"])
        self.assertFalse(payload["advanced_options"]["contribution_balance"])
        self.assertEqual(payload["advanced_options"]["layer_filter"], "9-15")

    def test_stable_seed_explicit_all_layers_keeps_all_layers(self):
        payload = options.build_preset_payload(
            constants.PRESET_STABLE_SEED,
            layer_mode=constants.LAYER_MODE_ALL,
        )

        self.assertEqual(payload["advanced_options"]["layer_filter"], "")

    def test_block_map_groups_layers_and_keeps_timing_visible(self):
        block_map = chain_tools.format_artist_block_map(
            ["wlop", "krenz", "hiten"],
            ["0-1", "2-3", ""],
            ["0.0-0.5", "0.5-1.0", ""],
            num_blocks=4,
            target_blocks=[0, 1, 2, 3],
        )

        self.assertIn("L0-L1: wlop%0.00-0.50, hiten", block_map)
        self.assertIn("L2-L3: krenz%0.50-1.00, hiten", block_map)

    def test_external_cross_attention_wrapper_is_reported(self):
        class PlainCrossAttn:
            context_dim = 1024

        class ExternalWrapper:
            def __init__(self):
                self.original = PlainCrossAttn()

        class Block:
            def __init__(self, cross_attn):
                self.cross_attn = cross_attn

        dm = types.SimpleNamespace(blocks=[Block(PlainCrossAttn()), Block(ExternalWrapper())])
        hints = patching.describe_external_cross_attn_patches(dm, [0, 1])

        self.assertEqual(len(hints), 1)
        self.assertIn("L1", hints[0])
        self.assertIn("ExternalWrapper", hints[0])


class ChainBuilderTest(unittest.TestCase):
    def test_chain_builder_creates_layer_scheduled_chain(self):
        chain, report = chain_tools.build_artist_chain_from_rows(
            constants.CHAIN_LAYOUT_LAYER_SCHEDULED,
            [
                ("@wlop", 1.2, "", ""),
                ("krenz", 0.8, "", ""),
                ("", 1.0, "", ""),
            ],
            num_blocks=28,
        )

        self.assertEqual(
            chain,
            "::@wlop::1.2@0-8%0.0-0.45\n::krenz::0.8@9-18%0.35-0.85",
        )
        self.assertIn("L0-L8: @wlop%0.00-0.45", report)
        self.assertIn("L9-L18: krenz%0.35-0.85", report)

    def test_chain_preview_reports_invalid_timing_before_clip_encoding(self):
        cleaned, report = chain_tools.format_artist_chain_preview(
            "wlop@0,2,4%0.0-0.5, bad%0.5-0.5",
            num_blocks=28,
        )

        self.assertEqual(cleaned, "wlop@0,2,4%0.0-0.5\nbad%0.5-0.5")
        self.assertIn("L0: wlop%0.00-0.50", report)
        self.assertIn("invalid timing", report)

    def test_chain_preview_reports_percent_layer_windows(self):
        cleaned, report = chain_tools.format_artist_chain_preview(
            "@background@0%-33%, @character@0.33-0.67, @clothing@67%-100%",
            num_blocks=28,
        )

        self.assertEqual(
            cleaned,
            "@background@0%-33%\n@character@0.33-0.67\n@clothing@67%-100%",
        )
        self.assertIn("L0-L8: @background", report)
        self.assertIn("L9-L18: @character", report)
        self.assertIn("L19-L27: @clothing", report)

    def test_chain_builder_accepts_manual_percent_layer_windows(self):
        chain, report = chain_tools.build_artist_chain_from_rows(
            constants.CHAIN_LAYOUT_MANUAL,
            [
                ("@background", 1.0, "0%-33%", ""),
                ("@character", 1.0, "0.33-0.67", ""),
                ("@clothing", 1.0, "67%-100%", ""),
            ],
            num_blocks=28,
        )

        self.assertEqual(
            chain,
            "@background@0%-33%\n@character@0.33-0.67\n@clothing@67%-100%",
        )
        self.assertNotIn("invalid layer route ignored", report)
        self.assertIn("L0-L8: @background", report)
        self.assertIn("L9-L18: @character", report)
        self.assertIn("L19-L27: @clothing", report)

    def test_chain_preview_flags_negative_weights(self):
        _, report = chain_tools.format_artist_chain_preview(
            "::wlop::-0.5, krenz", num_blocks=28,
        )
        self.assertIn("negative ::weight detected", report)

    def test_chain_builder_ignores_invalid_manual_routes(self):
        chain, report = chain_tools.build_artist_chain_from_rows(
            constants.CHAIN_LAYOUT_MANUAL,
            [("wlop", 1.0, "abc", "0.2-0.2")],
            num_blocks=28,
        )

        self.assertEqual(chain, "wlop")
        self.assertIn("invalid layer route ignored", report)
        self.assertIn("invalid timing route ignored", report)

    def test_chain_builder_table_supports_more_than_three_artists(self):
        rows = chain_tools.parse_builder_artist_table(
            "@a | 1.2\n"
            "b | 0.8\n"
            "c\n"
            "d"
        )
        chain, report = chain_tools.build_artist_chain_from_rows(
            constants.CHAIN_LAYOUT_LAYER_SCHEDULED,
            rows,
            num_blocks=28,
        )
        lines = chain.splitlines()

        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "::@a::1.2@0-6%0.00-0.33")
        self.assertEqual(lines[-1], "d@21-27%0.67-1.00")
        self.assertIn("artists: 4", report)

    def test_chain_builder_node_accepts_table_artists(self):
        result = AnimaArtistChainBuilder().build(
            constants.CHAIN_LAYOUT_LAYER_SCHEDULED,
            "a\nb\nc\nd\ne",
            "",
            1.0,
            "",
            1.0,
            "",
            1.0,
            num_blocks=28,
        )
        chain, report = result["result"]

        self.assertEqual(len(chain.splitlines()), 5)
        self.assertIn("artists: 5", report)
        self.assertIn("e@22-27%0.72-1.00", chain)

    def test_chain_builder_reports_invalid_table_weight(self):
        rows, warnings = chain_tools.parse_builder_artist_table(
            "wlop | not-a-number",
            return_warnings=True,
        )
        chain, report = chain_tools.build_artist_chain_from_rows(
            constants.CHAIN_LAYOUT_MANUAL,
            rows,
            extra_warnings=warnings,
        )

        self.assertEqual(chain, "wlop")
        self.assertIn("status: CHECK", report)
        self.assertIn("invalid weight for wlop", report)

    def test_chain_builder_reports_empty_artist_chain(self):
        chain, report = chain_tools.build_artist_chain_from_rows(
            constants.CHAIN_LAYOUT_LAYER_SCHEDULED,
            [],
        )

        self.assertEqual(chain, "")
        self.assertIn("status: CHECK", report)
        self.assertIn("no artists", report)

    def test_starter_recipe_outputs_connectable_payloads(self):
        result = AnimaArtistStarter().build(
            constants.PRESET_COMPATIBILITY_SAFE,
            "a\nb\nc\nd",
            constants.CHAIN_LAYOUT_LAYER_SCHEDULED,
            1.0,
            normalize_weights=True,
            layer_mode=constants.LAYER_MODE_AUTO,
            custom_layer_filter="",
            num_blocks=28,
        )
        chain, preset, advanced_options, guide = result["result"]

        self.assertEqual(len(chain.splitlines()), 4)
        self.assertEqual(preset["preset"], constants.PRESET_COMPATIBILITY_SAFE)
        self.assertTrue(advanced_options["compatibility_mode"])
        self.assertIn("artist_chain -> AnimaArtistPack.artist_chain", guide)
        self.assertIn("preset -> AnimaArtistPresetApply.preset", guide)
        self.assertNotIn("preset -> AnimaArtistCrossAttn.preset", guide)
        self.assertIn("status: OK", guide)

    def test_starter_guide_reports_norm_lock_settings(self):
        result = AnimaArtistStarter().build(
            constants.PRESET_FACE_LOCK,
            "@uof\n@kieed",
            constants.CHAIN_LAYOUT_LAYER_SCHEDULED,
            1.0,
            normalize_weights=True,
            layer_mode=constants.LAYER_MODE_AUTO,
            custom_layer_filter="",
            num_blocks=28,
        )
        _, _, _, guide = result["result"]

        self.assertIn("match_base_norm: on", guide)
        self.assertIn("norm_lock_mode: token", guide)
        self.assertIn("norm_lock_scope: per_artist", guide)

    def test_starter_guide_explains_drift_auto_preview_resolution(self):
        result = AnimaArtistStarter().build(
            constants.PRESET_DRIFT_AUTO,
            "@uof\n@kieed",
            constants.CHAIN_LAYOUT_LAYER_SCHEDULED,
            1.0,
            normalize_weights=True,
            layer_mode=constants.LAYER_MODE_AUTO,
            custom_layer_filter="",
            num_blocks=28,
        )
        _, preset, _, guide = result["result"]

        self.assertEqual(preset["preset"], constants.PRESET_DRIFT_AUTO)
        self.assertIn("drift_auto: resolves at runtime", guide)
        self.assertIn("preview ignores base_prompt", guide)
        self.assertIn("preview_resolved_preset: drift_soft", guide)

    def test_starter_guide_uses_artist_count_for_drift_auto_preview(self):
        result = AnimaArtistStarter().build(
            constants.PRESET_DRIFT_AUTO,
            "@uof\n@kieed\n@ciloranko\n@huanxiang_heitu",
            constants.CHAIN_LAYOUT_LAYER_SCHEDULED,
            1.0,
            normalize_weights=True,
            layer_mode=constants.LAYER_MODE_AUTO,
            custom_layer_filter="",
            num_blocks=28,
        )
        _, preset, _, guide = result["result"]

        self.assertEqual(preset["preset"], constants.PRESET_DRIFT_AUTO)
        self.assertIn("artists: 4", guide)
        self.assertIn("preview_resolved_preset: drift_soft", guide)
        self.assertIn("4+ artists", guide)

    def test_preset_summary_marks_drift_auto_preview_as_base_prompt_blind(self):
        result = AnimaArtistPreset().build(
            constants.PRESET_DRIFT_AUTO,
            1.0,
            True,
            constants.LAYER_MODE_AUTO,
            "",
        )
        _, preset, _, summary = (None, *result["result"])

        self.assertEqual(preset["preset"], constants.PRESET_DRIFT_AUTO)
        self.assertIn("preview ignores base_prompt", summary)


class WorkflowExampleTest(unittest.TestCase):
    def test_layer_role_example_workflow_contains_three_role_routes(self):
        workflow_path = Path(REPO_ROOT) / "workflow" / "artist-layer-role-routing.json"
        data = json.loads(workflow_path.read_text(encoding="utf-8"))
        nodes = data.get("nodes", [])
        pack_nodes = [node for node in nodes if node.get("type") == "AnimaArtistPack"]

        self.assertTrue(pack_nodes)
        artist_chain = str(pack_nodes[0].get("widgets_values", [""])[0])

        self.assertIn("@raisuta@0%-33%", artist_chain)
        self.assertIn("@yuchi \\(salmon-1000\\)@0.33-0.67", artist_chain)
        self.assertIn("@blue-senpai@67%-100%", artist_chain)

        parts = parsing.split_artist_chain(artist_chain)
        parts, timing_routes = parsing.parse_artist_timing_routes(parts)
        parts, layer_routes = parsing.parse_artist_layer_routes(parts)
        names, _, _ = parsing.parse_artist_weights(parts)
        resolved_layers, has_layers = parsing.resolve_artist_layer_routes(layer_routes, 28)

        self.assertEqual(
            names,
            ["@raisuta", "@yuchi \\(salmon-1000\\)", "@blue-senpai"],
        )
        self.assertEqual(timing_routes, ["", "", ""])
        self.assertTrue(has_layers)
        self.assertEqual(resolved_layers[0], set(range(0, 9)))
        self.assertEqual(resolved_layers[1], set(range(9, 19)))
        self.assertEqual(resolved_layers[2], set(range(19, 28)))

    def test_layer_role_example_uses_preset_apply_node(self):
        workflow_path = Path(REPO_ROOT) / "workflow" / "artist-layer-role-routing.json"
        data = json.loads(workflow_path.read_text(encoding="utf-8"))
        node_types = {node.get("type") for node in data.get("nodes", [])}

        self.assertIn("AnimaArtistPresetApply", node_types)

    def test_sample_workflow_uses_preset_apply_without_advanced_options_node(self):
        workflow_path = Path(REPO_ROOT) / "sample workflow.json"
        data = json.loads(workflow_path.read_text(encoding="utf-8"))
        node_types = {node.get("type") for node in data.get("nodes", [])}

        self.assertIn("AnimaArtistPreset", node_types)
        self.assertIn("AnimaArtistPresetApply", node_types)
        self.assertNotIn("AnimaArtistOptions", node_types)

    def test_pr4_api_examples_use_preset_apply_for_preset_paths(self):
        workflow_dir = Path(REPO_ROOT) / "workflow" / "pr4_self_test_api"
        workflow_paths = sorted(workflow_dir.glob("*.json"))

        self.assertTrue(workflow_paths)
        for workflow_path in workflow_paths:
            data = json.loads(workflow_path.read_text(encoding="utf-8"))
            node_types = {
                str(node.get("class_type", ""))
                for node in data.values()
                if isinstance(node, dict)
            }

            self.assertIn("AnimaArtistPresetApply", node_types, workflow_path.name)
            self.assertNotIn("AnimaArtistBasic", node_types, workflow_path.name)

    def test_node_usage_showcase_covers_every_registered_node(self):
        workflow_dir = Path(REPO_ROOT) / "workflow" / "node_usage_showcase"
        workflow_paths = sorted(workflow_dir.glob("*.json"))

        self.assertTrue(workflow_paths)

        import anima_mixer

        seen = set()
        for workflow_path in workflow_paths:
            data = json.loads(workflow_path.read_text(encoding="utf-8"))
            for node in data.values():
                if not isinstance(node, dict):
                    continue
                node_type = str(node.get("class_type", ""))
                if node_type.startswith("AnimaArtist"):
                    seen.add(node_type)

        self.assertEqual(seen, set(anima_mixer.NODE_CLASS_MAPPINGS))


class InspectorTest(unittest.TestCase):
    def _pack(self, **overrides):
        pack = {
            "labels": ["wlop"],
            "weights": [1.0],
            "layer_routes": [""],
            "timing_routes": [""],
            "has_explicit_weights": False,
            "base_prompt": "portrait",
        }
        pack.update(overrides)
        return pack

    def test_inspector_has_ok_status_without_forced_warning(self):
        report = AnimaArtistInspector().inspect(self._pack())["result"][0]

        self.assertIn("status: OK", report)
        self.assertIn("warnings:\n  - no obvious configuration risk", report)
        self.assertIn("notes:", report)

    def test_inspector_notes_negative_weights(self):
        report = AnimaArtistInspector().inspect(
            self._pack(labels=["wlop", "bad"], weights=[1.0, -0.5],
                       layer_routes=["", ""], timing_routes=["", ""],
                       has_explicit_weights=True),
        )["result"][0]

        self.assertIn("negative weights present", report)

    def test_inspector_reports_norm_lock_settings(self):
        report = AnimaArtistInspector().inspect(
            self._pack(),
            advanced_options={
                "norm_lock_mode": constants.NORM_LOCK_ROW,
                "norm_lock_scope": constants.NORM_LOCK_SCOPE_MIXED,
                "contribution_balance": False,
                "contribution_balance_alpha": 0.35,
                "mixed_delta_cap": True,
                "mixed_delta_cap_ratio": 0.8,
            },
        )["result"][0]

        self.assertIn("norm_lock_mode: row", report)
        self.assertIn("norm_lock_scope: mixed", report)
        self.assertIn("contribution_balance: off", report)
        self.assertIn("contribution_balance_alpha: 0.35", report)
        self.assertIn("mixed_delta_cap: on", report)
        self.assertIn("mixed_delta_cap_ratio: 0.80", report)

    def test_inspector_resolves_drift_auto_from_base_prompt(self):
        report = AnimaArtistInspector().inspect(
            self._pack(
                base_prompt=(
                    "1girl, solo, wide shot, full body, small figure, "
                    "cityscape, detailed background, daylight"
                ),
            ),
            preset=options.build_preset_payload(constants.PRESET_DRIFT_AUTO),
        )["result"][0]

        self.assertIn("preset: drift_auto", report)
        self.assertIn("resolved_preset: scene_lock", report)
        self.assertIn("drift_auto_reason: wide or background-heavy", report)


class PromptPassthroughRuntimeTest(unittest.TestCase):
    def test_prompt_passthrough_preserves_raw_unweighted_prompt(self):
        class Clip:
            def __init__(self):
                self.encoded_texts = []

            def tokenize(self, text):
                return {"text": text}

            def encode_from_tokens_scheduled(self, tokens):
                text = tokens["text"]
                self.encoded_texts.append(text)
                return {"conditioning": text}

        class Model:
            pass

        clip = Clip()
        model = Model()
        artist_pack = AnimaArtistPack().pack(
            clip,
            "@yuchi \\(salmon-1000\\), @uof",
            "1girl, solo",
        )[0]
        payload = options.build_preset_payload(constants.PRESET_PROMPT_PASSTHROUGH)

        patched_model, positive = AnimaArtistCrossAttn().patch(
            model,
            artist_pack,
            constants.COMBINE_OUTPUT_AVG,
            constants.FUSION_INTERPOLATE,
            1.0,
            True,
            False,
            preset=payload,
        )

        self.assertIs(patched_model, model)
        self.assertEqual(
            positive,
            {
                "conditioning": (
                    "@yuchi \\(salmon-1000\\), @uof\n\n1girl, solo"
                ),
            },
        )
        self.assertIn(
            "@yuchi \\(salmon-1000\\), @uof\n\n1girl, solo",
            clip.encoded_texts,
        )

    def test_prompt_passthrough_converts_explicit_artist_weights(self):
        class Clip:
            def __init__(self):
                self.encoded_texts = []

            def tokenize(self, text):
                return {"text": text}

            def encode_from_tokens_scheduled(self, tokens):
                text = tokens["text"]
                self.encoded_texts.append(text)
                return {"conditioning": text}

        class Model:
            pass

        clip = Clip()
        model = Model()
        artist_pack = AnimaArtistPack().pack(
            clip,
            "1.2::@yuchi \\(salmon-1000\\)::, @uof::0.8",
            "1girl, solo",
        )[0]
        payload = options.build_preset_payload(constants.PRESET_PROMPT_PASSTHROUGH)

        patched_model, positive = AnimaArtistCrossAttn().patch(
            model,
            artist_pack,
            constants.COMBINE_OUTPUT_AVG,
            constants.FUSION_INTERPOLATE,
            1.0,
            True,
            False,
            preset=payload,
        )

        self.assertIs(patched_model, model)
        self.assertEqual(
            positive,
            {
                "conditioning": (
                    "(@yuchi \\(salmon-1000\\):1.2), (@uof:0.8)\n\n1girl, solo"
                ),
            },
        )
        self.assertIn(
            "(@yuchi \\(salmon-1000\\):1.2), (@uof:0.8)\n\n1girl, solo",
            clip.encoded_texts,
        )

    def test_prompt_passthrough_rejects_mixer_only_routes(self):
        class Clip:
            def tokenize(self, text):
                return {"text": text}

            def encode_from_tokens_scheduled(self, tokens):
                return {"conditioning": tokens["text"]}

        class Model:
            pass

        artist_pack = AnimaArtistPack().pack(
            Clip(),
            "@wlop@0-8",
            "1girl",
        )[0]
        payload = options.build_preset_payload(constants.PRESET_PROMPT_PASSTHROUGH)

        with self.assertRaisesRegex(ValueError, "does not support layer routes"):
            AnimaArtistCrossAttn().patch(
                Model(),
                artist_pack,
                constants.COMBINE_OUTPUT_AVG,
                constants.FUSION_INTERPOLATE,
                1.0,
                True,
                False,
                preset=payload,
            )


class RecipeTest(unittest.TestCase):
    def test_legacy_embed_avg_recipe_falls_back_to_output_avg(self):
        # embed_avg was cut before the v26 release (it averaged
        # token-misaligned text embeddings and produced heavy artifacts).
        # Recipes saved with it must degrade to output_avg with a warning,
        # not crash.
        text = recipe.serialize_recipe(
            "wlop", constants.COMBINE_OUTPUT_AVG, constants.FUSION_INTERPOLATE, 1.0,
        ).replace('"output_avg"', '"embed_avg"')
        payload, warnings = recipe.deserialize_recipe(text)
        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertTrue(any("embed_avg" in w for w in warnings))

    def test_recipe_roundtrip(self):
        adv = options.base_advanced_options()
        adv["artist_ema_alpha"] = 0.25
        adv["low_vram_cache"] = True
        adv["norm_lock_mode"] = constants.NORM_LOCK_ROW
        adv["norm_lock_scope"] = constants.NORM_LOCK_SCOPE_MIXED
        adv["contribution_balance"] = False
        adv["contribution_balance_alpha"] = 0.4
        adv["mixed_delta_cap"] = True
        adv["mixed_delta_cap_ratio"] = 0.75
        text = recipe.serialize_recipe(
            "wlop, ::krenz::0.8", constants.COMBINE_LOWRANK_AVG,
            constants.FUSION_INTERPOLATE, 1.4, adv, notes="my mix",
        )
        payload, warnings = recipe.deserialize_recipe(text)

        self.assertEqual(warnings, [])
        self.assertEqual(payload["artist_chain"], "wlop, ::krenz::0.8")
        self.assertEqual(payload["combine_mode"], constants.COMBINE_LOWRANK_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_INTERPOLATE)
        self.assertAlmostEqual(payload["strength"], 1.4)
        self.assertEqual(payload["advanced_options"]["artist_ema_alpha"], 0.25)
        self.assertTrue(payload["advanced_options"]["low_vram_cache"])
        self.assertEqual(payload["advanced_options"]["norm_lock_mode"], constants.NORM_LOCK_ROW)
        self.assertEqual(payload["advanced_options"]["norm_lock_scope"], constants.NORM_LOCK_SCOPE_MIXED)
        self.assertFalse(payload["advanced_options"]["contribution_balance"])
        self.assertAlmostEqual(payload["advanced_options"]["contribution_balance_alpha"], 0.4)
        self.assertTrue(payload["advanced_options"]["mixed_delta_cap"])
        self.assertAlmostEqual(payload["advanced_options"]["mixed_delta_cap_ratio"], 0.75)
        self.assertEqual(payload["notes"], "my mix")

    def test_recipe_rejects_garbage(self):
        with self.assertRaises(ValueError):
            recipe.deserialize_recipe("not json at all {")

    def test_recipe_unknown_modes_fall_back_with_warning(self):
        payload, warnings = recipe.deserialize_recipe(
            '{"format": "anima-artist-recipe", "version": 1, '
            '"artist_chain": "a", "combine_mode": "wat", "fusion_mode": "huh"}'
        )
        self.assertEqual(payload["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(payload["fusion_mode"], constants.FUSION_INTERPOLATE)
        self.assertTrue(any("combine_mode" in w for w in warnings))
        self.assertTrue(any("fusion_mode" in w for w in warnings))

    def test_recipe_nodes_roundtrip(self):
        save_result = AnimaArtistRecipeSave().save(
            "wlop\nkrenz", constants.COMBINE_OUTPUT_AVG,
            constants.FUSION_INTERPOLATE, 1.0,
        )
        recipe_json = save_result["result"][0]
        chain, preset, adv, summary = AnimaArtistRecipeLoad().load(recipe_json)["result"]

        self.assertEqual(chain, "wlop\nkrenz")
        self.assertEqual(preset["combine_mode"], constants.COMBINE_OUTPUT_AVG)
        self.assertEqual(preset["fusion_mode"], constants.FUSION_INTERPOLATE)
        self.assertIn("status: OK", summary)
        self.assertIn("preset -> AnimaArtistPresetApply.preset", summary)
        self.assertNotIn("preset -> AnimaArtistCrossAttn.preset", summary)


class RegistryTest(unittest.TestCase):
    def test_basic_node_is_small_balanced_entrypoint(self):
        inputs = AnimaArtistBasic.INPUT_TYPES()["required"]

        self.assertEqual(list(inputs), [
            "model",
            "clip",
            "artist_chain",
            "base_prompt",
            "preset",
            "intensity",
            "enabled",
        ])
        self.assertEqual(inputs["preset"][1]["default"], constants.PRESET_BALANCED)
        self.assertEqual(inputs["preset"][0], constants.PRESET_RECOMMENDED_CHOICES)
        self.assertIn(constants.PRESET_PROMPT_PASSTHROUGH, inputs["preset"][0])

    def test_preset_apply_node_hides_manual_mixer_settings(self):
        inputs = AnimaArtistPresetApply.INPUT_TYPES()

        required = inputs["required"]
        optional = inputs["optional"]

        self.assertIn("preset", required)
        self.assertIn("advanced_options", optional)
        self.assertNotIn("combine_mode", required)
        self.assertNotIn("fusion_mode", required)
        self.assertNotIn("strength", required)

    def test_manual_cross_attn_keeps_compatibility_preset_socket(self):
        inputs = AnimaArtistCrossAttn.INPUT_TYPES()

        required = inputs["required"]
        optional = inputs["optional"]

        self.assertIn("combine_mode", required)
        self.assertIn("fusion_mode", required)
        self.assertIn("strength", required)
        self.assertIn("preset", optional)

    def test_starter_uses_recommended_presets_only(self):
        inputs = AnimaArtistStarter.INPUT_TYPES()["required"]

        self.assertEqual(inputs["recipe"][1]["default"], constants.PRESET_BALANCED)
        self.assertEqual(inputs["recipe"][0], constants.PRESET_RECOMMENDED_CHOICES)
        self.assertIn(constants.PRESET_PROMPT_PASSTHROUGH, inputs["recipe"][0])

    def test_preset_node_keeps_advanced_presets_available(self):
        inputs = AnimaArtistPreset.INPUT_TYPES()["required"]

        self.assertEqual(inputs["preset"][0], constants.PRESET_CHOICES)
        self.assertIn(constants.PRESET_COMPATIBILITY_SAFE, inputs["preset"][0])
        self.assertIn(constants.PRESET_FACE_LOCK, inputs["preset"][0])

    def test_node_mappings_complete(self):
        import anima_mixer
        self.assertEqual(
            set(anima_mixer.NODE_CLASS_MAPPINGS),
            set(anima_mixer.NODE_DISPLAY_NAME_MAPPINGS),
        )
        self.assertIn("AnimaArtistBasic", anima_mixer.NODE_CLASS_MAPPINGS)
        self.assertIn("AnimaArtistPresetApply", anima_mixer.NODE_CLASS_MAPPINGS)
        self.assertIn("AnimaArtistCrossAttn", anima_mixer.NODE_CLASS_MAPPINGS)
        self.assertIn("AnimaArtistRecipeSave", anima_mixer.NODE_CLASS_MAPPINGS)
        self.assertIn("AnimaArtistProbe", anima_mixer.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
