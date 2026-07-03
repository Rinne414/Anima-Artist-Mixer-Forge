"""Regression tests for the verified parsing/config-layer fixes.

One class per fix; unittest.TestCase style so both pytest and unittest
discovery collect them.
"""

import json
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from anima_mixer import chain_tools, constants, options, parsing, recipe  # noqa: E402
from anima_mixer.nodes_core import AnimaArtistPack  # noqa: E402
from anima_mixer.nodes_ui import (  # noqa: E402
    AnimaArtistRecipeLoad,
    AnimaArtistRecipeSave,
)


def _parse_chain(chain):
    """Run the full name/weight/route pipeline the Pack node uses."""
    parts = parsing.split_artist_chain(chain)
    parts, timings = parsing.parse_artist_timing_routes(parts)
    parts, layers = parsing.parse_artist_layer_routes(parts)
    names, weights, has_explicit = parsing.parse_artist_weights(parts)
    return names, weights, has_explicit, layers, timings


# ── FIX 1 ─────────────────────────────────────────────────────────────────
class CommaLayerRoutePhantomTest(unittest.TestCase):
    def test_prefix_weight_comma_route_yields_one_artist(self):
        names, weights, _, layers, _ = _parse_chain("1.2::wlop@0,2,4::")
        self.assertEqual(names, ["wlop"])
        self.assertEqual(weights, [1.2])
        resolved, _ = parsing.resolve_artist_layer_routes(layers, 28)
        self.assertEqual(resolved[0], {0, 2, 4})

    def test_postfix_weight_after_comma_route_yields_one_artist(self):
        names, weights, _, layers, _ = _parse_chain("wlop@0,2,4::1.2")
        self.assertEqual(names, ["wlop"])
        self.assertEqual(weights, [1.2])
        resolved, _ = parsing.resolve_artist_layer_routes(layers, 28)
        self.assertEqual(resolved[0], {0, 2, 4})

    def test_comma_route_then_plain_artist_still_splits(self):
        names, _, _, _, _ = _parse_chain("wlop@0,2,4, hiten")
        self.assertEqual(names, ["wlop", "hiten"])

    def test_comma_after_closing_marker_still_splits(self):
        names, _, _, layers, _ = _parse_chain("1.2::wlop@0,2::, krenz")
        self.assertEqual(names, ["wlop", "krenz"])
        resolved, _ = parsing.resolve_artist_layer_routes(layers, 28)
        self.assertEqual(resolved[0], {0, 2})


# ── FIX 2 ─────────────────────────────────────────────────────────────────
class LayerFilterOutOfRangeTest(unittest.TestCase):
    def test_out_of_range_range_returns_empty_with_warning(self):
        warnings = []
        self.assertEqual(parsing.parse_layer_filter("30-40", 28, warnings), [])
        self.assertTrue(any("matches no blocks" in m for m in warnings))

    def test_out_of_range_single_returns_empty_with_warning(self):
        warnings = []
        self.assertEqual(parsing.parse_layer_filter("99", 28, warnings), [])
        self.assertTrue(any("matches no blocks" in m for m in warnings))

    def test_partial_valid_with_unparseable_part(self):
        warnings = []
        self.assertEqual(
            parsing.parse_layer_filter("0-8,abc", 28, warnings), list(range(0, 9))
        )
        self.assertTrue(any("abc" in m for m in warnings))

    def test_empty_text_returns_none(self):
        self.assertIsNone(parsing.parse_layer_filter("", 28))

    def test_unrecognized_text_returns_none(self):
        self.assertIsNone(parsing.parse_layer_filter("abc", 28))

    def test_resolve_maps_empty_to_empty_set(self):
        routes, has_routes = parsing.resolve_artist_layer_routes(["30-40"], 28)
        self.assertTrue(has_routes)
        self.assertEqual(routes[0], set())

    def test_resolve_maps_no_filter_to_none(self):
        routes, has_routes = parsing.resolve_artist_layer_routes([""], 28)
        self.assertFalse(has_routes)
        self.assertIsNone(routes[0])

    def test_target_blocks_strict_raises_on_empty(self):
        with self.assertRaisesRegex(ValueError, "matches no blocks"):
            parsing.resolve_target_blocks_from_options(
                {"layer_filter": "99"}, 28, strict=True,
            )

    def test_target_blocks_nonstrict_returns_empty(self):
        self.assertEqual(
            parsing.resolve_target_blocks_from_options({"layer_filter": "99"}, 28),
            [],
        )


# ── FIX 3 ─────────────────────────────────────────────────────────────────
class CleanedChainRoundTripTest(unittest.TestCase):
    def test_explicit_unit_weight_preserved(self):
        cleaned, _ = chain_tools.format_artist_chain_preview("1.0::wlop::", num_blocks=28)
        self.assertEqual(cleaned, "1::wlop::")

    def test_weight_formatting_not_truncated(self):
        cleaned, _ = chain_tools.format_artist_chain_preview("1.2345::wlop::", num_blocks=28)
        self.assertEqual(cleaned, "1.2345::wlop::")

    def test_parse_of_cleaned_matches_original(self):
        chain = "1.2::wlop@0-8%0.0-0.45::, 1.0::krenz::, hiten"
        cleaned, _ = chain_tools.format_artist_chain_preview(chain, num_blocks=28)
        self.assertEqual(_parse_chain(chain), _parse_chain(cleaned))


# ── FIX 4 ─────────────────────────────────────────────────────────────────
class FullWidthCommaExpansionTest(unittest.TestCase):
    def test_fullwidth_comma_before_weight_expands(self):
        result = parsing.expand_prompt_weights("masterpiece，1.5::high quality::，1girl")
        self.assertIn("(high quality:1.5)", result)

    def test_mixed_width_prompt_expands(self):
        result = parsing.expand_prompt_weights("masterpiece, 1.5::high quality::，1girl")
        self.assertIn("(high quality:1.5)", result)


# ── FIX 5 ─────────────────────────────────────────────────────────────────
class NaNWeightTest(unittest.TestCase):
    def test_clamp_float_raises_on_nan(self):
        with self.assertRaises(ValueError):
            parsing.clamp_float(float("nan"), 0.0, 4.0)

    def test_nan_prefix_weight_kept_as_raw_not_explicit(self):
        names, weights, has_explicit = parsing.parse_artist_weights(["nan::wlop::"])
        self.assertEqual(weights, [1.0])
        self.assertFalse(has_explicit)
        self.assertIn("wlop", names[0])

    def test_inf_prefix_weight_clamps_to_max(self):
        names, weights, has_explicit = parsing.parse_artist_weights(["inf::wlop::"])
        self.assertEqual(names, ["wlop"])
        self.assertEqual(weights, [constants.WEIGHT_MAX])
        self.assertTrue(has_explicit)

    def test_builder_table_nan_weight_does_not_crash(self):
        rows, warnings = chain_tools.parse_builder_artist_table(
            "wlop | nan", return_warnings=True,
        )
        chain, _ = chain_tools.build_artist_chain_from_rows(
            constants.CHAIN_LAYOUT_MANUAL, rows,
        )
        self.assertEqual(chain, "wlop")
        self.assertTrue(any("invalid weight" in m for m in warnings))


# ── FIX 6 ─────────────────────────────────────────────────────────────────
class DecorativeAndDoubleWeightTest(unittest.TestCase):
    def test_decorative_both_sides_strips_colons(self):
        names, weights, has_explicit = parsing.parse_artist_weights(["::sakimichan::"])
        self.assertEqual(names, ["sakimichan"])
        self.assertEqual(weights, [1.0])
        self.assertFalse(has_explicit)

    def test_double_weight_prefix_wins(self):
        names, weights, has_explicit = parsing.parse_artist_weights(["1.5::wlop::0.8"])
        self.assertEqual(names, ["wlop"])
        self.assertEqual(weights, [1.5])
        self.assertTrue(has_explicit)

    def test_decorative_single_prefix_still_works(self):
        names, _, has_explicit = parsing.parse_artist_weights(["::wlop"])
        self.assertEqual(names, ["wlop"])
        self.assertFalse(has_explicit)

    def test_bare_route_with_trailing_marker_strips_colons(self):
        # `wlop@0-8::` (no weight): the route detaches, and the leftover
        # bare `::` must not stay in the CLIP-encoded name.
        names, weights, has_explicit, layers, _ = _parse_chain("wlop@0-8::")
        self.assertEqual(names, ["wlop"])
        self.assertEqual(weights, [1.0])
        self.assertFalse(has_explicit)
        resolved, _ = parsing.resolve_artist_layer_routes(layers, 28)
        self.assertEqual(resolved[0], set(range(0, 9)))


# ── FIX 7 ─────────────────────────────────────────────────────────────────
class TimingScalingTest(unittest.TestCase):
    def test_bare_percent_window_scales(self):
        self.assertEqual(parsing.parse_timing_filter("0-45"), (0.0, 0.45, 0.0))

    def test_bare_percent_midrange_scales(self):
        self.assertEqual(parsing.parse_timing_filter("50-80"), (0.5, 0.8, 0.0))

    def test_bare_percent_with_fade_scales(self):
        self.assertEqual(parsing.parse_timing_filter("20-80~10"), (0.2, 0.8, 0.1))

    def test_fractional_window_unchanged(self):
        self.assertEqual(parsing.parse_timing_filter("0.0-0.45"), (0.0, 0.45, 0.0))

    def test_over_100_returns_none(self):
        self.assertIsNone(parsing.parse_timing_filter("150-400"))


# ── FIX 8 ─────────────────────────────────────────────────────────────────
class RecipeRangeValidationTest(unittest.TestCase):
    def _recipe_with_adv(self, **adv_overrides):
        text = recipe.serialize_recipe(
            "wlop", constants.COMBINE_OUTPUT_AVG, constants.FUSION_INTERPOLATE, 1.0,
        )
        data = json.loads(text)
        data["advanced_options"].update(adv_overrides)
        return json.dumps(data)

    def test_string_bool_false_parses_to_false(self):
        payload, _ = recipe.deserialize_recipe(
            self._recipe_with_adv(normalize_weights="false"),
        )
        self.assertFalse(payload["advanced_options"]["normalize_weights"])

    def test_out_of_range_ema_clamped_with_warning(self):
        payload, warnings = recipe.deserialize_recipe(
            self._recipe_with_adv(artist_ema_alpha=999),
        )
        self.assertAlmostEqual(payload["advanced_options"]["artist_ema_alpha"], 0.95)
        self.assertTrue(any("artist_ema_alpha" in m for m in warnings))

    def test_nan_strength_falls_back_to_default(self):
        payload, warnings = recipe.deserialize_recipe(
            '{"format": "anima-artist-recipe", "version": 2, "strength": NaN}'
        )
        self.assertAlmostEqual(payload["strength"], 1.0)
        self.assertTrue(any("strength" in m for m in warnings))


# ── FIX 9 ─────────────────────────────────────────────────────────────────
class RecipeDriftAutoDeferralTest(unittest.TestCase):
    def test_drift_auto_recipe_roundtrip_preserves_dynamic_routing(self):
        preset = options.build_preset_payload(constants.PRESET_DRIFT_AUTO)
        recipe_json = AnimaArtistRecipeSave().save(
            "wlop, krenz",
            constants.COMBINE_OUTPUT_AVG,
            constants.FUSION_INTERPOLATE,
            1.0,
            preset=preset,
        )["result"][0]

        _, loaded_preset, adv, _ = AnimaArtistRecipeLoad().load(recipe_json)["result"]
        # The recipe must round-trip as drift_auto so the route stays dynamic.
        self.assertEqual(loaded_preset["preset"], constants.PRESET_DRIFT_AUTO)

        _, _, _, adv_closeup, _ = options.merge_runtime_options(
            constants.COMBINE_OUTPUT_AVG, constants.FUSION_INTERPOLATE, 1.0,
            adv, loaded_preset,
            base_prompt=(
                "1girl, close-up portrait, detailed eyes, looking at viewer, "
                "simple background"
            ),
        )
        _, _, _, adv_wide, _ = options.merge_runtime_options(
            constants.COMBINE_OUTPUT_AVG, constants.FUSION_INTERPOLATE, 1.0,
            adv, loaded_preset,
            base_prompt="1girl, wide shot, cityscape, detailed background, daylight",
        )
        self.assertNotEqual(
            adv_closeup["drift_auto_resolved_preset"],
            adv_wide["drift_auto_resolved_preset"],
        )

    def test_v1_recipe_without_preset_key_still_loads(self):
        v1 = (
            '{"format": "anima-artist-recipe", "version": 1, '
            '"artist_chain": "wlop", "combine_mode": "output_avg", '
            '"fusion_mode": "interpolate", "strength": 1.0}'
        )
        chain, preset, _, _ = AnimaArtistRecipeLoad().load(v1)["result"]
        self.assertEqual(chain, "wlop")
        self.assertEqual(preset["preset"], "recipe")


# ── FIX 10 ────────────────────────────────────────────────────────────────
class LintParsedArtistsTest(unittest.TestCase):
    def test_leftover_double_colon_flagged(self):
        warnings = chain_tools.lint_parsed_artists(
            ["nan::wlop"], [""], [""], "nan::wlop",
        )
        self.assertTrue(any("weight syntax" in m for m in warnings))

    def test_fullwidth_route_marker_flagged(self):
        warnings = chain_tools.lint_parsed_artists(
            ["wlop＠0-8"], [""], [""], "wlop＠0-8",
        )
        self.assertTrue(any("full-width" in m for m in warnings))

    def test_swallowed_route_tail_flagged(self):
        warnings = chain_tools.lint_parsed_artists(
            ["wlop%0.5-0.5"], [""], [""], "wlop%0.5-0.5",
        )
        self.assertTrue(any("swallowed route" in m for m in warnings))

    def test_at_percent_confusion_flagged(self):
        warnings = chain_tools.lint_parsed_artists(
            ["wlop", "krenz"], ["0.0-0.5", "0.5-1.0"], ["", ""],
            "wlop@0.0-0.5, krenz@0.5-1.0",
        )
        self.assertTrue(any("LAYER range" in m for m in warnings))

    def test_clean_chain_has_no_lint(self):
        warnings = chain_tools.lint_parsed_artists(
            ["wlop", "krenz"], ["0-8", "9-18"], ["", ""], "wlop@0-8, krenz@9-18",
        )
        self.assertEqual(warnings, [])


class MaxArtistsTruncationTest(unittest.TestCase):
    class _Clip:
        def tokenize(self, text):
            return {"text": text}

        def encode_from_tokens_scheduled(self, tokens):
            return {"conditioning": tokens["text"]}

    def test_truncation_recomputes_has_explicit(self):
        # The only explicit weight sits past the MAX_ARTISTS cutoff; it must
        # not disable normalization for the surviving weight-1.0 artists.
        chain = ", ".join(f"artist{i}" for i in range(constants.MAX_ARTISTS))
        chain += ", 1.5::overflow_artist::"
        pack = AnimaArtistPack().pack(self._Clip(), chain, "1girl")[0]
        self.assertEqual(len(pack["labels"]), constants.MAX_ARTISTS)
        self.assertFalse(pack["has_explicit_weights"])

    def test_explicit_weight_inside_limit_survives_truncation(self):
        chain = "1.5::artist0::, " + ", ".join(
            f"artist{i}" for i in range(1, constants.MAX_ARTISTS + 4)
        )
        pack = AnimaArtistPack().pack(self._Clip(), chain, "1girl")[0]
        self.assertEqual(len(pack["labels"]), constants.MAX_ARTISTS)
        self.assertTrue(pack["has_explicit_weights"])


if __name__ == "__main__":
    unittest.main()
