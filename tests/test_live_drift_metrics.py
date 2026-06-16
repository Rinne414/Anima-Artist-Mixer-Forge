"""Unit tests for the live drift A/B image descriptors."""

import os
import sys
import tempfile
import unittest

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - optional integration dependency
    Image = None
    ImageDraw = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_ROOT = os.path.join(REPO_ROOT, "tests")
for path in (REPO_ROOT, TESTS_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

if Image is not None:
    import live_drift_ab  # noqa: E402


@unittest.skipIf(Image is None, "Pillow is required for drift descriptor tests")
class DriftDescriptorTest(unittest.TestCase):
    def _portrait_like(self, background, shirt, face):
        img = Image.new("RGB", (128, 128), background)
        draw = ImageDraw.Draw(img)
        draw.rectangle((42, 58, 86, 118), fill=shirt)
        draw.ellipse((48, 16, 80, 54), fill=face)
        return img

    def test_foreground_descriptor_downweights_background_changes(self):
        base = self._portrait_like((40, 80, 120), (210, 40, 40), (235, 185, 145))
        bg_changed = self._portrait_like((120, 80, 40), (210, 40, 40), (235, 185, 145))

        base_desc = live_drift_ab._descriptors_for_loaded_image(base)
        changed_desc = live_drift_ab._descriptors_for_loaded_image(bg_changed)

        full_dist = live_drift_ab._descriptor_distance(base_desc["full"], changed_desc["full"])
        foreground_dist = live_drift_ab._descriptor_distance(
            base_desc["foreground"], changed_desc["foreground"],
        )

        self.assertLess(foreground_dist, full_dist * 0.8)

    def test_foreground_descriptor_tracks_subject_changes(self):
        base = self._portrait_like((40, 80, 120), (210, 40, 40), (235, 185, 145))
        bg_changed = self._portrait_like((120, 80, 40), (210, 40, 40), (235, 185, 145))
        subject_changed = self._portrait_like((40, 80, 120), (40, 120, 220), (245, 205, 175))

        base_desc = live_drift_ab._descriptors_for_loaded_image(base)
        bg_desc = live_drift_ab._descriptors_for_loaded_image(bg_changed)
        subject_desc = live_drift_ab._descriptors_for_loaded_image(subject_changed)

        bg_dist = live_drift_ab._descriptor_distance(base_desc["foreground"], bg_desc["foreground"])
        subject_dist = live_drift_ab._descriptor_distance(
            base_desc["foreground"], subject_desc["foreground"],
        )

        self.assertGreater(subject_dist, bg_dist * 1.5)

    def test_metric_pairwise_distance_reports_each_descriptor(self):
        images = [
            self._portrait_like((40, 80, 120), (210, 40, 40), (235, 185, 145)),
            self._portrait_like((120, 80, 40), (210, 40, 40), (235, 185, 145)),
            self._portrait_like((40, 80, 120), (40, 120, 220), (245, 205, 175)),
        ]
        descs = [live_drift_ab._descriptors_for_loaded_image(img) for img in images]

        distances = live_drift_ab._pairwise_metric_distances(descs)

        self.assertIn("full", distances)
        self.assertIn("center", distances)
        self.assertIn("upper_center", distances)
        self.assertIn("foreground", distances)
        self.assertGreater(distances["foreground"], 0.0)

    def test_reduction_summary_flags_cross_metric_risk(self):
        reductions = {
            "a": 0.2,
            "b": 0.1,
        }
        metric_reductions = {
            "full": {"a": 0.1, "b": 0.2},
            "center": {"a": -0.05, "b": 0.01},
            "upper_center": {"a": 0.3, "b": 0.0},
            "foreground": {"a": 0.2, "b": 0.1},
        }

        summary = live_drift_ab._summarize_reductions(
            reductions, metric_reductions, comparison_metric="foreground",
        )

        self.assertEqual(summary["best_by_metric"]["foreground"], "a")
        self.assertEqual(summary["best_by_metric"]["full"], "b")
        self.assertFalse(summary["configs"]["a"]["all_positive"])
        self.assertTrue(summary["configs"]["b"]["all_positive"])
        self.assertEqual(summary["configs"]["a"]["negative_metrics"], ["center"])

    def test_live_config_registry_includes_lightweight_presets(self):
        configs = live_drift_ab.build_config_registry()

        self.assertEqual(configs["balanced_preset"]["use_preset"], "balanced")
        self.assertEqual(
            configs["compatibility_safe_preset"]["use_preset"],
            "compatibility_safe",
        )
        self.assertEqual(configs["identity_guard_preset"]["use_preset"], "identity_guard")

    def test_live_config_registry_includes_closeup_hybrid_candidates(self):
        configs = live_drift_ab.build_config_registry()

        self.assertEqual(configs["face_lock_k5_base_preserve"]["fusion"], "base_preserve")
        self.assertEqual(
            configs["face_lock_k5_base_preserve"]["opts"]["static_capture_k"],
            5,
        )
        self.assertEqual(configs["face_lock_k6_base_preserve"]["fusion"], "base_preserve")
        self.assertEqual(
            configs["face_lock_k6_no_norm_base_preserve"]["opts"]["static_capture_k"],
            6,
        )
        self.assertFalse(
            configs["face_lock_k6_no_norm_base_preserve"]["opts"]["match_base_norm"]
        )

    def test_live_config_registry_includes_delta_cap_candidates(self):
        configs = live_drift_ab.build_config_registry()

        self.assertTrue(configs["stable_seed_delta_cap075"]["opts"]["mixed_delta_cap"])
        self.assertAlmostEqual(
            configs["stable_seed_delta_cap075"]["opts"]["mixed_delta_cap_ratio"],
            0.75,
        )
        self.assertEqual(configs["face_lock_delta_cap100"]["fusion"], "base_preserve")
        self.assertTrue(configs["face_lock_delta_cap100"]["opts"]["mixed_delta_cap"])
        self.assertAlmostEqual(
            configs["face_lock_delta_cap100"]["opts"]["mixed_delta_cap_ratio"],
            1.0,
        )

    def test_aggregate_reduction_summaries_tracks_stability(self):
        summaries = [
            {
                "pairwise_distance_reduction": {"a": 0.2, "b": 0.1},
                "reduction_summary": {
                    "best_by_comparison_metric": "a",
                    "configs": {
                        "a": {"all_positive": True, "negative_metrics": []},
                        "b": {"all_positive": False, "negative_metrics": ["center"]},
                    },
                },
            },
            {
                "pairwise_distance_reduction": {"a": -0.1, "b": 0.05},
                "reduction_summary": {
                    "best_by_comparison_metric": "b",
                    "configs": {
                        "a": {"all_positive": False, "negative_metrics": ["full"]},
                        "b": {"all_positive": True, "negative_metrics": []},
                    },
                },
            },
        ]

        aggregate = live_drift_ab.aggregate_reduction_summaries(summaries)

        self.assertEqual(aggregate["runs"], 2)
        self.assertEqual(aggregate["best_by_average_reduction"], "b")
        self.assertAlmostEqual(aggregate["configs"]["a"]["average_reduction"], 0.05)
        self.assertAlmostEqual(aggregate["configs"]["b"]["average_reduction"], 0.075)
        self.assertEqual(aggregate["configs"]["a"]["winner_count"], 1)
        self.assertEqual(aggregate["configs"]["b"]["winner_count"], 1)
        self.assertEqual(aggregate["configs"]["a"]["all_positive_count"], 1)
        self.assertEqual(aggregate["configs"]["b"]["all_positive_count"], 1)
        self.assertEqual(aggregate["configs"]["a"]["negative_metric_counts"], {"full": 1})
        self.assertEqual(aggregate["configs"]["a"]["negative_reduction_count"], 1)
        self.assertEqual(aggregate["configs"]["b"]["negative_reduction_count"], 0)
        self.assertAlmostEqual(aggregate["configs"]["a"]["average_regret"], 0.075)
        self.assertAlmostEqual(aggregate["configs"]["b"]["average_regret"], 0.05)
        self.assertEqual(aggregate["best_by_average_regret"], "b")

    def test_aggregate_best_average_ignores_partial_configs(self):
        summaries = [
            {
                "pairwise_distance_reduction": {"a": 0.2, "b": 0.1, "partial": 0.9},
                "reduction_summary": {
                    "best_by_comparison_metric": "partial",
                    "configs": {
                        "a": {"all_positive": True, "negative_metrics": []},
                        "b": {"all_positive": True, "negative_metrics": []},
                        "partial": {"all_positive": True, "negative_metrics": []},
                    },
                },
            },
            {
                "pairwise_distance_reduction": {"a": 0.1, "b": 0.3},
                "reduction_summary": {
                    "best_by_comparison_metric": "b",
                    "configs": {
                        "a": {"all_positive": True, "negative_metrics": []},
                        "b": {"all_positive": True, "negative_metrics": []},
                    },
                },
            },
        ]

        aggregate = live_drift_ab.aggregate_reduction_summaries(summaries)

        self.assertEqual(aggregate["best_by_average_reduction"], "b")
        self.assertEqual(aggregate["best_by_available_average_reduction"], "partial")
        self.assertFalse(aggregate["configs"]["partial"]["complete"])

    def test_extract_summary_accepts_full_result_or_summary(self):
        summary = {"pairwise_distance_reduction": {"a": 0.1}}

        self.assertIs(live_drift_ab.extract_summary({"summary": summary}), summary)
        self.assertIs(live_drift_ab.extract_summary(summary), summary)

    def test_write_json_result_creates_parent_directory(self):
        result = {
            "summary": {
                "pairwise_distance_reduction": {"a": 0.1},
                "reduction_summary": {
                    "best_by_comparison_metric": "a",
                    "configs": {"a": {"all_positive": True, "negative_metrics": []}},
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "result.json")

            live_drift_ab.write_json_result(path, result)
            aggregate = live_drift_ab._aggregate_files([path])

        self.assertEqual(aggregate["runs"], 1)
        self.assertEqual(aggregate["best_by_average_reduction"], "a")


if __name__ == "__main__":
    unittest.main()
