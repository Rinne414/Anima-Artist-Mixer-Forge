"""Tests for the bundled Danbooru tag vocabulary lookup (v27.3).

unittest.TestCase style so both pytest and unittest discovery collect them.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from anima_mixer import tag_vocab  # noqa: E402


class NormalizeTagTests(unittest.TestCase):
    def test_at_prefix_and_escapes(self):
        self.assertEqual(tag_vocab.normalize_tag("@uof"), "uof")
        self.assertEqual(
            tag_vocab.normalize_tag("@yuchi \\(salmon-1000\\)"),
            "yuchi_(salmon-1000)",
        )

    def test_spaces_fullwidth_and_case(self):
        self.assertEqual(tag_vocab.normalize_tag("  Hatsune  Miku "), "hatsune_miku")
        self.assertEqual(tag_vocab.normalize_tag("yuchi （salmon-1000）"), "yuchi_(salmon-1000)")

    def test_empty(self):
        self.assertEqual(tag_vocab.normalize_tag(""), "")
        self.assertEqual(tag_vocab.normalize_tag(None), "")


class LookupWithInjectedVocabTests(unittest.TestCase):
    def setUp(self):
        tag_vocab._VOCAB_CACHE = (
            {
                "wlop": (1, 365),
                "hatsune_miku": (4, 106634),
                "watercolor_(medium)": (0, 5000),
            },
            {"wlop_alias": "wlop", "ghost_alias": "not_a_real_tag"},
        )

    def tearDown(self):
        tag_vocab._VOCAB_CACHE = None

    def test_artist_hit(self):
        res = tag_vocab.lookup("@wlop")
        self.assertEqual(res["status"], "artist")
        self.assertEqual(res["count"], 365)

    def test_other_category(self):
        res = tag_vocab.lookup("Hatsune Miku")
        self.assertEqual(res["status"], "other_category")
        self.assertEqual(res["category"], 4)

    def test_alias_resolves_to_canonical(self):
        res = tag_vocab.lookup("wlop_alias")
        self.assertEqual(res["status"], "alias")
        self.assertEqual(res["canonical"], "wlop")
        self.assertEqual(res["count"], 365)

    def test_alias_to_missing_canonical_is_not_found(self):
        self.assertEqual(tag_vocab.lookup("ghost_alias")["status"], "not_found")

    def test_not_found(self):
        self.assertEqual(tag_vocab.lookup("zzqqx9999")["status"], "not_found")

    def test_literal_at_prefixed_tag_found_via_fallback(self):
        # A handful of real Danbooru tags start with '@'; the Anima marker
        # strip must not hide them (review finding, 2026-07-05).
        tag_vocab._VOCAB_CACHE[0]["@shun"] = (1, 90)
        res = tag_vocab.lookup("@shun")
        self.assertEqual(res["status"], "artist")
        self.assertEqual(res["canonical"], "@shun")

    def test_describe_lines(self):
        self.assertIn("known artist tag", tag_vocab.describe("wlop"))
        self.assertIn("365", tag_vocab.describe("wlop"))
        self.assertIn("character tag", tag_vocab.describe("hatsune_miku"))
        self.assertIn("not an artist", tag_vocab.describe("hatsune_miku"))
        self.assertIn("alias of 'wlop'", tag_vocab.describe("wlop_alias"))
        self.assertIn("not in the bundled", tag_vocab.describe("zzqqx9999"))

    def test_low_count_artist_notes_weak_prior(self):
        tag_vocab._VOCAB_CACHE[0]["tiny_artist"] = (1, 12)
        self.assertIn("low post count", tag_vocab.describe("tiny_artist"))

    def test_report_lines_one_per_name(self):
        lines = tag_vocab.report_lines(["@wlop", "zzqqx9999"])
        self.assertEqual(len(lines), 3)  # header + 2 entries
        self.assertIn("danbooru", lines[0].lower())
        self.assertIn("@wlop", lines[1])
        self.assertIn("zzqqx9999", lines[2])


class SuggestionTests(unittest.TestCase):
    def setUp(self):
        tag_vocab._VOCAB_CACHE = (
            {
                "yuchi_(salmon-1000)": (1, 208),
                "sakimichan": (1, 1014),
                "sakimori": (1, 30),
                "hatsune_miku": (4, 106634),  # non-artist: never suggested
            },
            {},
        )

    def tearDown(self):
        tag_vocab._VOCAB_CACHE = None

    def test_missing_disambiguator_suggested(self):
        self.assertIn("yuchi_(salmon-1000)", tag_vocab.suggest_artists("yuchi"))

    def test_typo_fuzzy_suggested(self):
        self.assertIn("sakimichan", tag_vocab.suggest_artists("sakimichann"))

    def test_gibberish_gets_no_suggestions(self):
        self.assertEqual(tag_vocab.suggest_artists("zzqqxx9999"), [])

    def test_non_artist_categories_never_suggested(self):
        self.assertNotIn("hatsune_miku", tag_vocab.suggest_artists("hatsune_mik"))

    def test_describe_not_found_includes_did_you_mean(self):
        line = tag_vocab.describe("@yuchi")
        self.assertIn("not in the bundled", line)
        self.assertIn("did you mean", line)
        self.assertIn("yuchi_(salmon-1000)", line)

    def test_describe_not_found_without_candidates_stays_clean(self):
        line = tag_vocab.describe("zzqqxx9999")
        self.assertIn("not in the bundled", line)
        self.assertNotIn("did you mean", line)


class UnavailableVocabTests(unittest.TestCase):
    def setUp(self):
        tag_vocab._VOCAB_CACHE = False  # load previously failed

    def tearDown(self):
        tag_vocab._VOCAB_CACHE = None

    def test_lookup_reports_unavailable(self):
        self.assertEqual(tag_vocab.lookup("wlop")["status"], "unavailable")

    def test_report_lines_collapse_to_single_note(self):
        lines = tag_vocab.report_lines(["a", "b"])
        self.assertEqual(len(lines), 1)
        self.assertIn("unavailable", lines[0])


class BundledFileSmokeTests(unittest.TestCase):
    """Exercises the real gz shipped in anima_mixer/data/."""

    def setUp(self):
        tag_vocab._VOCAB_CACHE = None

    tearDown = setUp

    def test_bundled_file_loads_and_knows_the_world(self):
        tags, aliases = tag_vocab.load_vocab()
        self.assertIsNotNone(tags)
        self.assertGreater(len(tags), 100_000)
        self.assertGreater(len(aliases), 10_000)
        self.assertEqual(tag_vocab.lookup("wlop")["status"], "artist")
        self.assertEqual(tag_vocab.lookup("hatsune_miku")["category"], 4)
        self.assertEqual(tag_vocab.lookup("zzqqxnotanartist9999")["status"], "not_found")
        # '#'-prefixed tags must survive the provenance-header filter and
        # '@'-prefixed tags must be reachable via the literal fallback
        # (review findings, 2026-07-05).
        self.assertEqual(tag_vocab.lookup("#b7282e")["status"], "artist")
        self.assertEqual(tag_vocab.lookup("@shun")["status"], "artist")


class NodeIntegrationTests(unittest.TestCase):
    def setUp(self):
        tag_vocab._VOCAB_CACHE = (
            {"wlop": (1, 365)},
            {},
        )

    def tearDown(self):
        tag_vocab._VOCAB_CACHE = None

    def test_chain_preview_report_gains_vocab_section(self):
        from anima_mixer.nodes_ui import AnimaArtistChainPreview

        out = AnimaArtistChainPreview().preview("wlop, zzqqx9999")
        report = out["result"][1]
        self.assertIn("danbooru", report.lower())
        self.assertIn("known artist tag", report)
        self.assertIn("not in the bundled", report)

    def test_tagcheck_report_gains_vocab_section(self):
        import torch

        from anima_mixer.nodes_diagnostics import AnimaArtistTagCheck

        def cond(vec):
            return [[vec.reshape(1, 1, -1).repeat(1, 4, 1), {}]]

        base = torch.zeros(8)
        base[0] = 1.0
        artist = torch.zeros(8)
        artist[1] = 1.0
        pack = {
            "conditionings": [cond(artist)],
            "labels": ["wlop"],
            "weights": [1.0],
            "base_conditioning": cond(base),
        }
        report = AnimaArtistTagCheck().check(pack)["result"][0]
        self.assertIn("known artist tag", report)


if __name__ == "__main__":
    unittest.main()
