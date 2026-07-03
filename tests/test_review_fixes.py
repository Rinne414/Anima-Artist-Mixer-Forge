"""Regression tests for the 2026-07-04 review findings (parsing layer).

Covers three confirmed parsing bugs:
  B1 - `weight::name@a,b,c::` split a phantom artist off the last route index.
  B2 - an out-of-range `@route` silently resolved to "all layers".
  B3 - a full-width CJK comma broke `expand_prompt_weights`.
"""
import unittest

from anima_mixer.parsing import (
    expand_prompt_weights,
    parse_artist_layer_routes,
    parse_artist_weights,
    parse_layer_filter,
    resolve_artist_layer_routes,
    split_artist_chain,
)


class TestB1PhantomArtistSplit(unittest.TestCase):
    """`1.2::wlop@0,2,4::` must stay one artist with route {0,2,4}."""

    def test_trailing_boundary_does_not_split_comma_route(self):
        parts = split_artist_chain("1.2::wlop@0,2,4::")
        assert parts == ["1.2::wlop@0,2,4::"]

    def test_full_chain_weight_route_boundary(self):
        parts = split_artist_chain("1.2::wlop@0,2,4::")
        names, weights, has_explicit = parse_artist_weights(parts)
        assert names == ["wlop@0,2,4"]
        assert weights == [1.2]
        assert has_explicit is True
        clean, routes = parse_artist_layer_routes(names)
        assert clean == ["wlop"]
        resolved, has_routes = resolve_artist_layer_routes(routes, 28)
        assert has_routes is True
        assert resolved == [{0, 2, 4}]

    def test_no_phantom_artist_created(self):
        parts = split_artist_chain("1.2::wlop@0,2,4::")
        names, _, _ = parse_artist_weights(parts)
        assert "4::" not in names
        assert len(names) == 1

    def test_route_without_trailing_boundary_still_works(self):
        # This form already worked; guard against regression.
        parts = split_artist_chain("1.2::wlop@0,2,4")
        names, weights, _ = parse_artist_weights(parts)
        clean, routes = parse_artist_layer_routes(names)
        assert clean == ["wlop"]
        resolved, _ = resolve_artist_layer_routes(routes, 28)
        assert resolved == [{0, 2, 4}]

    def test_comma_separated_artists_still_split(self):
        # A postfix-weighted second artist must not be swallowed as a route.
        assert split_artist_chain("wlop, sakimichan::1.5") == [
            "wlop",
            "sakimichan::1.5",
        ]

    def test_comma_route_then_new_artist_splits(self):
        parts = split_artist_chain("wlop@0,2, krenz")
        assert parts == ["wlop@0,2", "krenz"]


class TestB2OutOfRangeRoute(unittest.TestCase):
    """A specified but out-of-range route resolves to no blocks, never all-layers.

    A fully out-of-range route returns [] (explicitly "no blocks" + warning)
    rather than the earlier clamp-to-nearest-block behavior, so an artist the
    user routed to a non-existent layer is silent instead of applied to a
    surprise block or to every layer.
    """

    def test_range_fully_above_returns_no_blocks(self):
        warnings = []
        assert parse_layer_filter("50-60", 28, warnings) == []
        assert any("matches no blocks" in m for m in warnings)

    def test_range_partially_above_keeps_valid_subrange(self):
        assert parse_layer_filter("20-100", 28) == list(range(20, 28))

    def test_single_index_above_returns_no_blocks(self):
        warnings = []
        assert parse_layer_filter("100", 28, warnings) == []
        assert any("matches no blocks" in m for m in warnings)

    def test_negative_last_block_still_works(self):
        assert parse_layer_filter("-1", 28) == [27]

    def test_in_range_unchanged(self):
        assert parse_layer_filter("0-8", 28) == list(range(0, 9))

    def test_unparseable_still_none(self):
        assert parse_layer_filter("abc", 28) is None

    def test_empty_still_none(self):
        assert parse_layer_filter("", 28) is None


class TestB3FullWidthCommaExpansion(unittest.TestCase):
    """Full-width comma before a weight prefix must not break expansion."""

    def test_fullwidth_comma_before_weight(self):
        assert (
            expand_prompt_weights("1girl，1.5::masterpiece::")
            == "1girl，(masterpiece:1.5)"
        )

    def test_ascii_comma_still_works(self):
        # Pre-existing behavior drops the boundary whitespace before the
        # weight; only assert the expansion itself is correct.
        assert (
            expand_prompt_weights("masterpiece, 1.5::high quality::, ok")
            == "masterpiece,(high quality:1.5), ok"
        )

    def test_fullwidth_comma_inside_target_preserved(self):
        assert (
            expand_prompt_weights("1.3::detailed，intricate::, 1girl")
            == "(detailed，intricate:1.3), 1girl"
        )


if __name__ == "__main__":
    unittest.main()
