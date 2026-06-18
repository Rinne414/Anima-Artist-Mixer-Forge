"""Tensor-math and runtime-helper tests (require a real torch install)."""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from anima_mixer import math_utils, patching  # noqa: E402
from anima_mixer.anchor import _context_fingerprint, make_sigma_capture, maybe_run_anchor  # noqa: E402
from anima_mixer.wrapper import CrossAttnWrapper, _combine_concat  # noqa: E402


class LowRankTest(unittest.TestCase):
    def test_k_at_least_n_is_identity(self):
        d = torch.randn(3, 16)
        out = math_utils.lowrank_rows_deterministic(d, 3)
        self.assertTrue(torch.equal(out, d))

    def test_projection_reduces_rank(self):
        torch.manual_seed(0)
        d = torch.randn(4, 32)
        out = math_utils.lowrank_rows_deterministic(d, 1)
        # Rank-1 reconstruction: every row is a multiple of the same vector.
        rank = torch.linalg.matrix_rank(out, tol=1e-4).item()
        self.assertEqual(rank, 1)

    def test_projection_is_deterministic(self):
        torch.manual_seed(1)
        d = torch.randn(5, 64)
        a = math_utils.lowrank_rows_deterministic(d, 2)
        b = math_utils.lowrank_rows_deterministic(d.clone(), 2)
        self.assertTrue(torch.allclose(a, b))

    def test_projection_preserves_rowspace_energy_ordering(self):
        torch.manual_seed(2)
        base = torch.randn(1, 16)
        # Rows mostly aligned with one direction plus small noise.
        d = base.repeat(4, 1) + 0.01 * torch.randn(4, 16)
        out = math_utils.lowrank_rows_deterministic(d, 1)
        # The rank-1 reconstruction should be very close to the input.
        self.assertLess((out - d).norm().item() / d.norm().item(), 0.05)


class ProjectPerpendicularTest(unittest.TestCase):
    def test_result_is_orthogonal_to_base_per_token(self):
        torch.manual_seed(3)
        base = torch.randn(2, 5, 8)
        delta = torch.randn(2, 5, 8)
        perp = math_utils.project_perpendicular(delta, base)
        dots = (perp * base).sum(dim=-1)
        self.assertTrue(torch.allclose(dots, torch.zeros_like(dots), atol=1e-5))

    def test_parallel_delta_vanishes(self):
        base = torch.randn(1, 3, 4)
        delta = 2.5 * base
        perp = math_utils.project_perpendicular(delta, base)
        self.assertTrue(torch.allclose(perp, torch.zeros_like(perp), atol=1e-5))


class TimingFadeTest(unittest.TestCase):
    # Window: sigma hi=10 (start), fade_in_lo=8, fade_out_hi=0.5, lo=0.1 (end).
    ROUTE = (0.1, 10.0, 8.0, 0.5)

    def test_outside_window_is_zero(self):
        self.assertEqual(math_utils.timing_fade_factor(self.ROUTE, 11.0), 0.0)
        self.assertEqual(math_utils.timing_fade_factor(self.ROUTE, 0.05), 0.0)

    def test_plateau_is_one(self):
        self.assertEqual(math_utils.timing_fade_factor(self.ROUTE, 5.0), 1.0)

    def test_fade_in_midpoint_is_half(self):
        # sigma 9.0 is halfway between hi=10 and fade_in_lo=8 -> smoothstep(0.5)=0.5
        self.assertAlmostEqual(
            math_utils.timing_fade_factor(self.ROUTE, 9.0), 0.5, places=6,
        )

    def test_fade_out_midpoint_is_half(self):
        # sigma 0.3 is halfway between fade_out_hi=0.5 and lo=0.1
        self.assertAlmostEqual(
            math_utils.timing_fade_factor(self.ROUTE, 0.3), 0.5, places=6,
        )

    def test_no_fade_route_is_binary(self):
        route = (0.1, 10.0, 10.0, 0.1)  # fade edges collapse onto the window
        self.assertEqual(math_utils.timing_fade_factor(route, 10.0), 1.0)
        self.assertEqual(math_utils.timing_fade_factor(route, 0.1), 1.0)
        self.assertEqual(math_utils.timing_fade_factor(route, 10.1), 0.0)

    def test_none_route_or_sigma_is_one(self):
        self.assertEqual(math_utils.timing_fade_factor(None, 5.0), 1.0)
        self.assertEqual(math_utils.timing_fade_factor(self.ROUTE, None), 1.0)


class ResolveMaskTest(unittest.TestCase):
    def test_exact_length_markers(self):
        mask = patching.resolve_mask([0, 1], 2, False, {})
        self.assertEqual(mask, [True, False])

    def test_chunk_expansion_for_batched_latents(self):
        # 2 latents per cond entry: rows [cond, cond, uncond, uncond].
        state = {}
        mask = patching.resolve_mask([0, 1], 4, False, state)
        self.assertEqual(mask, [True, True, False, False])
        self.assertNotIn("_warned", state)

    def test_apply_to_uncond_injects_everywhere(self):
        mask = patching.resolve_mask([0, 1], 4, True, {})
        self.assertEqual(mask, [True] * 4)

    def test_unusable_markers_fall_back_with_warning(self):
        state = {}
        mask = patching.resolve_mask([0, 1], 3, False, state)
        self.assertEqual(mask, [True] * 3)
        self.assertTrue(state.get("_warned"))

    def test_missing_markers_fall_back_with_warning(self):
        state = {}
        mask = patching.resolve_mask(None, 2, False, state)
        self.assertEqual(mask, [True] * 2)
        self.assertTrue(state.get("_warned"))


class BroadcastBatchTest(unittest.TestCase):
    def test_expand_single(self):
        t = torch.randn(1, 4, 8)
        out = patching.broadcast_batch(t, 3)
        self.assertEqual(tuple(out.shape), (3, 4, 8))
        self.assertTrue(torch.equal(out[0], out[2]))

    def test_repeat_divisible(self):
        t = torch.randn(2, 4, 8)
        out = patching.broadcast_batch(t, 4)
        self.assertEqual(tuple(out.shape), (4, 4, 8))
        self.assertTrue(torch.equal(out[0], out[2]))


class _KVMeanAttn(torch.nn.Module):
    """Stub cross-attention: returns the per-batch mean of the K/V tokens,
    broadcast to the query's token count. Lets fusion math be tested without
    a real attention module."""

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        mean = context.mean(dim=1, keepdim=True)
        return mean.expand(x.shape[0], x.shape[1], context.shape[-1])


class WrapperHelpersTest(unittest.TestCase):
    def _wrapper(self, state=None):
        return CrossAttnWrapper(torch.nn.Identity(), state or {}, 0)

    def test_combine_concat_scales_and_concatenates(self):
        a = torch.ones(1, 2, 4)
        b = torch.ones(1, 3, 4)
        out = _combine_concat([a, b], [0.5, 2.0])
        self.assertEqual(tuple(out.shape), (1, 5, 4))
        self.assertTrue(torch.allclose(out[0, 0], torch.full((4,), 0.5)))
        self.assertTrue(torch.allclose(out[0, 2], torch.full((4,), 2.0)))

    def test_effective_weights_normalize_then_fade(self):
        # Normalization must run BEFORE the fade multiplies in, otherwise a
        # single fading artist would renormalize back to 1.0 (fade no-op).
        w = self._wrapper({"normalize_weights": True})
        ws, comp = w._effective_weights([1.0], [0.5])
        self.assertEqual(ws, [0.5])
        self.assertAlmostEqual(comp, 0.5)

    def test_effective_weights_no_fade_no_compensation(self):
        w = self._wrapper({"normalize_weights": True})
        ws, comp = w._effective_weights([2.0, 2.0], [1.0, 1.0])
        self.assertEqual(ws, [0.5, 0.5])
        self.assertAlmostEqual(comp, 0.0)

    def test_output_avg_fade_blends_toward_base(self):
        # Single artist at fade 0.5 with normalize on: the output must be the
        # midpoint between the artist attention output and the base output,
        # not a renormalized full-strength artist. match_base_norm is off so
        # the zero-valued stub base output does not rescale the assertion.
        state = {
            "normalize_weights": True, "apply_to_uncond": False,
            "match_base_norm": False,
        }
        w = CrossAttnWrapper(_KVMeanAttn(), state, 0)
        x = torch.zeros(1, 2, 4)
        base_ctx = torch.zeros(1, 3, 4)      # base attention output -> 0
        artist = torch.ones(1, 3, 4)         # artist attention output -> 1
        out = w._fwd_output_avg(
            x, base_ctx, None, {}, [artist], [1.0], [0.5],
            [True], "interpolate", 1.0,
        )
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.5)))

    def test_dispatch_zero_fade_does_not_renormalize_remaining_artist(self):
        state = {
            "individuals": [
                torch.full((1, 3, 1), 14.0),
                torch.full((1, 3, 1), 20.0),
            ],
            "real_lens": [3, 3],
            "combine_mode": "output_avg",
            "fusion_mode": "interpolate",
            "strength": 1.0,
            "user_weights": [1.0, 1.0],
            "normalize_weights": True,
            "apply_to_uncond": False,
            "match_base_norm": False,
            "has_artist_timing_routes": True,
            "artist_layer_routes": [None, None],
            "artist_timing_routes": [(0.1, 10.0, 8.0, 0.5), None],
            "current_sigma": 11.0,
        }
        w = CrossAttnWrapper(_KVMeanAttn(), state, 0)
        x = torch.zeros(1, 2, 1)
        base_ctx = torch.full((1, 3, 1), 10.0)

        out = w._dispatch(x, base_ctx, None, {"cond_or_uncond": [0]})

        self.assertTrue(torch.allclose(out, torch.full_like(out, 15.0)))

    def test_output_avg_explicit_weight_scales_artist_delta(self):
        state = {
            "normalize_weights": False, "apply_to_uncond": False,
            "match_base_norm": False,
            "has_explicit_weights": True,
        }
        w = CrossAttnWrapper(_KVMeanAttn(), state, 0)
        x = torch.zeros(1, 2, 1)
        base_ctx = torch.full((1, 3, 1), 10.0)
        artist = torch.full((1, 3, 1), 14.0)

        out = w._fwd_output_avg(
            x, base_ctx, None, {}, [artist], [0.25], [1.0],
            [True], "interpolate", 1.0,
        )

        self.assertTrue(torch.allclose(out, torch.full_like(out, 11.0)))

    def test_output_avg_negative_weight_subtracts_artist_delta(self):
        state = {
            "normalize_weights": False, "apply_to_uncond": False,
            "match_base_norm": False,
            "has_explicit_weights": True,
        }
        w = CrossAttnWrapper(_KVMeanAttn(), state, 0)
        x = torch.zeros(1, 2, 1)
        base_ctx = torch.full((1, 3, 1), 10.0)
        artist = torch.full((1, 3, 1), 14.0)

        out = w._fwd_output_avg(
            x, base_ctx, None, {}, [artist], [-0.5], [1.0],
            [True], "interpolate", 1.0,
        )

        self.assertTrue(torch.allclose(out, torch.full_like(out, 8.0)))

    def test_artist_chunks_split_by_limit(self):
        w = self._wrapper({"max_batch_artists": 2})
        items = [torch.zeros(1)] * 5
        chunks = w._artist_chunks(items)
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])

    def test_artist_chunks_no_limit(self):
        w = self._wrapper({"max_batch_artists": 0})
        items = [torch.zeros(1)] * 5
        chunks = w._artist_chunks(items)
        self.assertEqual([len(c) for c in chunks], [5])

    def test_apply_fusion_interpolate_respects_mask(self):
        w = self._wrapper()
        base = torch.zeros(2, 3, 4)
        artist = torch.ones(2, 3, 4)
        out = w._apply_fusion(base, artist, [True, False], "interpolate", 0.5)
        self.assertTrue(torch.allclose(out[0], torch.full((3, 4), 0.5)))
        self.assertTrue(torch.allclose(out[1], torch.zeros(3, 4)))

    def test_apply_fusion_base_preserve_keeps_base_direction(self):
        w = self._wrapper()
        base = torch.zeros(1, 2, 4)
        base[..., 0] = 1.0  # base points along e0
        artist = torch.zeros(1, 2, 4)
        artist[..., 0] = 3.0  # parallel to base: should be stripped entirely
        artist[..., 1] = 2.0  # perpendicular: should survive
        out = w._apply_fusion(base, artist, [True], "base_preserve", 1.0)
        self.assertTrue(torch.allclose(out[..., 0], base[..., 0]))
        self.assertTrue(torch.allclose(out[..., 1], torch.full((1, 2), 2.0)))


class MatchBaseNormTest(unittest.TestCase):
    def _wrapper(self, state=None):
        return CrossAttnWrapper(torch.nn.Identity(), state or {}, 0)

    def test_rescales_artist_to_base_rms(self):
        w = self._wrapper()
        base = torch.full((1, 4, 8), 2.0)     # RMS 2
        artist = torch.full((1, 4, 8), 1.0)   # RMS 1
        out = w._match_base_norm(artist, base, [True])
        self.assertTrue(torch.allclose(out, torch.full_like(out, 2.0)))

    def test_token_norm_lock_matches_each_token_rms_by_default(self):
        w = self._wrapper()
        base = torch.tensor([[[1.0, 0.0], [0.0, 4.0]]])
        artist = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])

        out = w._match_base_norm(artist, base, [True])

        self.assertTrue(torch.allclose(out, base))

    def test_row_norm_lock_keeps_legacy_single_row_scale(self):
        w = self._wrapper({"norm_lock_mode": "row"})
        base = torch.tensor([[[1.0, 0.0], [0.0, 4.0]]])
        artist = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])

        out = w._match_base_norm(artist, base, [True])

        self.assertFalse(torch.allclose(out, base))
        self.assertAlmostEqual(
            out.pow(2).mean().sqrt().item(),
            base.pow(2).mean().sqrt().item(),
            places=6,
        )

    def test_scale_is_clamped(self):
        w = self._wrapper()
        base = torch.full((1, 4, 8), 10.0)    # would need 10x
        artist = torch.full((1, 4, 8), 1.0)
        out = w._match_base_norm(artist, base, [True])
        self.assertTrue(torch.allclose(out, torch.full_like(out, 2.0)))  # 2x cap
        base = torch.full((1, 4, 8), 0.1)     # would need 0.1x
        out = w._match_base_norm(artist, base, [True])
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.5)))  # 0.5x floor

    def test_unmasked_rows_untouched(self):
        w = self._wrapper()
        base = torch.full((2, 4, 8), 2.0)
        artist = torch.full((2, 4, 8), 1.0)
        out = w._match_base_norm(artist, base, [True, False])
        self.assertTrue(torch.allclose(out[0], torch.full((4, 8), 2.0)))
        self.assertTrue(torch.allclose(out[1], torch.full((4, 8), 1.0)))

    def test_anchor_base_norm_reference_overrides_current_seed_base(self):
        w = self._wrapper({
            "anchor_base_norm_ref": True,
            "_anchor_base_cache": {0: torch.full((1, 4, 8), 4.0)},
        })
        current_seed_base = torch.full((1, 4, 8), 2.0)
        artist = torch.full((1, 4, 8), 2.0)

        out = w._match_base_norm(artist, current_seed_base, [True])

        self.assertTrue(torch.allclose(out, torch.full_like(out, 4.0)))

    def test_anchor_base_norm_reference_falls_back_on_shape_mismatch(self):
        w = self._wrapper({
            "anchor_base_norm_ref": True,
            "_anchor_base_cache": {0: torch.full((1, 5, 8), 4.0)},
        })
        current_seed_base = torch.full((1, 4, 8), 2.0)
        artist = torch.full((1, 4, 8), 2.0)

        out = w._match_base_norm(artist, current_seed_base, [True])

        self.assertTrue(torch.allclose(out, torch.full_like(out, 2.0)))

    def test_preserves_direction(self):
        torch.manual_seed(7)
        w = self._wrapper()
        base = torch.randn(1, 4, 8)
        artist = torch.randn(1, 4, 8)
        out = w._match_base_norm(artist, base, [True])
        cos = torch.nn.functional.cosine_similarity(
            out, artist, dim=-1,
        )
        self.assertTrue(torch.all(cos > 0.9999))

    def test_per_artist_norm_balances_artist_energy_before_mix(self):
        class TwoArtistMeanAttn(torch.nn.Module):
            def forward(self, x, context=None, rope_emb=None, transformer_options=None):
                mean = context.mean(dim=1, keepdim=True)
                return mean.expand(x.shape[0], x.shape[1], context.shape[-1])

        state = {
            "normalize_weights": True,
            "apply_to_uncond": False,
            "match_base_norm": True,
            "norm_lock_mode": "token",
            "norm_lock_scope": "per_artist",
        }
        w = CrossAttnWrapper(TwoArtistMeanAttn(), state, 0)
        x = torch.zeros(1, 2, 2)
        base_ctx = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
        strong_x = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
        strong_y = torch.tensor([[[0.0, 10.0], [0.0, 10.0]]])

        out = w._fwd_output_avg(
            x, base_ctx, None, {},
            [strong_x, strong_y], [1.0, 1.0], [1.0, 1.0],
            [True], "interpolate", 1.0,
        )

        expected_rms = torch.full((1, 2), 2 ** -0.5)
        self.assertTrue(torch.allclose(out.pow(2).mean(dim=-1).sqrt(), expected_rms, atol=1e-5))

    def test_contribution_balance_equalizes_artist_delta_strengths(self):
        state = {
            "contribution_balance": True,
            "contribution_balance_alpha": 1.0,
        }
        w = self._wrapper(state)
        base = torch.zeros(1, 2, 2)
        weak = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        strong = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])

        out_weak, out_strong = w._balance_artist_deltas(
            [weak, strong], base, [0.5, 0.5], [True],
        )

        weak_rms = (out_weak - base).pow(2).mean(dim=-1).sqrt()
        strong_rms = (out_strong - base).pow(2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(weak_rms, strong_rms, atol=1e-6))
        self.assertTrue(torch.all(out_weak[..., 0] > weak[..., 0]))
        self.assertTrue(torch.all(out_strong[..., 1] < strong[..., 1]))

    def test_contribution_balance_keeps_weight_ratio_in_final_mix(self):
        w = self._wrapper({
            "contribution_balance": True,
            "contribution_balance_alpha": 1.0,
        })
        base = torch.zeros(1, 1, 2)
        a = torch.tensor([[[1.0, 0.0]]])
        b = torch.tensor([[[0.0, 2.0]]])

        out_a, out_b = w._balance_artist_deltas(
            [a, b], base, [1.0, 3.0], [True],
        )

        a_rms = (out_a - base).pow(2).mean(dim=-1).sqrt()
        b_rms = (out_b - base).pow(2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(b_rms, a_rms, atol=1e-6))

        mixed_delta = (out_a - base) * 0.25 + (out_b - base) * 0.75
        weighted_a = ((out_a - base) * 0.25).pow(2).mean(dim=-1).sqrt()
        weighted_b = ((out_b - base) * 0.75).pow(2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(weighted_b, weighted_a * 3.0, atol=1e-6))
        self.assertTrue(torch.allclose(mixed_delta, torch.tensor([[[0.3750, 1.1250]]])))

    def test_contribution_balance_alpha_can_blend_or_disable(self):
        base = torch.zeros(1, 1, 2)
        weak = torch.tensor([[[1.0, 0.0]]])
        strong = torch.tensor([[[0.0, 9.0]]])

        disabled = self._wrapper({"contribution_balance": False})
        out = disabled._balance_artist_deltas([weak, strong], base, [1.0, 1.0], [True])
        self.assertTrue(torch.equal(out[0], weak))
        self.assertTrue(torch.equal(out[1], strong))

        partial = self._wrapper({
            "contribution_balance": True,
            "contribution_balance_alpha": 0.5,
        })
        out_weak, out_strong = partial._balance_artist_deltas(
            [weak, strong], base, [1.0, 1.0], [True],
        )
        weak_rms = (out_weak - base).pow(2).mean(dim=-1).sqrt()
        strong_rms = (out_strong - base).pow(2).mean(dim=-1).sqrt()

        self.assertGreater(weak_rms.item(), (weak - base).pow(2).mean(dim=-1).sqrt().item())
        self.assertLess(strong_rms.item(), (strong - base).pow(2).mean(dim=-1).sqrt().item())
        self.assertLess(weak_rms.item(), strong_rms.item())

    def test_contribution_balance_respects_unmasked_rows(self):
        w = self._wrapper({"contribution_balance": True})
        base = torch.zeros(2, 1, 2)
        weak = torch.zeros(2, 1, 2)
        weak[..., 0] = 1.0
        strong = torch.zeros(2, 1, 2)
        strong[..., 1] = 9.0

        out_weak, out_strong = w._balance_artist_deltas(
            [weak, strong], base, [1.0, 1.0], [True, False],
        )

        self.assertFalse(torch.equal(out_weak[0], weak[0]))
        self.assertFalse(torch.equal(out_strong[0], strong[0]))
        self.assertTrue(torch.equal(out_weak[1], weak[1]))
        self.assertTrue(torch.equal(out_strong[1], strong[1]))

    def test_contribution_balance_ignores_zero_weight_artist_strength(self):
        w = self._wrapper({"contribution_balance": True})
        base = torch.zeros(1, 1, 2)
        active = torch.tensor([[[1.0, 0.0]]])
        faded_out = torch.tensor([[[0.0, 9.0]]])

        out_active, out_faded = w._balance_artist_deltas(
            [active, faded_out], base, [1.0, 0.0], [True],
        )

        self.assertTrue(torch.equal(out_active, active))
        self.assertTrue(torch.equal(out_faded, faded_out))

    def test_output_avg_uses_contribution_balance_before_mixing(self):
        class ContextSelectAttn(torch.nn.Module):
            def forward(self, x, context=None, rope_emb=None, transformer_options=None):
                return context[:, :1, :].expand(x.shape[0], x.shape[1], context.shape[-1])

        state = {
            "normalize_weights": True,
            "apply_to_uncond": False,
            "match_base_norm": False,
            "contribution_balance": True,
            "contribution_balance_alpha": 1.0,
        }
        w = CrossAttnWrapper(ContextSelectAttn(), state, 0)
        x = torch.zeros(1, 1, 2)
        base_ctx = torch.zeros(1, 1, 2)
        weak = torch.tensor([[[1.0, 0.0]]])
        strong = torch.tensor([[[0.0, 2.0]]])

        out = w._fwd_output_avg(
            x, base_ctx, None, {},
            [weak, strong], [1.0, 1.0], [1.0, 1.0],
            [True], "interpolate", 1.0,
        )

        self.assertTrue(torch.allclose(out, torch.tensor([[[0.75, 0.75]]]), atol=1e-6))

    def test_static_capture_output_mode_freezes_artist_output(self):
        class BaseMovesArtistFrozenAttn(torch.nn.Module):
            def forward(self, x, context=None, rope_emb=None, transformer_options=None):
                if context.mean().item() < 5.0:
                    return x + 1.0
                return torch.full_like(x, 10.0)

        state = {
            "normalize_weights": True,
            "apply_to_uncond": False,
            "match_base_norm": False,
            "artist_static_capture": True,
            "static_capture_k": 1,
            "static_capture_mode": "output",
            "current_sigma": 10.0,
        }
        w = CrossAttnWrapper(BaseMovesArtistFrozenAttn(), state, 0)
        base_ctx = torch.zeros(1, 1, 2)
        artist_ctx = torch.full((1, 1, 2), 10.0)

        first = w._fwd_output_avg(
            torch.zeros(1, 1, 2), base_ctx, None, {},
            [artist_ctx], [1.0], [1.0], [True], "interpolate", 1.0,
        )
        state["current_sigma"] = 9.0
        second = w._fwd_output_avg(
            torch.full((1, 1, 2), 100.0), base_ctx, None, {},
            [artist_ctx], [1.0], [1.0], [True], "interpolate", 1.0,
        )

        self.assertTrue(torch.allclose(first, torch.full((1, 1, 2), 10.0)))
        self.assertTrue(torch.allclose(second, torch.full((1, 1, 2), 10.0)))

    def test_static_capture_delta_mode_preserves_current_base_motion(self):
        class BaseMovesArtistFrozenAttn(torch.nn.Module):
            def forward(self, x, context=None, rope_emb=None, transformer_options=None):
                if context.mean().item() < 5.0:
                    return x + 1.0
                return torch.full_like(x, 10.0)

        state = {
            "normalize_weights": True,
            "apply_to_uncond": False,
            "match_base_norm": False,
            "artist_static_capture": True,
            "static_capture_k": 1,
            "static_capture_mode": "delta",
            "current_sigma": 10.0,
        }
        w = CrossAttnWrapper(BaseMovesArtistFrozenAttn(), state, 0)
        base_ctx = torch.zeros(1, 1, 2)
        artist_ctx = torch.full((1, 1, 2), 10.0)

        first = w._fwd_output_avg(
            torch.zeros(1, 1, 2), base_ctx, None, {},
            [artist_ctx], [1.0], [1.0], [True], "interpolate", 1.0,
        )
        state["current_sigma"] = 9.0
        second = w._fwd_output_avg(
            torch.full((1, 1, 2), 100.0), base_ctx, None, {},
            [artist_ctx], [1.0], [1.0], [True], "interpolate", 1.0,
        )

        self.assertTrue(torch.allclose(first, torch.full((1, 1, 2), 10.0)))
        self.assertTrue(torch.allclose(second, torch.full((1, 1, 2), 110.0)))

    def test_static_capture_blend_mode_interpolates_output_and_delta_paths(self):
        class BaseMovesArtistFrozenAttn(torch.nn.Module):
            def forward(self, x, context=None, rope_emb=None, transformer_options=None):
                if context.mean().item() < 5.0:
                    return x + 1.0
                return torch.full_like(x, 10.0)

        state = {
            "normalize_weights": True,
            "apply_to_uncond": False,
            "match_base_norm": False,
            "artist_static_capture": True,
            "static_capture_k": 1,
            "static_capture_mode": "blend",
            "static_capture_blend_alpha": 0.25,
            "current_sigma": 10.0,
        }
        w = CrossAttnWrapper(BaseMovesArtistFrozenAttn(), state, 0)
        base_ctx = torch.zeros(1, 1, 2)
        artist_ctx = torch.full((1, 1, 2), 10.0)

        first = w._fwd_output_avg(
            torch.zeros(1, 1, 2), base_ctx, None, {},
            [artist_ctx], [1.0], [1.0], [True], "interpolate", 1.0,
        )
        state["current_sigma"] = 9.0
        second = w._fwd_output_avg(
            torch.full((1, 1, 2), 100.0), base_ctx, None, {},
            [artist_ctx], [1.0], [1.0], [True], "interpolate", 1.0,
        )

        self.assertTrue(torch.allclose(first, torch.full((1, 1, 2), 10.0)))
        self.assertTrue(torch.allclose(second, torch.full((1, 1, 2), 35.0)))

    def test_static_capture_blend_perp_filters_style_parallel_base_motion(self):
        class DirectionalBaseMovesAttn(torch.nn.Module):
            def forward(self, x, context=None, rope_emb=None, transformer_options=None):
                if context.mean().item() < 5.0:
                    return x + torch.tensor([[[1.0, 0.0]]], device=x.device, dtype=x.dtype)
                return torch.tensor([[[10.0, 0.0]]], device=x.device, dtype=x.dtype)

        state = {
            "normalize_weights": True,
            "apply_to_uncond": False,
            "match_base_norm": False,
            "artist_static_capture": True,
            "static_capture_k": 1,
            "static_capture_mode": "blend_perp",
            "static_capture_blend_alpha": 0.25,
            "current_sigma": 10.0,
        }
        w = CrossAttnWrapper(DirectionalBaseMovesAttn(), state, 0)
        base_ctx = torch.zeros(1, 1, 2)
        artist_ctx = torch.full((1, 1, 2), 10.0)

        first = w._fwd_output_avg(
            torch.zeros(1, 1, 2), base_ctx, None, {},
            [artist_ctx], [1.0], [1.0], [True], "interpolate", 1.0,
        )
        state["current_sigma"] = 9.0
        second = w._fwd_output_avg(
            torch.full((1, 1, 2), 100.0), base_ctx, None, {},
            [artist_ctx], [1.0], [1.0], [True], "interpolate", 1.0,
        )

        self.assertTrue(torch.allclose(first, torch.tensor([[[10.0, 0.0]]])))
        self.assertTrue(torch.allclose(second, torch.tensor([[[10.0, 25.0]]])))

    def test_contribution_balance_mixes_deltas_not_base_multiple(self):
        class ContextSelectAttn(torch.nn.Module):
            def forward(self, x, context=None, rope_emb=None, transformer_options=None):
                return context[:, :1, :].expand(x.shape[0], x.shape[1], context.shape[-1])

        state = {
            "normalize_weights": False,
            "apply_to_uncond": False,
            "match_base_norm": False,
            "contribution_balance": True,
            "contribution_balance_alpha": 1.0,
        }
        w = CrossAttnWrapper(ContextSelectAttn(), state, 0)
        x = torch.zeros(1, 1, 2)
        base_ctx = torch.tensor([[[10.0, 10.0]]])
        weak = torch.tensor([[[11.0, 10.0]]])
        strong = torch.tensor([[[10.0, 12.0]]])

        out = w._fwd_output_avg(
            x, base_ctx, None, {},
            [weak, strong], [1.0, 1.0], [1.0, 1.0],
            [True], "interpolate", 1.0,
        )

        self.assertTrue(torch.allclose(out, torch.tensor([[[11.5, 11.5]]]), atol=1e-6))

    def test_mixed_delta_cap_disabled_is_noop(self):
        w = self._wrapper({
            "mixed_delta_cap": False,
            "mixed_delta_cap_ratio": 0.5,
        })
        base = torch.ones(1, 1, 1)
        artist = torch.full((1, 1, 1), 9.0)

        out = w._cap_mixed_delta(
            artist, base, [True], "interpolate", strength=1.0,
        )

        self.assertTrue(torch.equal(out, artist))

    def test_mixed_delta_cap_limits_final_interpolate_delta(self):
        w = self._wrapper({
            "mixed_delta_cap": True,
            "mixed_delta_cap_ratio": 1.0,
        })
        base = torch.ones(1, 1, 1)
        artist = torch.full((1, 1, 1), 9.0)

        capped = w._cap_mixed_delta(
            artist, base, [True], "interpolate", strength=2.0,
        )
        out = w._apply_fusion(base, capped, [True], "interpolate", 2.0)

        self.assertTrue(torch.allclose(out, torch.full((1, 1, 1), 2.0), atol=1e-6))

    def test_mixed_delta_cap_respects_unmasked_rows(self):
        w = self._wrapper({
            "mixed_delta_cap": True,
            "mixed_delta_cap_ratio": 0.5,
        })
        base = torch.ones(2, 1, 1)
        artist = torch.full((2, 1, 1), 9.0)

        out = w._cap_mixed_delta(
            artist, base, [True, False], "interpolate", strength=1.0,
        )

        self.assertTrue(torch.allclose(out[0], torch.full((1, 1), 1.5), atol=1e-6))
        self.assertTrue(torch.equal(out[1], artist[1]))

    def test_mixed_delta_cap_measures_base_preserve_perpendicular_delta(self):
        w = self._wrapper({
            "mixed_delta_cap": True,
            "mixed_delta_cap_ratio": 1.0,
        })
        base = torch.tensor([[[1.0, 0.0]]])
        artist = torch.tensor([[[1.0, 8.0]]])

        capped = w._cap_mixed_delta(
            artist, base, [True], "base_preserve", strength=1.0,
        )
        out = w._apply_fusion(base, capped, [True], "base_preserve", 1.0)

        self.assertTrue(torch.allclose(out, torch.tensor([[[1.0, 1.0]]]), atol=1e-6))

    def test_output_avg_applies_mixed_delta_cap_before_fusion(self):
        class ContextSelectAttn(torch.nn.Module):
            def forward(self, x, context=None, rope_emb=None, transformer_options=None):
                return context[:, :1, :].expand(x.shape[0], x.shape[1], context.shape[-1])

        state = {
            "normalize_weights": True,
            "apply_to_uncond": False,
            "match_base_norm": False,
            "mixed_delta_cap": True,
            "mixed_delta_cap_ratio": 0.5,
        }
        w = CrossAttnWrapper(ContextSelectAttn(), state, 0)
        x = torch.zeros(1, 1, 1)
        base_ctx = torch.ones(1, 1, 1)
        artist = torch.full((1, 1, 1), 9.0)

        out = w._fwd_output_avg(
            x, base_ctx, None, {}, [artist], [1.0], [1.0],
            [True], "interpolate", 1.0,
        )

        self.assertTrue(torch.allclose(out, torch.full((1, 1, 1), 1.5), atol=1e-6))


class AnchorFingerprintTest(unittest.TestCase):
    def test_same_content_same_fingerprint(self):
        a = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
        b = a.clone()
        self.assertEqual(_context_fingerprint(a), _context_fingerprint(b))

    def test_different_content_different_fingerprint(self):
        a = torch.zeros(1, 8, 8)
        b = torch.ones(1, 8, 8)
        self.assertNotEqual(_context_fingerprint(a), _context_fingerprint(b))

    def test_none_is_none(self):
        self.assertIsNone(_context_fingerprint(None))

    def test_anchor_prerun_uses_patched_apply_model(self):
        class FakeDM:
            pass

        state = {
            "dm_ref": FakeDM(),
            "anchor_seeds_count": 1,
            "low_vram_cache": False,
            "_anchor_cache": {},
            "_anchor_failed": False,
        }
        user_x = torch.zeros(1, 2, 3)
        timestep = torch.tensor([1.0])
        context = torch.ones(1, 4, 3)
        calls = []

        def fake_apply_model(x, ts, **kwargs):
            calls.append((x.clone(), ts.clone(), kwargs))
            self.assertTrue(state["_in_anchor_run"])
            state["_anchor_cache"][0] = torch.full_like(x, 7.0)
            return x

        maybe_run_anchor(
            state,
            user_x,
            timestep,
            {"context": context, "transformer_options": {"cond_or_uncond": [0]}},
            apply_model=fake_apply_model,
        )

        self.assertEqual(len(calls), 1)
        self.assertFalse(state["_in_anchor_run"])
        self.assertFalse(state.get("_anchor_failed", False))
        self.assertIn(0, state["_anchor_cache"])
        self.assertTrue(torch.allclose(state["_anchor_cache"][0], torch.full_like(user_x, 7.0)))
        self.assertNotIn("cond_or_uncond", calls[0][2]["transformer_options"])
        self.assertIn("c_crossattn", calls[0][2])
        self.assertNotIn("context", calls[0][2])

    def test_anchor_prerun_accepts_comfy_c_crossattn(self):
        class FakeDM:
            pass

        state = {
            "dm_ref": FakeDM(),
            "anchor_seeds_count": 1,
            "low_vram_cache": False,
            "_anchor_cache": {},
            "_anchor_failed": False,
        }
        user_x = torch.zeros(2, 2, 3)
        timestep = torch.tensor([1.0, 1.0])
        c_crossattn = torch.stack([
            torch.full((4, 3), 3.0),
            torch.full((4, 3), 5.0),
        ])
        calls = []

        def fake_apply_model(x, ts, **kwargs):
            calls.append((x.clone(), ts.clone(), kwargs))
            state["_anchor_cache"][0] = torch.full_like(x, 11.0)
            return x

        maybe_run_anchor(
            state,
            user_x,
            timestep,
            {
                "c_crossattn": c_crossattn,
                "transformer_options": {"cond_or_uncond": [1, 0]},
            },
            apply_model=fake_apply_model,
        )

        self.assertEqual(len(calls), 1)
        self.assertIn(0, state["_anchor_cache"])
        used_context = calls[0][2]["c_crossattn"]
        self.assertEqual(tuple(used_context.shape), tuple(c_crossattn.shape))
        self.assertTrue(torch.allclose(used_context[0], c_crossattn[1]))
        self.assertTrue(torch.allclose(used_context[1], c_crossattn[1]))
        self.assertNotIn("context", calls[0][2])

    def test_anchor_prerun_averages_base_output_cache(self):
        class FakeDM:
            pass

        state = {
            "dm_ref": FakeDM(),
            "anchor_seeds_count": 2,
            "low_vram_cache": False,
            "_anchor_cache": {},
            "_anchor_base_cache": {},
            "_anchor_failed": False,
        }
        user_x = torch.zeros(1, 2, 3)
        timestep = torch.tensor([1.0])
        context = torch.ones(1, 4, 3)
        calls = []

        def fake_apply_model(x, ts, **kwargs):
            calls.append((x.clone(), ts.clone(), kwargs))
            state["_anchor_cache"][0] = torch.full_like(x, float(len(calls)))
            state["_anchor_base_cache"][0] = torch.full_like(x, float(len(calls) + 10))
            return x

        maybe_run_anchor(
            state,
            user_x,
            timestep,
            {"context": context, "transformer_options": {"cond_or_uncond": [0]}},
            apply_model=fake_apply_model,
        )

        self.assertEqual(len(calls), 2)
        self.assertIn(0, state["_anchor_cache"])
        self.assertIn(0, state["_anchor_base_cache"])
        self.assertTrue(torch.allclose(state["_anchor_cache"][0], torch.full_like(user_x, 1.5)))
        self.assertTrue(torch.allclose(state["_anchor_base_cache"][0], torch.full_like(user_x, 11.5)))

    def test_anchor_refresh_each_step_bypasses_start_only_cache_gate(self):
        class FakeDM:
            pass

        state = {
            "dm_ref": FakeDM(),
            "artist_anchor_q": True,
            "anchor_refresh_each_step": True,
            "anchor_seeds_count": 1,
            "low_vram_cache": False,
            "_anchor_failed": False,
            "_anchor_cache": {0: torch.ones(1, 2, 3)},
            "_anchor_last_sigma": 1.0,
        }
        calls = []

        def fake_apply_model(x, timestep, **kwargs):
            calls.append((x, timestep, kwargs))
            state["_anchor_cache"][0] = torch.full_like(x, 9.0)
            return x

        def prev_wrapper(apply_model, options):
            return "done"

        wrapped = make_sigma_capture(state, prev_wrapper)
        out = wrapped(fake_apply_model, {
            "input": torch.zeros(1, 2, 3),
            "timestep": torch.tensor([0.5]),
            "c": {"c_crossattn": torch.ones(1, 4, 3)},
        })

        self.assertEqual(out, "done")
        self.assertEqual(len(calls), 1)

    def test_anchor_cache_hit_skips_rerun_when_refresh_disabled(self):
        class FakeDM:
            pass

        state = {
            "dm_ref": FakeDM(),
            "anchor_seeds_count": 1,
            "low_vram_cache": False,
            "_anchor_cache": {0: torch.ones(1, 2, 3)},
            "_anchor_cache_key": (
                (1, 2, 3),
                _context_fingerprint(torch.ones(1, 4, 3)),
                1.0,
            ),
            "_anchor_failed": False,
        }
        calls = []

        maybe_run_anchor(
            state,
            torch.zeros(1, 2, 3),
            torch.tensor([1.0]),
            {"c_crossattn": torch.ones(1, 4, 3)},
            apply_model=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
