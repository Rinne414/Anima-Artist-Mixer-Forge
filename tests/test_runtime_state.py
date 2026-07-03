"""Runtime-state lifecycle regression tests (require a real torch install).

Covers the verified runtime-state fixes: layer-failure handling, the unified
run-start reset, concat_with_base CFG isolation, per-forward stabilizer
fingerprints, anchor cond-row pairing, combined-path fade invalidation, probe
scoping/reset, low-vram EMA caching, and shared-helper cleanups.
"""

import os
import sys
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from anima_mixer import patching  # noqa: E402
from anima_mixer.anchor import make_sigma_capture, maybe_run_anchor  # noqa: E402
from anima_mixer.nodes_core import _ProbeCrossAttnWrapper  # noqa: E402
from anima_mixer.patching import (  # noqa: E402
    _forward_fingerprint,
    _in_stabilizer_window,
    reset_run_state,
)
from anima_mixer.wrapper import CrossAttnWrapper, _should_reraise  # noqa: E402


class _AddOneAttn(nn.Module):
    """Returns x + 1, so its output tracks the query input."""

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        return x + 1.0


class _KVMeanAttn(nn.Module):
    """Returns the per-batch mean of the K/V tokens, broadcast over the query."""

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        mean = context.mean(dim=1, keepdim=True)
        return mean.expand(x.shape[0], x.shape[1], context.shape[-1])


class _KVLenAttn(nn.Module):
    """Returns the K/V token count per call, recording the lengths it saw."""

    def __init__(self):
        super().__init__()
        self.kv_lens = []

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        self.kv_lens.append(context.shape[1])
        val = float(context.shape[1])
        return torch.full((x.shape[0], x.shape[1], context.shape[-1]), val,
                          device=x.device, dtype=x.dtype)


# ------------------------------------------------------------------- FIX 1

class LayerFailureTest(unittest.TestCase):
    def test_generic_failure_disables_layer_via_shared_state_set(self):
        state = {"enabled": True}
        w = CrossAttnWrapper(_AddOneAttn(), state, 0)
        calls = {"n": 0}

        def boom(*args, **kwargs):
            calls["n"] += 1
            raise RuntimeError("injection failed")

        w._dispatch = boom
        ctx = torch.zeros(1, 1, 2)
        x = torch.zeros(1, 1, 2)

        out1 = w.forward(x, context=ctx)
        out2 = w.forward(x, context=ctx)

        # First failure records the layer in the shared set; the second call is
        # short-circuited by the disabled check (dispatch not re-entered).
        self.assertEqual(calls["n"], 1)
        self.assertIn(0, state["_disabled_layers"])
        self.assertTrue(torch.allclose(out1, torch.ones(1, 1, 2)))
        self.assertTrue(torch.allclose(out2, torch.ones(1, 1, 2)))

    def test_should_not_reraise_generic_exception(self):
        self.assertFalse(_should_reraise(RuntimeError("boom")))
        self.assertFalse(_should_reraise(ValueError("bad")))

    def test_should_reraise_oom(self):
        oom_cls = (
            getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
            or getattr(torch, "OutOfMemoryError", None)
        )
        if oom_cls is None:
            self.skipTest("no OutOfMemoryError class in this torch build")
        self.assertTrue(_should_reraise(oom_cls("out of memory")))

    def test_should_reraise_interrupt_and_forward_propagates(self):
        fake = types.ModuleType("comfy.model_management")

        class InterruptProcessingException(Exception):
            pass

        fake.InterruptProcessingException = InterruptProcessingException
        saved_pkg = sys.modules.get("comfy")
        saved_mod = sys.modules.get("comfy.model_management")
        sys.modules["comfy"] = sys.modules.get("comfy") or types.ModuleType("comfy")
        sys.modules["comfy.model_management"] = fake
        try:
            self.assertTrue(_should_reraise(InterruptProcessingException()))

            state = {"enabled": True}
            w = CrossAttnWrapper(_AddOneAttn(), state, 0)

            def interrupt(*args, **kwargs):
                raise InterruptProcessingException()

            w._dispatch = interrupt
            with self.assertRaises(InterruptProcessingException):
                w.forward(torch.zeros(1, 1, 2), context=torch.zeros(1, 1, 2))
            # The interrupt must not have disabled the layer.
            self.assertNotIn(0, state.get("_disabled_layers", set()))
        finally:
            if saved_pkg is None:
                sys.modules.pop("comfy", None)
            else:
                sys.modules["comfy"] = saved_pkg
            if saved_mod is None:
                sys.modules.pop("comfy.model_management", None)
            else:
                sys.modules["comfy.model_management"] = saved_mod


# ------------------------------------------------------------------- FIX 2

class RunStartResetTest(unittest.TestCase):
    def _static_state(self):
        return {
            "normalize_weights": True, "apply_to_uncond": False,
            "match_base_norm": False, "artist_static_capture": True,
            "static_capture_k": 1, "static_capture_mode": "output",
        }

    def test_static_cache_resets_on_requeue_with_same_schedule(self):
        state = self._static_state()
        w = CrossAttnWrapper(_AddOneAttn(), state, 0)
        artist = torch.zeros(1, 1, 2)
        base_ctx = torch.zeros(1, 1, 2)

        out = None
        for sig, xval in [(10.0, 0.0), (9.0, 0.0), (8.0, 0.0)]:
            state["current_sigma"] = sig
            out = w._fwd_output_avg(
                torch.full((1, 1, 2), xval), base_ctx, None, {},
                [artist], [1.0], [1.0], [True], "interpolate", 1.0,
            )
        self.assertTrue(torch.allclose(out, torch.ones(1, 1, 2)))  # frozen at x=0

        # Re-queue: identical schedule, sigma jumps back to the start. The old
        # max-sigma tracking never reset here (cur == prev_max); last-sigma
        # tracking does, so the artist re-freezes to the new x.
        state["current_sigma"] = 10.0
        out2 = w._fwd_output_avg(
            torch.full((1, 1, 2), 5.0), base_ctx, None, {},
            [artist], [1.0], [1.0], [True], "interpolate", 1.0,
        )
        self.assertTrue(torch.allclose(out2, torch.full((1, 1, 2), 6.0)))

    def test_disabled_layer_reinjects_after_run_start_reset(self):
        state = {"enabled": True}
        w = CrossAttnWrapper(_AddOneAttn(), state, 0)
        fail = {"on": True}

        def maybe_fail(*args, **kwargs):
            if fail["on"]:
                raise RuntimeError("boom")
            return torch.full((1, 1, 2), 99.0)

        w._dispatch = maybe_fail
        ctx = torch.zeros(1, 1, 2)
        x = torch.zeros(1, 1, 2)

        w.forward(x, context=ctx)
        self.assertIn(0, state["_disabled_layers"])

        reset_run_state(state)   # a new sampling run starts
        fail["on"] = False
        out2 = w.forward(x, context=ctx)
        self.assertTrue(torch.allclose(out2, torch.full((1, 1, 2), 99.0)))

    def test_sigma_capture_resets_run_state_only_on_sigma_jump(self):
        state = {"_disabled_layers": {3}, "_static_cache": {"x": 1}}

        def prev_wrapper(apply_model, options):
            return "done"

        wrapped = make_sigma_capture(state, prev_wrapper)

        def step(sigma):
            return wrapped(lambda *a, **k: None, {
                "input": torch.zeros(1, 2, 2),
                "timestep": torch.tensor([sigma]),
                "c": {"c_crossattn": torch.ones(1, 2, 2)},
            })

        step(8.0)                                   # run 1 start -> reset
        self.assertEqual(state["_disabled_layers"], set())
        self.assertEqual(state["_static_cache"], {})

        state["_disabled_layers"].add(5)
        step(7.0)                                   # sigma decreases -> no reset
        self.assertIn(5, state["_disabled_layers"])

        step(8.0)                                   # run 2 start -> reset
        self.assertEqual(state["_disabled_layers"], set())


# ------------------------------------------------------------------- FIX 3

class ConcatWithBaseCFGTest(unittest.TestCase):
    def test_uncond_row_stays_pure_base(self):
        attn = _KVLenAttn()
        w = CrossAttnWrapper(attn, {}, 0)
        x = torch.zeros(2, 2, 2)
        context = torch.zeros(2, 3, 2)       # base K/V = 3 tokens
        combined = torch.ones(1, 2, 2)       # artist K/V = 2 tokens

        out = w._fwd_with_combined(
            x, context, None, {}, combined, [True, False], "concat_with_base", 1.0,
        )

        # cond row = merged forward (3 + 2 tokens), uncond row = pure base (3).
        self.assertTrue(torch.allclose(out[0], torch.full((2, 2), 5.0)))
        self.assertTrue(torch.allclose(out[1], torch.full((2, 2), 3.0)))
        self.assertEqual(sorted(attn.kv_lens), [3, 5])   # merged + base

    def test_all_masked_rows_use_single_forward(self):
        attn = _KVLenAttn()
        w = CrossAttnWrapper(attn, {}, 0)
        x = torch.zeros(2, 2, 2)
        context = torch.zeros(2, 3, 2)
        combined = torch.ones(1, 2, 2)

        out = w._fwd_with_combined(
            x, context, None, {}, combined, [True, True], "concat_with_base", 1.0,
        )

        self.assertTrue(torch.allclose(out, torch.full((2, 2, 2), 5.0)))
        self.assertEqual(attn.kv_lens, [5])   # no extra base forward


# ------------------------------------------------------------------- FIX 4

class PerForwardFingerprintTest(unittest.TestCase):
    def test_ema_entries_independent_per_context(self):
        state = {
            "artist_ema_alpha": 0.5, "artist_static_capture": False,
            "current_sigma": 10.0,
        }
        w = CrossAttnWrapper(nn.Identity(), state, 0)
        ctx_a = torch.zeros(1, 2, 2)
        ctx_b = torch.ones(1, 2, 2)
        fp_a = _forward_fingerprint(state, ctx_a)
        fp_b = _forward_fingerprint(state, ctx_b)
        self.assertNotEqual(fp_a, fp_b)

        a1 = w._apply_ema(torch.full((1, 1, 1), 10.0), "interpolate", fp=fp_a)
        b1 = w._apply_ema(torch.full((1, 1, 1), 100.0), "interpolate", fp=fp_b)
        a2 = w._apply_ema(torch.full((1, 1, 1), 20.0), "interpolate", fp=fp_a)

        # B never blends with A (independent entries); A blends only with A.
        self.assertTrue(torch.equal(a1, torch.full((1, 1, 1), 10.0)))
        self.assertTrue(torch.equal(b1, torch.full((1, 1, 1), 100.0)))
        self.assertTrue(torch.allclose(a2, torch.full((1, 1, 1), 15.0)))

    def test_static_entries_independent_per_context(self):
        state = {
            "normalize_weights": True, "apply_to_uncond": False,
            "match_base_norm": False, "artist_static_capture": True,
            "static_capture_k": 1, "static_capture_mode": "output",
            "current_sigma": 10.0,
        }
        w = CrossAttnWrapper(_AddOneAttn(), state, 0)
        artist = torch.zeros(1, 1, 2)
        ctx_a = torch.zeros(1, 1, 2)
        ctx_b = torch.ones(1, 1, 2)
        fp_a = _forward_fingerprint(state, ctx_a)
        fp_b = _forward_fingerprint(state, ctx_b)

        out_a = w._fwd_output_avg(
            torch.zeros(1, 1, 2), ctx_a, None, {},
            [artist], [1.0], [1.0], [True], "interpolate", 1.0, fp=fp_a,
        )
        out_b = w._fwd_output_avg(
            torch.full((1, 1, 2), 50.0), ctx_b, None, {},
            [artist], [1.0], [1.0], [True], "interpolate", 1.0, fp=fp_b,
        )

        self.assertTrue(torch.allclose(out_a, torch.ones(1, 1, 2)))       # x=0
        self.assertTrue(torch.allclose(out_b, torch.full((1, 1, 2), 51.0)))  # x=50
        self.assertEqual(len(state["_static_cache"]), 2)


# ------------------------------------------------------------------- FIX 5

class AnchorCondRowTest(unittest.TestCase):
    def test_pairs_cond_row_ids_under_uncond_first(self):
        class RecordingDM:
            def __init__(self):
                self.seen_ids = None

            def preprocess_text_embeds(self, ctx, ids, t5xxl_weights=None):
                self.seen_ids = ids
                return ctx

        dm = RecordingDM()
        state = {
            "dm_ref": dm, "anchor_seeds_count": 1, "low_vram_cache": False,
            "_anchor_cache": {}, "_anchor_base_cache": {}, "_anchor_failed": False,
        }
        user_x = torch.zeros(1, 2, 3)                       # bsz = 1
        context = torch.stack([torch.full((4, 3), 1.0), torch.full((4, 3), 2.0)])
        ids = torch.tensor([[7, 7, 7, 7], [9, 9, 9, 9]])   # row1 = cond ids

        def fake_apply_model(x, ts, **kwargs):
            state["_anchor_cache"][0] = torch.full_like(x, 1.0)
            return x

        maybe_run_anchor(
            state, user_x, torch.tensor([1.0]),
            {
                "c_crossattn": context, "t5xxl_ids": ids,
                "transformer_options": {"cond_or_uncond": [1, 0]},
            },
            apply_model=fake_apply_model,
        )

        # cou=[1, 0] -> cond row is index 1; its ids (9) must be the ones used.
        self.assertIsNotNone(dm.seen_ids)
        self.assertTrue(torch.all(dm.seen_ids == 9))

    def test_skips_forward_without_cond_row(self):
        class FakeDM:
            pass

        state = {
            "dm_ref": FakeDM(), "anchor_seeds_count": 1, "low_vram_cache": False,
            "_anchor_cache": {}, "_anchor_base_cache": {}, "_anchor_failed": False,
        }
        calls = []

        maybe_run_anchor(
            state, torch.zeros(2, 2, 3), torch.tensor([1.0, 1.0]),
            {
                "c_crossattn": torch.ones(2, 4, 3),
                "transformer_options": {"cond_or_uncond": [1]},
            },
            apply_model=lambda *a, **k: calls.append(1),
        )

        self.assertEqual(calls, [])
        self.assertEqual(state["_anchor_cache"], {})   # no pre-run built


# ------------------------------------------------------------------- FIX 6

class CombinedFadeInvalidationTest(unittest.TestCase):
    def test_fade_change_invalidates_frozen_combined_output(self):
        state = {
            "normalize_weights": True, "apply_to_uncond": False,
            "match_base_norm": False, "artist_static_capture": True,
            "static_capture_k": 1, "static_capture_mode": "output",
            "current_sigma": 10.0,
        }
        w = CrossAttnWrapper(_KVMeanAttn(), state, 0)
        x = torch.zeros(1, 1, 2)
        context = torch.zeros(1, 1, 2)     # base attention output -> 0

        out1 = w._fwd_with_combined(
            x, context, None, {}, torch.full((1, 2, 2), 4.0),
            [True], "interpolate", 1.0, fp=None, extra_fp=(4.0,),
        )
        self.assertTrue(torch.allclose(out1, torch.full((1, 1, 2), 4.0)))

        # Fade drops: the combined tokens change and so does extra_fp, so the
        # frozen entry is invalidated instead of locking the stale weight.
        state["current_sigma"] = 9.0
        out2 = w._fwd_with_combined(
            x, context, None, {}, torch.full((1, 2, 2), 1.0),
            [True], "interpolate", 1.0, fp=None, extra_fp=(1.0,),
        )
        self.assertTrue(torch.allclose(out2, torch.full((1, 1, 2), 1.0)))


# ------------------------------------------------------------------- FIX 7

class ProbeTest(unittest.TestCase):
    def _probe_state(self, budget=6, sigma=10.0):
        return {
            "raws": [None], "individuals": [torch.ones(1, 1, 2)], "real_lens": [1],
            "probe_stats": {}, "probe_steps": budget, "_probe_seen_sigmas": set(),
            "_probe_forward_count": 0, "current_sigma": sigma,
            "_disable_batched": True,
        }

    def test_excludes_uncond_rows_from_measurement(self):
        state = self._probe_state()
        # cond context matches the artist (delta 0); uncond context differs.
        state["individuals"] = [torch.ones(1, 1, 2)]
        w = _ProbeCrossAttnWrapper(_KVMeanAttn(), state, 0)
        context = torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]])   # row0 cond, row1 uncond
        x = torch.zeros(2, 1, 2)

        w._dispatch(x, context, None, {"cond_or_uncond": [0, 1]})

        layer_stats = state["probe_stats"][0]
        # Only the cond row is measured; its delta is zero (artist == base there).
        # Including the uncond row would have produced a non-zero influence.
        self.assertAlmostEqual(layer_stats[0][0], 0.0, places=6)
        self.assertEqual(layer_stats[0][1], 1)

    def test_forward_counter_enforces_budget_without_sigma(self):
        state = self._probe_state(budget=2, sigma=None)
        w = _ProbeCrossAttnWrapper(_KVMeanAttn(), state, 0)
        context = torch.ones(1, 1, 2)
        for _ in range(5):
            w._dispatch(torch.zeros(1, 1, 2), context, None, {"cond_or_uncond": [0]})
        # Budget caps the measured forwards even with no visible sigma.
        self.assertEqual(state["probe_stats"][0][0][1], 2)

    def test_reset_clears_probe_accumulators_in_place(self):
        stats = {0: [[1.0, 3]]}
        seen = {10.0, 9.0}
        state = {"probe_stats": stats, "_probe_seen_sigmas": seen,
                 "_probe_forward_count": 4}

        reset_run_state(state)

        self.assertEqual(state["probe_stats"], {})
        self.assertEqual(state["_probe_seen_sigmas"], set())
        self.assertEqual(state["_probe_forward_count"], 0)
        # In-place clear so the probe registry's shared reference sees the reset.
        self.assertIs(state["probe_stats"], stats)
        self.assertIs(state["_probe_seen_sigmas"], seen)


# ------------------------------------------------------------------- FIX 9

class EmaLowVramTest(unittest.TestCase):
    def test_ema_cache_round_trips_through_low_vram_helpers(self):
        state = {
            "artist_ema_alpha": 0.5, "artist_static_capture": False,
            "current_sigma": 10.0, "low_vram_cache": True,
        }
        w = CrossAttnWrapper(nn.Identity(), state, 0)

        w._apply_ema(torch.full((1, 1, 1), 10.0), "interpolate", fp="fpX")
        second = w._apply_ema(torch.full((1, 1, 1), 20.0), "interpolate", fp="fpX")

        # The low-vram path stores detached/offloaded tensors and loads them
        # back to blend; the EMA math must still be correct.
        self.assertTrue(torch.allclose(second, torch.full((1, 1, 1), 15.0)))
        self.assertEqual(state["_ema_cache"][(0, "fpX")].device.type, "cpu")


# ------------------------------------------------------------------ FIX 10

class SharedHelperTest(unittest.TestCase):
    def test_reset_run_state_clears_expected_keys(self):
        state = {
            "_disabled_layers": {1, 2}, "_disable_batched": True,
            "_warned_batched": True, "_warned": True, "_warned_svd": True,
            "_ema_cache": {"k": 1}, "_static_cache": {"k": 1},
            "_ctx_fp_memo": {1: 2}, "_anchor_failed": True,
            "_anchor_cache": {0: torch.ones(1)},          # must survive
            "_anchor_cache_key": ("keep",),               # must survive
        }
        reset_run_state(state)

        self.assertEqual(state["_disabled_layers"], set())
        self.assertFalse(state["_disable_batched"])
        self.assertFalse(state["_warned_batched"])
        self.assertFalse(state["_warned"])
        self.assertFalse(state["_warned_svd"])
        self.assertEqual(state["_ema_cache"], {})
        self.assertEqual(state["_static_cache"], {})
        self.assertEqual(state["_ctx_fp_memo"], {})
        self.assertFalse(state["_anchor_failed"])
        # Content-keyed anchor caches survive across runs by design.
        self.assertIn(0, state["_anchor_cache"])
        self.assertEqual(state["_anchor_cache_key"], ("keep",))

    def test_cleanup_residual_wrappers_removed(self):
        self.assertFalse(hasattr(patching, "cleanup_residual_wrappers"))

    def test_in_stabilizer_window_is_shared_helper(self):
        from anima_mixer import anchor, wrapper
        self.assertIs(wrapper._in_stabilizer_window, _in_stabilizer_window)
        self.assertIs(anchor._in_stabilizer_window, _in_stabilizer_window)

    def test_in_stabilizer_window_semantics(self):
        self.assertTrue(_in_stabilizer_window({}))
        self.assertTrue(_in_stabilizer_window({"stabilizer_min_sigma": 5.0}))
        self.assertTrue(_in_stabilizer_window(
            {"stabilizer_min_sigma": 5.0, "current_sigma": 6.0}))
        self.assertFalse(_in_stabilizer_window(
            {"stabilizer_min_sigma": 5.0, "current_sigma": 4.0}))


if __name__ == "__main__":
    unittest.main()
