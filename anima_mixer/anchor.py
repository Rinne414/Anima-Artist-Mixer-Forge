"""Anchor-Q pre-run: fixed-seed hidden-state capture for cross-seed stability."""

import logging

import torch

from .constants import ANCHOR_SEEDS_MAX, ANCHOR_SEEDS_POOL
from .patching import _context_fingerprint, _in_stabilizer_window, reset_run_state

logger = logging.getLogger(__name__)


def _call_diffusion_forward(dm, x, timestep, context, transformer_options):
    """Invoke the diffusion model's forward, preferring the private
    ``_forward`` (skips outer wrappers, avoiding recursion) but falling back
    to the public call path when the private API is missing or changes."""
    if hasattr(dm, "_forward"):
        return dm._forward(
            x, timestep, context, transformer_options=transformer_options,
        )
    return dm(x, timestep, context, transformer_options=transformer_options)


def _get_crossattn_context(c_dict):
    context = c_dict.get("c_crossattn")
    if context is None:
        context = c_dict.get("context")
    return context


def make_sigma_capture(state, prev_wrapper):
    """Wrap the model forward to:
    1. Capture the current sigma.
    2. Trigger the anchor pre-run when the cache misses.
    3. Chain to any previously installed wrapper.
    """
    def wrapper(apply_model, options):
        ts = options.get("timestep")
        cur_sigma = None
        if ts is not None:
            try:
                cur_sigma = float(ts.flatten()[0].item())
                state["current_sigma"] = cur_sigma
            except Exception:
                pass

        # Unified run-start reset: a sigma jump upward (or the very first
        # forward) means a new sampling pass, so clear the per-run caches and
        # one-shot warnings. This is the single reset point every configuration
        # relies on, which is why the capture wrapper is always installed.
        prev_run_sigma = state.get("_run_last_sigma")
        if prev_run_sigma is None or (
            cur_sigma is not None and cur_sigma > prev_run_sigma + 1e-3
        ):
            reset_run_state(state)
        state["_run_last_sigma"] = cur_sigma

        # The anchor cache survives sigma jumps on purpose: the same prompt
        # across seeds shares a fingerprint and hits the cache. Only a
        # fingerprint change (shape / context content / first timestep)
        # triggers a re-run. The cache check itself runs only at the start
        # of a sampling run (sigma jump upward) — the cache key includes the
        # first-step sigma, so checking it again mid-run would miss on every
        # step and re-run the anchor pre-pass each time.
        if (
            state.get("artist_anchor_q", False)
            and not state.get("_anchor_failed", False)
            and _in_stabilizer_window(state)
        ):
            prev_sigma = state.get("_anchor_last_sigma")
            is_run_start = (
                prev_sigma is None
                or (cur_sigma is not None and cur_sigma > prev_sigma + 1e-3)
            )
            state["_anchor_last_sigma"] = cur_sigma
            if (
                state.get("anchor_refresh_each_step", False)
                or is_run_start
                or not state.get("_anchor_cache")
            ):
                user_x = options.get("input")
                user_ts = options.get("timestep")
                c_dict = options.get("c", {}) or {}
                if user_x is not None and user_ts is not None and c_dict:
                    maybe_run_anchor(state, user_x, user_ts, c_dict, apply_model=apply_model)

        if prev_wrapper is not None:
            return prev_wrapper(apply_model, options)
        return apply_model(options["input"], options["timestep"], **options["c"])
    return wrapper


def maybe_run_anchor(state, user_x, user_timestep, c_dict, apply_model=None):
    """Run the anchor pre-pass when the cache misses.

    Generates fixed-seed noise, runs a full model forward with
    ``state["_in_anchor_run"] = True`` so each CrossAttnWrapper captures its
    layer input into ``state["_anchor_cache"]`` without injecting artists.

    Called from the model_function_wrapper before the main forward starts.
    When ComfyUI provides ``apply_model``, use it so object patches are active
    and the CrossAttnWrapper instances can capture per-layer hidden states.
    """
    base_context = _get_crossattn_context(c_dict)
    if base_context is None:
        return
    original_context = base_context

    # Under CFG, use the cond row as the anchor conditioning. Pick the cond
    # row index once so context and token ids/weights stay paired: with
    # uncond-first ordering (cou=[1, 0]) a fixed row-0 slice would feed the
    # cond embedding with uncond ids. When this forward carries no cond row
    # (cou present and 0 not in it), defer to a later forward that has one.
    transformer_options = c_dict.get("transformer_options", {}) or {}
    cou = transformer_options.get("cond_or_uncond")
    cond_idx = 0
    if cou is not None:
        if 0 not in cou:
            return
        cond_idx = cou.index(0)
    if base_context.dim() >= 2 and base_context.shape[0] > 1:
        row = cond_idx if cond_idx < base_context.shape[0] else 0
        base_context = base_context[row:row + 1]

    cache_key = state.get("_anchor_cache_key")
    try:
        sigma_key = round(float(user_timestep.flatten()[0].item()), 4)
    except Exception:
        sigma_key = None
    new_key = (
        tuple(user_x.shape),
        _context_fingerprint(original_context),
        sigma_key,
    )
    if cache_key == new_key and state.get("_anchor_cache"):
        return  # cache hit

    dm = state["dm_ref"]

    state["_anchor_cache"] = {}
    state["_anchor_base_cache"] = {}
    state["_in_anchor_run"] = True

    bsz = user_x.shape[0]
    if base_context.shape[0] != bsz:
        if base_context.shape[0] == 1:
            ctx_for_anchor = base_context.expand(bsz, -1, -1)
        else:
            ctx_for_anchor = base_context[:1].expand(bsz, -1, -1)
    else:
        ctx_for_anchor = base_context
    ctx_for_anchor = ctx_for_anchor.contiguous().to(device=user_x.device, dtype=user_x.dtype)

    anchor_kwargs = {}
    for key in ("t5xxl_ids", "t5xxl_weights"):
        v = c_dict.get(key)
        if v is None or not torch.is_tensor(v):
            continue
        if v.shape[0] != bsz:
            if v.shape[0] == 1:
                v = v.expand(bsz, *v.shape[1:])
            else:
                # Reduce to the cond row (same index used for the context) so
                # ids/weights stay paired with the anchor conditioning.
                row = cond_idx if cond_idx < v.shape[0] else 0
                v = v[row:row + 1].expand(bsz, *v.shape[1:])
        anchor_kwargs[key] = v.contiguous()

    # Isolate transformer_options: no cond_or_uncond / patches leak through.
    safe_opts = dict(transformer_options) if isinstance(transformer_options, dict) else {}
    safe_opts.pop("cond_or_uncond", None)
    safe_opts.pop("patches", None)

    try:
        with torch.no_grad():
            t5xxl_ids = anchor_kwargs.pop("t5xxl_ids", None)
            t5xxl_weights = anchor_kwargs.pop("t5xxl_weights", None)
            if t5xxl_ids is not None and hasattr(dm, "preprocess_text_embeds"):
                processed_ctx = dm.preprocess_text_embeds(
                    ctx_for_anchor, t5xxl_ids, t5xxl_weights=t5xxl_weights,
                )
            else:
                processed_ctx = ctx_for_anchor

            seeds_count = max(1, min(int(state.get("anchor_seeds_count", 1)), ANCHOR_SEEDS_MAX))
            seeds = ANCHOR_SEEDS_POOL[:seeds_count]

            accumulator = {}   # layer_idx -> fp32 sum of hidden states
            base_accumulator = {}   # layer_idx -> fp32 sum of base outputs
            for seed in seeds:
                gen = torch.Generator(device=user_x.device)
                gen.manual_seed(seed)
                anchor_x_k = torch.randn(
                    user_x.shape, generator=gen,
                    device=user_x.device, dtype=user_x.dtype,
                )
                state["_anchor_cache"] = {}
                state["_anchor_base_cache"] = {}
                if apply_model is not None:
                    apply_model(
                        anchor_x_k,
                        user_timestep,
                        c_crossattn=processed_ctx,
                        transformer_options=safe_opts,
                    )
                else:
                    _call_diffusion_forward(
                        dm, anchor_x_k, user_timestep, processed_ctx, safe_opts,
                    )
                for layer_idx, hidden in state["_anchor_cache"].items():
                    if layer_idx not in accumulator:
                        accumulator[layer_idx] = hidden.to(torch.float32)
                    else:
                        accumulator[layer_idx] = accumulator[layer_idx] + hidden.to(torch.float32)
                for layer_idx, base_out in state.get("_anchor_base_cache", {}).items():
                    if layer_idx not in base_accumulator:
                        base_accumulator[layer_idx] = base_out.to(torch.float32)
                    else:
                        base_accumulator[layer_idx] = (
                            base_accumulator[layer_idx] + base_out.to(torch.float32)
                        )

            inv = 1.0 / max(1, seeds_count)
            avg_dtype = user_x.dtype
            low_vram = bool(state.get("low_vram_cache", False))
            anchor_cache = {}
            for idx, acc in accumulator.items():
                avg = (acc * inv).to(avg_dtype)
                anchor_cache[idx] = avg.cpu() if low_vram else avg
            state["_anchor_cache"] = anchor_cache
            anchor_base_cache = {}
            for idx, acc in base_accumulator.items():
                avg = (acc * inv).to(avg_dtype)
                anchor_base_cache[idx] = avg.cpu() if low_vram else avg
            state["_anchor_base_cache"] = anchor_base_cache
    except Exception as e:
        logger.warning(
            "[AnimaCrossAttn] anchor pre-run failed; anchor_q is disabled "
            "for this session: %s", e,
        )
        state["_anchor_cache"] = {}
        state["_anchor_base_cache"] = {}
        state["_anchor_failed"] = True
    finally:
        state["_in_anchor_run"] = False

    if state["_anchor_cache"]:
        state["_anchor_cache_key"] = new_key
        if not state.get("_warned_anchor_ok", False):
            logger.info(
                "[AnimaCrossAttn] anchor pre-run captured %d layers of hidden state",
                len(state["_anchor_cache"]),
            )
            state["_warned_anchor_ok"] = True
    elif not state.get("_anchor_failed", False):
        # No exception, but our wrappers captured nothing — a later cross-attn
        # patch likely overrode ours. Disable anchor_q for this run so the next
        # steps do not each trigger a full (fruitless) pre-run.
        state["_anchor_failed"] = True
        if not state.get("_warned_anchor_empty", False):
            logger.warning(
                "[AnimaCrossAttn] anchor pre-run captured no hidden states; the "
                "cross-attn wrappers may have been overridden by a later patch. "
                "anchor_q is disabled for this run."
            )
            state["_warned_anchor_empty"] = True
