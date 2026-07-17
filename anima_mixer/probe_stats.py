"""Pure computations for the probe report (v27.2): contribution shares,
plain-language verdicts, and per-step influence curves.

No torch here — everything operates on the plain accumulator structures the
probe wrapper fills ([sum, count] pairs), so the pytest suite exercises it
without a GPU.
"""

_CURVE_RAMP = " .:-=+*#@"
_LABEL_WIDTH = 20


def contribution_shares(scores):
    """Per-artist mean influence and share of the total.

    ``scores[artist][layer]`` holds mean relative deltas. Returns
    ``(totals, shares)``: ``totals[i]`` is artist i's mean over layers,
    ``shares[i]`` its fraction of the summed totals (all zeros when nothing
    was measured).
    """
    totals = [sum(row) / len(row) if row else 0.0 for row in scores]
    denom = sum(totals)
    if denom <= 0:
        return totals, [0.0] * len(totals)
    return totals, [total / denom for total in totals]


SUGGEST_RATIO_CAP = 3.0


def suggest_equalizing_weights(totals, ratio_cap=SUGGEST_RATIO_CAP):
    """Per-artist weights that would equalize the measured influence.

    The probe measures unweighted per-artist delta strength ``m_i`` and the
    mixed contribution scales as ``w_i * m_i``, so ``w_i ~ 1/m_i`` levels the
    split. The weights are meant to be emitted as explicit ``::w::`` values,
    which bypass runtime normalization — so they are normalized to SUM to
    1.0, keeping the total injection strength of a normalized chain instead
    of multiplying it by the artist count. Correction ratios are clamped to
    [1/ratio_cap, ratio_cap] around the mean as a measurement-noise guard;
    immeasurable (~zero influence) artists get the max ratio.

    Returns a list of ``(weight, status)`` with status in {"ok", "clamped",
    "immeasurable"}, or None when fewer than two artists are measurable.
    """
    if len(totals) < 2:
        return None
    inverses = []
    for total in totals:
        try:
            total = float(total)
        except (TypeError, ValueError):
            total = 0.0
        inverses.append(1.0 / total if total > 1e-6 else None)
    finite = [v for v in inverses if v is not None]
    if len(finite) < 2:
        return None
    mean_inv = sum(finite) / len(finite)
    lo, hi = 1.0 / ratio_cap, ratio_cap
    ratios, statuses = [], []
    for inv in inverses:
        if inv is None:
            ratios.append(hi)
            statuses.append("immeasurable")
            continue
        ratio = inv / mean_inv
        clamped = min(max(ratio, lo), hi)
        ratios.append(clamped)
        statuses.append("ok" if abs(clamped - ratio) < 1e-9 else "clamped")
    denom = sum(ratios)
    return [(round(r / denom, 2), s) for r, s in zip(ratios, statuses)]


def share_verdict(share, artist_count):
    """Plain-language verdict for one artist's share vs an equal split."""
    if artist_count <= 0:
        return ""
    ratio = share * artist_count
    if ratio >= 1.5:
        return "dominant"
    if ratio >= 0.5:
        return "balanced"
    if ratio >= 0.15:
        return "weak"
    return "negligible"


def render_step_curves(step_stats, labels):
    """ASCII curves of per-step influence, one row per artist.

    ``step_stats`` maps a rounded sigma to per-artist ``[sum, count]`` pairs
    (each forward at that step adds one layer sample). Sampling order is
    sigma descending. Returns [] when there is nothing to draw, so callers
    can skip the section for pre-v27.2 probe data.
    """
    if not isinstance(step_stats, dict) or not step_stats or not labels:
        return []
    keys = sorted(step_stats, reverse=True)
    series = []
    for i in range(len(labels)):
        vals = []
        for key in keys:
            rows = step_stats.get(key) or []
            if i < len(rows) and rows[i][1] > 0:
                vals.append(rows[i][0] / rows[i][1])
            else:
                vals.append(0.0)
        series.append(vals)
    top = max((max(vals) for vals in series), default=0.0)
    if top <= 0:
        return []
    width = min(max(len(str(label)) for label in labels), _LABEL_WIDTH)
    lines = [
        "per-step influence (mean over layers; left = sampling start / high sigma):",
    ]
    for label, vals in zip(labels, series):
        chars = "".join(_CURVE_RAMP[int(round((len(_CURVE_RAMP) - 1) * v / top))] for v in vals)
        lines.append(f"  {str(label)[:_LABEL_WIDTH]:<{width}} {chars}  ({vals[0]:.2f} -> {vals[-1]:.2f})")
    lines.append(f"  sigma {keys[0]:g} -> {keys[-1]:g}; '{_CURVE_RAMP[-1]}' = {top:.3f}")
    return lines
