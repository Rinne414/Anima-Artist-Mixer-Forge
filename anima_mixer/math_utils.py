"""Pure tensor math used by the cross-attention wrapper."""

import torch


def lowrank_rows_deterministic(d_mat, k):
    """Deterministic top-k low-rank reconstruction of row vectors.

    ``D_k = U_k U_k^T D`` via Gram eigendecomposition (no randomized SVD),
    so results are reproducible across runs.
    """
    n = int(d_mat.shape[0])
    if k >= n:
        return d_mat
    work = d_mat.to(torch.float32)
    gram = work @ work.transpose(0, 1)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    basis = eigvecs[:, order[:k]]
    return basis @ (basis.transpose(0, 1) @ work)


def project_perpendicular(delta, base):
    """Strip the component of ``delta`` parallel to ``base``.

    Projection is per token (inner product over the last dim):
    ``delta_perp = delta - (delta . base_unit) * base_unit``
    """
    delta_f32 = delta.to(torch.float32)
    base_f32 = base.to(torch.float32)
    base_norm_sq = (base_f32 * base_f32).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    proj_coef = (delta_f32 * base_f32).sum(dim=-1, keepdim=True) / base_norm_sq
    return (delta_f32 - proj_coef * base_f32).to(delta.dtype)


def smoothstep(t):
    """Classic smoothstep ramp on a scalar in [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def timing_fade_factor(sigma_route, current_sigma):
    """Per-artist injection multiplier for a sigma-space timing window.

    ``sigma_route`` is ``(lo, hi, fade_in_lo, fade_out_hi)`` produced at patch
    time from the percent-space window (sigma decreases as sampling
    progresses, so ``hi`` is the start of the window and ``lo`` the end):

      - ``hi``          sigma at window start percent
      - ``fade_in_lo``  sigma at ``start + fade`` (== hi when fade == 0)
      - ``fade_out_hi`` sigma at ``end - fade``   (== lo when fade == 0)
      - ``lo``          sigma at window end percent

    Returns 0.0 outside the window, 1.0 in the plateau, and a smoothstep
    ramp inside the fade regions.
    """
    if sigma_route is None:
        return 1.0
    if current_sigma is None:
        return 1.0
    lo, hi, fade_in_lo, fade_out_hi = sigma_route
    c = float(current_sigma)
    if c > hi or c < lo:
        return 0.0
    # Fade-in region: sigma between fade_in_lo and hi.
    if c > fade_in_lo and hi > fade_in_lo:
        return smoothstep((hi - c) / (hi - fade_in_lo))
    # Fade-out region: sigma between lo and fade_out_hi.
    if c < fade_out_hi and fade_out_hi > lo:
        return smoothstep((c - lo) / (fade_out_hi - lo))
    return 1.0
