"""Loss functions for the windowed encoder training.

L_total = c_window_gate * [
    λ1 * L_cont(z_v_proj, z_f_proj)           -- contrastive (contact sub-batch)
  + λ2 * ||g_v(z_v_repr)  − F_window||²        -- vision-only regression
  + λ3 * ||g_vp([z_v_repr, z_p_repr]) − F_window||²  -- fused regression
]

Gating strategy: contact-only sub-batch.
  - MSE terms: averaged only over contact samples.
  - Contrastive: run on the contact sub-batch exclusively (B_c × B_c matrix).
    If a batch contains zero contact samples the loss is zero and training
    skips the backward pass (handled in train.py).
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# ── Contrastive loss ───────────────────────────────────────────────────────────

def contrastive_loss(
    z_a: torch.Tensor,        # (B, D) L2-normalised
    z_b: torch.Tensor,        # (B, D) L2-normalised
    logit_scale: torch.Tensor, # scalar parameter (log temperature)
) -> torch.Tensor:
    """Symmetric InfoNCE loss over a batch of positive pairs.

    Uses einsum for the scaled cosine-similarity matrix, matching the approach
    from ContrastiveLoss in mmai_hw2_edward_rivera.py.

    Parameters
    ----------
    z_a, z_b    : already L2-normalised embeddings of the same shape (B, D).
    logit_scale : learnable log(1/temperature); clamped to prevent explosion.

    Returns
    -------
    Scalar loss (mean of symmetric cross-entropy rows + cols).
    """
    B = z_a.shape[0]
    scale = logit_scale.exp().clamp(max=100.0)
    # (B, B) scaled cosine similarity matrix
    sim = torch.einsum("id,jd->ij", z_a, z_b) * scale
    labels = torch.arange(B, device=z_a.device)
    loss_a2b = F.cross_entropy(sim,   labels)
    loss_b2a = F.cross_entropy(sim.T, labels)
    return (loss_a2b + loss_b2a) / 2.0


# ── Combined gated loss ────────────────────────────────────────────────────────

def combined_loss(
    outputs: Dict[str, torch.Tensor],
    f_window: torch.Tensor,              # (B, 3) — target force direction
    c_window: torch.Tensor,              # (B,)   — int64 contact gate
    lambdas: Tuple[float, float, float], # (λ1, λ2, λ3)
    logit_scale: torch.Tensor,           # learnable scalar
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the c_window-gated combined loss.

    Returns
    -------
    total        : scalar tensor (differentiable).
    components   : dict with float values for logging:
                   {"n_contact", "l_cont", "l_mse_v", "l_mse_vp", "total"}
                   Returns zeros for all components when no contact samples.
    """
    λ1, λ2, λ3 = lambdas
    contact_mask = c_window.bool()  # (B,)
    n_contact = int(contact_mask.sum().item())

    zero = f_window.new_tensor(0.0)

    if n_contact == 0:
        components = {
            "n_contact": 0,
            "l_cont":    0.0,
            "l_mse_v":   0.0,
            "l_mse_vp":  0.0,
            "total":     0.0,
        }
        return zero, components

    # ── Contrastive on contact sub-batch ──────────────────────────────────────
    z_v_proj_c = outputs["z_v_proj"][contact_mask]   # (B_c, 64)
    z_f_proj_c = outputs["z_f_proj"][contact_mask]   # (B_c, 64)
    l_cont = contrastive_loss(z_v_proj_c, z_f_proj_c, logit_scale)

    # ── MSE regression on contact sub-batch ───────────────────────────────────
    f_target   = f_window[contact_mask]                # (B_c, 3)
    F_pred_v   = outputs["F_pred_v"][contact_mask]     # (B_c, 3)
    F_pred_vp  = outputs["F_pred_vp"][contact_mask]    # (B_c, 3)

    l_mse_v  = F.mse_loss(F_pred_v,  f_target)
    l_mse_vp = F.mse_loss(F_pred_vp, f_target)

    total = λ1 * l_cont + λ2 * l_mse_v + λ3 * l_mse_vp

    components = {
        "n_contact": n_contact,
        "l_cont":    l_cont.item(),
        "l_mse_v":   l_mse_v.item(),
        "l_mse_vp":  l_mse_vp.item(),
        "total":     total.item(),
    }
    return total, components
