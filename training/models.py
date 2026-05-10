"""Encoder architectures and combined training model.

Architecture summary
--------------------
VisionEncoder
    ResNet18 backbone (ImageNet pretrained), fc replaced with Linear(512, 128)
    → z_v_repr (128-d representation)
    → z_v_proj (64-d projection, separate head)

ProprioEncoder
    Flatten(k*12=60) → Linear(60, 128) → ReLU → Linear(128, 128)
    → z_p_repr (128-d representation)

ForceEncoder
    Linear(3, 64) → ReLU → Linear(64, 64)
    → z_f_proj (64-d projection, lives in the contrastive embedding space)

ForcePredV  (g_v)
    Linear(128, 64) → ReLU → Linear(64, 3)
    Predicts F_window from vision representation alone.

ForcePredVP  (g_vp)
    Linear(256, 128) → ReLU → Linear(128, 3)
    Predicts F_window from concatenated [z_v_repr, z_p_repr].

EncoderTrainingModel
    Owns all encoders, heads, and a learnable logit_scale (for contrastive).
    forward() returns a dict of all intermediate representations + predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ── Vision encoder ─────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """ResNet18 backbone → 128-d representation + 64-d projection head.

    The ResNet18 fc is replaced with a Linear(512, 128) to produce z_v_repr.
    A separate linear projection maps z_v_repr → z_v_proj (64-d) for use in
    the contrastive alignment loss with z_f_proj.

    All pretrained convolutional layers are frozen; only backbone.fc and proj
    are trained. This prevents overfitting when contact-labelled windows are
    scarce relative to the 11M-parameter backbone.
    """

    def __init__(self, repr_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Linear(512, repr_dim)
        self.backbone = backbone
        self.proj = nn.Linear(repr_dim, proj_dim)

        # Freeze pretrained conv layers; leave backbone.fc + proj trainable
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W)  →  z_repr (B, 128), z_proj (B, 64)"""
        z_repr = self.backbone(x)           # (B, 128)
        z_proj = self.proj(z_repr)          # (B, 64)
        return z_repr, z_proj


# ── Proprio encoder ────────────────────────────────────────────────────────────

class ProprioEncoder(nn.Module):
    """2-layer MLP: (k*12=60) → 128-d representation.

    Input is the flattened, z-scored proprio window (k timesteps × 12 joints).
    """

    def __init__(self, in_dim: int = 60, repr_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 60)  →  z_p_repr (B, 128)"""
        return self.net(x)


# ── Force encoder ──────────────────────────────────────────────────────────────

class ForceEncoder(nn.Module):
    """2-layer MLP: (3,) → 64-d projection.

    Encodes the F_window unit vector into the same space as z_v_proj for
    contrastive alignment.
    """

    def __init__(self, in_dim: int = 3, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3)  →  z_f_proj (B, 64)"""
        return self.net(x)


# ── F_window prediction heads ──────────────────────────────────────────────────

class ForcePredV(nn.Module):
    """g_v: predict F_window from vision representation alone.

    Linear(128, 64) → ReLU → Linear(64, 3)
    """

    def __init__(self, repr_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, z_v_repr: torch.Tensor) -> torch.Tensor:
        """z_v_repr: (B, 128)  →  F_pred (B, 3)"""
        return self.net(z_v_repr)


class ForcePredVP(nn.Module):
    """g_vp: predict F_window from [z_v_repr, z_p_repr] concatenation.

    Linear(256, 128) → ReLU → Linear(128, 3)
    """

    def __init__(self, v_repr_dim: int = 128, p_repr_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(v_repr_dim + p_repr_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, z_v_repr: torch.Tensor, z_p_repr: torch.Tensor) -> torch.Tensor:
        """z_v_repr: (B, 128), z_p_repr: (B, 128)  →  F_pred (B, 3)"""
        return self.net(torch.cat([z_v_repr, z_p_repr], dim=-1))


# ── Combined model ─────────────────────────────────────────────────────────────

class EncoderTrainingModel(nn.Module):
    """Holds all encoders, prediction heads, and the learnable logit_scale.

    forward() accepts a full batch and returns a dict of representations and
    predictions for use by the loss functions.

    Output dict keys
    ----------------
    z_v_repr  : (B, 128)
    z_v_proj  : (B, 64)   — L2-normalised
    z_p_repr  : (B, 128)
    z_f_proj  : (B, 64)   — L2-normalised
    F_pred_v  : (B, 3)
    F_pred_vp : (B, 3)
    """

    def __init__(
        self,
        proprio_in_dim: int = 60,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.vision_encoder  = VisionEncoder(repr_dim=128, proj_dim=64)
        self.proprio_encoder = ProprioEncoder(in_dim=proprio_in_dim, repr_dim=128)
        self.force_encoder   = ForceEncoder(in_dim=3, proj_dim=64)
        self.g_v             = ForcePredV(repr_dim=128)
        self.g_vp            = ForcePredVP(v_repr_dim=128, p_repr_dim=128)
        # Learnable temperature; initialised to match provided temperature
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1.0 / temperature)
        )

    def forward(
        self,
        image: torch.Tensor,          # (B, 3, H, W)
        proprio_window: torch.Tensor, # (B, 60)
        f_window: torch.Tensor,       # (B, 3)
    ) -> dict:
        z_v_repr, z_v_proj_raw = self.vision_encoder(image)
        z_p_repr                = self.proprio_encoder(proprio_window)
        z_f_proj_raw            = self.force_encoder(f_window)

        # L2-normalise projections so dot product = cosine similarity
        z_v_proj = F.normalize(z_v_proj_raw, dim=-1)
        z_f_proj = F.normalize(z_f_proj_raw, dim=-1)

        F_pred_v  = self.g_v(z_v_repr)
        F_pred_vp = self.g_vp(z_v_repr, z_p_repr)

        return {
            "z_v_repr":  z_v_repr,
            "z_v_proj":  z_v_proj,
            "z_p_repr":  z_p_repr,
            "z_f_proj":  z_f_proj,
            "F_pred_v":  F_pred_v,
            "F_pred_vp": F_pred_vp,
        }
