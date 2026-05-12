"""Linear evaluation probes for contact and success classification.

Both probes keep all encoder parameters frozen; only the linear head is trained.

ContactProbe
    Input  : image (B, 3, H, W)
    Encoder: VisionEncoder → z_v_repr (B, 128)
    Head   : Linear(128, 2)

SuccessProbe
    Input  : image, proprio_window, f_window
    Encoders: VisionEncoder → z_v_repr (B, 128)
              ProprioEncoder → z_p_repr (B, 128)
              ForceEncoder → z_f_proj (B, 64)  [L2-normalised]
    Head   : Linear(320, 2)
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from .models import EncoderTrainingModel


# ── Encoder loading ────────────────────────────────────────────────────────────

def build_encoders(
    checkpoint_path: Optional[str],
    device: torch.device,
    proprio_in_dim: int = 60,
) -> EncoderTrainingModel:
    """Load or freshly initialise an EncoderTrainingModel with all params frozen.

    Parameters
    ----------
    checkpoint_path
        Path to a saved ``state_dict`` from :class:`EncoderTrainingModel`.
        If ``None``, a freshly initialised model is returned (baseline).
    device
        Target device.
    proprio_in_dim
        Must match the value used during encoder training (default 60).

    Returns
    -------
    EncoderTrainingModel in eval mode, all parameters frozen.
    """
    model = EncoderTrainingModel(proprio_in_dim=proprio_in_dim).to(device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


# ── Probe modules ──────────────────────────────────────────────────────────────

class ContactProbe(nn.Module):
    """Binary contact classifier from RGB alone.

    The vision encoder is shared from an :class:`EncoderTrainingModel` and
    remains frozen.  Only ``head`` is trained.
    """

    def __init__(self, encoder_model: EncoderTrainingModel):
        super().__init__()
        self.vision_encoder = encoder_model.vision_encoder  # frozen externally
        self.head = nn.Linear(128, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, 3, H, W)  →  logits (B, 2)"""
        with torch.no_grad():
            z_v_repr, _ = self.vision_encoder(image)
        return self.head(z_v_repr)


class SuccessProbe(nn.Module):
    """Binary success classifier from vision + proprio + force.

    All three encoders are shared from an :class:`EncoderTrainingModel` and
    remain frozen.  Only ``head`` is trained.
    """

    def __init__(self, encoder_model: EncoderTrainingModel):
        super().__init__()
        self.vision_encoder  = encoder_model.vision_encoder
        self.proprio_encoder = encoder_model.proprio_encoder
        self.force_encoder   = encoder_model.force_encoder
        self.head = nn.Linear(128 + 128 + 64, 2)  # 320 → 2

    def forward(
        self,
        image: torch.Tensor,           # (B, 3, H, W)
        proprio_window: torch.Tensor,  # (B, 60)
        f_window: torch.Tensor,        # (B, 3)
    ) -> torch.Tensor:
        """→  logits (B, 2)"""
        with torch.no_grad():
            z_v_repr, _    = self.vision_encoder(image)
            z_p_repr       = self.proprio_encoder(proprio_window)
            z_f_proj_raw   = self.force_encoder(f_window)
            z_f_proj       = F.normalize(z_f_proj_raw, dim=-1)
        z = torch.cat([z_v_repr, z_p_repr, z_f_proj], dim=-1)  # (B, 320)
        return self.head(z)


# ── Probe training ─────────────────────────────────────────────────────────────

def _eval_probe_metrics(
    probe: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> Tuple[float, float, float]:
    """Run inference and return (val_loss, f1, primary_metric).

    primary_metric is recall for contact, precision for success.
    """
    probe.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            if task == "contact":
                img    = batch["image"].to(device, non_blocking=True)
                labels = batch["c_window"].to(device, non_blocking=True)
                logits = probe(img)
            else:
                img    = batch["image"].to(device, non_blocking=True)
                prop   = batch["proprio_window"].to(device, non_blocking=True)
                force  = batch["f_window"].to(device, non_blocking=True)
                labels = batch["success"].to(device, non_blocking=True)
                logits = probe(img, prop, force)

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            n_batches  += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    preds  = torch.cat(all_logits,  dim=0).argmax(dim=1).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()

    f1 = float(f1_score(labels_np, preds, pos_label=1, average="binary", zero_division=0))
    if task == "contact":
        primary = float(recall_score(labels_np, preds, pos_label=1, average="binary", zero_division=0))
    else:
        primary = float(precision_score(labels_np, preds, pos_label=1, average="binary", zero_division=0))

    val_loss = total_loss / max(n_batches, 1)
    return val_loss, f1, primary


def train_probe(
    probe:       nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    task:        str,            # "contact" or "success"
    save_path:   Optional[str] = None,
    epochs:      int           = 15,
    lr:          float         = 1e-3,
    weight_decay: float        = 1e-4,
    patience:    int           = 5,
    device:      Optional[torch.device] = None,
) -> Dict:
    """Train the linear head of a ContactProbe or SuccessProbe.

    Only ``probe.head`` parameters are updated; encoder weights are frozen.

    Parameters
    ----------
    task
        ``"contact"`` or ``"success"``; determines which batch keys are read
        and which primary metric is used for checkpointing (recall / precision).
    save_path
        If provided, the best head ``state_dict`` is written here.

    Returns
    -------
    dict with keys: best_f1, best_primary, best_epoch, history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe = probe.to(device)

    optimizer = optim.Adam(
        probe.head.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_f1      = -1.0
    best_epoch   = 0
    no_improve   = 0
    best_primary = 0.0
    history: List[Dict] = []
    start = time.time()

    print(f"\n{'='*56}")
    print(f"  Probe: {task}  |  epochs={epochs}  lr={lr}  wd={weight_decay}")
    print(f"{'='*56}")

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        probe.train()
        probe.head.train()
        train_loss = 0.0
        n_train    = 0

        for batch in train_loader:
            if task == "contact":
                img    = batch["image"].to(device, non_blocking=True)
                labels = batch["c_window"].to(device, non_blocking=True)
                logits = probe(img)
            else:
                img    = batch["image"].to(device, non_blocking=True)
                prop   = batch["proprio_window"].to(device, non_blocking=True)
                force  = batch["f_window"].to(device, non_blocking=True)
                labels = batch["success"].to(device, non_blocking=True)
                logits = probe(img, prop, force)

            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train    += 1

        avg_train = train_loss / max(n_train, 1)

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, f1, primary = _eval_probe_metrics(probe, val_loader, device, task)

        elapsed = time.time() - start
        metric_name = "recall" if task == "contact" else "precision"
        print(
            f"  Ep {epoch:3d}/{epochs} | "
            f"tr_loss={avg_train:.4f}  val_loss={val_loss:.4f}  "
            f"f1={f1:.4f}  {metric_name}={primary:.4f}  "
            f"[{elapsed:.0f}s]"
        )
        history.append({
            "epoch": epoch, "train_loss": avg_train,
            "val_loss": val_loss, "f1": f1, metric_name: primary,
        })

        if f1 > best_f1:
            best_f1      = f1
            best_primary = primary
            best_epoch   = epoch
            no_improve   = 0
            if save_path is not None:
                torch.save(probe.head.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [EarlyStopping] No F1 improvement for {patience} epochs.")
                break

    if save_path is None:
        # If no path was given, keep best weights in memory
        pass

    print(f"\n  Best F1={best_f1:.4f}  {metric_name}={best_primary:.4f}  epoch={best_epoch}")
    return {
        "best_f1":      best_f1,
        "best_primary": best_primary,
        "best_epoch":   best_epoch,
        "history":      history,
    }
