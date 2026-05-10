"""Training loop and hyperparameter sweep for the windowed encoder model.

Adapted from train_fusion_model / run_all_fusions in mmai_hw2_edward_rivera.py,
updated for the self-supervised combined loss (contrastive + MSE regression)
with c_window gating.

Two checkpoints are saved per run:
  *_mse.pt  — best raw val_mse_vp  (primary: regression quality, λ-independent)
  *.pt      — best total val_loss   (used only for early stopping signal)

Sweep winner selection uses best_val_mse_vp so cross-run comparison is
λ-independent. ret_acc is logged as a contrastive alignment sanity check only.
"""

import time
import tracemalloc
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .models import EncoderTrainingModel
from .losses import combined_loss


# ── Retrieval accuracy (sanity check) ─────────────────────────────────────────

@torch.no_grad()
def _contrastive_retrieval_acc(
    model: EncoderTrainingModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Top-1 retrieval: for each contact z_v_proj, find nearest z_f_proj.

    Within a single pass over the val loader we accumulate all contact
    embeddings and compute a global nearest-neighbour accuracy, which is more
    meaningful than a per-batch estimate on small batches.

    Returns accuracy in [0, 1] (returns 0.0 if no contact samples found).
    """
    model.eval()
    all_zv, all_zf = [], []

    for batch in loader:
        img   = batch["image"].to(device, non_blocking=True)
        prop  = batch["proprio_window"].to(device, non_blocking=True)
        force = batch["f_window"].to(device, non_blocking=True)
        cw    = batch["c_window"].to(device, non_blocking=True)

        out = model(img, prop, force)
        mask = cw.bool()
        if mask.sum() == 0:
            continue
        all_zv.append(out["z_v_proj"][mask].cpu())
        all_zf.append(out["z_f_proj"][mask].cpu())

    if not all_zv:
        return 0.0

    zv = torch.cat(all_zv, dim=0)  # (N_c, 64)
    zf = torch.cat(all_zf, dim=0)  # (N_c, 64)

    # Cosine similarity matrix (already normalised)
    sim = zv @ zf.T   # (N_c, N_c)
    preds = sim.argmax(dim=1)
    labels = torch.arange(len(zv))
    return float((preds == labels).float().mean().item())


# ── Single-run training function ──────────────────────────────────────────────

def train_encoder_model(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    lambdas:      Tuple[float, float, float] = (0.1, 1.0, 1.0),
    lr:           float = 1e-3,
    weight_decay: float = 1e-3,
    epochs:       int   = 30,
    patience:     int   = 7,
    max_minutes:  float = 30.0,
    proprio_in_dim: int = 60,
    temperature:  float = 0.07,
    save_path:    Optional[str] = None,
    model_name:   str  = "encoder_model",
    device:       Optional[torch.device] = None,
) -> Dict:
    """Train EncoderTrainingModel for one hyperparameter configuration.

    Returns
    -------
    dict with keys:
        best_val_mse_vp  : float  (primary — raw val mse_vp at best mse_vp epoch)
        best_val_mse_v   : float  (raw val mse_v at best mse_vp epoch)
        best_mse_epoch   : int
        mse_ckpt_path    : str    (checkpoint saved at best mse_vp epoch)
        best_val_loss    : float  (total val loss; drives early stopping)
        best_val_ret_acc : float  (ret_acc at best val_loss epoch)
        best_epoch       : int    (epoch of best val_loss)
        train_time_s     : float
        peak_mem_mb      : float
        n_params         : int
        hyperparams      : dict
        history          : list of per-epoch dicts
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EncoderTrainingModel(
        proprio_in_dim=proprio_in_dim,
        temperature=temperature,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=6, factor=0.5, min_lr=1e-6
    )

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    else:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()

    n_params     = sum(p.numel() for p in model.parameters())
    n_trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  λ=({lambdas[0]}, {lambdas[1]}, {lambdas[2]})  lr={lr}  wd={weight_decay}")
    print(f"  {n_params:,} params total  |  {n_trainable:,} trainable  |  device={device}")
    print(f"{'='*60}")

    best_val_loss    = float("inf")
    best_val_ret_acc = 0.0
    best_epoch       = 0
    no_improve       = 0

    best_val_mse_vp  = float("inf")
    best_val_mse_v   = float("inf")
    best_mse_epoch   = 0

    start_time       = time.time()
    best_time_s      = 0.0
    max_seconds      = max_minutes * 60
    history: List[Dict] = []

    ckpt_path     = save_path or f"best_{model_name}.pt"
    mse_ckpt_path = ckpt_path.replace(".pt", "_mse.pt")

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_totals: Dict[str, float] = {
            "n_contact": 0, "l_cont": 0.0,
            "l_mse_v": 0.0, "l_mse_vp": 0.0, "total": 0.0,
        }
        n_batches_with_contact = 0

        for batch in train_loader:
            img   = batch["image"].to(device, non_blocking=True)
            prop  = batch["proprio_window"].to(device, non_blocking=True)
            force = batch["f_window"].to(device, non_blocking=True)
            cw    = batch["c_window"].to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out  = model(img, prop, force)
                    loss, comps = combined_loss(
                        out, force, cw, lambdas, model.logit_scale
                    )
            else:
                out  = model(img, prop, force)
                loss, comps = combined_loss(
                    out, force, cw, lambdas, model.logit_scale
                )

            if comps["n_contact"] == 0:
                continue  # skip batches with no contact samples

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            for k in train_totals:
                train_totals[k] += comps[k]
            n_batches_with_contact += 1

        n_bwc = max(n_batches_with_contact, 1)
        avg_train = {k: train_totals[k] / n_bwc for k in train_totals}

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_totals: Dict[str, float] = {
            "n_contact": 0, "l_cont": 0.0,
            "l_mse_v": 0.0, "l_mse_vp": 0.0, "total": 0.0,
        }
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                img   = batch["image"].to(device, non_blocking=True)
                prop  = batch["proprio_window"].to(device, non_blocking=True)
                force = batch["f_window"].to(device, non_blocking=True)
                cw    = batch["c_window"].to(device, non_blocking=True)

                out = model(img, prop, force)
                _, comps = combined_loss(
                    out, force, cw, lambdas, model.logit_scale
                )

                if comps["n_contact"] == 0:
                    continue
                for k in val_totals:
                    val_totals[k] += comps[k]
                n_val_batches += 1

        n_vb = max(n_val_batches, 1)
        avg_val = {k: val_totals[k] / n_vb for k in val_totals}
        scheduler.step(avg_val["total"])

        val_ret_acc = _contrastive_retrieval_acc(model, val_loader, device)

        elapsed = time.time() - start_time
        print(
            f"  Ep {epoch:3d}/{epochs} | "
            f"tr_loss={avg_train['total']:.4f} "
            f"(cont={avg_train['l_cont']:.4f} "
            f"mse_v={avg_train['l_mse_v']:.4f} "
            f"mse_vp={avg_train['l_mse_vp']:.4f}) | "
            f"val_loss={avg_val['total']:.4f} "
            f"(mse_v={avg_val['l_mse_v']:.4f} "
            f"mse_vp={avg_val['l_mse_vp']:.4f}) "
            f"ret_acc={val_ret_acc:.3f}"
        )

        row = {
            "epoch": epoch,
            "train": avg_train,
            "val":   avg_val,
            "val_ret_acc": val_ret_acc,
            "elapsed_s":   elapsed,
        }
        history.append(row)

        # ── Primary checkpoint: best raw val_mse_vp (λ-independent) ──────────
        if avg_val["l_mse_vp"] < best_val_mse_vp:
            best_val_mse_vp = avg_val["l_mse_vp"]
            best_val_mse_v  = avg_val["l_mse_v"]
            best_mse_epoch  = epoch
            torch.save(model.state_dict(), mse_ckpt_path)

        # ── Early-stopping checkpoint: best total val_loss ─────────────────
        if avg_val["total"] < best_val_loss:
            best_val_loss    = avg_val["total"]
            best_val_ret_acc = val_ret_acc
            best_epoch       = epoch
            best_time_s      = elapsed
            no_improve       = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        if elapsed > max_seconds:
            print(f"  [!] Time limit ({max_minutes:.0f} min) reached at epoch {epoch}. Stopping.")
            break
        if no_improve >= patience:
            print(f"  [EarlyStopping] No improvement for {patience} epochs.")
            break

    # ── Memory stats ──────────────────────────────────────────────────────────
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem_mb = peak / 1024**2

    print(
        f"\n  Best mse_vp:  {best_val_mse_vp:.4f}  |  "
        f"mse_v: {best_val_mse_v:.4f}  |  "
        f"epoch: {best_mse_epoch}  →  {mse_ckpt_path}"
    )
    print(
        f"  Best val_loss: {best_val_loss:.4f}  |  "
        f"ret_acc: {best_val_ret_acc:.3f}  |  "
        f"epoch: {best_epoch}  |  {best_time_s:.1f}s  |  "
        f"peak mem: {peak_mem_mb:.1f} MB"
    )

    return {
        "best_val_mse_vp":  best_val_mse_vp,
        "best_val_mse_v":   best_val_mse_v,
        "best_mse_epoch":   best_mse_epoch,
        "mse_ckpt_path":    mse_ckpt_path,
        "best_val_loss":    best_val_loss,
        "best_val_ret_acc": best_val_ret_acc,
        "best_epoch":       best_epoch,
        "train_time_s":     best_time_s,
        "peak_mem_mb":      peak_mem_mb,
        "n_params":         n_params,
        "hyperparams":      {
            "lambdas": lambdas, "lr": lr, "weight_decay": weight_decay,
        },
        "history":          history,
        "ckpt_path":        ckpt_path,
    }


# ── Hyperparameter sweep ───────────────────────────────────────────────────────

SWEEP = [
    {"lambdas": (0.1, 1.0, 1.0), "lr": 1e-3,  "wd": 1e-3},
    {"lambdas": (0.1, 1.0, 1.0), "lr": 5e-4,  "wd": 1e-4},
    {"lambdas": (0.5, 1.0, 1.0), "lr": 5e-4,  "wd": 1e-4},
    {"lambdas": (0.1, 0.5, 1.0), "lr": 5e-4,  "wd": 1e-4},
    {"lambdas": (0.1, 1.0, 0.5), "lr": 5e-4,  "wd": 1e-4},
]

# ── Sweep 2: fix λ1:λ2 = 1:10, explore λ2:λ3 ratio at fixed lr/wd ────────────
# Motivation: SWEEP confirmed λ1=0.1, lr=5e-4, wd=1e-4 as the stable base.
# λ3 is varied across {0.25, 0.5, 1.0, 2.0} to map the λ2:λ3 trade-off.
# Winner selected by best_val_mse_vp (λ-independent regression quality).
SWEEP_2 = [
    {"lambdas": (0.1, 1.0, 0.25), "lr": 5e-4, "wd": 1e-4},
    {"lambdas": (0.1, 1.0, 0.50), "lr": 5e-4, "wd": 1e-4},
    {"lambdas": (0.1, 1.0, 1.00), "lr": 5e-4, "wd": 1e-4},
    {"lambdas": (0.1, 1.0, 2.00), "lr": 5e-4, "wd": 1e-4},
]


def run_lambda_sweep(
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    sweep:          List[Dict] = SWEEP,
    epochs:         int   = 30,
    patience:       int   = 10,
    max_minutes:    float = 30.0,
    proprio_in_dim: int   = 60,
    output_dir:     str   = ".",
    device:         Optional[torch.device] = None,
) -> Dict[str, Dict]:
    """Run train_encoder_model for each entry in sweep; return best results.

    Selection criterion: lowest raw val_mse_vp (λ-independent regression quality).
    ret_acc is shown as a contrastive alignment sanity check only.

    Returns
    -------
    dict mapping a human-readable run name → metrics dict from train_encoder_model.
    Also prints a summary table.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results: Dict[str, Dict] = {}
    best_run_name   = None
    best_mse_vp     = float("inf")

    for i, hp in enumerate(sweep):
        lambdas = hp["lambdas"]
        lr      = hp["lr"]
        wd      = hp["wd"]
        name    = (
            f"run{i:02d}_l{lambdas[0]:.2f}-{lambdas[1]:.2f}-{lambdas[2]:.2f}"
            f"_lr{lr}_wd{wd}"
        )
        ckpt = f"{output_dir}/best_{name}.pt"

        metrics = train_encoder_model(
            train_loader   = train_loader,
            val_loader     = val_loader,
            lambdas        = lambdas,
            lr             = lr,
            weight_decay   = wd,
            epochs         = epochs,
            patience       = patience,
            max_minutes    = max_minutes,
            proprio_in_dim = proprio_in_dim,
            save_path      = ckpt,
            model_name     = name,
            device         = device,
        )
        all_results[name] = metrics

        if metrics["best_val_mse_vp"] < best_mse_vp:
            best_mse_vp   = metrics["best_val_mse_vp"]
            best_run_name = name

    # ── Summary table ──────────────────────────────────────────────────────────
    col_w = 52
    print(f"\n{'='*100}")
    print(
        f"{'Run':<{col_w}} {'MseVp':>8} {'MseV':>8} {'MseEp':>6} "
        f"{'ValLoss':>9} {'RetAcc':>7} {'Time(s)':>8} {'Params':>10}"
    )
    print(f"{'-'*100}")
    for name, res in all_results.items():
        marker = " *" if name == best_run_name else ""
        print(
            f"{name + marker:<{col_w}} "
            f"{res['best_val_mse_vp']:>8.4f} "
            f"{res['best_val_mse_v']:>8.4f} "
            f"{res['best_mse_epoch']:>6d} "
            f"{res['best_val_loss']:>9.4f} "
            f"{res['best_val_ret_acc']:>7.3f} "
            f"{res['train_time_s']:>8.1f} "
            f"{res['n_params']:>10,}"
        )
    print(f"{'='*100}")
    print(f"Best run: {best_run_name}  (val_mse_vp={best_mse_vp:.4f})")

    return all_results
