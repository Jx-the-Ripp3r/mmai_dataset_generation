"""Run probe inference on each noise-sweep level and produce degradation curves.

Reads the four trained head checkpoints and the saved probe_meta.json from
``--probes``, then iterates over every ``level_<m>`` directory under
``--noise_sweep``.  For each level it builds the appropriate probe datasets,
runs inference with both the aligned and baseline probes, and records:

    Contact task : recall  (contact = positive)
                   F1
    Success task : precision  (success = positive)
                   F1

Two figures are saved:
    <output>/contact_degradation.png
    <output>/success_degradation.png

A CSV with every data-point is saved to:
    <output>/metrics.csv  (columns: level, task, model, metric, value)

Usage
-----
    python evaluate_probes.py \\
        --probes      probes/ \\
        --noise_sweep dataset_noise_sweep \\
        --checkpoint  best_run00_l0.10-1.00-1.00_lr0.0005_wd0.0001_mse.pt \\
        --output      figures/

The ``--checkpoint`` argument must point to the same aligned-model checkpoint
that was used when running train_probes.py (it can also be read from
``probes/probe_meta.json`` automatically if the flag is omitted).
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from training.data import (
    ContactProbeDataset,
    SuccessProbeDataset,
    collect_episodes,
    collect_noise_sweep_levels,
)
from training.probes import ContactProbe, SuccessProbe, build_encoders


# ── Inference helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def _run_contact_inference(
    probe: ContactProbe,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    probe.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        img    = batch["image"].to(device, non_blocking=True)
        labels = batch["c_window"]
        logits = probe(img)
        preds  = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


@torch.no_grad()
def _run_success_inference(
    probe: SuccessProbe,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    probe.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        img    = batch["image"].to(device, non_blocking=True)
        prop   = batch["proprio_window"].to(device, non_blocking=True)
        force  = batch["f_window"].to(device, non_blocking=True)
        labels = batch["success"]
        logits = probe(img, prop, force)
        preds  = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


def _compute_contact_metrics(
    preds: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    return {
        "recall": float(recall_score(labels, preds, pos_label=1, average="binary", zero_division=0)),
        "f1":     float(f1_score(   labels, preds, pos_label=1, average="binary", zero_division=0)),
    }


def _compute_success_metrics(
    preds: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    return {
        "precision": float(precision_score(labels, preds, pos_label=1, average="binary", zero_division=0)),
        "f1":        float(f1_score(        labels, preds, pos_label=1, average="binary", zero_division=0)),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

_METRIC_COLORS = {
    "recall":    "#1f77b4",
    "f1":        "#ff7f0e",
    "precision": "#2ca02c",
}

_MODEL_STYLES = {
    "aligned":  "-",
    "baseline": "--",
}


def _plot_degradation(
    levels: List[float],
    curves: Dict[str, Dict[str, List[float]]],
    task: str,
    metrics: List[str],
    output_path: str,
) -> None:
    """Render a degradation curve figure.

    Parameters
    ----------
    levels
        Sorted noise multipliers (x-axis).
    curves
        ``{model_name: {metric_name: [value_per_level, ...]}}``
    task
        ``"contact"`` or ``"success"`` (used for title / y-label).
    metrics
        Which metric names to plot (keys in the inner dict).
    output_path
        Path to write the PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, metric_dict in curves.items():
        linestyle = _MODEL_STYLES.get(model_name, "-")
        for metric_name in metrics:
            vals  = metric_dict.get(metric_name, [])
            color = _METRIC_COLORS.get(metric_name, "black")
            label = f"{metric_name} ({model_name})"
            ax.plot(
                levels, vals,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                marker="o",
                markersize=4,
                label=label,
            )

    ax.set_xlabel("Noise multiplier (m)", fontsize=12)
    ax.set_ylabel("Metric value", fontsize=12)
    ax.set_title(f"{task.capitalize()} probe — degradation vs noise", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(levels)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)

    # Compact legend annotations
    model_lines = [
        plt.Line2D([0], [0], color="gray", linestyle="-",  linewidth=2, label="aligned  (solid)"),
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="baseline (dotted)"),
    ]
    handles, labels_text = ax.get_legend_handles_labels()
    ax.legend(
        handles + model_lines,
        labels_text + ["aligned (solid)", "baseline (dotted)"],
        loc="lower left",
        fontsize=9,
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained probes over noise-sweep levels"
    )
    parser.add_argument(
        "--probes", required=True,
        help="Directory containing head_*.pt files and probe_meta.json",
    )
    parser.add_argument(
        "--noise_sweep", required=True,
        help="Root directory of noise-sweep splits (output of generate_noise_sweep.py)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Aligned encoder checkpoint path (overrides probe_meta.json if given)",
    )
    parser.add_argument("--output",  default="figures")
    parser.add_argument("--batch",   type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load probe meta ───────────────────────────────────────────────────────
    meta_path = os.path.join(args.probes, "probe_meta.json")
    with open(meta_path) as fh:
        probe_meta = json.load(fh)

    checkpoint_path = args.checkpoint or probe_meta["checkpoint"]
    mean_p = np.array(probe_meta["mean_p"], dtype=np.float32)
    std_p  = np.array(probe_meta["std_p"],  dtype=np.float32)
    stats  = {"mean_p": mean_p, "std_p": std_p}

    print(f"Aligned checkpoint : {checkpoint_path}")

    # ── Build frozen encoders ─────────────────────────────────────────────────
    aligned_model  = build_encoders(checkpoint_path, device)
    baseline_model = build_encoders(None,            device)

    # ── Load probe heads ──────────────────────────────────────────────────────
    def _load_contact_probe(enc_model, model_name: str) -> ContactProbe:
        probe = ContactProbe(enc_model).to(device)
        head_path = os.path.join(args.probes, f"head_contact_{model_name}.pt")
        probe.head.load_state_dict(torch.load(head_path, map_location=device))
        probe.eval()
        return probe

    def _load_success_probe(enc_model, model_name: str) -> SuccessProbe:
        probe = SuccessProbe(enc_model).to(device)
        head_path = os.path.join(args.probes, f"head_success_{model_name}.pt")
        probe.head.load_state_dict(torch.load(head_path, map_location=device))
        probe.eval()
        return probe

    contact_aligned  = _load_contact_probe(aligned_model,  "aligned")
    contact_baseline = _load_contact_probe(baseline_model, "baseline")
    success_aligned  = _load_success_probe(aligned_model,  "aligned")
    success_baseline = _load_success_probe(baseline_model, "baseline")

    # ── Discover noise-sweep levels ───────────────────────────────────────────
    levels_eps = collect_noise_sweep_levels(args.noise_sweep)
    if not levels_eps:
        raise RuntimeError(
            f"No noise-sweep levels found under '{args.noise_sweep}'. "
            "Run generate_noise_sweep.py first."
        )

    sorted_levels = sorted(levels_eps.keys())
    print(f"\nNoise levels found: {sorted_levels}")

    pin = torch.cuda.is_available()

    # Accumulate results: {model: {metric: [value_per_level]}}
    contact_curves: Dict[str, Dict[str, List[float]]] = {
        "aligned":  {"recall": [], "f1": []},
        "baseline": {"recall": [], "f1": []},
    }
    success_curves: Dict[str, Dict[str, List[float]]] = {
        "aligned":  {"precision": [], "f1": []},
        "baseline": {"precision": [], "f1": []},
    }

    csv_rows: List[Dict] = []

    for m in sorted_levels:
        eps = levels_eps[m]
        print(f"\n  Level m={m:g} — {len(eps)} episodes")

        contact_ds = ContactProbeDataset(eps, stats)
        success_ds = SuccessProbeDataset(eps, stats)

        contact_loader = DataLoader(
            contact_ds, batch_size=args.batch, shuffle=False,
            num_workers=0, pin_memory=pin,
        )
        success_loader = DataLoader(
            success_ds, batch_size=args.batch, shuffle=False,
            num_workers=0, pin_memory=pin,
        )

        # ── Contact inference ──────────────────────────────────────────────────
        for model_name, probe in [("aligned", contact_aligned), ("baseline", contact_baseline)]:
            preds, labels = _run_contact_inference(probe, contact_loader, device)
            met = _compute_contact_metrics(preds, labels)
            contact_curves[model_name]["recall"].append(met["recall"])
            contact_curves[model_name]["f1"].append(met["f1"])
            for metric_name, val in met.items():
                csv_rows.append({
                    "level": m, "task": "contact",
                    "model": model_name, "metric": metric_name, "value": val,
                })
            print(
                f"    contact/{model_name:8s}: recall={met['recall']:.4f}  f1={met['f1']:.4f}"
                f"  (n={len(labels)}, pos={labels.sum()})"
            )

        # ── Success inference ──────────────────────────────────────────────────
        for model_name, probe in [("aligned", success_aligned), ("baseline", success_baseline)]:
            preds, labels = _run_success_inference(probe, success_loader, device)
            met = _compute_success_metrics(preds, labels)
            success_curves[model_name]["precision"].append(met["precision"])
            success_curves[model_name]["f1"].append(met["f1"])
            for metric_name, val in met.items():
                csv_rows.append({
                    "level": m, "task": "success",
                    "model": model_name, "metric": metric_name, "value": val,
                })
            print(
                f"    success/{model_name:8s}: precision={met['precision']:.4f}  f1={met['f1']:.4f}"
                f"  (n={len(labels)}, pos={labels.sum()})"
            )

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_degradation(
        levels   = sorted_levels,
        curves   = contact_curves,
        task     = "contact",
        metrics  = ["recall", "f1"],
        output_path = os.path.join(args.output, "contact_degradation.png"),
    )
    _plot_degradation(
        levels   = sorted_levels,
        curves   = success_curves,
        task     = "success",
        metrics  = ["precision", "f1"],
        output_path = os.path.join(args.output, "success_degradation.png"),
    )

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output, "metrics.csv")
    fieldnames = ["level", "task", "model", "metric", "value"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Saved CSV    → {csv_path}")

    print(f"\nDone.  Results in '{args.output}'.")


if __name__ == "__main__":
    main()
