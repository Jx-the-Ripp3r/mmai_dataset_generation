"""Debug diagnostics: contact ratio, force plots, histograms, summary."""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------
# Contact ratio
# ------------------------------------------------------------------

def check_contact_ratio(dataset_dir: str, threshold: float = 0.5) -> float:
    contact_frames = 0
    total_frames = 0

    for ep_dir in sorted(glob.glob(os.path.join(dataset_dir, "episode_*"))):
        force = np.load(os.path.join(ep_dir, "force.npy"))
        mag = np.linalg.norm(force[:, :3], axis=1)
        contact_frames += int(np.sum(mag > threshold))
        total_frames += len(mag)

    ratio = contact_frames / max(total_frames, 1)
    print(f"Contact ratio: {ratio:.3f}  "
          f"({contact_frames}/{total_frames} frames above {threshold} N)")
    return ratio


# ------------------------------------------------------------------
# Force-vs-time plot (sample of episodes)
# ------------------------------------------------------------------

def plot_force_vs_time(
    dataset_dir: str,
    output_dir: str,
    n_episodes: int = 5,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ep_dirs = sorted(glob.glob(os.path.join(dataset_dir, "episode_*")))
    if not ep_dirs:
        return

    indices = np.linspace(0, len(ep_dirs) - 1, min(n_episodes, len(ep_dirs)),
                          dtype=int)

    fig, axes = plt.subplots(len(indices), 1,
                             figsize=(8, 2.5 * len(indices)),
                             squeeze=False)

    for row, idx in enumerate(indices):
        ep_dir = ep_dirs[idx]
        force = np.load(os.path.join(ep_dir, "force.npy"))
        mag = np.linalg.norm(force[:, :3], axis=1)

        with open(os.path.join(ep_dir, "metadata.json")) as f:
            meta = json.load(f)

        ax = axes[row, 0]
        ax.plot(mag, linewidth=1.0)
        ax.set_ylabel("|F| (N)")
        label = "success" if meta.get("success") else "fail"
        ax.set_title(f"Episode {meta.get('episode_id', idx)}  [{label}]",
                     fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Timestep")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "force_vs_time.png"), dpi=120)
    plt.close(fig)
    print(f"  Saved force_vs_time.png")


# ------------------------------------------------------------------
# Force histogram
# ------------------------------------------------------------------

def plot_force_histogram(dataset_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    all_mags: list[float] = []

    for ep_dir in sorted(glob.glob(os.path.join(dataset_dir, "episode_*"))):
        force = np.load(os.path.join(ep_dir, "force.npy"))
        mag = np.linalg.norm(force[:, :3], axis=1)
        all_mags.extend(mag.tolist())

    if not all_mags:
        return

    arr = np.array(all_mags)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=80, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("EE Contact Force Magnitude (N)")
    ax.set_ylabel("Frame Count")
    ax.set_title("Distribution of Per-Frame EE Force Magnitudes")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1,
               label="threshold = 0.5 N")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "force_histogram.png"), dpi=120)
    plt.close(fig)
    print(f"  Saved force_histogram.png")


# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------

def print_summary(dataset_dir: str) -> None:
    ep_dirs = sorted(glob.glob(os.path.join(dataset_dir, "episode_*")))
    if not ep_dirs:
        print("No episodes found.")
        return

    successes = 0
    contact_ratios: list[float] = []
    max_forces: list[float] = []

    for ep_dir in ep_dirs:
        with open(os.path.join(ep_dir, "metadata.json")) as f:
            meta = json.load(f)
        successes += int(meta.get("success", 0))
        contact_ratios.append(meta.get("contact_ratio", 0.0))
        max_forces.append(meta.get("max_contact_force", 0.0))

    n = len(ep_dirs)
    print("=" * 50)
    print(f"  Episodes        : {n}")
    print(f"  Success rate    : {successes}/{n}  ({100*successes/n:.1f}%)")
    print(f"  Mean contact %  : {100*np.mean(contact_ratios):.1f}%")
    print(f"  Max force (mean): {np.mean(max_forces):.2f} N")
    print(f"  Max force (max) : {np.max(max_forces):.2f} N")
    print("=" * 50)


# ------------------------------------------------------------------
# Run all
# ------------------------------------------------------------------

def run_all(dataset_dir: str, diagnostics_dir: str,
            threshold: float = 0.5) -> None:
    print("\n--- Diagnostics ---")
    check_contact_ratio(dataset_dir, threshold)
    plot_force_vs_time(dataset_dir, diagnostics_dir)
    plot_force_histogram(dataset_dir, diagnostics_dir)
    print_summary(dataset_dir)
    print(f"Plots saved to {diagnostics_dir}\n")
