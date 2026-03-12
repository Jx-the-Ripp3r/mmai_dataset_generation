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
# Force-vs-time plot (all episodes, 2-column grid)
# ------------------------------------------------------------------

def plot_force_vs_time(
    dataset_dir: str,
    output_dir: str,
    n_episodes: int | None = None,
) -> None:
    """Plot force components + magnitude for every episode in a 2-column grid.

    Pass n_episodes=None (default) to include all episodes found on disk.
    Pass an integer to cap at that many (evenly sampled).
    """
    os.makedirs(output_dir, exist_ok=True)
    ep_dirs = sorted(glob.glob(os.path.join(dataset_dir, "episode_*")))
    if not ep_dirs:
        return

    if n_episodes is None:
        indices = list(range(len(ep_dirs)))
    else:
        indices = list(
            np.linspace(0, len(ep_dirs) - 1, min(n_episodes, len(ep_dirs)),
                        dtype=int)
        )

    n_cols = 2
    n_rows = (len(indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12, 3.0 * n_rows),
        squeeze=False,
    )

    component_styles = [
        ("Fx", "tab:blue",   0),
        ("Fy", "tab:orange", 1),
        ("Fz", "tab:green",  2),
    ]

    for plot_pos, idx in enumerate(indices):
        row, col = divmod(plot_pos, n_cols)
        ep_dir = ep_dirs[idx]
        force = np.load(os.path.join(ep_dir, "force.npy"))
        mag = np.linalg.norm(force[:, :3], axis=1)

        with open(os.path.join(ep_dir, "metadata.json")) as f:
            meta = json.load(f)

        ax = axes[row, col]
        success = bool(meta.get("success"))
        is_hard = bool(meta.get("is_hard", False))
        ep_id = meta.get("episode_id", idx)
        peg_off = meta.get("peg_offset", [0.0, 0.0])
        off_mm = [round(v * 1000, 1) for v in peg_off]

        # Subtle background tint: red for fail, green for success
        bg_color = "#fff0f0" if not success else "#f0fff0"
        ax.set_facecolor(bg_color)

        # Plot per-component forces as thin lines
        for label, color, ci in component_styles:
            ax.plot(force[:, ci], linewidth=0.8, color=color,
                    alpha=0.6, label=label)

        # Plot magnitude as a bold black line
        ax.plot(mag, linewidth=1.4, color="black", label="|F|")

        ax.set_ylabel("|F| (N)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

        outcome = "SUCCESS" if success else "FAIL"
        difficulty = "HARD" if is_hard else "easy"
        ax.set_title(
            f"Ep {ep_id}  [{outcome}]  {difficulty}  "
            f"offset=({off_mm[0]},{off_mm[1]}) mm",
            fontsize=8,
            color="darkred" if not success else "darkgreen",
        )

        if plot_pos == 0:
            ax.legend(fontsize=6, loc="upper right", ncol=4)

    # Hide unused subplots; label x-axis on the last filled subplot in each column
    for plot_pos in range(len(indices), n_rows * n_cols):
        row, col = divmod(plot_pos, n_cols)
        axes[row, col].set_visible(False)

    for col in range(n_cols):
        # Find the last row in this column that has a plot
        filled_rows = [
            plot_pos // n_cols
            for plot_pos in range(col, len(indices), n_cols)
        ]
        if filled_rows:
            axes[filled_rows[-1], col].set_xlabel("Timestep", fontsize=7)

    fig.suptitle(
        "Force vs Time — All Episodes  (red bg = fail, green bg = success)",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "force_vs_time.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved force_vs_time.png  ({len(indices)} episodes)")


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
    plot_force_vs_time(dataset_dir, diagnostics_dir, n_episodes=None)
    plot_force_histogram(dataset_dir, diagnostics_dir)
    print_summary(dataset_dir)
    print(f"Plots saved to {diagnostics_dir}\n")
