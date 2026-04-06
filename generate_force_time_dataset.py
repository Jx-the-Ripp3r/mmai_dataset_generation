"""Generate force-time VQA datasets from the existing episode dataset.

Creates two datasets:
  force-time-dataset-train/  — 21 entries (7 success, 7 wall jam, 7 rim jam)
  force-time-dataset-test/   — 4 entries  (2 success, 1 wall jam, 1 rim jam)

Each dataset contains:
  images/N.jpg   — force-vs-time plot for one episode (no labels leaked)
  data.jsonl     — VQA annotations

Usage:
    python generate_force_time_dataset.py [--seed SEED]
"""

import argparse
import glob
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTION = "What type of interaction is it: rim jam, wall jam, or success?"

TRAIN_COUNTS = {"success": 7, "wall jam": 7, "rim jam": 7}
TEST_COUNTS  = {"success": 2, "wall jam": 1, "rim jam": 1}

COMPONENT_STYLES = [
    ("Fx", "tab:blue",   0),
    ("Fy", "tab:orange", 1),
    ("Fz", "tab:green",  2),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_episode(meta: dict) -> str:
    """Return 'success', 'wall jam', or 'rim jam' for an episode metadata dict."""
    if bool(meta.get("success")):
        return "success"
    if bool(meta.get("angular_jam", False)):
        return "wall jam"
    return "rim jam"


def load_episodes(dataset_dir: str) -> dict[str, list[str]]:
    """Return {label: [ep_dir, ...]} bucketed by classification."""
    ep_dirs = sorted(glob.glob(os.path.join(dataset_dir, "episode_*")))
    if not ep_dirs:
        sys.exit(f"ERROR: No episode_* folders found in '{dataset_dir}'.")

    buckets: dict[str, list[str]] = {"success": [], "wall jam": [], "rim jam": []}
    for ep_dir in ep_dirs:
        meta_path = os.path.join(ep_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        label = classify_episode(meta)
        buckets[label].append(ep_dir)

    return buckets


def check_availability(buckets: dict[str, list[str]]) -> None:
    """Error out if there aren't enough episodes for both splits combined."""
    needed = {label: TRAIN_COUNTS[label] + TEST_COUNTS[label]
              for label in TRAIN_COUNTS}
    short = {
        label: (needed[label], len(buckets[label]))
        for label in needed
        if len(buckets[label]) < needed[label]
    }
    if short:
        lines = [f"  {label}: need {need}, have {have}"
                 for label, (need, have) in short.items()]
        sys.exit("ERROR: Not enough episodes in existing dataset:\n" + "\n".join(lines))


def plot_force_episode(force: np.ndarray, out_path: str) -> None:
    """Save a single-episode force-vs-time JPEG (white background, no title)."""
    mag = np.linalg.norm(force[:, :3], axis=1)

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for label, color, ci in COMPONENT_STYLES:
        ax.plot(force[:, ci], linewidth=0.9, color=color, alpha=0.7, label=label)

    ax.plot(mag, linewidth=1.5, color="black", label="|F|")

    ax.set_xlabel("Timestep", fontsize=9)
    ax.set_ylabel("|F| (N)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), borderaxespad=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100, format="jpeg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_dataset(
    ep_dirs_by_label: dict[str, list[str]],
    counts: dict[str, int],
    out_dir: str,
) -> None:
    """Generate images and data.jsonl for one split."""
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    entries = []
    img_idx = 1

    for label, n in counts.items():
        for ep_dir in ep_dirs_by_label[label][:n]:
            force_path = os.path.join(ep_dir, "force.npy")
            force = np.load(force_path)

            img_filename = f"{img_idx}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            plot_force_episode(force, img_path)

            entries.append({
                "image": f"images/{img_filename}",
                "question": QUESTION,
                "answer": label,
            })
            img_idx += 1

    random.shuffle(entries)

    jsonl_path = os.path.join(out_dir, "data.jsonl")
    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"  Saved {len(entries)} entries -> {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate force-time VQA datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_root, "dataset")

    if not os.path.isdir(dataset_dir):
        sys.exit(f"ERROR: dataset directory not found at '{dataset_dir}'.\n"
                 "Run generate_dataset.py first.")

    print("Loading episodes...")
    buckets = load_episodes(dataset_dir)

    for label, eps in buckets.items():
        print(f"  {label}: {len(eps)} episodes")

    check_availability(buckets)

    # Shuffle each bucket, then split train / test (non-overlapping)
    shuffled: dict[str, list[str]] = {}
    for label, eps in buckets.items():
        eps_copy = list(eps)
        random.shuffle(eps_copy)
        shuffled[label] = eps_copy

    train_eps = {label: shuffled[label][:TRAIN_COUNTS[label]]
                 for label in TRAIN_COUNTS}
    test_eps  = {label: shuffled[label][TRAIN_COUNTS[label]:
                                        TRAIN_COUNTS[label] + TEST_COUNTS[label]]
                 for label in TEST_COUNTS}

    train_dir = os.path.join(project_root, "force-time-dataset-train")
    test_dir  = os.path.join(project_root, "force-time-dataset-test")

    print("\nBuilding force-time-dataset-train ...")
    build_dataset(train_eps, TRAIN_COUNTS, train_dir)

    print("Building force-time-dataset-test ...")
    build_dataset(test_eps, TEST_COUNTS, test_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
