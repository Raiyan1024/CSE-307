"""CSE-307 Track 2: Disk Scheduling with a Learned Scheduler Selector.

Implements FCFS, SCAN, C-SCAN, and SSTF; trains a lightweight decision-tree
classifier to select the scheduler with the lowest seek time for a request
window; evaluates on both held-out synthetic windows and a shifted timeline.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt

ALGORITHMS = ["FCFS", "SCAN", "C-SCAN", "SSTF"]
DISK_MAX = 199
DEFAULT_HEAD = 100


def fcfs(requests: np.ndarray, head: int = DEFAULT_HEAD) -> int:
    cur = head
    total = 0
    for r in requests:
        total += abs(int(r) - cur)
        cur = int(r)
    return total


def sstf(requests: np.ndarray, head: int = DEFAULT_HEAD) -> int:
    pending = [int(x) for x in requests]
    cur = head
    total = 0
    while pending:
        i = min(range(len(pending)), key=lambda k: abs(pending[k] - cur))
        nxt = pending.pop(i)
        total += abs(nxt - cur)
        cur = nxt
    return total


def scan(requests: np.ndarray, head: int = DEFAULT_HEAD, disk_max: int = DISK_MAX) -> int:
    """Elevator/SCAN moving upward first, touching the end before reversing."""
    req = sorted(int(x) for x in requests)
    up = [r for r in req if r >= head]
    down = [r for r in req if r < head][::-1]
    cur = head
    total = 0
    for r in up:
        total += abs(r - cur); cur = r
    if down:
        total += abs(disk_max - cur); cur = disk_max
        for r in down:
            total += abs(r - cur); cur = r
    return total


def cscan(requests: np.ndarray, head: int = DEFAULT_HEAD, disk_max: int = DISK_MAX) -> int:
    """Circular SCAN moving upward; wrap cost from max cylinder to zero is counted."""
    req = sorted(int(x) for x in requests)
    up = [r for r in req if r >= head]
    low = [r for r in req if r < head]
    cur = head
    total = 0
    for r in up:
        total += abs(r - cur); cur = r
    if low:
        total += abs(disk_max - cur); cur = disk_max
        total += disk_max; cur = 0
        for r in low:
            total += abs(r - cur); cur = r
    return total


ALG_FUNCS: Dict[str, Callable[[np.ndarray, int], int]] = {
    "FCFS": fcfs,
    "SCAN": scan,
    "C-SCAN": cscan,
    "SSTF": sstf,
}


def generate_window(kind: str, n: int, rng: np.random.Generator, disk_max: int = DISK_MAX) -> np.ndarray:
    """Generate one request window with a known request pattern."""
    if kind == "sequential":
        start = rng.integers(0, disk_max - 40)
        step = rng.choice([1, 2, 3, 4])
        noise = rng.integers(-2, 3, size=n)
        arr = start + np.arange(n) * step + noise
        # Reflect instead of clipping too aggressively.
        arr = np.mod(arr, disk_max + 1)
        return arr.astype(int)
    if kind == "random":
        return rng.integers(0, disk_max + 1, size=n).astype(int)
    if kind == "bursty":
        centers = rng.choice(np.arange(15, disk_max - 15), size=rng.integers(2, 5), replace=False)
        choices = rng.choice(centers, size=n)
        arr = choices + rng.normal(0, rng.uniform(3, 9), size=n)
        return np.clip(np.rint(arr), 0, disk_max).astype(int)
    if kind == "mixed":
        half = n // 2
        return np.concatenate([
            generate_window("sequential", half, rng, disk_max),
            generate_window(rng.choice(["random", "bursty"]), n - half, rng, disk_max),
        ])
    raise ValueError(f"Unknown workload kind: {kind}")


def features(requests: np.ndarray, head: int) -> Dict[str, float]:
    req = requests.astype(float)
    diffs = np.diff(req)
    absdiff = np.abs(diffs) if len(diffs) else np.array([0.0])
    return {
        "head": float(head),
        "mean": float(np.mean(req)),
        "std": float(np.std(req)),
        "min": float(np.min(req)),
        "max": float(np.max(req)),
        "range": float(np.max(req) - np.min(req)),
        "mean_abs_delta": float(np.mean(absdiff)),
        "std_abs_delta": float(np.std(absdiff)),
        "locality_fraction_delta_le_10": float(np.mean(absdiff <= 10)),
        "monotone_up_fraction": float(np.mean(diffs >= 0)) if len(diffs) else 0.0,
        "unique_ratio": float(len(np.unique(req)) / len(req)),
        "distance_head_to_mean": float(abs(head - np.mean(req))),
    }


def seek_times(requests: np.ndarray, head: int) -> Dict[str, int]:
    return {name: fn(requests, head) for name, fn in ALG_FUNCS.items()}


def build_dataset(samples_per_kind: int, n_requests: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for kind in ["sequential", "random", "bursty", "mixed"]:
        for _ in range(samples_per_kind):
            head = int(rng.integers(0, DISK_MAX + 1))
            req = generate_window(kind, n_requests, rng)
            seeks = seek_times(req, head)
            best = min(seeks, key=seeks.get)
            row = {"kind": kind, "best_scheduler": best, **features(req, head), **{f"seek_{k}": v for k, v in seeks.items()}}
            rows.append(row)
    return pd.DataFrame(rows)


def train_classifier(df: pd.DataFrame, seed: int):
    feature_cols = [c for c in df.columns if c not in {"kind", "best_scheduler"} and not c.startswith("seek_")]
    X = df[feature_cols]
    y = df["best_scheduler"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=seed)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)
    conf = prob.max(axis=1)
    correct = pred == y_test.to_numpy()
    eval_summary = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else float("nan"),
        "mean_confidence_wrong": float(conf[~correct].mean()) if (~correct).any() else float("nan"),
        "n_test": int(len(y_test)),
        "feature_cols": feature_cols,
        "classes": list(clf.classes_),
        "classification_report": classification_report(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=clf.classes_).tolist(),
        "tree": export_text(clf, feature_names=feature_cols),
    }
    return clf, feature_cols, eval_summary


def shifted_timeline(clf, feature_cols: List[str], n_windows: int, n_requests: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 999)
    # Deliberate shift: sequential/locality -> random -> bursty. This stresses the selector.
    kinds = (["sequential"] * (n_windows // 3) +
             ["random"] * (n_windows // 3) +
             ["bursty"] * (n_windows - 2 * (n_windows // 3)))
    rows = []
    for i, kind in enumerate(kinds, start=1):
        head = int(rng.integers(0, DISK_MAX + 1))
        req = generate_window(kind, n_requests, rng)
        seeks = seek_times(req, head)
        feat = features(req, head)
        X = pd.DataFrame([{k: feat[k] for k in feature_cols}])
        pred = str(clf.predict(X)[0])
        conf = float(clf.predict_proba(X).max())
        best = min(seeks, key=seeks.get)
        rows.append({
            "window": i,
            "kind": kind,
            "head": head,
            "best_scheduler": best,
            "predicted_scheduler": pred,
            "prediction_correct": pred == best,
            "confidence": conf,
            "selector_seek": seeks[pred],
            "oracle_seek": seeks[best],
            **{f"seek_{k}": v for k, v in seeks.items()},
        })
    return pd.DataFrame(rows)


def plot_results(timeline: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    totals = {alg: timeline[f"seek_{alg}"].sum() for alg in ALGORITHMS}
    totals["Learned selector"] = timeline["selector_seek"].sum()
    totals["Oracle best"] = timeline["oracle_seek"].sum()
    plt.figure(figsize=(8, 4.6))
    colors = ["#8da0cb"] * 4 + ["#66c2a5", "#fc8d62"]
    plt.bar(totals.keys(), totals.values(), color=colors)
    plt.ylabel("Total seek distance (cylinders)")
    plt.title("Total seek time on shifted workload timeline")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "total_seek_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.6))
    for alg in ["FCFS", "SCAN", "C-SCAN", "SSTF"]:
        plt.plot(timeline["window"], timeline[f"seek_{alg}"], marker="o", linewidth=1, alpha=0.65, label=alg)
    plt.plot(timeline["window"], timeline["selector_seek"], marker="s", linewidth=2.2, color="#1b9e77", label="Learned selector")
    for x in [timeline["window"].max() / 3 + 0.5, 2 * timeline["window"].max() / 3 + 0.5]:
        plt.axvline(x, color="black", linestyle="--", alpha=0.4)
    plt.xlabel("Request window")
    plt.ylabel("Seek distance")
    plt.title("Per-window seek distance before and after workload shifts")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "timeline_seek.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    correct = timeline["prediction_correct"].map({True: "correct", False: "wrong"})
    plt.scatter(timeline["window"], timeline["confidence"], c=timeline["prediction_correct"].map({True: "#1b9e77", False: "#d95f02"}), s=70)
    plt.ylim(0, 1.05)
    plt.xlabel("Request window")
    plt.ylabel("Classifier confidence")
    plt.title("Prediction confidence on shifted timeline")
    for _, r in timeline.iterrows():
        plt.text(r["window"], r["confidence"] + 0.025, correct.loc[r.name], ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "confidence_timeline.png", dpi=180)
    plt.close()


def write_summary(eval_summary: dict, timeline: pd.DataFrame, outpath: Path) -> None:
    totals = {alg: int(timeline[f"seek_{alg}"].sum()) for alg in ALGORITHMS}
    totals["Learned selector"] = int(timeline["selector_seek"].sum())
    totals["Oracle best"] = int(timeline["oracle_seek"].sum())
    lines = []
    lines.append("# Experiment Summary\n")
    lines.append(f"Held-out classifier accuracy: {eval_summary['accuracy']:.3f}\n")
    lines.append(f"Mean confidence when correct: {eval_summary['mean_confidence_correct']:.3f}\n")
    lines.append(f"Mean confidence when wrong: {eval_summary['mean_confidence_wrong']:.3f}\n")
    lines.append("\n## Total seek distance on shifted timeline\n")
    for k, v in totals.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## Decision tree\n\n```text\n")
    lines.append(eval_summary["tree"])
    lines.append("\n```\n\n## Classification report\n\n```text\n")
    lines.append(eval_summary["classification_report"])
    lines.append("\n```\n")
    outpath.write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-kind", type=int, default=350)
    ap.add_argument("--requests-per-window", type=int, default=60)
    ap.add_argument("--timeline-windows", type=int, default=18)
    ap.add_argument("--seed", type=int, default=307)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    figdir = args.out / "figures"

    df = build_dataset(args.samples_per_kind, args.requests_per_window, args.seed)
    df.to_csv(args.out / "training_dataset.csv", index=False)
    clf, feature_cols, eval_summary = train_classifier(df, args.seed)
    timeline = shifted_timeline(clf, feature_cols, args.timeline_windows, args.requests_per_window, args.seed)
    timeline.to_csv(args.out / "shifted_timeline_results.csv", index=False)
    plot_results(timeline, figdir)
    write_summary(eval_summary, timeline, args.out / "summary.md")
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
