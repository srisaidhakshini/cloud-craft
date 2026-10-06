"""Comparison plots and a mean +/- std table from the CSV logs (FR-12).

    python analysis/analyze.py --results results --out analysis/output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {"A": "A: FedAvg (all 5)", "B": "B: Random K", "C": "C: Perf-aware", "D": "D: Perf-aware + explore"}
SPLIT_NAME = {"iid": "IID", "dirichlet": "non-IID (Dirichlet 0.5)"}


def load_runs(results: Path) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        log, meta = run_dir / "log.csv", run_dir / "meta.json"
        if not log.exists():
            continue  # not a run directory
        if not (log.exists() and meta.exists()) or not json.loads(meta.read_text()).get("complete"):
            print(f"warning: skipping incomplete run {run_dir.name}")
            continue
        df = pd.read_csv(log)
        df["split"] = json.loads(meta.read_text())["split"]
        frames.append(df)
    if not frames:
        raise SystemExit("no complete runs found")
    return pd.concat(frames, ignore_index=True)


def per_run_metrics(df: pd.DataFrame, target: float) -> pd.DataFrame:
    rows = []
    for (cond, split, seed), g in df.groupby(["condition", "split", "seed"]):
        rounds = g.drop_duplicates("round").sort_values("round")
        hit = rounds[rounds["global_accuracy"] >= target]
        sel = g.groupby("client_id")["selected"].mean()
        rows.append({
            "condition": cond, "split": split, "seed": seed,
            "time_to_target_s": hit["wall_clock"].iloc[0] if len(hit) else np.nan,
            "final_accuracy": rounds["global_accuracy"].iloc[-1],
            "mean_round_time_s": rounds["round_time"].mean(),
            "min_client_coverage": sel.min(),
        })
    return pd.DataFrame(rows)


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["time_to_target_s", "final_accuracy", "mean_round_time_s", "min_client_coverage"]
    out = []
    for (split, cond), g in runs.groupby(["split", "condition"]):
        row = {"split": split, "condition": cond, "runs": len(g),
               "target_reached": f"{g['time_to_target_s'].notna().sum()}/{len(g)}"}
        for m in metrics:
            row[m] = f"{g[m].mean():.3f} ± {g[m].std(ddof=1) if len(g) > 1 else 0.0:.3f}"
        out.append(row)
    return pd.DataFrame(out)


def plot_accuracy_vs_time(df, out: Path, target: float) -> None:
    splits = sorted(df["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    for ax, split in zip(axes[0], splits):
        for cond in sorted(df["condition"].unique()):
            g = df[(df["split"] == split) & (df["condition"] == cond)].drop_duplicates(["seed", "round"])
            runs = [r.sort_values("wall_clock") for _, r in g.groupby("seed")]
            tmax = min(r["wall_clock"].max() for r in runs)
            grid = np.linspace(0, tmax, 200)
            ys = np.array([np.interp(grid, r["wall_clock"], r["global_accuracy"]) for r in runs])
            ax.plot(grid, ys.mean(0), label=LABELS.get(cond, cond))
            if len(runs) > 1:
                ax.fill_between(grid, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0), alpha=0.15)
        ax.axhline(target, ls="--", c="grey", lw=0.8)
        ax.set(title=SPLIT_NAME.get(split, split), xlabel="wall-clock (s)", ylabel="global test accuracy")
        ax.set_ylim(max(0.0, target - 0.15), 1.0)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_time.png", dpi=150)
    plt.close(fig)


def plot_round_time(df, out: Path) -> None:
    splits = sorted(df["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    for ax, split in zip(axes[0], splits):
        for cond in sorted(df["condition"].unique()):
            g = df[(df["split"] == split) & (df["condition"] == cond)].drop_duplicates(["seed", "round"])
            m = g.groupby("round")["round_time"].mean()
            ax.plot(m.index, m.values, label=LABELS.get(cond, cond))
        ax.set(title=SPLIT_NAME.get(split, split), xlabel="round", ylabel="round time (s)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "round_time.png", dpi=150)
    plt.close(fig)


def plot_selection_heatmaps(df, out: Path) -> None:
    for (cond, split), g in df[df["condition"].isin(["C", "D"])].groupby(["condition", "split"]):
        mat = g.groupby(["client_id", "round"])["selected"].mean().unstack("round")
        fig, ax = plt.subplots(figsize=(8, 2.8))
        im = ax.imshow(mat.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(len(mat.index)), [f"client {c}" for c in mat.index])
        ax.set_xticks(range(len(mat.columns)), mat.columns)
        ax.set(xlabel="round", title=f"Selection frequency across seeds, {LABELS[cond]}, {SPLIT_NAME.get(split, split)}")
        fig.colorbar(im, ax=ax, label="share of seeds selected")
        fig.tight_layout()
        fig.savefig(out / f"selection_heatmap_{cond}_{split}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="analysis/output")
    p.add_argument("--target", type=float, default=0.97)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs(Path(args.results))
    runs = per_run_metrics(df, args.target)
    runs.to_csv(out / "per_run_metrics.csv", index=False)
    table = summary_table(runs)
    table.to_csv(out / "summary_table.csv", index=False)
    (out / "summary_table.md").write_text(table.to_markdown(index=False) if _has_tabulate() else table.to_string(index=False))
    plot_accuracy_vs_time(df, out, args.target)
    plot_round_time(df, out)
    plot_selection_heatmaps(df, out)
    print(table.to_string(index=False))
    print(f"\nWrote plots and tables to {out}")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
