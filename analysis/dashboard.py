"""Results dashboard for the CloudCraft experiments.

    streamlit run analysis/dashboard.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import LABELS, SPLIT_NAME, per_run_metrics, summary_table  # noqa: E402

st.set_page_config(page_title="CloudCraft results", layout="wide")

DEFAULT_KEY = r"C:\Users\kinja\Documents\IMPORTANT\EC2 INSTANCE\cloudcraftAccess.pem"


def signature(folder: Path) -> tuple:
    if not folder.exists():
        return ()
    files = list(folder.glob("*/log.csv")) + list(folder.glob("*/meta.json"))
    return tuple(sorted((str(p), p.stat().st_mtime) for p in files))


@st.cache_data(show_spinner=False)
def load(folder: str, sig: tuple):
    frames, status = [], []
    for run_dir in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        log, meta = run_dir / "log.csv", run_dir / "meta.json"
        if not log.exists():
            continue
        info = json.loads(meta.read_text()) if meta.exists() else {}
        complete = bool(info.get("complete"))
        cond, split, seed = run_dir.name.split("_")
        df = pd.read_csv(log)
        status.append({
            "run": run_dir.name, "condition": cond, "split": split,
            "seed": int(seed.replace("seed", "")), "rows": info.get("rows", len(df)),
            "expected_rows": info.get("expected_rows"),
            "status": "complete" if complete else "incomplete",
        })
        if complete:
            df["split"] = split
            frames.append(df)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return data, pd.DataFrame(status)


def sync(host: str, key: str, remote_dir: str, local: Path):
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", "-i", key, "-o", "StrictHostKeyChecking=accept-new", "-r",
           f"ubuntu@{host}:{remote_dir}/{local.name}", str(local.parent)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900)


def curves(df: pd.DataFrame, x: str, target: float):
    splits = sorted(df["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    for ax, split in zip(axes[0], splits):
        for cond in sorted(df["condition"].unique()):
            g = df[(df["split"] == split) & (df["condition"] == cond)].drop_duplicates(["seed", "round"])
            if g.empty:
                continue
            if x == "round":
                m = g.groupby("round")["global_accuracy"].agg(["mean", "std"]).fillna(0)
                ax.plot(m.index, m["mean"], label=LABELS.get(cond, cond))
                if g["seed"].nunique() > 1:
                    ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.15)
            else:
                runs = [r.sort_values("wall_clock") for _, r in g.groupby("seed")]
                grid = np.linspace(0, min(r["wall_clock"].max() for r in runs), 200)
                ys = np.array([np.interp(grid, r["wall_clock"], r["global_accuracy"]) for r in runs])
                ax.plot(grid, ys.mean(0), label=LABELS.get(cond, cond))
                if len(runs) > 1:
                    ax.fill_between(grid, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0), alpha=0.15)
        ax.axhline(target, ls="--", c="grey", lw=0.8)
        ax.set(title=SPLIT_NAME.get(split, split), ylabel="global test accuracy",
               xlabel="round" if x == "round" else "wall-clock (s)")
        ax.set_ylim(max(0.0, target - 0.15), 1.0)
        ax.legend()
    fig.tight_layout()
    return fig


def round_times(df: pd.DataFrame):
    splits = sorted(df["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    for ax, split in zip(axes[0], splits):
        for cond in sorted(df["condition"].unique()):
            g = df[(df["split"] == split) & (df["condition"] == cond)].drop_duplicates(["seed", "round"])
            if not g.empty:
                m = g.groupby("round")["round_time"].mean()
                ax.plot(m.index, m.values, label=LABELS.get(cond, cond))
        ax.set(title=SPLIT_NAME.get(split, split), xlabel="round", ylabel="round time (s)")
        ax.legend()
    fig.tight_layout()
    return fig


def heatmap(df: pd.DataFrame, cond: str, split: str):
    g = df[(df["condition"] == cond) & (df["split"] == split)]
    mat = g.groupby(["client_id", "round"])["selected"].mean().unstack("round")
    fig, ax = plt.subplots(figsize=(9, 2.8))
    im = ax.imshow(mat.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(mat.index)), [f"client {c}" for c in mat.index])
    ax.set_xticks(range(len(mat.columns)), mat.columns)
    ax.set(xlabel="round", title=f"{LABELS[cond]}, {SPLIT_NAME.get(split, split)}")
    fig.colorbar(im, ax=ax, label="share of seeds selected")
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("Data")
    folder = Path(st.text_input("Results folder", "results_full1"))
    target = st.slider("Target accuracy", 0.90, 0.999, 0.97, 0.001, format="%.3f")
    st.caption("Choose the target from the baseline (A) before looking at B, C and D.")

    st.header("Sync from EC2")
    host = st.text_input("Client instance public IP")
    key = st.text_input("Key file", DEFAULT_KEY)
    remote_dir = st.text_input("Remote repo folder", "~/cloud-craft")
    if st.button("Sync now", disabled=not host):
        with st.spinner("Copying results from the client instance..."):
            try:
                out = sync(host, key, remote_dir, folder)
                if out.returncode == 0:
                    st.success("Synced.")
                else:
                    st.error(out.stderr.strip() or "scp failed")
            except Exception as exc:
                st.error(str(exc))
    if st.button("Reload files"):
        st.cache_data.clear()

st.title("CloudCraft: performance-aware federated learning")

if not folder.exists():
    st.info(f"No folder named '{folder}' yet. Sync from the client instance, or fix the folder name in the sidebar.")
    st.stop()

data, status = load(str(folder), signature(folder))
if status.empty:
    st.info("The folder has no runs yet. Finished runs appear after the runner copies them back.")
    st.stop()

done = int((status["status"] == "complete").sum())
c1, c2, c3 = st.columns(3)
c1.metric("Runs found", len(status))
c2.metric("Complete", done)
c3.metric("Incomplete or failed", len(status) - done)

if data.empty:
    st.warning("No complete runs yet, so there is nothing to analyse.")
    st.dataframe(status, hide_index=True)
    st.stop()

tab_sum, tab_acc, tab_time, tab_sel, tab_runs = st.tabs(
    ["Summary", "Accuracy", "Round time", "Selection", "Runs"])

with tab_sum:
    runs = per_run_metrics(data, target)
    st.dataframe(summary_table(runs), hide_index=True, use_container_width=True)
    st.caption(
        "Values are mean ± std across seeds. A std of 0.000 with one seed means it was not "
        "measured. 'target_reached' counts runs that reached the target; the time to target "
        "averages only those runs. Coverage includes round 1, when every client is selected."
    )
    with st.expander("Per-run metrics"):
        st.dataframe(runs, hide_index=True, use_container_width=True)

with tab_acc:
    st.pyplot(curves(data, "wall_clock", target))
    st.pyplot(curves(data, "round", target))

with tab_time:
    st.pyplot(round_times(data))

with tab_sel:
    late = (data[data["round"] >= 2].groupby(["split", "condition", "client_id"])["selected"]
            .mean().unstack("client_id").round(2))
    st.subheader("How often each client is selected, from round 2 on")
    st.dataframe(late, use_container_width=True)
    perf = sorted(set(data["condition"]) & {"C", "D"})
    if perf:
        a, b = st.columns(2)
        cond = a.selectbox("Condition", perf)
        split = b.selectbox("Split", sorted(data["split"].unique()))
        st.pyplot(heatmap(data, cond, split))

with tab_runs:
    st.dataframe(status, hide_index=True, use_container_width=True)