"""Results dashboard and EC2 control panel for the CloudCraft experiments.

    streamlit run analysis/dashboard.py
"""
from __future__ import annotations

import io
import json
import shlex
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

DEFAULT_KEY = ""  # paste the full path to your .pem file in the sidebar


# ------------------------------------------------------------------ data
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
        parts = run_dir.name.split("_")
        if not log.exists() or len(parts) != 3:
            continue
        cond, split, seed = parts
        info = json.loads(meta.read_text()) if meta.exists() else {}
        complete = bool(info.get("complete"))
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


#ssh
def sync(host: str, key: str, remote_dir: str, local: Path):
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", "-i", key, "-o", "StrictHostKeyChecking=accept-new", "-r",
           f"ubuntu@{host}:{remote_dir}/{local.name}", str(local.parent)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900)


def ssh(host: str, key: str, cmd: str, timeout: int = 30):
    return subprocess.run(
        ["ssh", "-i", key, "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
         f"ubuntu@{host}", cmd], capture_output=True, text=True, timeout=timeout)


def via_server(server_ip: str, cmd: str) -> str:
    """Run a command on the server instance by hopping through the client instance."""
    return f"ssh -o StrictHostKeyChecking=accept-new ubuntu@{server_ip} {shlex.quote(cmd)}"


def out_text(r) -> str:
    return (r.stdout + r.stderr).strip() or "(no output)"


#plots
def curves(df: pd.DataFrame, x: str, target: float):
    splits = sorted(df["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(4.5 * len(splits), 3), squeeze=False)
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
    fig, axes = plt.subplots(1, len(splits), figsize=(4.5 * len(splits), 3), squeeze=False)
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
    fig, ax = plt.subplots(figsize=(7, 2.4))
    im = ax.imshow(mat.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(mat.index)), [f"client {c}" for c in mat.index])
    ax.set_xticks(range(len(mat.columns)), mat.columns)
    ax.set(xlabel="round", title=f"{LABELS[cond]}, {SPLIT_NAME.get(split, split)}")
    fig.colorbar(im, ax=ax, label="share of seeds selected")
    fig.tight_layout()
    return fig


#sidebar
with st.sidebar:
    st.header("Data")
    folder = Path(st.text_input("Results folder", "results_full2"))
    target = st.slider("Target accuracy", 0.90, 0.999, 0.97, 0.001, format="%.3f")
    st.caption("Choose the target from the baseline (A) before looking at B, C and D.")
    warmup = st.number_input("Warm-up rounds (selection table skips them)", 1, 10, 2)

    st.header("EC2 connection")
    host = st.text_input("Client instance public IP")
    key = st.text_input("Key file", DEFAULT_KEY)
    remote_dir = st.text_input("Remote repo folder", "~/cloud-craft")
    server_ip = st.text_input("Server private IP", "172.31.22.232")
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

#control panel
with st.expander("EC2 control panel", expanded=True):
    st.caption("Start both instances in the AWS console first. Uses the IP, key and server IP from the sidebar.")
    c1, c2, c3 = st.columns(3)
    conds = c1.multiselect("Conditions", list("ABCD"), list("ABCD"))
    splits = c2.multiselect("Splits", ["iid", "dirichlet"], ["iid", "dirichlet"])
    seeds = " ".join(s for s in c3.text_input("Seeds", "0 1 2").split() if s.isdigit())
    cfg_path = st.text_input("Config on the instance", "configs/experiment.yaml")
    resume = st.checkbox("Skip runs that are already complete (--resume)")
    ready = bool(host and conds and splits and seeds)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Check connection", disabled=not host):
        r = ssh(host, key, "hostname; tmux ls 2>&1; echo containers: $(docker ps -q | wc -l); df -h / | tail -1")
        st.code(out_text(r), language=None)

    if b2.button("Start sweep", disabled=not ready):
        run_args = (f"--config {cfg_path} --results-dir {folder.name} --remote-host {server_ip} "
                    f"--ssh-key ~/.ssh/id_ed25519 --conditions {' '.join(conds)} "
                    f"--splits {' '.join(splits)} --seeds {seeds}" + (" --resume" if resume else ""))
        inner = f"python3 -m cloudcraft.runner {run_args}; echo SWEEP FINISHED; sleep 86400"
        r = ssh(host, key, f"cd ~/cloud-craft && tmux new-session -d -s sweep {shlex.quote(inner)}")
        st.code(out_text(r) if r.returncode else "Started in tmux session 'sweep'.", language=None)

    if b3.button("Stop sweep", disabled=not host):
        ssh(host, key, "tmux kill-session -t sweep; cd ~/cloud-craft && docker compose down", 120)
        ssh(host, key, via_server(server_ip, "docker stop $(docker ps -q)"))
        st.warning("Stopped the sweep and the containers.")

    if b4.button("Open native terminal", disabled=not host or sys.platform != "win32"):
        subprocess.Popen(
            ["ssh", "-t", "-i", key, "-o", "StrictHostKeyChecking=accept-new",
             f"ubuntu@{host}", "tmux attach -t sweep || exec bash -l"],
            creationflags=subprocess.CREATE_NEW_CONSOLE)

    live = st.toggle("Live terminal (refreshes every 5 s)")

    @st.fragment(run_every=5)
    def live_terminal():
        if host and live:
            r = ssh(host, key, "tmux capture-pane -p -t sweep -S -60 2>&1 | tail -n 40", 15)
            st.code(out_text(r), language=None)

    live_terminal()

    if st.button("Show the run in progress", disabled=not host):
        cmd = via_server(server_ip, "d=$(ls -t ~/cloud-craft/results | head -1); echo $d; "
                                    "cat ~/cloud-craft/results/$d/log.csv")
        r = ssh(host, key, cmd, 60)
        name, _, body = r.stdout.partition("\n")
        try:
            live_df = pd.read_csv(io.StringIO(body))
            st.write(f"**{name}**: round {int(live_df['round'].max())}, "
                     f"accuracy {live_df['global_accuracy'].iloc[-1]:.4f}")
            st.line_chart(live_df.drop_duplicates("round").set_index("round")["global_accuracy"])
            st.dataframe(live_df[live_df["round"] == live_df["round"].max()], hide_index=True)
        except Exception:
            st.code(out_text(r), language=None)

#results
if not folder.exists():
    st.info(f"No folder named '{folder}' yet. Sync from the client instance, or fix the folder name in the sidebar.")
    st.stop()

data, status = load(str(folder), signature(folder))
if status.empty:
    st.info("The folder has no runs yet. Finished runs appear after the runner copies them back.")
    st.stop()

done = int((status["status"] == "complete").sum())
m1, m2, m3 = st.columns(3)
m1.metric("Runs found", len(status))
m2.metric("Complete", done)
m3.metric("Incomplete or failed", len(status) - done)

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
        "averages only those runs. Coverage includes the warm-up rounds, when every client is selected."
    )
    with st.expander("Per-run metrics"):
        st.dataframe(runs, hide_index=True, use_container_width=True)

with tab_acc:
    st.pyplot(curves(data, "wall_clock", target), use_container_width=False)
    st.pyplot(curves(data, "round", target), use_container_width=False)

with tab_time:
    st.pyplot(round_times(data), use_container_width=False)

with tab_sel:
    late = (data[data["round"] > warmup].groupby(["split", "condition", "client_id"])["selected"]
            .mean().unstack("client_id").round(2))
    st.subheader(f"How often each client is selected, after the {warmup} warm-up rounds")
    st.dataframe(late, use_container_width=True)
    perf = sorted(set(data["condition"]) & {"C", "D"})
    if perf:
        a, b = st.columns(2)
        cond = a.selectbox("Condition", perf)
        split = b.selectbox("Split", sorted(data["split"].unique()))
        st.pyplot(heatmap(data, cond, split), use_container_width=False)

with tab_runs:
    st.dataframe(status, hide_index=True, use_container_width=True)
    pick = st.selectbox("Open a run", list(status["run"]))
    row = status[status["run"] == pick].iloc[0]
    g = data[(data["condition"] == row["condition"]) & (data["split"] == row["split"])
             & (data["seed"] == row["seed"])]
    if not g.empty:
        per_round = g.drop_duplicates("round").set_index("round")
        st.line_chart(per_round[["global_accuracy"]])
        st.line_chart(per_round[["round_time"]])
        st.dataframe(g.pivot(index="round", columns="client_id", values="selected"))
    for log_name in ("server.log", "clients.log"):
        if st.button(f"Load {log_name} from the instance", key=f"{pick}-{log_name}", disabled=not host):
            r = ssh(host, key, f"tail -n 150 ~/cloud-craft/{folder.name}/{pick}/{log_name}")
            st.code(out_text(r), language=None)