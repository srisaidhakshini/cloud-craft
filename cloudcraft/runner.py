"""Experiment runner (FR-11): every condition x split x seed from a YAML config.

Locally it starts the server and the 5 clients as subprocesses for each run.
Complete runs (meta.json says complete) are skipped, so an interrupted sweep
can be resumed. Use --force to redo them.

    python -m cloudcraft.runner --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from cloudcraft import data as data_mod
from cloudcraft.config import load_config


def _wait_for_port(port: int, timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def _is_complete(run_dir: Path) -> bool:
    meta = run_dir / "meta.json"
    return meta.exists() and json.loads(meta.read_text()).get("complete", False)


def run_one(cfg_path, cfg, condition, split, seed, results_dir, data_dir, port, run_timeout) -> bool:
    run_id = f"{condition}_{split}_seed{seed}"
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").unlink(missing_ok=True)
    data_mod.make_and_save_split(
        data_dir, split, seed, len(cfg["clients"]), cfg["data"].get("dirichlet_alpha", 0.5)
    )
    py = sys.executable
    server = subprocess.Popen(
        [py, "-m", "cloudcraft.server", "--config", cfg_path, "--condition", condition,
         "--split", split, "--seed", str(seed), "--data-dir", data_dir,
         "--run-dir", str(run_dir), "--address", f"127.0.0.1:{port}"],
        stdout=open(run_dir / "server.log", "w"), stderr=subprocess.STDOUT,
    )
    clients = []
    try:
        if not _wait_for_port(port):
            raise RuntimeError("server did not start listening")
        for cid, profile in cfg["clients"].items():
            clients.append(subprocess.Popen(
                [py, "-m", "cloudcraft.client", "--config", cfg_path, "--client-id", str(cid),
                 "--profile", profile, "--server-address", f"127.0.0.1:{port}",
                 "--split", split, "--seed", str(seed), "--data-dir", data_dir],
                stdout=open(run_dir / f"client{cid}.log", "w"), stderr=subprocess.STDOUT,
            ))
        server.wait(timeout=run_timeout)
    except Exception as exc:  # timeouts, startup failures
        print(f"  ! {run_id}: {exc}", flush=True)
    finally:
        for proc in [server, *clients]:
            if proc.poll() is None:
                proc.terminate()
        for proc in [server, *clients]:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return _is_complete(run_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Run all experiment conditions")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--force", action="store_true", help="redo runs that already completed")
    p.add_argument("--run-timeout", type=float, default=3600.0, help="seconds per run")
    args = p.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    combos = list(itertools.product(exp["splits"], exp["seeds"], exp["conditions"]))
    failed = []
    for i, (split, seed, cond) in enumerate(combos, 1):
        run_id = f"{cond}_{split}_seed{seed}"
        if not args.force and _is_complete(Path(args.results_dir) / run_id):
            print(f"[{i}/{len(combos)}] {run_id}: already complete, skipping", flush=True)
            continue
        print(f"[{i}/{len(combos)}] {run_id}: running", flush=True)
        t = time.time()
        ok = run_one(args.config, cfg, cond, split, seed, args.results_dir, args.data_dir,
                     int(exp.get("server_port", 8080)), args.run_timeout)
        print(f"    {'ok' if ok else 'FAILED'} in {time.time() - t:.0f}s", flush=True)
        if not ok:
            failed.append(run_id)
    if failed:
        print("Failed runs (rows missing, treated as failed):", ", ".join(failed))
        sys.exit(1)
    print("All runs complete.")


if __name__ == "__main__":
    main()
