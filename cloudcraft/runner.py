"""Experiment runner (FR-11): every condition x split x seed from a YAML config.

Locally it starts the server and the 5 clients as subprocesses for each run.
Every run executes each time; pass --resume to skip runs whose meta.json says complete.

    python -m cloudcraft.runner --config configs/experiment.yaml
"""
from __future__ import annotations

import os
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


def _wait_for_port(port: int, timeout: float = 60.0, host: str = "127.0.0.1") -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
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

def run_one_remote(cfg_path, condition, split, seed, results_dir, host, key, remote_dir, run_timeout) -> bool:
    run_id = f"{condition}_{split}_seed{seed}"
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").unlink(missing_ok=True)
    port = 8080
    ssh = ["ssh", "-i", key, "-o", "StrictHostKeyChecking=accept-new", f"ubuntu@{host}"]
    server_cmd = (
        f"cd {remote_dir} && sudo rm -rf results/{run_id} && "
        f"docker run --rm -p {port}:{port} -v $PWD/data:/app/data -v $PWD/results:/app/results cloudcraft "
        f"python -m cloudcraft.server --config {cfg_path} --condition {condition} "
        f"--split {split} --seed {seed} --address 0.0.0.0:{port}"
    )
    server = subprocess.Popen(ssh + [server_cmd], stdout=open(run_dir / "server.log", "w"), stderr=subprocess.STDOUT)
    env = {**os.environ, "SERVER_ADDRESS": f"{host}:{port}", "CONFIG": cfg_path, "SPLIT": split, "SEED": str(seed)}
    try:
        if not _wait_for_port(port, timeout=120, host=host):
            raise RuntimeError("remote server did not start listening")
        subprocess.run(["docker", "compose", "up"], env=env, timeout=run_timeout,
                       stdout=open(run_dir / "clients.log", "w"), stderr=subprocess.STDOUT)
        server.wait(timeout=120)
    except Exception as exc:
        print(f"  ! {run_id}: {exc}", flush=True)
    finally:
        subprocess.run(["docker", "compose", "down"], env=env)
        if server.poll() is None:
            server.terminate()
        subprocess.run(["scp", "-i", key, "-r", f"ubuntu@{host}:{remote_dir}/results/{run_id}", str(Path(results_dir))])
    return _is_complete(run_dir)

def main() -> None:
    p = argparse.ArgumentParser(description="Run all experiment conditions")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--resume", action="store_true", help="skip runs that already completed")
    p.add_argument("--run-timeout", type=float, default=3600.0, help="seconds per run")
    p.add_argument("--remote-host", default=None, help="EC2 address; run the server there, clients in local Docker")
    p.add_argument("--ssh-key", default=None)
    p.add_argument("--remote-dir", default="~/cloud-craft")
    p.add_argument("--conditions",nargs="+",default=None)
    p.add_argument("--splits",nargs="+",default= None)
    p.add_argument("--seeds",type=int,nargs="+",default=None)

    args = p.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    combos = list(itertools.product(
        args.splits or exp["splits"], args.seeds or exp["seeds"], args.conditions or exp["conditions"]))
    failed = []
    for i, (split, seed, cond) in enumerate(combos, 1):
        run_id = f"{cond}_{split}_seed{seed}"
        if args.resume and _is_complete(Path(args.results_dir) / run_id):
            print(f"[{i}/{len(combos)}] {run_id}: already complete, skipping", flush=True)
            continue
        print(f"[{i}/{len(combos)}] {run_id}: running", flush=True)
        t = time.time()
        if args.remote_host:
            ok = run_one_remote(args.config, cond, split, seed, args.results_dir, 
                                args.remote_host, args.ssh_key, args.remote_dir, args.run_timeout)
        else:
            ok = run_one(args.config, cfg, cond, split, seed, args.results_dir, args.data_dir,
                         int(exp.get("server_port", 8080)), args.run_timeout)
        if not ok:
            failed.append(run_id)
    if failed:
        print("Failed runs (rows missing, treated as failed):", ", ".join(failed))
        sys.exit(1)
    print("All runs complete.")


if __name__ == "__main__":
    main()
