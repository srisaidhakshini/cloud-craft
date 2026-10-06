"""Flower server entrypoint: one run = one condition x split x seed.

Run on EC2 (or locally). Writes <run-dir>/log.csv (FR-10) and meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters
from torch.utils.data import DataLoader, Subset

from cloudcraft import data as data_mod
from cloudcraft import model as model_mod
from cloudcraft.config import deep_update, load_config
from cloudcraft.scoring import ScoringConfig
from cloudcraft.strategy import PerformanceAwareStrategy


def build_strategy(cfg, condition: str, split: str, seed: int, run_dir: Path, data_dir: str):
    cond = deep_update(cfg["selection"], cfg["conditions"][condition])
    num_clients = len(cfg["clients"])
    run_id = f"{condition}_{split}_seed{seed}"

    test_set = data_mod.load_mnist(data_dir, train=False)
    n_test = cfg.get("data", {}).get("test_samples")
    if n_test:
        test_set = Subset(test_set, list(range(int(n_test))))
    test_loader = DataLoader(test_set, batch_size=1000)
    eval_model = model_mod.Net()

    def evaluate_fn(server_round, weights, _config):
        model_mod.set_weights(eval_model, weights)
        loss, acc = model_mod.evaluate(eval_model, test_loader)
        return loss, {"accuracy": acc}

    tr = cfg["training"]
    initial = ndarrays_to_parameters(model_mod.get_weights(model_mod.init_model(seed)))
    return PerformanceAwareStrategy(
        run_id=run_id,
        condition=condition,
        seed=seed,
        mode=cond["mode"],
        num_clients=num_clients,
        k=int(cond["k"]),
        epsilon=float(cond.get("epsilon", 0.0)),
        staleness_bonus=float(cond.get("staleness_bonus", 0.0)),
        scoring=ScoringConfig(**cfg["scoring"]),
        fit_config={
            "local_epochs": tr["local_epochs"], "batch_size": tr["batch_size"],
            "lr": tr["lr"], "momentum": tr["momentum"],
            "failure_mode": tr.get("failure_mode", "drop"), "round_timeout": tr["round_timeout"],
        },
        log_path=str(run_dir / "log.csv"),
        initial_parameters=initial,
        evaluate_fn=evaluate_fn,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="CloudCraft federated server")
    env = os.environ.get
    p.add_argument("--config", default=env("CONFIG", "configs/base.yaml"))
    p.add_argument("--condition", default=env("CONDITION", "C"))
    p.add_argument("--split", default=env("SPLIT", "iid"))
    p.add_argument("--seed", type=int, default=int(env("SEED", "0")))
    p.add_argument("--data-dir", default=env("DATA_DIR", "data"))
    p.add_argument("--run-dir", default=None)
    p.add_argument("--address", default=env("BIND_ADDRESS", "0.0.0.0:8080"))
    args = p.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir or f"results/{args.condition}_{args.split}_seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    strategy = build_strategy(cfg, args.condition, args.split, args.seed, run_dir, args.data_dir)
    rounds = int(cfg["training"]["rounds"])
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(
            num_rounds=rounds, round_timeout=float(cfg["training"]["round_timeout"])
        ),
        strategy=strategy,
    )
    strategy.close()

    expected = rounds * len(cfg["clients"])
    got = sum(1 for _ in open(run_dir / "log.csv")) - 1
    (run_dir / "meta.json").write_text(json.dumps({
        "condition": args.condition, "split": args.split, "seed": args.seed,
        "expected_rows": expected, "rows": got, "complete": got == expected,
    }))
    print(f"run finished: {got}/{expected} rows", flush=True)


if __name__ == "__main__":
    main()
