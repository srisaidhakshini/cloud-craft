"""Download MNIST and write the seeded split indices to disk (FR-1).

    python -m cloudcraft.prepare_data --split dirichlet --seed 0
"""
from __future__ import annotations

import argparse

from cloudcraft import data as data_mod
from cloudcraft.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split", nargs="+", default=["iid", "dirichlet"])
    p.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    cfg = load_config(args.config)
    for split in args.split:
        for seed in args.seed:
            path = data_mod.make_and_save_split(
                args.data_dir, split, seed, len(cfg["clients"]), cfg["data"].get("dirichlet_alpha", 0.5)
            )
            print("wrote", path)


if __name__ == "__main__":
    main()
