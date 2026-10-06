"""Flower client. One codebase; behaviour comes from PROFILE (FR-2).

Reports local training time and sample count in its fit metrics (FR-3).
Settings come from CLI flags, falling back to environment variables so the
same image works under Docker Compose.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import flwr as fl
import torch
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from torch.utils.data import DataLoader, Subset

from cloudcraft import data as data_mod
from cloudcraft import model as model_mod
from cloudcraft.config import load_config
from cloudcraft.profiles import Profile, profile_from_config, sleep

_OK = Status(code=Code.OK, message="OK")


class ProfiledClient(fl.client.Client):
    def __init__(self, client_id: str, profile_name: str, profile: Profile, dataset, indices, seed: int):
        self.client_id = str(client_id)
        self.profile_name = profile_name
        self.profile = profile
        self.dataset = dataset
        self.indices = indices
        self.seed = seed
        self.rng = random.Random(seed * 1000 + int(client_id))  # profile randomness
        self.model = model_mod.Net()

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=_OK, properties={"client_id": self.client_id, "profile": self.profile_name}
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status=_OK, parameters=ndarrays_to_parameters(model_mod.get_weights(self.model))
        )

    def fit(self, ins: FitIns) -> FitRes:
        cfg = ins.config
        rnd = int(cfg.get("server_round", 0))
        if self.profile.should_fail(self.rng):
            return self._simulated_failure(cfg)

        model_mod.set_weights(self.model, parameters_to_ndarrays(ins.parameters))
        gen = torch.Generator().manual_seed(self.seed * 100000 + int(self.client_id) * 1000 + rnd)
        loader = DataLoader(
            Subset(self.dataset, self.indices),
            batch_size=int(cfg.get("batch_size", 64)),
            shuffle=True,
            generator=gen,
        )
        start = time.perf_counter()
        model_mod.train(
            self.model, loader, int(cfg.get("local_epochs", 1)),
            float(cfg.get("lr", 0.01)), float(cfg.get("momentum", 0.9)),
        )
        sleep(self.profile.sample_train_delay(self.rng))  # simulated slow compute
        train_time = time.perf_counter() - start
        sleep(self.profile.sample_net_delay(self.rng))  # simulated network delay (not train time)

        return FitRes(
            status=_OK,
            parameters=ndarrays_to_parameters(model_mod.get_weights(self.model)),
            num_examples=len(self.indices),
            metrics={
                "client_id": self.client_id,
                "train_time": float(train_time),
                "num_samples": len(self.indices),
            },
        )

    def _simulated_failure(self, cfg) -> FitRes:
        if cfg.get("failure_mode", "drop") == "timeout":
            sleep(float(cfg.get("round_timeout", 30)) + 1.0)  # outlast the server's timeout
        return FitRes(
            status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="simulated client failure"),
            parameters=ndarrays_to_parameters([]),
            num_examples=0,
            metrics={},
        )


def main() -> None:
    p = argparse.ArgumentParser(description="CloudCraft federated client")
    env = os.environ.get
    p.add_argument("--client-id", default=env("CLIENT_ID", "1"))
    p.add_argument("--profile", default=env("PROFILE", "fast"))
    p.add_argument("--server-address", default=env("SERVER_ADDRESS", "127.0.0.1:8080"))
    p.add_argument("--split", default=env("SPLIT", "iid"))
    p.add_argument("--seed", type=int, default=int(env("SEED", "0")))
    p.add_argument("--data-dir", default=env("DATA_DIR", "data"))
    p.add_argument("--config", default=env("CONFIG", "configs/base.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    profile = profile_from_config(args.profile, cfg.get("profiles"))
    indices = data_mod.load_split_indices(args.data_dir, args.split, args.seed, args.client_id)
    cap = cfg.get("data", {}).get("samples_per_client")
    if cap:
        indices = indices[: int(cap)]
    dataset = data_mod.load_mnist(args.data_dir, train=True)

    client = ProfiledClient(args.client_id, args.profile, profile, dataset, indices, args.seed)
    fl.client.start_client(server_address=args.server_address, client=client, insecure=True)


if __name__ == "__main__":
    main()
