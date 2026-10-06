"""Client behaviour profiles (FR-2). One client codebase, behaviour from PROFILE."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Profile:
    train_delay: Tuple[float, float] = (0.0, 0.0)  # extra seconds added to local training
    net_delay: Tuple[float, float] = (0.0, 0.0)  # seconds added after training, outside train_time
    failure_prob: float = 0.0

    def sample_train_delay(self, rng: random.Random) -> float:
        return rng.uniform(*self.train_delay)

    def sample_net_delay(self, rng: random.Random) -> float:
        return rng.uniform(*self.net_delay)

    def should_fail(self, rng: random.Random) -> bool:
        return rng.random() < self.failure_prob


DEFAULT_PROFILES: Dict[str, Profile] = {
    "fast": Profile(),
    "medium": Profile(train_delay=(0.5, 1.0)),
    "slow": Profile(train_delay=(2.0, 4.0)),
    "high_latency": Profile(net_delay=(1.0, 2.0)),
    "unreliable": Profile(failure_prob=0.3),
}


def profile_from_config(name: str, overrides: Dict[str, dict] | None = None) -> Profile:
    if name not in DEFAULT_PROFILES:
        raise ValueError(f"unknown profile '{name}'; choose from {sorted(DEFAULT_PROFILES)}")
    base = DEFAULT_PROFILES[name]
    o = (overrides or {}).get(name)
    if not o:
        return base
    return Profile(
        train_delay=tuple(o.get("train_delay", base.train_delay)),
        net_delay=tuple(o.get("net_delay", base.net_delay)),
        failure_prob=o.get("failure_prob", base.failure_prob),
    )


def sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
