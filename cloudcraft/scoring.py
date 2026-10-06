"""Pure scoring and selection logic (no Flower, no network).

Stats in, scores out. See PRD "Scoring and selection specification".
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class ClientStats:
    """EMA-smoothed signals for one client. None = never measured."""

    train_time: Optional[float] = None
    latency: Optional[float] = None
    reliability: Optional[float] = None
    rounds_since_selected: int = 0


@dataclass(frozen=True)
class ScoringConfig:
    w_time: float = 0.4
    w_latency: float = 0.3
    w_reliability: float = 0.3
    ema_beta: float = 0.3

    def __post_init__(self) -> None:
        total = self.w_time + self.w_latency + self.w_reliability
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scoring weights must sum to 1, got {total}")
        if not 0.0 < self.ema_beta <= 1.0:
            raise ValueError("ema_beta must be in (0, 1]")


def ema(previous: Optional[float], observation: float, beta: float) -> float:
    """x_bar_t = beta * x_t + (1 - beta) * x_bar_{t-1}; first value seeds the EMA."""
    if previous is None:
        return observation
    return beta * observation + (1.0 - beta) * previous


def update_stats(
    stats: ClientStats,
    success: bool,
    train_time: Optional[float],
    latency: Optional[float],
    beta: float,
) -> ClientStats:
    """Return new stats after one round in which the client was asked to train.

    A failed/timed-out client gets success=0; its train time and latency are
    left untouched (PRD selection rule 4).
    """
    new = ClientStats(
        train_time=stats.train_time,
        latency=stats.latency,
        reliability=ema(stats.reliability, 1.0 if success else 0.0, beta),
        rounds_since_selected=0,
    )
    if success:
        if train_time is not None:
            new.train_time = ema(stats.train_time, train_time, beta)
        if latency is not None:
            new.latency = ema(stats.latency, max(latency, 0.0), beta)
    return new


def _minmax(values: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Min-max normalise across clients. Unmeasured -> 1.0 (worst)."""
    known = [v for v in values.values() if v is not None]
    if not known:
        return {k: 1.0 for k in values}
    lo, hi = min(known), max(known)
    out: Dict[str, float] = {}
    for k, v in values.items():
        if v is None:
            out[k] = 1.0
        elif hi - lo < 1e-12:
            out[k] = 0.0
        else:
            out[k] = (v - lo) / (hi - lo)
    return out


def score_clients(stats: Dict[str, ClientStats], cfg: ScoringConfig) -> Dict[str, float]:
    """S_i = w_T (1 - T_hat) + w_L (1 - L_hat) + w_R R_i, in [0, 1]."""
    t_hat = _minmax({k: s.train_time for k, s in stats.items()})
    l_hat = _minmax({k: s.latency for k, s in stats.items()})
    scores: Dict[str, float] = {}
    for k, s in stats.items():
        r = 0.0 if s.reliability is None else s.reliability
        scores[k] = (
            cfg.w_time * (1.0 - t_hat[k])
            + cfg.w_latency * (1.0 - l_hat[k])
            + cfg.w_reliability * r
        )
    return scores


def select_top_k(
    scores: Dict[str, float],
    k: int,
    rng: random.Random,
    epsilon: float = 0.0,
    staleness: Optional[Dict[str, int]] = None,
    staleness_bonus: float = 0.0,
) -> List[str]:
    """Top-K by score with optional exploration.

    - staleness_bonus > 0: add bonus * rounds_since_selected to each score
      before ranking (alternative exploration method).
    - epsilon > 0: each selected slot is swapped, with probability epsilon,
      for a random currently-unselected client.
    Ties break by client id for determinism.
    """
    k = min(k, len(scores))
    adjusted = dict(scores)
    if staleness_bonus > 0 and staleness:
        for cid in adjusted:
            adjusted[cid] += staleness_bonus * staleness.get(cid, 0)
    ranked = sorted(adjusted, key=lambda c: (-adjusted[c], c))
    chosen = ranked[:k]
    if epsilon > 0:
        pool = [c for c in ranked if c not in chosen]
        for i in range(len(chosen)):
            if pool and rng.random() < epsilon:
                j = rng.randrange(len(pool))
                chosen[i], pool[j] = pool[j], chosen[i]
    return sorted(chosen)


def select_random_k(client_ids: Sequence[str], k: int, rng: random.Random) -> List[str]:
    ids = sorted(client_ids)
    return sorted(rng.sample(ids, min(k, len(ids))))
