"""Performance-aware FedAvg strategy (FR-4..FR-10).

One class covers all four experiment conditions via `mode`:
  all           A  FedAvg, every client every round
  random        B  random K each round
  perf          C  round 1 all, then top-K by score
  perf_explore  D  C plus exploration (epsilon swap or staleness bonus)
"""
from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetPropertiesIns,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from cloudcraft.scoring import (
    ClientStats,
    ScoringConfig,
    score_clients,
    select_random_k,
    select_top_k,
    update_stats,
)

MODES = ("all", "random", "perf", "perf_explore")

CSV_COLUMNS = [
    "run_id", "condition", "seed", "round", "client_id", "selected", "train_time",
    "rtt", "latency", "success", "score", "global_accuracy", "global_loss",
    "wall_clock", "round_time",
]


class TimedProxy(ClientProxy):
    """Wraps a ClientProxy to measure round-trip time and capture failures (FR-4, FR-5)."""

    def __init__(self, inner: ClientProxy, client_id: str, outcomes: Dict[str, dict]):
        super().__init__(inner.cid)
        self._inner = inner
        self._client_id = client_id
        self._outcomes = outcomes
        self.properties = inner.properties

    def get_properties(self, ins, timeout, group_id):
        return self._inner.get_properties(ins, timeout, group_id)

    def get_parameters(self, ins, timeout, group_id):
        return self._inner.get_parameters(ins, timeout, group_id)

    def evaluate(self, ins, timeout, group_id):
        return self._inner.evaluate(ins, timeout, group_id)

    def reconnect(self, ins, timeout, group_id):
        return self._inner.reconnect(ins, timeout, group_id)

    def fit(self, ins, timeout, group_id):
        start = time.perf_counter()
        try:
            res = self._inner.fit(ins, timeout, group_id)
        except BaseException:
            self._outcomes[self._client_id] = {"rtt": time.perf_counter() - start, "ok": False}
            raise
        self._outcomes[self._client_id] = {
            "rtt": time.perf_counter() - start,
            "ok": res.status.code == Code.OK,
        }
        return res


class PerformanceAwareStrategy(FedAvg):
    def __init__(
        self,
        *,
        run_id: str,
        condition: str,
        seed: int,
        mode: str,
        num_clients: int,
        k: int,
        scoring: ScoringConfig,
        epsilon: float = 0.0,
        staleness_bonus: float = 0.0,
        fit_config: Optional[Dict[str, Scalar]] = None,
        log_path: str,
        connect_timeout: float = 300.0,
        **fedavg_kwargs,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=1,
            min_available_clients=num_clients,
            accept_failures=True,
            **fedavg_kwargs,
        )
        self.run_id, self.condition, self.seed = run_id, condition, seed
        self.mode, self.k, self.scoring = mode, k, scoring
        self.epsilon, self.staleness_bonus = epsilon, staleness_bonus
        self.fit_config = dict(fit_config or {})
        self.connect_timeout = connect_timeout
        self.rng = random.Random(seed)
        self.stats: Dict[str, ClientStats] = {}
        self.cid_to_client: Dict[str, str] = {}
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.log_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()

        self._t0: Optional[float] = None
        self._eval_total = 0.0
        self._round_start = 0.0
        self._prev_wall = 0.0
        self._selected: List[str] = []
        self._outcomes: Dict[str, dict] = {}
        self._pending: Dict[str, dict] = {}
        self._wall = 0.0
        self._round_time = 0.0

    # ----- selection -----------------------------------------------------
    def _identify(self, proxies: Dict[str, ClientProxy], server_round: int) -> Dict[str, ClientProxy]:
        """Map each connected proxy to its stable client_id via get_properties."""
        by_client: Dict[str, ClientProxy] = {}
        for cid, proxy in proxies.items():
            if cid not in self.cid_to_client:
                props = proxy.get_properties(GetPropertiesIns(config={}), 30.0, server_round).properties
                self.cid_to_client[cid] = str(props["client_id"])
            by_client[self.cid_to_client[cid]] = proxy
        return by_client

    def choose_clients(self, server_round: int, client_ids: List[str]) -> List[str]:
        for cid in client_ids:
            self.stats.setdefault(cid, ClientStats())
        if self.mode == "all":
            return sorted(client_ids)
        if self.mode == "random":
            return select_random_k(client_ids, self.k, self.rng)
        if server_round == 1:
            return sorted(client_ids)
        scores = score_clients(self.stats, self.scoring)
        explore = self.mode == "perf_explore"
        return select_top_k(
            scores,
            self.k,
            self.rng,
            epsilon=self.epsilon if explore else 0.0,
            staleness={c: s.rounds_since_selected for c, s in self.stats.items()},
            staleness_bonus=self.staleness_bonus if explore else 0.0,
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        if not client_manager.wait_for(self.min_available_clients, timeout=self.connect_timeout):
            raise RuntimeError(
                f"only {client_manager.num_available()} of {self.min_available_clients} clients connected"
            )
        by_client = self._identify(dict(client_manager.all()), server_round)
        self._selected = self.choose_clients(server_round, list(by_client))
        self._outcomes = {}
        cfg = dict(self.fit_config, server_round=server_round)
        if self._t0 is None:
            self._t0 = time.perf_counter()
        self._round_start = time.perf_counter()
        ins = FitIns(parameters, cfg)
        return [(TimedProxy(by_client[c], c, self._outcomes), ins) for c in self._selected]

    # ----- aggregation + stats -------------------------------------------
    def aggregate_fit(self, server_round, results, failures):
        self._pending = {}
        ok_metrics = {
            str(res.metrics.get("client_id")): res.metrics for _, res in results
        }
        for cid in self._selected:
            out = self._outcomes.get(cid, {"rtt": None, "ok": False})
            ok = bool(out["ok"]) and cid in ok_metrics
            train_time = float(ok_metrics[cid]["train_time"]) if ok else None
            rtt = out["rtt"]
            latency = (rtt - train_time) if ok and rtt is not None else None
            self.stats[cid] = update_stats(
                self.stats[cid], ok, train_time, latency, self.scoring.ema_beta
            )
            self._pending[cid] = {
                "selected": 1, "train_time": train_time, "rtt": rtt,
                "latency": latency, "success": int(ok),
            }
        for cid, s in self.stats.items():
            if cid not in self._selected:
                s.rounds_since_selected += 1

        now = time.perf_counter()
        self._wall = now - (self._t0 or now) - self._eval_total
        self._round_time = now - self._round_start
        return super().aggregate_fit(server_round, results, failures)

    # ----- evaluation + logging ------------------------------------------
    def evaluate(self, server_round: int, parameters: Parameters):
        start = time.perf_counter()
        result = super().evaluate(server_round, parameters)
        self._eval_total += time.perf_counter() - start
        if server_round == 0 or result is None:
            return result
        loss, metrics = result
        self._write_round(server_round, float(loss), float(metrics.get("accuracy", float("nan"))))
        return result

    def _write_round(self, server_round: int, loss: float, acc: float) -> None:
        scores = score_clients(self.stats, self.scoring)
        for cid in sorted(self.stats):
            row = {
                "selected": 0, "train_time": "", "rtt": "", "latency": "", "success": "",
                **self._pending.get(cid, {}),
            }
            row = {k: ("" if v is None else v) for k, v in row.items()}
            self._writer.writerow({
                "run_id": self.run_id, "condition": self.condition, "seed": self.seed,
                "round": server_round, "client_id": cid, **row,
                "score": round(scores[cid], 6),
                "global_accuracy": acc, "global_loss": loss,
                "wall_clock": round(self._wall, 4), "round_time": round(self._round_time, 4),
            })
        self._csv_file.flush()

    def close(self) -> None:
        self._csv_file.close()
