# CloudCraft: Performance-Aware Federated Learning

A scoring and top-K client-selection layer on Flower's FedAvg. It rates each client on training time, network latency and reliability, then trains only the best K per round, with an exploration rule so slow clients never drop out for good. See [PRD.md](PRD.md).

## Layout

| Path | What |
| --- | --- |
| `cloudcraft/scoring.py` | Pure scorer + selection (EMA, min-max, score, top-K, exploration). No Flower/network. |
| `cloudcraft/strategy.py` | `PerformanceAwareStrategy` (FedAvg subclass). Covers conditions A–D, RTT/latency/failure tracking, CSV log. |
| `cloudcraft/server.py` / `client.py` | Flower server and the profile-driven client. |
| `cloudcraft/data.py`, `prepare_data.py` | Seeded IID / Dirichlet(0.5) splits, indices saved to `data/splits/`. |
| `cloudcraft/runner.py` | Runs every condition × split × seed from a YAML config. |
| `analysis/analyze.py` | Plots and mean ± std table from the CSV logs. |
| `configs/` | `base.yaml` (all tunables), `experiment.yaml` (full 24-run sweep), `smoke.yaml` (fast check). |
| `docker-compose.yml` | 5 clients, one profile each, pointed at `SERVER_ADDRESS`. |

## Setup

```bash
python -m venv ~/venvs/cloudcraft && source ~/venvs/cloudcraft/bin/activate   # any location
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pytest tests          # scorer unit tests, no Flower needed
```

## Run

```bash
python -m cloudcraft.runner --config configs/smoke.yaml        # ~1-2 min end-to-end check
python -m cloudcraft.runner --config configs/experiment.yaml   # full 24 runs, resumable
python analysis/analyze.py --results results --out analysis/output
```

Each run writes `results/<condition>_<split>_seed<n>/log.csv` (one row per client per round, columns per FR-10, plus `round_time`) and `meta.json`. A run whose row count is short is marked incomplete and ignored by the analysis.

Notes on the log: `score` is the client's score after that round (the one used to pick the next round); `wall_clock` is training time since round 1 started, excluding server-side evaluation; unselected clients have blank `train_time`/`rtt`/`latency`/`success`.

## Conditions

| | mode | |
| --- | --- | --- |
| A | `all` | FedAvg, all 5 clients |
| B | `random` | random K = 3 |
| C | `perf` | round 1 all, then top-K by score |
| D | `perf_explore` | C + exploration (`epsilon` swap; optional `staleness_bonus`) |

## Running on EC2

Run one server per run on the instance: `python -m cloudcraft.server --condition C --split dirichlet --seed 0 --address 0.0.0.0:8080`, then start the clients elsewhere with `SERVER_ADDRESS=<elastic-ip>:8080 docker compose up --build`. Open port 8080 only to your team's IPs. Set `NETEM_DELAY=800ms` to add real `tc netem` delay to client 4 (on top of the software delay in its profile). `runner.py` currently drives local runs only; scripting remote runs is not built yet.

## Config you will likely tune

`training.round_timeout`, `selection.k`, `selection.epsilon`, `scoring.*`, `profiles.*`, `training.failure_mode` (`drop` = client reports failure; `timeout` = client hangs past the timeout). Open questions in the PRD (target accuracy, K, weights, exploration method) are config values, so they can be settled without code changes.
