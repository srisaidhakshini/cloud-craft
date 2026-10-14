# CloudCraft: Performance-Aware Federated Learning

A scoring and top-K client-selection layer on Flower's FedAvg. It rates each client on training time, network latency and reliability, then trains only the best K per round, with an exploration rule so slow clients never drop out for good. See [PRD.md](PRD.md).

## Layout

| Path | What |
| --- | --- |
| `cloudcraft/scoring.py` | Pure scorer + selection (EMA, min-max, score, top-K, exploration). No Flower/network. |
| `cloudcraft/strategy.py` | `PerformanceAwareStrategy` (FedAvg subclass). Covers conditions A–D, RTT/latency/failure tracking, CSV log. |
| `cloudcraft/server.py` / `client.py` | Flower server and the profile-driven client. |
| `cloudcraft/data.py`, `prepare_data.py` | Seeded IID / Dirichlet(0.5) splits, indices saved to `data/splits/`. |
| `cloudcraft/runner.py` | Runs every condition × split × seed from a YAML config, locally or against an EC2 server (`--remote-host`). |
| `analysis/analyze.py` | Plots and mean ± std table from the CSV logs. |
| `configs/` | `base.yaml` (all tunables), `experiment.yaml` (full 24-run sweep), `medium.yaml` (100 rounds, small data, 2 splits × 1 seed), `smoke.yaml` (4-round check). |
| `docker-compose.yml` | 5 clients, one profile each, pointed at `SERVER_ADDRESS`. |

## Setup

```bash
python -m venv ~/venvs/cloudcraft && source ~/venvs/cloudcraft/bin/activate   # Python 3.13; any location
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pytest tests          # scorer unit tests, no Flower needed
```

## Run

```bash
python -m cloudcraft.runner --config configs/smoke.yaml        # ~1-2 min end-to-end check
python -m cloudcraft.runner --config configs/medium.yaml       # 100-round run on reduced data
python -m cloudcraft.runner --config configs/experiment.yaml   # full 24 runs
python -m cloudcraft.runner --config configs/experiment.yaml --resume   # skip runs already complete
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

The runner can drive the whole sweep with the server on EC2 and the 5 clients in local Docker Compose:

```bash
python -m cloudcraft.runner --config configs/experiment.yaml   --remote-host <elastic-ip> --ssh-key <key.pem> --remote-dir ~/cloud-craft
```

For each run it SSHes in (as `ubuntu`) and starts the server in a `cloudcraft` Docker container on port 8080, brings up the local clients with `docker compose up`, then copies `results/<run_id>` back. The instance needs the repo and the built `cloudcraft` image under `--remote-dir`, with `data/` prepared. Open port 8080 only to your team's IPs. To do it by hand: run `python -m cloudcraft.server --condition C --split dirichlet --seed 0 --address 0.0.0.0:8080` on the instance and `SERVER_ADDRESS=<elastic-ip>:8080 docker compose up --build` locally. Set `NETEM_DELAY=800ms` to add real `tc netem` delay to client 4 (on top of the software delay in its profile). Stop the instance when no experiment is running.

## Config you will likely tune

`training.round_timeout`, `selection.k`, `selection.epsilon`, `scoring.*` (current weights: time 0.3, latency 0.2, reliability 0.5), `profiles.*`, `training.failure_mode` (`drop` = client reports failure; `timeout` = client hangs past the timeout). Open questions in the PRD (target accuracy, K, weights, exploration method) are config values, so they can be settled without code changes.

## Status

Built: scorer and tests, strategy, client profiles, splits, runner (local and EC2), analysis, Docker Compose. Not yet done: the 98% single-client check, a full-length run validating the selection behaviour, the 24-run sweep on EC2, and the report. The smoke test and unit tests were run on Flower 1.13.1; the pins have since moved to Flower 1.39.0 and Python 3.13, so re-run `pytest tests` and the smoke config after installing. Details and open questions are in [PRD.md](PRD.md).
