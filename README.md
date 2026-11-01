# CloudCraft: Performance-Aware Federated Learning

CloudCraft adds performance scoring and client selection to Flower's FedAvg. It ranks clients using training time, estimated communication latency and reliability, then selects K clients per round. Optional exploration lets lower-ranked clients re-enter training.

**Team:** Kinjal Nandi and Sri Saidhakshini V.
**Documentation updated:** November 2, 2026.

See the [product requirements](PRD.md) and [detailed project report](REPORT.md) for the design, evaluation plan, implementation status and limitations.

## Current status

Implemented: scorer, selection, Flower strategy, five client profiles, seeded MNIST splits, local/EC2 runner, CSV analysis, Streamlit dashboard and Docker Compose clients. This checkout contains no experiment results, so speedup and accuracy targets remain unverified. Earlier documentation records 15 scorer tests and a four-round smoke sweep on Flower 1.13.1; compatibility with the current dependency pins still needs validation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `cloudcraft/scoring.py` | EMA statistics, weighted scores, top-K, random selection and exploration |
| `cloudcraft/strategy.py` | FedAvg integration, warm-up, timing, failures and CSV logging |
| `cloudcraft/server.py`, `cloudcraft/client.py` | Server and profile-driven client entry points |
| `cloudcraft/model.py` | MNIST CNN, training, evaluation and parameter conversion |
| `cloudcraft/data.py`, `cloudcraft/prepare_data.py` | Seeded IID and Dirichlet splits |
| `cloudcraft/runner.py` | Local or remote experiment orchestration and resume support |
| `analysis/analyze.py` | Comparison plots and per-run/summary metrics |
| `analysis/dashboard.py` | Streamlit results viewer and SSH-based EC2 controls |
| `configs/base.yaml`, `configs/experiment.yaml` | Defaults and 30-round full sweep override |
| `tests/test_scoring.py` | 15 scorer and selector unit tests |
| `docker/`, `docker-compose.yml` | Container setup for five clients |

## Setup

The requirements declare Python 3.13 and pinned packages, including Flower 1.39.0 and PyTorch 2.6.0. Successful installation of this combination has not been established in this checkout.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pytest tests
```

On Linux/macOS, activate with `source .venv/bin/activate`. Run commands from the repository root. Initial data preparation downloads MNIST and requires network access.

## Run experiments

```bash
# One 20-round baseline as an initial integration check
python -m cloudcraft.runner --config configs/base.yaml --conditions A --splits iid --seeds 0

# Full sweep: 4 conditions x 2 splits x 3 seeds, 30 rounds per run
python -m cloudcraft.runner --config configs/experiment.yaml
python -m cloudcraft.runner --config configs/experiment.yaml --resume

# Analyse complete runs
python analysis/analyze.py --results results --out analysis/output --target 0.97

# Open the dashboard
streamlit run analysis/dashboard.py
```

The dashboard defaults to `results_full2`; change its Results folder to `results` for the runner's default output. Historical `smoke.yaml` and `medium.yaml` references have been removed because those files are absent. For shorter checks, create an override YAML beside `base.yaml`; the loader merges it with that file.

| Condition | Mode | Behaviour |
| --- | --- | --- |
| A | `all` | All five clients every round |
| B | `random` | Random K = 3 each round |
| C | `perf` | All clients in warm-up rounds 1–2, then top-K |
| D | `perf_explore` | C plus epsilon swaps; optional staleness bonus |

Current defaults: K = 3, epsilon = 0.1, EMA beta = 0.3, weights 0.3/0.2/0.5 for time/latency/reliability, timeout = 60 s and timeout-style simulated failures. Effective profile delays are Medium 3–5 s, Slow 8–12 s and High-latency 3–5 s; Unreliable fails with probability 0.3.

## Outputs and interpretation

Each run writes `results/<condition>_<split>_seed<n>/log.csv`, `meta.json` and process logs. There is one CSV row per client per round, including unselected clients. The server marks completion by checking the row count against rounds x clients; analysis includes only runs marked complete.

`score` is computed after the logged round. Unselected clients have blank timing and success fields. `wall_clock` excludes server evaluation; `round_time` measures the fit phase up to aggregation entry. Estimated latency is RTT minus reported training time, so it includes transfer and other overhead rather than measuring pure network latency.

Analysis produces `per_run_metrics.csv`, `summary_table.csv`, `summary_table.md`, accuracy/time plots and selection heatmaps. Time to target requires accuracy to remain above target through the final round. Misses are omitted from its mean and counted in `target_reached`. Coverage includes warm-up; use the dashboard's post-warm-up table to assess continued participation.

## EC2 deployment

Prepare matching MNIST data and splits on the server and client host, build the `cloudcraft` image on the server and build the Compose clients before running a remote sweep. The remote path uses SSH/SCP, the `ubuntu` account and port 8080.

```bash
python -m cloudcraft.prepare_data --config configs/experiment.yaml
# On the server host:
docker build -t cloudcraft -f docker/Dockerfile .
# On the client host:
docker compose build
python -m cloudcraft.runner --config configs/experiment.yaml --remote-host <server-ip> --ssh-key <key.pem> --remote-dir ~/cloud-craft
```

Clients can run locally or on a separate EC2 instance. The dashboard can connect to the client instance, start the remote runner and synchronize results. Enter host/key values in its sidebar; verify the server address and remote paths for your deployment. Restrict inbound gRPC to the client host and SSH to authorized hosts. Optional `NETEM_DELAY` adds container-level delay on top of the software profile. Stop instances when experiments finish.

## Remaining validation

Validate the installed environment, single-client learning, Docker connectivity, timeout recovery and longer-run selection behaviour before collecting the full sweep. The proposed acceptance targets are at least 30% lower time to 97% accuracy versus full FedAvg, final accuracy within one percentage point, and continued participation after warm-up. Numerical conclusions require complete logs across both splits and all three seeds. See [REPORT.md](REPORT.md) for the reporting protocol.
