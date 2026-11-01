# CloudCraft: Detailed Project Report

**Project:** Performance-Aware Federated Learning
**Team members:** Kinjal Nandi and Sri Saidhakshini V
**Updated:** November 2, 2026
**Report scope:** Repository implementation review and experimental evaluation protocol. Individual contributions are not recorded in the repository and are not assigned here.

## Abstract

CloudCraft investigates whether measurement-based client selection can reduce federated learning time under heterogeneous device and communication conditions. A Flower FedAvg strategy tracks client training time, estimated communication latency and reliability, computes weighted scores and selects the best K clients. An exploration variant occasionally includes lower-scoring clients to reduce the risk of losing their data distribution. The implementation provides five simulated clients, IID and non-IID MNIST partitions, four comparison conditions, an experiment runner and analysis tools. No experiment logs are available in this checkout; this report documents the implemented method and how to evaluate it, without claiming measured improvements.

## 1. Problem and objectives

Synchronous federated learning aggregates updates from selected clients. Slow computation, communication delays and failures can extend a round and reduce useful progress per second. Selecting fewer clients may improve speed, but can also reduce data coverage, especially when clients hold different label distributions.

CloudCraft aims to balance these effects through a configurable scoring layer. Its research question is whether selecting clients using measured system performance improves time to accuracy compared with both full participation and random selection of the same number of clients. The exploration variant tests the cost and potential accuracy benefit of including otherwise sidelined clients.

The proposed acceptance criteria are a 30% reduction in time to 97% global test accuracy relative to full FedAvg, final accuracy within one percentage point of that baseline, and lower average round time than both full and random selection. These are targets. The required evidence has not been collected in this checkout.

## 2. Implemented architecture

The server initializes a seeded CNN, distributes parameters, chooses clients, collects fit responses and aggregates updates using sample-count-weighted FedAvg. A timing proxy measures each fit request's round-trip duration. The strategy updates per-client statistics and writes one record for every client each round. Centralized evaluation uses the MNIST test set.

| Component | Responsibility |
| --- | --- |
| `cloudcraft/model.py` | Two 5x5 convolution layers (16 and 32 channels), max pooling, a 128-unit dense layer and ten output logits; SGD and cross-entropy training |
| `cloudcraft/data.py` | MNIST loading and saved seeded client partitions |
| `cloudcraft/profiles.py`, `client.py` | Configurable computation delay, communication delay and simulated failures |
| `cloudcraft/scoring.py` | Pure statistics, scoring and selection functions |
| `cloudcraft/strategy.py` | Flower FedAvg integration, warm-up, request timing, failure tracking and logging |
| `cloudcraft/runner.py` | Condition/split/seed iteration, local processes, remote Docker orchestration and resume |
| `analysis/analyze.py` | Complete-run filtering, metric aggregation and plots |
| `analysis/dashboard.py` | Interactive metrics, post-warm-up selection, remote synchronization and sweep controls |

The local runner launches a server and five client subprocesses. The remote runner starts the server container through SSH and runs clients through Compose on the runner host. The dashboard also supports operating from a client EC2 instance connected to a separate server instance. These paths exist in code; deployment success must be verified separately.

## 3. Data and client heterogeneity

MNIST is partitioned among five clients using either a seeded IID split or a Dirichlet split with alpha 0.5. Saved indices make partition reuse possible. The non-IID case is essential because system-based selection may repeatedly omit a client holding underrepresented labels.

| Client | Profile | Effective base configuration |
| --- | --- | --- |
| 1 | Fast | No added delay or simulated failure |
| 2 | Medium | Uniform 3–5 s extra local training delay |
| 3 | Slow | Uniform 8–12 s extra local training delay |
| 4 | High-latency | Uniform 3–5 s delay outside reported training time |
| 5 | Unreliable | Failure probability 0.3 per fit attempt |

These values come from `configs/base.yaml` merged with Python profile defaults. The smaller Medium/Slow/High-latency values in `profiles.py` are overridden by YAML. Optional container-level `tc netem` delay is additional to software delay. Artificial delays model controlled heterogeneity; they do not establish behaviour on physical edge devices.

## 4. Scoring method

For an observed value x, the exponential moving average is:

`EMA_new = beta * x + (1 - beta) * EMA_previous`, with beta = 0.3.

The first observation seeds the statistic directly. Successful fits update training-time and latency EMAs. Every selected fit updates reliability using a binary success indicator. Failed fits preserve previous timing estimates and lower the reliability estimate. Latency is estimated as RTT minus reported training time and clamped to zero for scoring. The raw logged latency can still be negative.

Training time and latency are min–max normalized across client statistics. An unknown value is assigned normalized cost 1; identical known values receive cost 0. Reliability is the success EMA and is not min–max normalized.

`score_i = 0.3 * (1 - normalized_time_i) + 0.2 * (1 - normalized_latency_i) + 0.5 * reliability_i`

The YAML weights prioritize avoiding unreliable clients, whose failures can consume the timeout budget. This is a design rationale rather than an experimentally established optimal setting. The standalone `ScoringConfig` defaults are 0.4/0.3/0.3; the server explicitly uses YAML values, so report the effective configuration for every experiment.

## 5. Selection and aggregation

Condition A selects all five clients each round. B chooses three randomly each round. C and D select all clients during the configured warm-up, which defaults to rounds 1–2, and switch to ranked selection in round 3. Reliability is updated during both warm-up rounds; timing updates begin at the final warm-up round to avoid using initial startup timings.

C selects the three highest-scoring clients, with client ID breaking ties. D starts from the same ranking and attempts an epsilon swap for each selected slot, with epsilon = 0.1. A swap draws from the currently unselected pool, preserving selection size and uniqueness. An optional staleness bonus adds `bonus * rounds_since_selected` before ranking; it defaults to zero.

Exploration makes future participation possible but cannot guarantee that every client appears in a finite post-warm-up window. Report actual selection frequencies. Whole-run coverage alone is insufficient because warm-up already selects every client.

The strategy accepts failed clients and uses standard FedAvg for successful responses. The configured round timeout is 60 s and the default simulated failure mode is `timeout`; `drop` is also available. Behaviour when no useful updates arrive and subsequent reconnection should be integration-tested before interpreting experiment outcomes.

## 6. Experiment protocol

The full design is four conditions x two splits x three seeds = 24 runs. `configs/experiment.yaml` overrides the base round count from 20 to 30. Each round uses one local epoch, batch size 64, learning rate 0.01 and SGD momentum 0.9. Seeds are 0, 1 and 2. Keep these settings, profiles and deployment topology fixed across conditions.

| Condition | Purpose |
| --- | --- |
| A: All clients | Full-participation FedAvg reference |
| B: Random K | Controls for the benefit of merely training fewer clients |
| C: Performance top-K | Measures the effect of ranking without exploration |
| D: Performance plus exploration | Tests the speed/coverage trade-off |

Before the sweep, validate dependency installation, scorer tests, one-client learning, an end-to-end baseline, failure recovery and post-warm-up selection. Historical smoke and medium configs are absent; create a short override beside `base.yaml` when a reduced integration run is needed. Do not change the accuracy target after inspecting the alternative conditions. If 30 rounds are insufficient, record that limitation and define a consistent longer protocol for all conditions.

The local and remote commands are provided in [README.md](README.md). Archive effective merged configs, Python/package versions, commit ID, hardware, instance types, regions, data indices and any added network delay with the final results.

## 7. Logs and metric definitions

Each run creates a client-by-round CSV with `run_id`, `condition`, `seed`, `round`, `client_id`, `selected`, `train_time`, `rtt`, `latency`, `success`, `score`, `global_accuracy`, `global_loss`, `wall_clock` and `round_time`. Unselected rows retain blank timing and success values. Scores reflect statistics after the logged round.

The server writes metadata with expected/observed row counts and marks the run complete when these match. At 30 rounds and five clients, expect 150 rows per run and 3,600 rows across the full sweep. Row-count completion does not by itself verify distinct client-round pairs, finite metrics or scientific validity; inspect these before final reporting. Analysis trusts the completion flag and excludes incomplete runs.

| Metric | Current analysis definition and reporting rule |
| --- | --- |
| Time to target | First recorded round at or above 0.97 whose accuracy stays at or above 0.97 through the final round |
| Target reached | Count of runs meeting that sustained criterion; target misses are NaN and omitted from time averages |
| Final accuracy | Global test accuracy in the final logged round |
| Mean round time | Average `round_time`; fit phase from configuration end to entry into aggregation, excluding centralized evaluation |
| Training wall-clock | Elapsed time from first fit configuration to aggregation entry, minus centralized evaluation time; not total job duration |
| Minimum client coverage | Minimum client selection fraction over the whole run, including warm-up |
| Post-warm-up coverage | Inspect separately in the dashboard after removing configured warm-up rounds |

The summary uses the sample standard deviation across seeds. A single run receives displayed standard deviation zero, which is not evidence of stability. When all runs miss target, report time to target as unavailable. When only some seeds reach target, report their count prominently to avoid overstating speed.

For paired successful seed/split comparisons, compute speedup as `100 * (time_A - time_variant) / time_A`. Report final-accuracy differences in percentage points. Distinguish paired comparisons from ratios of group means, and include B when judging whether ranking adds value.

## 8. Analysis and dashboard deliverables

Offline analysis generates per-run metrics, CSV/Markdown summary tables, accuracy versus wall-clock curves, round-duration curves and selection heatmaps for C/D. Curves average interpolated seed trajectories over their shared time range; time-to-target metrics use recorded rounds rather than interpolated crossings.

The Streamlit dashboard provides summary, accuracy, round-time, selection and run-detail tabs. It displays completion counts and post-warm-up selection frequencies, synchronizes files with SCP and exposes SSH-based EC2 controls. Its default result folder is `results_full2`, while the runner defaults to `results`; select the intended folder explicitly. Connection fields and remote paths need deployment-specific values.

## 9. Evidence and current results

| Item | Evidence available in this checkout | Conclusion |
| --- | --- | --- |
| Scorer and selector | Implementation and 15 test functions | Unit-test coverage exists; historical pass claim requires current execution to reproduce |
| Integration | Strategy, client/server, runner and Compose source | Implemented; successful current-version deployment not established |
| Analysis and dashboard | Source files | Implemented; no local result set available for numerical review |
| Historical smoke run | Earlier PRD records a four-round A–D sweep on Flower 1.13.1 | Historical statement only; supporting logs absent |
| Full 24-run sweep | No results directory or metric artifacts present | Cannot assess speedup, final accuracy or seed variability |
| Review presentation | `deck_work/` contains a build script and presentation assets | Artifact exists; numerical claims have not been validated here |

No measured score ordering, accuracy, time savings or EC2 execution success is asserted. The detailed report is complete as an implementation review; an empirical results section still requires logs.

## 10. Risks and limitations

System performance can correlate with data distribution, causing selection bias. Exploration mitigates exclusion probabilistically and may also reintroduce expensive clients. Relative min–max scores can change when the observed population changes; stale timing values may miss a client's recovery or deterioration. RTT-minus-training is an overhead estimate that includes serialization, transfer and scheduling, rather than a direct network measurement.

Timeout-style failures can dominate small experiments. Simulated delays and five clients limit generalization, and a short MNIST run may not reveal convergence differences. Shared local CPU resources can confound client timing; record host contention and compare deployment arrangements consistently.

The declared Python 3.13 and dependency pins have not been demonstrated to install together in this checkout. Earlier integration evidence used a different Flower version. Resolve environment compatibility before collecting results and preserve the validated environment rather than treating version strings as proof of reproducibility.

The prototype does not implement secure aggregation or differential privacy. Restrict cloud access to authorized hosts, keep keys outside version control and stop idle instances. Remote runner/dashboard controls assume particular SSH accounts, paths and networking; validate these assumptions for the chosen deployment.

## 11. Completion checklist and future work

1. Validate the runtime and reproduce scorer tests and an integration run.
2. Confirm baseline learning and choose the target and run length before comparison.
3. Check all profiles, timeout recovery, and C/D selection after warm-up.
4. Execute the 24-run sweep with fixed settings and retain complete logs and metadata.
5. Audit client-round records and target misses; publish per-seed metrics and mean ± standard deviation for each split.
6. Add measured comparisons, post-warm-up coverage, plots and interpretation to this report and the review presentation.

Future extensions include more clients, larger datasets, contribution-aware scoring, combination with FedProx, checkpoint recovery and systematic exploration/weight ablations. Any improvement claim should remain scoped to the measured model, hardware, network conditions and data partitions.

## Repository references

- [PRD](PRD.md): requirements, acceptance targets and milestones.
- [README](README.md): setup, execution and deployment instructions.
- `configs/base.yaml` and `configs/experiment.yaml`: effective experimental defaults.
- `cloudcraft/scoring.py`, `cloudcraft/strategy.py`: method implementation.
- `analysis/analyze.py`, `analysis/dashboard.py`: metric definitions and result inspection.
