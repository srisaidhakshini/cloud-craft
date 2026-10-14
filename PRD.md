# PRD: Performance-Aware Federated Learning

Oct 6, 2026 (updated Oct 14, 2026) · CloudCraft (Kinjal Nandi, Sri Saidhakshini V)

> Living version: https://claude.ai/code/artifact/e48be38e-3bff-4dd0-9e52-21269d76b8fc

## Overview

We will add a performance-scoring and client-selection layer to FedAvg that picks the best K clients each round, cutting wall-clock training time without losing accuracy. The Flower server runs on AWS EC2; five simulated clients with fixed speed, latency and reliability profiles connect to it over gRPC.

| Item | Detail |
| --- | --- |
| Project | Performance-Aware Federated Learning |
| Team | CloudCraft: Kinjal Nandi, Sri Saidhakshini V |
| Type | Academic / research prototype |
| Status | Build in progress: core code and runner (local and EC2) written; full-length validation and the 24-run experiments are pending (see Implementation status) |

## Problem

Standard FedAvg waits for every selected client before it can aggregate, so the slowest or least reliable device sets the length of every round.

- **Stragglers:** one slow client (2–4 s extra per round) stretches every round for everyone.
- **Failures:** an unreliable client that drops out about 30% of the time either stalls the round until timeout or is lost entirely.
- **Existing fixes miss round time:** FedProx tackles client drift, not round duration.
- **No lightweight option:** there is no simple, measurement-based way to skip slow clients without permanently losing their data.

## Goals and success metrics

Success means the performance-aware strategy reaches the target accuracy faster than both full FedAvg and random selection, with no meaningful accuracy loss.

**Goals**

1. Build a scorer that rates each client on training time, network latency and reliability.
2. Build a top-K selection strategy on Flower's FedAvg that uses those scores.
3. Keep slow clients' data in training through an exploration rule.
4. Prove the result on a real cloud deployment with repeatable experiments.

**Success metrics** (targets are proposed; confirm before Phase 4)

| Metric | Definition | Target |
| --- | --- | --- |
| Time to target accuracy | Wall-clock seconds until global test accuracy reaches 97% | At least 30% faster than full FedAvg |
| Final accuracy | Global test accuracy after the last round | Within 1 percentage point of full FedAvg |
| Mean round time | Average seconds per round | Lower than full FedAvg and random K |
| Client coverage | Share of rounds each client participates in | Every client above 0% (no client fully excluded) |
| Reproducibility | Std. dev. across 3 seeded runs | Reported for every metric |

**Non-goals**

- Production-grade security (secure aggregation, differential privacy).
- Real mobile or edge hardware; heterogeneity is simulated.
- Large models or datasets beyond a small CNN on MNIST.
- Fixing client drift (left to FedProx; see Future scope).

## Stakeholders

| Who | Needs from the project |
| --- | --- |
| Project team (CloudCraft) | A working prototype, clean experiment logs, and plots for the report |
| Evaluators / faculty | A clear comparison against baselines, with repeatable numbers |
| FL practitioners (eventual users) | A drop-in Flower strategy they can configure (K, weights, exploration rate) |

## Functional requirements

P0 items are required for the core result; P1 items strengthen it.

| ID | Requirement | Priority | Status |
| --- | --- | --- | --- |
| FR-1 | Split MNIST across 5 clients with a fixed seed; support both an IID split and a non-IID Dirichlet split (alpha = 0.5). Save split indices to disk. | P0 | Built |
| FR-2 | One client codebase whose behaviour comes from a PROFILE setting: Fast (no delay), Medium (0.5–1 s), Slow (2–4 s), High-latency (network delay), Unreliable (about 30% failure). | P0 | Built |
| FR-3 | Each client reports its local training time and sample count in its fit metrics. | P0 | Built |
| FR-4 | The server measures each client's round-trip time and derives latency as round-trip time minus training time. | P0 | Built |
| FR-5 | The server records failures and timeouts per client and tracks reliability as a running success rate. | P0 | Built |
| FR-6 | The scorer computes a 0–1 score per client from those three signals (see Scoring specification). | P0 | Built, unit-tested |
| FR-7 | The strategy selects every client in round 1, then the top K by score from round 2 on. | P0 | Built |
| FR-8 | An exploration rule keeps low-scoring clients participating occasionally. | P0 | Built (ε-swap; staleness bonus also available) |
| FR-9 | Rounds have a configurable timeout; the strategy accepts failures and aggregates whatever arrived. | P0 | Built |
| FR-10 | Log one CSV row per client per round: run_id, condition, seed, round, client_id, selected, train_time, rtt, latency, success, score, global_accuracy, global_loss, wall_clock. | P0 | Built (plus a `round_time` column) |
| FR-11 | A single experiment runner executes every condition × split × seed from YAML configs without manual steps. | P0 | Built (local and `--remote-host`) |
| FR-12 | Analysis scripts produce the comparison plots and a mean ± std table from the CSV logs. | P0 | Built, smoke-tested only |
| FR-13 | K, scoring weights, EMA factor, exploration rate and timeout are all config values, not hard-coded. | P1 | Built |
| FR-14 | Docker Compose starts all 5 clients, pointed at a configurable server address. | P1 | Built |

## Scoring and selection specification

Each client's score is a weighted sum of its normalised speed, latency and reliability, smoothed over rounds; the top K scores train each round, plus occasional exploration.

**Per-round signals for client i** (smoothed with an exponential moving average, factor β = 0.3 by default)

$$
\bar{x}_i^{(t)} = \beta \, x_i^{(t)} + (1 - \beta) \, \bar{x}_i^{(t-1)}
$$

**Score** (current config weights 0.3 / 0.2 / 0.5, changed from the originally proposed 0.4 / 0.3 / 0.3; T = training time, L = latency, R = reliability, each min–max normalised across clients)

$$
S_i = w_T \,(1 - \hat{T}_i) + w_L \,(1 - \hat{L}_i) + w_R \, R_i, \quad w_T + w_L + w_R = 1
$$

**Selection rules**

1. Round 1: select all clients to collect first measurements.
2. From round 2: rank clients by score and select the top K (default K = 3 of 5).
3. Exploration: each selected slot is swapped for a random unselected client with probability ε (default 0.1). Alternative to compare: a staleness bonus that raises a client's score for each round it sits out.
4. A failed or timed-out client gets success = 0 for that round; its training time and latency are not updated.
5. Aggregation is standard FedAvg, weighted by each client's sample count.

**Worked example** (from the pitch deck, after round 1; illustrative, not produced by the current weights)

| Client | Score | Selected in round 2 |
| --- | --- | --- |
| 1 Fast | 0.91 | Yes |
| 2 Medium | 0.78 | Yes |
| 4 High-latency | 0.65 | Yes |
| 3 Slow | 0.40 | No, unless picked by exploration |
| 5 Unreliable | 0.20 | No, unless picked by exploration |

## Architecture and tech stack

The new layer is two components inside the Flower server: a scorer that turns client measurements into scores, and a selector that picks who trains next round.

```
┌─────────────────── AWS EC2: Flower server (public IP, gRPC) ───────────────────┐
│                                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐ │
│  │ Round orchestrator   │──▶│ Scorer  [NEW]        │──▶│ Selector + FedAvg    │ │
│  │ runs rounds, timeout │    │ EMA of time, latency,│    │ [NEW] top-K +        │ │
│  │ writes CSV log       │    │ success → score 0–1  │    │ exploration, avg     │ │
│  └──────────▲───────────┘    └──────────────────────┘    └──────────┬───────────┘ │
│             └──────────── selected clients for the next round ──────┘             │
└───────────────────────────────┬──────────────────▲──────────────────────────────┘
                     global model│                  │weights + train time
        ┌──────────────┬─────────┴────┬─────────────┴┬──────────────┐
   ┌────┴────┐   ┌─────┴────┐   ┌─────┴────┐   ┌─────┴──────┐  ┌────┴───────┐
   │  Fast   │   │  Medium  │   │   Slow   │   │High-latency│  │ Unreliable │
   │no delay │   │ 0.5–1 s  │   │  2–4 s   │   │network dly │  │ fails ~30% │
   └─────────┘   └──────────┘   └──────────┘   └────────────┘  └────────────┘
          5 Docker clients, one profile each, MNIST shard per client
```

Each round, clients send back weights plus training time; the scorer updates its stats, and the selector picks the next round's clients and averages the updates.

| Layer | Choice |
| --- | --- |
| ML framework | PyTorch 2.6.0 (CPU), Python 3.13 |
| FL framework | Flower 1.39.0 (pinned; was 1.13.1 when first built) |
| Model and data | Small CNN on MNIST |
| Server | AWS EC2, public IP, gRPC over HTTP/2 |
| Clients | 5 Docker containers via Docker Compose (local machine; server on EC2) |
| Network delay | tc netem inside the High-latency container |
| Experiments | YAML configs + one Python runner |
| Analysis | pandas + matplotlib |

## Experiment design

We run 4 conditions × 2 data splits × 3 seeds = 24 runs. Random selection (B) is the key control: it shows whether gains come from choosing smartly or just from training fewer clients.

| Condition | Selection | What it shows |
| --- | --- | --- |
| A | FedAvg, all 5 clients | Baseline |
| B | Random K = 3 | Effect of fewer clients alone |
| C | Performance-aware top K = 3 | The proposed method |
| D | C + exploration | Cost of keeping every client's data in play |

- **Splits:** IID and non-IID (Dirichlet alpha = 0.5). On IID MNIST, dropping clients barely hurts accuracy, so the non-IID split is what tests the accuracy claim.
- **Fixed across runs:** model, optimiser, learning rate, local epochs, batch size, number of rounds, round timeout, client profiles.
- **Outputs:** accuracy vs wall-clock time (one line per condition), round time per round, a client × round selection heatmap, and a mean ± std table.

## Non-functional requirements

| Area | Requirement |
| --- | --- |
| Reproducibility | Seeds fixed for data split, model init and profile randomness; Python, PyTorch and Flower versions pinned in requirements.txt |
| Fault tolerance | No round may hang: a timeout bounds every round, and the server survives any client failure |
| Testability | The scorer is a pure function (stats in, scores out) with unit tests that need no Flower or network |
| Security | The EC2 security group opens the gRPC port only to the team's IP addresses; no credentials in the repo |
| Cost | A small instance (t3.small or t3.medium); stopped whenever no experiment is running |
| Observability | Every run writes a complete CSV log; a run with missing rows is treated as failed |

## Milestones

Five phases run in order; each ends with a check that must pass before the next starts. Dates are not set yet; the effort split is a guide. Status as of Oct 14, 2026.

1. **Setup (about 15% of effort):** repo, pinned environment, IID and non-IID splits, 5 client profiles.
   - Done when: one client trained alone reaches about 98% on MNIST.
   - Status: code done; the 98% check has not been run.
2. **Baseline (about 20%):** full FedAvg locally with Docker Compose, then the server on EC2 with timeouts and CSV logging.
   - Done when: a full baseline run on EC2 completes with a complete log.
   - Status: local baseline runs end to end; the EC2 runner path is written, but a complete EC2 baseline run is not yet confirmed.
3. **Selection logic (about 25%):** scorer, top-K selection, exploration, unit tests.
   - Done when: tests pass and logs show Slow and Unreliable clients picked less often, but not never.
   - Status: code and 15 unit tests done. The selection behaviour is not yet confirmed on a full-length run (a 4-round smoke run was too short to show it); `configs/medium.yaml` (100 rounds) exists for this check.
4. **Experiments (about 25%):** all 24 runs through the scripted runner.
   - Done when: every run has a complete log, and the EC2 instance is stopped.
   - Status: runner built (resumable with `--resume`); sweep not run.
5. **Wrap-up (about 15%):** plots, results table, report, final presentation.
   - Done when: the report and deck include the scoring formula, the results and the selection heatmap.
   - Status: analysis script built; report and deck not started.

## Implementation status

Written and pushed: scorer and selection (`cloudcraft/scoring.py`), strategy with timing, failure tracking and CSV logging (`strategy.py`), profile-driven client, seeded IID and Dirichlet splits, experiment runner (local and EC2 via `--remote-host`), analysis script, Docker Compose, and configs for smoke, medium (100 rounds) and the full sweep.

Verified: 15 scorer unit tests; a 4-round smoke sweep of conditions A–D with complete logs, on Flower 1.13.1.

Not verified yet: anything on the current pins (Flower 1.39.0, Python 3.13); full-length runs; the single-client 98% check; the EC2 path; the Docker Compose path; the success metrics in this document.

## Risks, open questions and future scope

**Risks**

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Skipping slow clients drops their data, hurting accuracy on non-IID splits | The accuracy claim fails | Exploration (condition D); report per-client coverage |
| Real internet latency to EC2 swamps the simulated delays | Profiles become hard to tell apart | Measure baseline round-trip time first; scale simulated delays above it |
| Flower API changes between versions | Code breaks mid-project | Pin the version; follow that version's docs only |
| IID MNIST is too easy to show any accuracy difference | Results look trivial | Non-IID split is mandatory, not future scope |
| EC2 networking (ports, public IP changes) eats time | Phase 2 slips | Get one remote client connected in week one; use an Elastic IP |
| Flower was bumped from 1.13.1 to 1.39.0 after the code was written and smoke-tested | Strategy or client code may break or behave differently | Re-run unit tests and the smoke config on the new pins before any experiment |

**Open questions**

- [ ] Final deadline and dates for each phase?
- [ ] Target accuracy for the time-to-accuracy metric: 97% or another value?
- [ ] Default K and scoring weights: config now uses K = 3 and 0.3 / 0.2 / 0.5; confirm these are final, and note the reason in the report.
- [ ] Exploration method: ε-greedy, staleness bonus, or compare both?
- [x] Clients from laptops over the internet, or from a second EC2 instance? Current setup: clients in local Docker, server on EC2 (confirm this is final).

**Future scope**

- Add data-quality or contribution signals to the score (FedCS-style).
- Combine with FedProx to target round time and client drift together.
- Server-side fault tolerance: checkpointing and restart policies.
- Scale beyond 5 clients and to larger models or datasets.
