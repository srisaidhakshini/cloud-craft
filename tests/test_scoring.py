import random

import pytest

from cloudcraft.scoring import (
    ClientStats,
    ScoringConfig,
    ema,
    score_clients,
    select_random_k,
    select_top_k,
    update_stats,
)

CFG = ScoringConfig()


def _stats():
    return {
        "1": ClientStats(train_time=1.0, latency=0.05, reliability=1.0),
        "2": ClientStats(train_time=1.8, latency=0.05, reliability=1.0),
        "3": ClientStats(train_time=4.0, latency=0.05, reliability=1.0),
        "4": ClientStats(train_time=1.0, latency=1.5, reliability=1.0),
        "5": ClientStats(train_time=1.0, latency=0.05, reliability=0.3),
    }


def test_ema_first_value_seeds():
    assert ema(None, 2.0, 0.3) == 2.0


def test_ema_formula():
    assert ema(10.0, 0.0, 0.3) == pytest.approx(7.0)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ScoringConfig(w_time=0.5, w_latency=0.5, w_reliability=0.5)


def test_scores_in_unit_interval_and_ordering():
    s = score_clients(_stats(), CFG)
    assert all(0.0 <= v <= 1.0 for v in s.values())
    assert s["1"] == max(s.values())
    assert s["1"] > s["2"] > s["3"]
    assert s["1"] > s["4"] and s["1"] > s["5"]


def test_identical_clients_get_equal_scores():
    same = {c: ClientStats(1.0, 0.1, 1.0) for c in "abc"}
    s = score_clients(same, CFG)
    assert len(set(round(v, 9) for v in s.values())) == 1


def test_unmeasured_client_scores_worst_on_time_and_latency():
    st = _stats()
    st["6"] = ClientStats()
    s = score_clients(st, CFG)
    assert s["6"] == 0.0


def test_failure_updates_reliability_not_time_or_latency():
    st = ClientStats(train_time=2.0, latency=0.5, reliability=1.0)
    new = update_stats(st, success=False, train_time=None, latency=None, beta=0.3)
    assert new.train_time == 2.0 and new.latency == 0.5
    assert new.reliability == pytest.approx(0.7)


def test_success_updates_all_signals():
    st = ClientStats(train_time=2.0, latency=0.5, reliability=0.5)
    new = update_stats(st, success=True, train_time=1.0, latency=0.1, beta=0.5)
    assert new.train_time == pytest.approx(1.5)
    assert new.latency == pytest.approx(0.3)
    assert new.reliability == pytest.approx(0.75)


def test_negative_latency_clamped():
    new = update_stats(ClientStats(), True, 1.0, -0.2, 0.3)
    assert new.latency == 0.0


def test_top_k_without_exploration_is_deterministic():
    s = score_clients(_stats(), CFG)
    picks = {tuple(select_top_k(s, 3, random.Random(i))) for i in range(20)}
    assert len(picks) == 1
    assert "3" not in picks.pop()


def test_exploration_sometimes_picks_low_scorers_but_keeps_k():
    s = score_clients(_stats(), CFG)
    rng = random.Random(0)
    seen_low = 0
    for _ in range(300):
        chosen = select_top_k(s, 3, rng, epsilon=0.3)
        assert len(chosen) == 3 and len(set(chosen)) == 3
        seen_low += ("3" in chosen) or ("5" in chosen)
    assert 0 < seen_low < 300


def test_epsilon_zero_never_explores():
    s = score_clients(_stats(), CFG)
    rng = random.Random(1)
    assert all("3" not in select_top_k(s, 3, rng, epsilon=0.0) for _ in range(100))


def test_staleness_bonus_eventually_selects_sidelined_client():
    s = score_clients(_stats(), CFG)
    stale = {c: 0 for c in s}
    stale["3"] = 50
    chosen = select_top_k(s, 3, random.Random(0), staleness=stale, staleness_bonus=0.05)
    assert "3" in chosen


def test_k_larger_than_pool():
    s = score_clients(_stats(), CFG)
    assert len(select_top_k(s, 99, random.Random(0))) == 5


def test_random_k_size_and_seed_determinism():
    a = select_random_k(list("12345"), 3, random.Random(7))
    b = select_random_k(list("12345"), 3, random.Random(7))
    assert a == b and len(a) == 3
