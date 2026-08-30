from __future__ import annotations

import pytest
from google.api_core.exceptions import Aborted

from service.rate_limits import LimitWindow, RateLimiter, _retry_on_contention


def test_rate_limit_blocks_after_limit_and_reports_reset(fake_db) -> None:
    now = [120.0]
    limiter = RateLimiter(
        fake_db,
        rules={"chat": (LimitWindow(60, 2),)},
        clock=lambda: now[0],
    )

    assert limiter.consume("user-a", "chat").allowed
    second = limiter.consume("user-a", "chat")
    assert second.allowed
    assert second.remaining == 0

    blocked = limiter.consume("user-a", "chat")
    assert not blocked.allowed
    assert blocked.retry_after == 60

    now[0] = 180.0
    assert limiter.consume("user-a", "chat").allowed


def test_rate_limits_are_per_user_and_survive_limiter_recreation(fake_db) -> None:
    rules = {"chat": (LimitWindow(86_400, 1),)}
    first = RateLimiter(fake_db, rules=rules, clock=lambda: 100.0)
    assert first.consume("user-a", "chat").allowed
    assert first.consume("user-b", "chat").allowed

    recreated = RateLimiter(fake_db, rules=rules, clock=lambda: 100.0)
    assert not recreated.status("user-a", "chat").allowed
    assert not recreated.consume("user-b", "chat").allowed


def test_short_and_long_windows_are_both_enforced(fake_db) -> None:
    now = [0.0]
    limiter = RateLimiter(
        fake_db,
        rules={"chat": (LimitWindow(60, 2), LimitWindow(86_400, 3))},
        clock=lambda: now[0],
    )
    assert limiter.consume("user-a", "chat").allowed
    assert limiter.consume("user-a", "chat").allowed
    assert not limiter.consume("user-a", "chat").allowed

    now[0] = 60.0
    assert limiter.consume("user-a", "chat").allowed
    assert not limiter.consume("user-a", "chat").allowed


def test_global_limit_cannot_be_bypassed_with_multiple_users(fake_db) -> None:
    limiter = RateLimiter(
        fake_db,
        rules={"chat": (LimitWindow(60, 10),)},
        global_rules={"chat": (LimitWindow(86_400, 2),)},
        clock=lambda: 100.0,
    )

    assert limiter.consume("user-a", "chat").allowed
    assert limiter.consume("user-b", "chat").allowed
    blocked = limiter.consume("user-c", "chat")
    assert not blocked.allowed
    assert blocked.scope == "global"


def test_retry_on_contention_recovers_after_transient_aborts(monkeypatch) -> None:
    # Reproduces the 2026-08-30 bug live: 3 concurrent "Start breaking down"
    # paper adds for the same user all hit the same rate-limit counter
    # documents at once; Firestore aborts the losers under contention, and
    # that previously reached the caller as an unhandled 500 with zero
    # retries (the abort surfaces during the transaction's read phase, which
    # the Firestore client's own retry does not cover - only a failed commit
    # is retried). This exercises the fix's retry loop directly, without
    # needing to fake real Firestore transaction internals.
    monkeypatch.setattr("service.rate_limits.time.sleep", lambda _seconds: None)
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise Aborted("409 Aborted due to cross-transaction contention")
        return "ok"

    assert _retry_on_contention(flaky) == "ok"
    assert calls["count"] == 3


def test_retry_on_contention_gives_up_after_exhausting_attempts(monkeypatch) -> None:
    monkeypatch.setattr("service.rate_limits.time.sleep", lambda _seconds: None)

    def always_aborted():
        raise Aborted("409 Aborted due to cross-transaction contention")

    with pytest.raises(Aborted):
        _retry_on_contention(always_aborted)


def test_retry_on_contention_does_not_swallow_other_errors(monkeypatch) -> None:
    monkeypatch.setattr("service.rate_limits.time.sleep", lambda _seconds: None)

    def raises_value_error():
        raise ValueError("not a contention error")

    with pytest.raises(ValueError):
        _retry_on_contention(raises_value_error)
