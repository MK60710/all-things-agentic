from __future__ import annotations

from service.rate_limits import LimitWindow, RateLimiter


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
