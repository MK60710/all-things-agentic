"""Persistent, per-user fixed-window limits for paid operations."""

from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from google.api_core.exceptions import Aborted
from google.cloud import firestore


def _retry_on_contention(fn: Callable[[], Any]) -> Any:
    """Retry a Firestore transaction on write-contention aborts.

    A concurrent batch (e.g. "Start breaking down N papers" firing N ingest
    requests for the same user at once) makes every one of those
    transactions read/write the same rate-limit counter documents in the
    same instant. Firestore aborts the losers under contention (409
    Aborted). That error surfaces during the transaction's read phase, and
    the Firestore client's own transaction retry only covers a failed
    commit, not a failed read - so left uncaught, this previously reached
    the caller as an unhandled 500 and silently dropped that request
    (confirmed live 2026-08-30: 3 concurrent paper adds, one lost this way).
    """
    last_error: Aborted | None = None
    for attempt in range(6):
        try:
            return fn()
        except Aborted as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(0.05 * (2**attempt) + random.uniform(0, 0.05))
    assert last_error is not None
    raise last_error


@dataclass(frozen=True)
class LimitWindow:
    seconds: int
    limit: int


RATE_LIMITS: dict[str, tuple[LimitWindow, ...]] = {
    "chat": (LimitWindow(60, 20), LimitWindow(86_400, 100)),
    "paper_ingest": (LimitWindow(3_600, 3), LimitWindow(86_400, 6)),
    "guide": (LimitWindow(3_600, 6), LimitWindow(86_400, 20)),
    "contradictions": (LimitWindow(3_600, 10), LimitWindow(86_400, 30)),
    "feynman": (LimitWindow(3_600, 20), LimitWindow(86_400, 50)),
    "gaps": (LimitWindow(3_600, 10), LimitWindow(86_400, 30)),
}

# Backstop for account multiplication or a leaked authenticated browser
# session. These are intentionally conservative for a free hackathon app and
# can be raised after observing real request/token costs.
GLOBAL_RATE_LIMITS: dict[str, tuple[LimitWindow, ...]] = {
    "chat": (LimitWindow(86_400, 500),),
    "paper_ingest": (LimitWindow(86_400, 15),),
    "guide": (LimitWindow(86_400, 40),),
    "contradictions": (LimitWindow(86_400, 60),),
    "feynman": (LimitWindow(86_400, 100),),
    "gaps": (LimitWindow(86_400, 60),),
}


@dataclass(frozen=True)
class RateLimitStatus:
    action: str
    allowed: bool
    remaining: int
    retry_after: int
    reset_at: str | None
    scope: str = "user"


class RateLimiter:
    def __init__(
        self,
        db_client: Any,
        *,
        rules: dict[str, tuple[LimitWindow, ...]] | None = None,
        global_rules: dict[str, tuple[LimitWindow, ...]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._db = db_client
        self._collection = db_client.collection("rate_limits")
        self._rules = rules or RATE_LIMITS
        self._global_rules = (
            global_rules
            if global_rules is not None
            else ({} if rules is not None else GLOBAL_RATE_LIMITS)
        )
        self._clock = clock
        self._lock = threading.Lock()

    @staticmethod
    def _document_id(uid: str, action: str, seconds: int, start: int) -> str:
        raw = f"{uid}:{action}:{seconds}:{start}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _windows(self, uid: str, action: str, now: float):
        try:
            user_rules = self._rules[action]
        except KeyError as exc:
            raise ValueError(f"unknown rate-limit action {action!r}") from exc
        windows = []
        for subject, rules in (
            (uid, user_rules),
            ("__global__", self._global_rules.get(action, ())),
        ):
            for rule in rules:
                start = int(now // rule.seconds) * rule.seconds
                reset = start + rule.seconds
                ref = self._collection.document(
                    self._document_id(subject, action, rule.seconds, start)
                )
                windows.append((rule, start, reset, ref))
        return windows

    @staticmethod
    def _decision(action: str, counts: list[tuple[int, int, int]], now: float) -> RateLimitStatus:
        # tuples are (count, limit, reset_epoch)
        exhausted = [item for item in counts if item[0] >= item[1]]
        allowed = not exhausted
        relevant = exhausted or counts
        reset_epoch = max(item[2] for item in relevant) if exhausted else None
        return RateLimitStatus(
            action=action,
            allowed=allowed,
            remaining=max(0, min(limit - count for count, limit, _ in counts)),
            retry_after=max(0, math.ceil(reset_epoch - now)) if reset_epoch else 0,
            reset_at=(
                datetime.fromtimestamp(reset_epoch, timezone.utc).isoformat()
                if reset_epoch
                else None
            ),
        )

    def status(self, uid: str, action: str) -> RateLimitStatus:
        now = self._clock()
        counts = []
        for rule, _start, reset, ref in self._windows(uid, action, now):
            snapshot = ref.get()
            count = int(snapshot.to_dict().get("count", 0)) if snapshot.exists else 0
            counts.append((count, rule.limit, reset))
        decision = self._decision(action, counts, now)
        user_count = len(self._rules[action])
        if not decision.allowed and any(
            count >= rule.limit
            for count, (rule, *_rest) in zip(counts[user_count:], self._windows(uid, action, now)[user_count:])
        ):
            return replace(decision, scope="global")
        return decision

    def consume(self, uid: str, action: str) -> RateLimitStatus:
        now = self._clock()
        windows = self._windows(uid, action, now)
        if hasattr(self._db, "transaction"):

            @firestore.transactional
            def increment(txn):
                snapshots = [ref.get(transaction=txn) for _, _, _, ref in windows]
                counts = [
                    int(snapshot.to_dict().get("count", 0)) if snapshot.exists else 0
                    for snapshot in snapshots
                ]
                decision = self._decision(
                    action,
                    [(count, rule.limit, reset) for count, (rule, _, reset, _) in zip(counts, windows)],
                    now,
                )
                if not decision.allowed:
                    user_count = len(self._rules[action])
                    global_exhausted = any(count >= rule.limit for count, (rule, *_rest) in zip(counts[user_count:], windows[user_count:]))
                    return replace(decision, scope="global" if global_exhausted else "user")
                for count, (rule, _start, reset, ref) in zip(counts, windows):
                    txn.set(
                        ref,
                        {
                            "uid": uid,
                            "action": action,
                            "window_seconds": rule.seconds,
                            "count": count + 1,
                            "expires_at": datetime.fromtimestamp(reset, timezone.utc)
                            + timedelta(days=1),
                        },
                    )
                return replace(self._decision(
                    action,
                    [(count + 1, rule.limit, reset) for count, (rule, _, reset, _) in zip(counts, windows)],
                    now,
                ), allowed=True)

            return _retry_on_contention(lambda: increment(self._db.transaction()))

        # Deterministic fake/local clients do not expose transactions. The
        # deployed Firestore client always takes the transactional path.
        with self._lock:
            snapshots = [ref.get() for _, _, _, ref in windows]
            counts = [
                int(snapshot.to_dict().get("count", 0)) if snapshot.exists else 0
                for snapshot in snapshots
            ]
            decision = self._decision(
                action,
                [(count, rule.limit, reset) for count, (rule, _, reset, _) in zip(counts, windows)],
                now,
            )
            if not decision.allowed:
                user_count = len(self._rules[action])
                global_exhausted = any(count >= rule.limit for count, (rule, *_rest) in zip(counts[user_count:], windows[user_count:]))
                return replace(decision, scope="global" if global_exhausted else "user")
            for count, (rule, _start, reset, ref) in zip(counts, windows):
                ref.set(
                    {
                        "uid": uid,
                        "action": action,
                        "window_seconds": rule.seconds,
                        "count": count + 1,
                        "expires_at": datetime.fromtimestamp(reset, timezone.utc)
                        + timedelta(days=1),
                    }
                )
            return replace(self._decision(
                action,
                [(count + 1, rule.limit, reset) for count, (rule, _, reset, _) in zip(counts, windows)],
                now,
            ), allowed=True)
