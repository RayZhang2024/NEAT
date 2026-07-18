"""Concurrency-safe daily quota storage for the hosted NEAT assistant."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Mapping, Optional


DEFAULT_DAILY_LIMIT = 20


class DailyQuotaExceeded(RuntimeError):
    """Raised when all globally shared requests have been reserved."""

    def __init__(self, *, reset_at_utc: str) -> None:
        super().__init__("The shared NEAT daily request limit has been reached.")
        self.reset_at_utc = reset_at_utc


class DuplicateRequestInProgress(RuntimeError):
    """Raised when a request ID is already reserved but has no final response."""


@dataclass(frozen=True)
class QuotaStatus:
    day_utc: str
    used: int
    limit: int
    remaining: int
    reset_at_utc: str


@dataclass(frozen=True)
class QuotaReservation:
    status: QuotaStatus
    cached_response: Optional[dict[str, object]] = None


def _utc_day(now: Optional[datetime] = None) -> tuple[str, str]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    day = current.date()
    reset = datetime.combine(day + timedelta(days=1), time(), tzinfo=timezone.utc)
    return day.isoformat(), reset.isoformat()


class SQLiteDailyQuota:
    """Reserve a global logical request using an atomic SQLite transaction.

    This backend is suitable for one hosted service instance. A horizontally
    scaled deployment should use a shared Redis or PostgreSQL implementation.
    """

    def __init__(
        self,
        database_path: Path,
        *,
        daily_limit: int = DEFAULT_DAILY_LIMIT,
    ) -> None:
        if daily_limit <= 0:
            raise ValueError("daily_limit must be positive")
        self.database_path = Path(database_path).expanduser()
        self.daily_limit = int(daily_limit)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS quota_days (
                    day_utc TEXT PRIMARY KEY,
                    used INTEGER NOT NULL CHECK (used >= 0)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS quota_requests (
                    request_id TEXT PRIMARY KEY,
                    day_utc TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    response_json TEXT,
                    error_code TEXT
                )
                """
            )

    def status(self, *, now: Optional[datetime] = None) -> QuotaStatus:
        day_utc, reset_at_utc = _utc_day(now)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT used FROM quota_days WHERE day_utc = ?",
                (day_utc,),
            ).fetchone()
        used = int(row["used"]) if row is not None else 0
        return QuotaStatus(
            day_utc=day_utc,
            used=used,
            limit=self.daily_limit,
            remaining=max(0, self.daily_limit - used),
            reset_at_utc=reset_at_utc,
        )

    def reserve(
        self,
        request_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> QuotaReservation:
        request_id = str(request_id or "").strip()
        if not request_id:
            raise ValueError("request_id is required")
        day_utc, reset_at_utc = _utc_day(now)
        created_at = (now or datetime.now(timezone.utc)).astimezone(
            timezone.utc
        ).isoformat()

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT state, response_json
                FROM quota_requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["state"] == "completed"
                    and existing["response_json"]
                ):
                    cached = json.loads(existing["response_json"])
                    connection.execute("COMMIT")
                    return QuotaReservation(
                        status=self.status(now=now),
                        cached_response=cached,
                    )
                connection.execute("ROLLBACK")
                raise DuplicateRequestInProgress(
                    "This shared request ID has already been used."
                )

            row = connection.execute(
                "SELECT used FROM quota_days WHERE day_utc = ?",
                (day_utc,),
            ).fetchone()
            used = int(row["used"]) if row is not None else 0
            if used >= self.daily_limit:
                connection.execute("ROLLBACK")
                raise DailyQuotaExceeded(reset_at_utc=reset_at_utc)

            next_used = used + 1
            connection.execute(
                """
                INSERT INTO quota_days(day_utc, used)
                VALUES (?, ?)
                ON CONFLICT(day_utc) DO UPDATE SET used = excluded.used
                """,
                (day_utc, next_used),
            )
            connection.execute(
                """
                INSERT INTO quota_requests(
                    request_id, day_utc, state, created_at_utc
                )
                VALUES (?, ?, 'reserved', ?)
                """,
                (request_id, day_utc, created_at),
            )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

        return QuotaReservation(
            status=QuotaStatus(
                day_utc=day_utc,
                used=next_used,
                limit=self.daily_limit,
                remaining=self.daily_limit - next_used,
                reset_at_utc=reset_at_utc,
            )
        )

    def complete(
        self,
        request_id: str,
        response: Mapping[str, object],
    ) -> None:
        serialized = json.dumps(
            dict(response),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE quota_requests
                SET state = 'completed', response_json = ?, error_code = NULL
                WHERE request_id = ? AND state = 'reserved'
                """,
                (serialized, request_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"No reserved quota request: {request_id}")

    def fail(self, request_id: str, *, error_code: str) -> None:
        """Mark an attempted provider call as failed without refunding its slot."""

        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE quota_requests
                SET state = 'failed', error_code = ?
                WHERE request_id = ? AND state = 'reserved'
                """,
                (str(error_code or "provider_error"), request_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"No reserved quota request: {request_id}")


__all__ = [
    "DEFAULT_DAILY_LIMIT",
    "DailyQuotaExceeded",
    "DuplicateRequestInProgress",
    "QuotaReservation",
    "QuotaStatus",
    "SQLiteDailyQuota",
]
