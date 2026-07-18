"""Tests for the hosted assistant's atomic global daily quota."""

from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.assistant_shared_quota import (
    DailyQuotaExceeded,
    SQLiteDailyQuota,
)


class SharedAssistantQuotaTests(unittest.TestCase):
    def test_twenty_requests_are_allowed_and_the_next_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            quota = SQLiteDailyQuota(Path(directory) / "quota.sqlite3")
            reservations = [
                quota.reserve(f"request-{index}") for index in range(20)
            ]
            self.assertEqual(reservations[-1].status.remaining, 0)
            with self.assertRaises(DailyQuotaExceeded):
                quota.reserve("request-21")

    def test_quota_resets_on_the_next_utc_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            quota = SQLiteDailyQuota(
                Path(directory) / "quota.sqlite3",
                daily_limit=1,
            )
            first_day = datetime(2026, 7, 17, 23, 59, tzinfo=timezone.utc)
            quota.reserve("first-day", now=first_day)
            with self.assertRaises(DailyQuotaExceeded):
                quota.reserve("same-day", now=first_day)

            next_day = first_day + timedelta(minutes=2)
            reservation = quota.reserve("next-day", now=next_day)
            self.assertEqual(reservation.status.used, 1)

    def test_completed_duplicate_returns_cached_response_without_new_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            quota = SQLiteDailyQuota(Path(directory) / "quota.sqlite3")
            quota.reserve("same-request")
            quota.complete("same-request", {"answer": "cached"})

            repeated = quota.reserve("same-request")

            self.assertEqual(repeated.cached_response, {"answer": "cached"})
            self.assertEqual(quota.status().used, 1)

    def test_concurrent_reservations_cannot_exceed_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            quota = SQLiteDailyQuota(
                Path(directory) / "quota.sqlite3",
                daily_limit=20,
            )

            def attempt(index: int) -> bool:
                try:
                    quota.reserve(f"concurrent-{index}")
                except DailyQuotaExceeded:
                    return False
                return True

            with ThreadPoolExecutor(max_workers=16) as executor:
                accepted = list(executor.map(attempt, range(50)))

            self.assertEqual(sum(accepted), 20)
            self.assertEqual(quota.status().used, 20)


if __name__ == "__main__":
    unittest.main()
