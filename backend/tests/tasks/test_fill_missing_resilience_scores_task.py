"""
Tests for the fill_missing_resilience_scores Celery task.

Covers that users are scored in batches (query count does not grow with the
number of users), that batched scores match the single-user scoring path, and
that one failing user does not block the rest of their batch.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from app.models import DataSource, HealthScore, SeriesTypeDefinition
from app.schemas.enums import HealthScoreCategory, ProviderName
from app.services.scores.resilience_service import resilience_score_service
from tests.factories import (
    DataPointSeriesFactory,
    DataSourceFactory,
    EventRecordFactory,
    SeriesTypeDefinitionFactory,
    SleepDetailsFactory,
    UserFactory,
)

TASK_MODULE = "app.integrations.celery.tasks.fill_missing_resilience_scores_task"


@contextmanager
def _count_selects(db: Session) -> Iterator[list[str]]:
    """Collect every SELECT statement executed on the session's connection."""
    selects: list[str] = []

    def _before_cursor_execute(*args: Any) -> None:
        statement = args[2]
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    bind = db.get_bind()
    event.listen(bind, "before_cursor_execute", _before_cursor_execute)
    try:
        yield selects
    finally:
        event.remove(bind, "before_cursor_execute", _before_cursor_execute)


def _run_task(db: Session, mock_session_local: Callable[[Session], MagicMock]) -> dict[str, int]:
    from app.integrations.celery.tasks.fill_missing_resilience_scores_task import fill_missing_resilience_scores

    with (
        patch(f"{TASK_MODULE}.SessionLocal", return_value=mock_session_local(db)),
        patch(f"{TASK_MODULE}.settings.ow_scores_enabled", True),
    ):
        return fill_missing_resilience_scores()


def _db_today(db: Session) -> date:
    """CURRENT_DATE as the task's missing-scores query sees it."""
    return db.execute(text("SELECT CURRENT_DATE")).scalar_one()


def _add_night(
    data_source: DataSource,
    rmssd_type: SeriesTypeDefinition,
    night: date,
    value: float,
    *,
    hrv_hour: int = 2,
    stages: list[dict] | None = None,
) -> None:
    """Add a 00:00–08:00 UTC sleep session on `night` and one RMSSD sample at `hrv_hour`."""
    record = EventRecordFactory(
        category="sleep",
        type_="sleep",
        start_datetime=datetime.combine(night, time(0), tzinfo=timezone.utc),
        end_datetime=datetime.combine(night, time(8), tzinfo=timezone.utc),
        data_source=data_source,
    )
    SleepDetailsFactory(event_record=record, sleep_stages=stages)
    DataPointSeriesFactory(
        data_source=data_source,
        series_type=rmssd_type,
        recorded_at=datetime.combine(night, time(hrv_hour), tzinfo=timezone.utc),
        value=Decimal(str(value)),
    )


def _seed_users(db: Session, today: date) -> dict[str, UUID]:
    """Seed users covering the scored, unscored and stage-filtered cases."""
    rmssd_type = SeriesTypeDefinitionFactory.get_or_create_heart_rate_variability_rmssd()
    users: dict[str, UUID] = {}

    def new_source(name: str) -> DataSource:
        source = DataSourceFactory(user=UserFactory())
        users[name] = source.user_id
        return source

    # Same RMSSD every night → CV 0 → resilience score 100.
    steady = new_source("steady")
    for days_back in range(1, 11):
        _add_night(steady, rmssd_type, today - timedelta(days=days_back), 45)

    # Varying RMSSD across the whole backfill + lookback window.
    variable = new_source("variable")
    for days_back in range(1, 38):
        _add_night(variable, rmssd_type, today - timedelta(days=days_back), 30 + (days_back * 7) % 25)

    # Fewer nights than min_days_required → never scored.
    sparse = new_source("sparse")
    for days_back in range(1, 4):
        _add_night(sparse, rmssd_type, today - timedelta(days=days_back), 50)

    # RMSSD recorded outside the sleep session → filtered out, never scored.
    daytime = new_source("daytime")
    for days_back in range(1, 11):
        _add_night(daytime, rmssd_type, today - timedelta(days=days_back), 50, hrv_hour=14)

    # Staged sleep: only the samples inside the asleep stage count.
    staged = new_source("staged")
    for days_back in range(1, 15):
        night = today - timedelta(days=days_back)
        stages = [
            {
                "stage": "awake",
                "start_time": datetime.combine(night, time(0), tzinfo=timezone.utc).isoformat(),
                "end_time": datetime.combine(night, time(1), tzinfo=timezone.utc).isoformat(),
            },
            {
                "stage": "light",
                "start_time": datetime.combine(night, time(1), tzinfo=timezone.utc).isoformat(),
                "end_time": datetime.combine(night, time(8), tzinfo=timezone.utc).isoformat(),
            },
        ]
        _add_night(staged, rmssd_type, night, 40 + days_back % 4 * 5, hrv_hour=3, stages=stages)
        DataPointSeriesFactory(  # inside the session, but during the awake stage
            data_source=staged,
            series_type=rmssd_type,
            recorded_at=datetime.combine(night, time(0, 30), tzinfo=timezone.utc),
            value=Decimal("200"),
        )

    db.flush()
    return users


def _saved_scores(db: Session, user_ids: list[UUID]) -> dict[tuple[UUID, date], tuple[float, dict]]:
    rows = (
        db.query(HealthScore)
        .filter(
            HealthScore.user_id.in_(user_ids),
            HealthScore.category == HealthScoreCategory.RESILIENCE,
            HealthScore.provider == ProviderName.INTERNAL,
        )
        .all()
    )
    return {(row.user_id, row.recorded_at.date()): (float(row.value), row.components) for row in rows}


class TestFillMissingResilienceScoresTask:
    """Test suite for fill_missing_resilience_scores task."""

    @pytest.mark.parametrize("user_count", [1, 10, 30])
    def test_select_count_does_not_grow_with_users(
        self,
        db: Session,
        mock_session_local: Callable[[Session], MagicMock],
        user_count: int,
    ) -> None:
        """One missing-scores query plus one sleep and one RMSSD query for the whole batch."""
        rmssd_type = SeriesTypeDefinitionFactory.get_or_create_heart_rate_variability_rmssd()
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        for _ in range(user_count):
            DataPointSeriesFactory(
                data_source=DataSourceFactory(user=UserFactory()),
                series_type=rmssd_type,
                value=Decimal("45"),
                recorded_at=yesterday,
            )
        db.flush()

        with _count_selects(db) as selects:
            _run_task(db, mock_session_local)

        assert len(selects) == 3

    def test_select_count_scales_with_chunks_not_users(
        self,
        db: Session,
        mock_session_local: Callable[[Session], MagicMock],
    ) -> None:
        """30 users in chunks of 10 → 1 + 2 * 3 SELECTs."""
        rmssd_type = SeriesTypeDefinitionFactory.get_or_create_heart_rate_variability_rmssd()
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        for _ in range(30):
            DataPointSeriesFactory(
                data_source=DataSourceFactory(user=UserFactory()),
                series_type=rmssd_type,
                value=Decimal("45"),
                recorded_at=yesterday,
            )
        db.flush()

        with patch(f"{TASK_MODULE}._USER_CHUNK_SIZE", 10), _count_selects(db) as selects:
            _run_task(db, mock_session_local)

        assert len(selects) == 7

    def test_batched_scores_match_single_user_path(
        self,
        db: Session,
        mock_session_local: Callable[[Session], MagicMock],
    ) -> None:
        """Scores saved in chunks equal get_hrv_cv_score computed one (user, date) at a time."""
        today = _db_today(db)
        users = _seed_users(db, today)
        user_ids = list(users.values())
        backfill_dates = [today - timedelta(days=i) for i in range(31)]

        expected: dict[tuple[UUID, date], tuple[float, dict]] = {}
        for user_id in user_ids:
            for ref_date in backfill_dates:
                result = resilience_score_service.get_hrv_cv_score(db, user_id, ref_date)
                if result.hrv_cv is None:
                    continue
                expected[(user_id, ref_date)] = (
                    result.hrv_cv,
                    {
                        "days_counted": {"value": result.days_counted, "qualifier": None},
                        "metric_type": {"value": None, "qualifier": result.metric_type},
                        "resilience_score": {"value": result.resilience_score, "qualifier": None},
                    },
                )

        # Chunks of 2 put users with different data in the same batch and split the rest.
        with patch(f"{TASK_MODULE}._USER_CHUNK_SIZE", 2):
            result = _run_task(db, mock_session_local)

        saved = _saved_scores(db, user_ids)
        assert saved.keys() == expected.keys()
        for key, (value, components) in saved.items():
            expected_value, expected_components = expected[key]
            assert value == pytest.approx(expected_value, abs=5e-4)
            assert components == expected_components

        assert result == {"saved": len(expected), "skipped": len(user_ids) * len(backfill_dates) - len(expected)}

        # Known values: constant RMSSD scores 100; unscorable users stay unscored.
        steady_scores = [v for (uid, _), v in saved.items() if uid == users["steady"]]
        assert steady_scores
        assert all(value == 0 and c["resilience_score"]["value"] == 100 for value, c in steady_scores)
        assert {uid for uid, _ in saved} == {users["steady"], users["variable"], users["staged"]}

        # Idempotent: nothing new on the next run.
        assert _run_task(db, mock_session_local)["saved"] == 0

    def test_failing_user_does_not_block_rest_of_chunk(
        self,
        db: Session,
        mock_session_local: Callable[[Session], MagicMock],
    ) -> None:
        """A batch failure is retried per user, so only the failing user is skipped."""
        today = _db_today(db)
        users = _seed_users(db, today)
        db.commit()  # keep the seed across the task's rollback
        bad_user = users["variable"]
        real_scoring = resilience_score_service.get_hrv_cv_scores_for_users

        def failing_for_bad_user(db_session: Session, dates_by_user: dict[UUID, list[date]]) -> Any:
            if bad_user in dates_by_user:
                raise RuntimeError("boom")
            return real_scoring(db_session, dates_by_user)

        with (
            patch(f"{TASK_MODULE}._USER_CHUNK_SIZE", 10),
            patch.object(resilience_score_service, "get_hrv_cv_scores_for_users", side_effect=failing_for_bad_user),
            patch(f"{TASK_MODULE}.log_and_capture_error") as mock_capture,
        ):
            _run_task(db, mock_session_local)

        scored_users = {uid for uid, _ in _saved_scores(db, list(users.values()))}
        assert scored_users == {users["steady"], users["staged"]}
        mock_capture.assert_called_once()
        assert mock_capture.call_args.kwargs["extra"]["user_id"] == str(bad_user)
