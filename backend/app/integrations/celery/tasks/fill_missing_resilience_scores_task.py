from collections import defaultdict
from datetime import date, datetime, timezone
from logging import getLogger
from uuid import UUID, uuid4

from celery import shared_task
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.algorithms.config_algorithms import resilience_config
from app.config import settings
from app.database import SessionLocal
from app.schemas.enums import HealthScoreCategory, ProviderName, SeriesType, get_series_type_id
from app.schemas.model_crud.activities.health_score import HealthScoreCreate, ScoreComponent
from app.services.health_score_service import health_score_service
from app.services.scores.resilience_service import resilience_score_service
from app.utils.sentry_helpers import log_and_capture_error
from app.utils.structured_logging import log_structured

logger = getLogger(__name__)

_RMSSD_TYPE_ID = get_series_type_id(SeriesType.heart_rate_variability_rmssd)
_SDNN_TYPE_ID = get_series_type_id(SeriesType.heart_rate_variability_sdnn)

# Users scored per batch: each batch costs one sleep-records query, one RMSSD query and
# one INSERT. RMSSD can arrive every ~5 min overnight (~100 samples/night), so a batch
# loads up to about users * (score_backfill_days + lookback_days) * 100 ≈ 370k rows at
# 100 users with the defaults — small enough for a worker, large enough that the query
# count stops mattering.
_USER_CHUNK_SIZE = 100

# Find all (user, reference_date) pairs that are missing an OW resilience score.
# active_users: users who have any RMSSD or SDNN data in the extended lookback window.
# We cross-join with a date series covering the backfill window and filter out any
# dates that already have a score, ensuring idempotency.
_MISSING_RESILIENCE_SCORES_QUERY = text("""
    SELECT DISTINCT
        active_users.user_id,
        gs.target_date::date AS reference_date
    FROM generate_series(
        CURRENT_DATE - :backfill_days * INTERVAL '1 day',
        CURRENT_DATE,
        INTERVAL '1 day'
    ) AS gs(target_date)
    CROSS JOIN (
        SELECT DISTINCT ds.user_id
        FROM data_point_series dp
        JOIN data_source ds ON ds.id = dp.data_source_id
        WHERE dp.series_type_definition_id IN (:rmssd_id, :sdnn_id)
          AND dp.recorded_at >= CURRENT_DATE - :data_lookback_days * INTERVAL '1 day'
    ) AS active_users
    LEFT JOIN health_score hs
           ON hs.user_id  = active_users.user_id
          AND hs.category = 'resilience'
          AND hs.provider = 'internal'
          AND hs.recorded_at::date = gs.target_date::date
    WHERE hs.id IS NULL
    ORDER BY active_users.user_id, reference_date DESC
""")


@shared_task
def fill_missing_resilience_scores() -> dict[str, int]:
    """Find (user, date) pairs without an OW resilience score and calculate them.

    Runs frequently so scores appear shortly after any sync path that delivers
    HRV data. Uses a LEFT JOIN against generate_series to guarantee idempotency
    — already-scored dates are never re-processed.

    Only dates where the service returns a non-null hrv_cv are persisted.
    Dates where there is insufficient HRV data remain unscored so they are
    retried on the next run once more data arrives.
    """
    if not settings.ow_scores_enabled:
        return {"saved": 0, "skipped": 0}

    with SessionLocal() as db:
        rows = db.execute(
            _MISSING_RESILIENCE_SCORES_QUERY,
            {
                "backfill_days": settings.score_backfill_days,
                "rmssd_id": _RMSSD_TYPE_ID,
                "sdnn_id": _SDNN_TYPE_ID,
                "data_lookback_days": settings.score_backfill_days + resilience_config.lookback_days,
            },
        ).fetchall()

        if not rows:
            return {"saved": 0, "skipped": 0}

        # Group reference dates per user
        dates_by_user: dict[UUID, list[date]] = defaultdict(list)
        for user_id, reference_date in rows:
            dates_by_user[UUID(str(user_id))].append(reference_date)

        log_structured(
            logger,
            "info",
            f"Found {len(rows)} resilience score(s) missing across {len(dates_by_user)} user(s)",
            task="fill_missing_resilience_scores",
        )

        total_saved = 0
        total_skipped = 0

        user_ids = list(dates_by_user)
        for chunk_start in range(0, len(user_ids), _USER_CHUNK_SIZE):
            chunk = {uid: dates_by_user[uid] for uid in user_ids[chunk_start : chunk_start + _USER_CHUNK_SIZE]}
            try:
                saved = _score_users(db, chunk)
            except Exception as e:
                # Fall back to one user at a time so a single bad user only skips themself,
                # instead of blocking the whole chunk on every run.
                db.rollback()
                log_structured(
                    logger,
                    "warning",
                    f"Batched resilience scoring failed for {len(chunk)} user(s), retrying per user: {e}",
                    task="fill_missing_resilience_scores",
                )
                saved = 0
                for uid, reference_dates in chunk.items():
                    try:
                        saved += _score_users(db, {uid: reference_dates})
                    except Exception as user_error:
                        db.rollback()
                        log_and_capture_error(
                            user_error,
                            logger,
                            f"Failed to calculate resilience scores for user {uid}",
                            extra={"user_id": str(uid), "task": "fill_missing_resilience_scores"},
                        )

            total_saved += saved
            # Dates with hrv_cv=None have insufficient data — leave unscored for retry.
            total_skipped += sum(len(reference_dates) for reference_dates in chunk.values()) - saved

        log_structured(
            logger,
            "info",
            f"Resilience score fill complete: {total_saved} saved, {total_skipped} skipped",
            task="fill_missing_resilience_scores",
            saved=total_saved,
            skipped=total_skipped,
        )

        return {"saved": total_saved, "skipped": total_skipped}


def _score_users(db: Session, dates_by_user: dict[UUID, list[date]]) -> int:
    """Calculate and persist resilience scores for a group of users; return how many were saved.

    Commits on success. Raises on any fetch or save failure, leaving the rollback to the caller.
    """
    scores_by_user = resilience_score_service.get_hrv_cv_scores_for_users(db, dates_by_user)

    scores_to_save = [
        HealthScoreCreate(
            id=uuid4(),
            user_id=uid,
            data_source_id=None,
            provider=ProviderName.INTERNAL,
            category=HealthScoreCategory.RESILIENCE,
            value=result.hrv_cv,
            recorded_at=datetime(ref_date.year, ref_date.month, ref_date.day, tzinfo=timezone.utc),
            components={
                "days_counted": ScoreComponent(value=result.days_counted),
                "metric_type": ScoreComponent(qualifier=result.metric_type),
                "resilience_score": ScoreComponent(value=result.resilience_score),
            },
        )
        for uid, scores_by_date in scores_by_user.items()
        for ref_date, result in scores_by_date.items()
        if result.hrv_cv is not None
    ]

    if scores_to_save:
        health_score_service.bulk_create(db, scores_to_save)
        db.commit()
    return len(scores_to_save)
