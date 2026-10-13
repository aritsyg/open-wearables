from datetime import date, datetime, timedelta, timezone
from typing import Any
from typing import cast as typing_cast
from uuid import UUID

from sqlalchemy import Date, and_, asc, cast, desc, select
from sqlalchemy.dialects.postgresql import insert

from app.database import DbSession
from app.models import DataSource, HealthScore
from app.repositories.repositories import CrudRepository
from app.schemas.enums import HealthScoreCategory
from app.schemas.model_crud.activities import HealthScoreCreate, HealthScoreQueryParams, HealthScoreUpdate
from app.utils.pagination import decode_cursor


class HealthScoreRepository(CrudRepository[HealthScore, HealthScoreCreate, HealthScoreUpdate]):
    def get_by_all_components(self, db_session: DbSession, components: list[str]) -> list[HealthScore]:
        """Return health scores whose components JSONB contains all specified keys (?& operator)."""
        return db_session.query(HealthScore).filter(HealthScore.components.has_all(components)).all()

    def get_by_any_component(self, db_session: DbSession, components: list[str]) -> list[HealthScore]:
        """Return health scores whose components JSONB contains any of the specified keys (?| operator)."""
        return db_session.query(HealthScore).filter(HealthScore.components.has_any(components)).all()

    def get_with_filters(
        self,
        db_session: DbSession,
        user_id: UUID,
        params: HealthScoreQueryParams,
    ) -> tuple[list[HealthScore], int]:
        filters = [HealthScore.user_id == user_id]

        if params.category:
            filters.append(HealthScore.category == params.category)
        if params.provider:
            filters.append(HealthScore.provider == params.provider)
        if params.data_source_id:
            filters.append(HealthScore.data_source_id == params.data_source_id)
        if params.start_datetime:
            filters.append(HealthScore.recorded_at >= params.start_datetime)
        if params.end_datetime:
            filters.append(HealthScore.recorded_at < params.end_datetime)

        query = db_session.query(HealthScore).filter(and_(*filters))

        total_count = query.count()
        results = query.order_by(desc(HealthScore.recorded_at)).offset(params.offset).limit(params.limit).all()
        return results, total_count

    def bulk_create(self, db_session: DbSession, creators: list[HealthScoreCreate]) -> None:
        """Bulk insert health scores, doing nothing on conflict with the unique constraint."""
        if not creators:
            return

        values = [c.model_dump() for c in creators]

        stmt = insert(HealthScore).values(values).on_conflict_do_nothing()
        db_session.execute(stmt)
        # Caller is responsible for commit — allows batching with other operations

    def get_latest_by_category(
        self,
        db_session: DbSession,
        user_id: UUID,
        category: HealthScoreCategory,
    ) -> HealthScore | None:
        """Return the most recent health score for a given category and user."""
        return (
            db_session.query(HealthScore)
            .filter(HealthScore.user_id == user_id, HealthScore.category == category)
            .order_by(desc(HealthScore.recorded_at))
            .first()
        )

    def delete_for_user_date(
        self,
        db_session: DbSession,
        user_id: UUID,
        score_date: date,
        category: HealthScoreCategory,
        provider: str = "internal",
    ) -> int:
        """Delete health scores matching user/category/provider/date without loading objects.

        Caller is responsible for commit. Returns deleted row count.
        Matches the whole day rather than an exact timestamp: sleep scores are anchored to
        the session's local wake time, and older rows sit at midnight of the same date.
        """
        day_start = datetime(score_date.year, score_date.month, score_date.day, tzinfo=timezone.utc)
        return (
            db_session.query(HealthScore)
            .filter(
                HealthScore.user_id == user_id,
                HealthScore.provider == provider,
                HealthScore.category == category,
                HealthScore.recorded_at >= day_start,
                HealthScore.recorded_at < day_start + timedelta(days=1),
            )
            .delete(synchronize_session=False)
        )

    def get_recovery_summaries(
        self,
        db_session: DbSession,
        user_id: UUID,
        start_date: datetime,
        end_date: datetime,
        cursor: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get recovery health scores for a date range with cursor-based pagination.

        Returns list of dicts with keys: recovery_date, provider, source, device_model,
        device_type, record_id, recorded_at, recovery_score, resting_heart_rate,
        hrv_rmssd_milli, spo2_percentage.
        Pages by date: returns each source's first score of the day for the next limit + 1
        dates after the cursor, so the caller can keep one score per date and still fill a
        page and detect has_more without a separate COUNT query.
        Ordering matches get_sleep_summaries: ASC by default, DESC when paginating backward.
        """
        recovery_date = cast(HealthScore.recorded_at, Date)
        recovery_filters = [
            HealthScore.user_id == user_id,
            HealthScore.category == HealthScoreCategory.RECOVERY,
            HealthScore.recorded_at >= start_date,
            HealthScore.recorded_at < end_date,
        ]

        # The page's dates. A date is one summary however many providers scored it, so
        # the limit counts dates, and a date's scores are never split across pages. The
        # cursor's record id is ignored: one summary per date makes the date enough.
        page_dates = (
            db_session.query(recovery_date.label("recovery_date")).filter(*recovery_filters).group_by(recovery_date)
        )
        backwards = False
        if cursor:
            cursor_ts, _, direction = decode_cursor(cursor)
            backwards = direction == "prev"
            if backwards:
                page_dates = page_dates.filter(recovery_date < cursor_ts.date())
            else:
                page_dates = page_dates.filter(recovery_date > cursor_ts.date())
        page_dates = page_dates.order_by(desc(recovery_date) if backwards else asc(recovery_date))
        page_dates_subquery = page_dates.limit(limit + 1).subquery()

        # A provider can send many scores a day (Suunto sends one per Balance sample),
        # and the caller keeps a source's earliest score of the day anyway, so only that
        # one is loaded.
        first_per_source = (
            db_session.query(HealthScore.id)
            .filter(
                *recovery_filters,
                recovery_date.in_(select(page_dates_subquery.c.recovery_date)),
            )
            .distinct(recovery_date, HealthScore.provider, HealthScore.data_source_id)
            .order_by(
                recovery_date,
                HealthScore.provider,
                HealthScore.data_source_id,
                asc(HealthScore.recorded_at),
                asc(HealthScore.id),
            )
            .subquery()
        )

        # Outer join so scores without a data_source (older rows) still come back.
        query = (
            db_session.query(HealthScore, DataSource)
            .outerjoin(DataSource, HealthScore.data_source_id == DataSource.id)
            .filter(HealthScore.id.in_(select(first_per_source.c.id)))
        )

        # Backward pagination returns the dates before the cursor, nearest first
        date_order = desc(recovery_date) if backwards else asc(recovery_date)
        rows = query.order_by(date_order, asc(HealthScore.recorded_at), asc(HealthScore.id)).all()

        return [
            {
                "recovery_date": row.recorded_at.date(),
                "provider": row.provider,
                "source": data_source.source if data_source else None,
                "device_model": data_source.device_model if data_source else None,
                "device_type": data_source.device_type if data_source else None,
                "record_id": row.id,
                "recorded_at": row.recorded_at,
                "recovery_score": int(row.value) if row.value is not None else None,
                "resting_heart_rate": typing_cast(dict, row.components or {})
                .get("resting_heart_rate", {})
                .get("value"),
                "hrv_rmssd_milli": typing_cast(dict, row.components or {}).get("hrv_rmssd_milli", {}).get("value"),
                "spo2_percentage": typing_cast(dict, row.components or {}).get("spo2_percentage", {}).get("value"),
            }
            for row, data_source in rows
        ]

    def get_latest_per_category(
        self,
        db_session: DbSession,
        user_id: UUID,
    ) -> list[HealthScore]:
        """Return the most recent score for each category for a given user.

        Uses PostgreSQL DISTINCT ON (category) for efficiency.
        """
        return (
            db_session.query(HealthScore)
            .filter(HealthScore.user_id == user_id)
            .distinct(HealthScore.category)
            .order_by(HealthScore.category, desc(HealthScore.recorded_at))
            .all()
        )
