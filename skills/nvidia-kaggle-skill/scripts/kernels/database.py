"""SQLite database for competition-scoped kernel metadata."""

import sqlite3
from pathlib import Path
from typing import Any, Optional

from .models import (
    CompetitionInfo,
    CompetitionSummary,
    KernelMetadata,
)

_KERNELS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS kernels (
    competition_id TEXT NOT NULL,
    kernel_ref     TEXT NOT NULL,
    title          TEXT NOT NULL,
    author         TEXT NOT NULL,
    total_votes    INTEGER NOT NULL DEFAULT 0,
    last_run_time  TEXT,
    is_private     INTEGER NOT NULL DEFAULT 0,
    ingested_at    TEXT NOT NULL,
    PRIMARY KEY (competition_id, kernel_ref)
);
"""

_COMPETITION_INFO_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS competition_info (
    competition_id    TEXT PRIMARY KEY,
    title             TEXT NOT NULL DEFAULT '',
    description       TEXT NOT NULL DEFAULT '',
    evaluation_metric TEXT NOT NULL DEFAULT '',
    url               TEXT NOT NULL DEFAULT '',
    deadline          TEXT,
    updated_at        TEXT NOT NULL
);
"""

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_kernels_competition ON kernels(competition_id);
CREATE INDEX IF NOT EXISTS idx_kernels_votes ON kernels(competition_id, total_votes DESC);
"""


class KernelDatabase:
    """Competition-scoped SQLite database for kernel metadata."""

    def __init__(self, db_path: str | Path = "data/kernels.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _initialize_schema(self) -> None:
        self._conn.execute(_KERNELS_TABLE_SQL)
        self._conn.execute(_COMPETITION_INFO_TABLE_SQL)
        self._conn.executescript(_INDEXES_SQL)
        self._conn.commit()

    # ── Kernel CRUD ──────────────────────────────────────────────────────

    def upsert_kernels(self, kernels: list[KernelMetadata]) -> tuple[int, int]:
        """Insert or update kernels. Returns (inserted, updated) counts."""
        inserted = 0
        updated = 0
        for kernel in kernels:
            existing = self._conn.execute(
                "SELECT 1 FROM kernels WHERE competition_id = ? AND kernel_ref = ?",
                (kernel.competition_id, kernel.ref),
            ).fetchone()

            self._conn.execute(
                """
                INSERT INTO kernels
                    (competition_id, kernel_ref, title, author,
                     total_votes, last_run_time, is_private, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(competition_id, kernel_ref) DO UPDATE SET
                    title = excluded.title,
                    author = excluded.author,
                    total_votes = excluded.total_votes,
                    last_run_time = excluded.last_run_time,
                    is_private = excluded.is_private,
                    ingested_at = excluded.ingested_at
                """,
                (
                    kernel.competition_id,
                    kernel.ref,
                    kernel.title,
                    kernel.author,
                    kernel.total_votes,
                    kernel.last_run_time.isoformat() if getattr(kernel.last_run_time, "isoformat", None) else kernel.last_run_time,
                    int(kernel.is_private),
                    kernel.ingested_at.isoformat() if getattr(kernel.ingested_at, "isoformat", None) else kernel.ingested_at,
                ),
            )
            if existing:
                updated += 1
            else:
                inserted += 1

        self._conn.commit()
        return inserted, updated

    def query_kernels(
        self,
        competition_id: str,
        *,
        search: Optional[str] = None,
        min_votes: Optional[int] = None,
        author: Optional[str] = None,
        sort_by: str = "total_votes",
        sort_order: str = "DESC",
        limit: int = 50,
        offset: int = 0,
    ) -> list[KernelMetadata]:
        """Query kernels with filtering and sorting."""
        clauses = ["competition_id = ?"]
        params: list[Any] = [competition_id]

        if search:
            clauses.append("(title LIKE ? OR author LIKE ? OR kernel_ref LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like])
        if min_votes is not None:
            clauses.append("total_votes >= ?")
            params.append(min_votes)
        if author:
            clauses.append("author LIKE ?")
            params.append(f"%{author}%")

        allowed_sort = {"total_votes", "last_run_time", "title", "author", "ingested_at"}
        if sort_by not in allowed_sort:
            sort_by = "total_votes"
        sort_order = "DESC" if sort_order.upper() == "DESC" else "ASC"

        sql = f"""
            SELECT * FROM kernels
            WHERE {' AND '.join(clauses)}
            ORDER BY {sort_by} {sort_order}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_kernel(row) for row in rows]

    def get_kernel(self, competition_id: str, kernel_ref: str) -> Optional[KernelMetadata]:
        """Get a single kernel by competition and ref."""
        row = self._conn.execute(
            "SELECT * FROM kernels WHERE competition_id = ? AND kernel_ref = ?",
            (competition_id, kernel_ref),
        ).fetchone()
        return self._row_to_kernel(row) if row else None

    # ── Aggregates ────────────────────────────────────────────────────────

    def list_competitions(self) -> list[dict[str, Any]]:
        """List all competitions with kernel counts."""
        rows = self._conn.execute(
            """
            SELECT competition_id, COUNT(*) AS cnt,
                   MIN(ingested_at) AS first_ingest, MAX(ingested_at) AS last_ingest
            FROM kernels
            GROUP BY competition_id
            ORDER BY cnt DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def competition_summary(self, competition_id: str) -> CompetitionSummary:
        """Get detailed stats for a competition."""
        kernel_count = self._conn.execute(
            "SELECT COUNT(*) FROM kernels WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()[0]

        author_rows = self._conn.execute(
            """
            SELECT author, COUNT(*) AS cnt
            FROM kernels
            WHERE competition_id = ?
            GROUP BY author
            ORDER BY cnt DESC
            LIMIT 10
            """,
            (competition_id,),
        ).fetchall()
        top_authors = [(row["author"], row["cnt"]) for row in author_rows]

        vote_row = self._conn.execute(
            """
            SELECT MIN(total_votes) AS min_v, MAX(total_votes) AS max_v, AVG(total_votes) AS avg_v
            FROM kernels
            WHERE competition_id = ?
            """,
            (competition_id,),
        ).fetchone()
        vote_stats = {
            "min": vote_row["min_v"] or 0,
            "max": vote_row["max_v"] or 0,
            "avg": round(vote_row["avg_v"] or 0, 1),
        }

        date_row = self._conn.execute(
            """
            SELECT MIN(last_run_time) AS earliest, MAX(last_run_time) AS latest
            FROM kernels
            WHERE competition_id = ? AND last_run_time IS NOT NULL
            """,
            (competition_id,),
        ).fetchone()
        date_range = None
        if date_row["earliest"] and date_row["latest"]:
            date_range = (date_row["earliest"], date_row["latest"])

        comp_info = self.get_competition_info(competition_id)

        return CompetitionSummary(
            competition_id=competition_id,
            kernel_count=kernel_count,
            top_authors=top_authors,
            vote_stats=vote_stats,
            date_range=date_range,
            competition_info=comp_info,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_kernel(row: sqlite3.Row) -> KernelMetadata:
        return KernelMetadata(
            competition_id=row["competition_id"],
            ref=row["kernel_ref"],
            title=row["title"],
            author=row["author"],
            total_votes=row["total_votes"],
            last_run_time=row["last_run_time"],
            is_private=bool(row["is_private"]),
            ingested_at=row["ingested_at"],
        )

    # ── Competition info ─────────────────────────────────────────────────

    def upsert_competition_info(self, info: CompetitionInfo) -> None:
        """Insert or update competition metadata."""
        self._conn.execute(
            """
            INSERT INTO competition_info
                (competition_id, title, description, evaluation_metric, url, deadline, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(competition_id) DO UPDATE SET
                title = excluded.title,
                description = excluded.description,
                evaluation_metric = excluded.evaluation_metric,
                url = excluded.url,
                deadline = excluded.deadline,
                updated_at = excluded.updated_at
            """,
            (
                info.competition_id,
                info.title,
                info.description,
                info.evaluation_metric,
                info.url,
                info.deadline,
                info.updated_at.isoformat() if getattr(info.updated_at, "isoformat", None) else info.updated_at,
            ),
        )
        self._conn.commit()

    def get_competition_info(self, competition_id: str) -> Optional[CompetitionInfo]:
        """Get cached competition metadata."""
        row = self._conn.execute(
            "SELECT * FROM competition_info WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        if not row:
            return None
        return CompetitionInfo(
            competition_id=row["competition_id"],
            title=row["title"],
            description=row["description"],
            evaluation_metric=row["evaluation_metric"],
            url=row["url"],
            deadline=row["deadline"],
            updated_at=row["updated_at"],
        )
