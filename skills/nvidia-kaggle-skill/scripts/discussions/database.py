"""SQLite database for competition-scoped discussion metadata and comments."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from .models import (
    CompetitionInfo,
    CompetitionSummary,
    DiscussionComment,
    DiscussionRecord,
)

_DISCUSSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS discussions (
    competition_id   TEXT NOT NULL,
    discussion_id    INTEGER NOT NULL,
    title            TEXT NOT NULL DEFAULT '',
    author           TEXT NOT NULL DEFAULT '',
    author_username  TEXT NOT NULL DEFAULT '',
    author_tier      TEXT NOT NULL DEFAULT '',
    votes            INTEGER NOT NULL DEFAULT 0,
    comment_count    INTEGER NOT NULL DEFAULT 0,
    body_markdown    TEXT NOT NULL DEFAULT '',
    url              TEXT NOT NULL DEFAULT '',
    tags             TEXT NOT NULL DEFAULT '[]',
    created_at       TEXT,
    updated_at       TEXT,
    last_fetched_at  TEXT,
    ingested_at      TEXT NOT NULL,
    PRIMARY KEY (competition_id, discussion_id)
);
"""

_COMMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS discussion_comments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    discussion_id    INTEGER NOT NULL,
    competition_id   TEXT NOT NULL,
    author           TEXT NOT NULL DEFAULT '',
    author_username  TEXT NOT NULL DEFAULT '',
    author_tier      TEXT NOT NULL DEFAULT '',
    votes            INTEGER NOT NULL DEFAULT 0,
    body_markdown    TEXT NOT NULL DEFAULT '',
    created_at       TEXT,
    FOREIGN KEY (competition_id, discussion_id)
        REFERENCES discussions(competition_id, discussion_id)
        ON DELETE CASCADE ON UPDATE CASCADE
);
"""

_COMPETITION_INFO_TABLE = """
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

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_disc_competition ON discussions(competition_id);
CREATE INDEX IF NOT EXISTS idx_disc_votes ON discussions(competition_id, votes DESC);
CREATE INDEX IF NOT EXISTS idx_comments_disc ON discussion_comments(competition_id, discussion_id);
"""


class DiscussionDatabase:
    """SQLite database for discussion metadata and comments."""

    def __init__(self, db_path: str | Path = "data/discussions.db"):
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
        self._conn.execute(_DISCUSSIONS_TABLE)
        self._conn.execute(_COMMENTS_TABLE)
        self._conn.execute(_COMPETITION_INFO_TABLE)
        self._conn.executescript(_INDEXES)
        self._conn.commit()

    # ── Discussion CRUD ─────────────────────────────────────────────

    def upsert_discussions(self, discussions: list[DiscussionRecord]) -> tuple[int, int]:
        inserted = 0
        updated = 0
        for d in discussions:
            existing = self._conn.execute(
                "SELECT 1 FROM discussions WHERE competition_id = ? AND discussion_id = ?",
                (d.competition_id, d.discussion_id),
            ).fetchone()

            tags_json = json.dumps(d.tags) if d.tags else "[]"
            ingested = d.ingested_at.isoformat() if hasattr(d.ingested_at, "isoformat") else str(d.ingested_at)

            self._conn.execute(
                """
                INSERT INTO discussions
                    (competition_id, discussion_id, title, author, author_username, author_tier,
                     votes, comment_count, body_markdown, url, tags,
                     created_at, updated_at, last_fetched_at, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(competition_id, discussion_id) DO UPDATE SET
                    title = excluded.title,
                    author = excluded.author,
                    author_username = excluded.author_username,
                    author_tier = excluded.author_tier,
                    votes = excluded.votes,
                    comment_count = excluded.comment_count,
                    body_markdown = excluded.body_markdown,
                    url = excluded.url,
                    tags = excluded.tags,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    last_fetched_at = excluded.last_fetched_at,
                    ingested_at = excluded.ingested_at
                """,
                (
                    d.competition_id, d.discussion_id, d.title, d.author,
                    d.author_username, d.author_tier,
                    d.votes, d.comment_count, d.body_markdown, d.url,
                    tags_json, d.created_at, d.updated_at, d.last_fetched_at, ingested,
                ),
            )
            if existing:
                updated += 1
            else:
                inserted += 1

        self._conn.commit()
        return inserted, updated

    def query_discussions(
        self,
        competition_id: str,
        *,
        search: Optional[str] = None,
        min_votes: Optional[int] = None,
        author: Optional[str] = None,
        sort_by: str = "votes",
        sort_order: str = "DESC",
        limit: int = 50,
        offset: int = 0,
    ) -> list[DiscussionRecord]:
        clauses = ["competition_id = ?"]
        params: list[Any] = [competition_id]

        if search:
            clauses.append("(title LIKE ? OR author LIKE ? OR body_markdown LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like])
        if min_votes is not None:
            clauses.append("votes >= ?")
            params.append(min_votes)
        if author:
            clauses.append("author LIKE ?")
            params.append(f"%{author}%")

        allowed_sort = {"votes", "created_at", "updated_at", "title", "comment_count", "ingested_at"}
        if sort_by not in allowed_sort:
            sort_by = "votes"
        sort_order = "DESC" if sort_order.upper() == "DESC" else "ASC"

        sql = f"""
            SELECT * FROM discussions
            WHERE {' AND '.join(clauses)}
            ORDER BY {sort_by} {sort_order}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_discussion(row) for row in rows]

    def get_discussion(self, competition_id: str, discussion_id: int) -> Optional[DiscussionRecord]:
        row = self._conn.execute(
            "SELECT * FROM discussions WHERE competition_id = ? AND discussion_id = ?",
            (competition_id, discussion_id),
        ).fetchone()
        return self._row_to_discussion(row) if row else None

    # ── Comments ────────────────────────────────────────────────────

    def upsert_comments(self, comments: list[DiscussionComment]) -> int:
        if not comments:
            return 0
        disc_id = comments[0].discussion_id
        comp_id = comments[0].competition_id
        self._conn.execute(
            "DELETE FROM discussion_comments WHERE competition_id = ? AND discussion_id = ?",
            (comp_id, disc_id),
        )
        for c in comments:
            self._conn.execute(
                """
                INSERT INTO discussion_comments
                    (discussion_id, competition_id, author, author_username, author_tier,
                     votes, body_markdown, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (c.discussion_id, c.competition_id, c.author, c.author_username, c.author_tier,
                 c.votes, c.body_markdown, c.created_at),
            )
        self._conn.commit()
        return len(comments)

    def get_comments(self, competition_id: str, discussion_id: int) -> list[DiscussionComment]:
        rows = self._conn.execute(
            """
            SELECT * FROM discussion_comments
            WHERE competition_id = ? AND discussion_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (competition_id, discussion_id),
        ).fetchall()
        return [self._row_to_comment(row) for row in rows]

    def sync_comment_counts(self, competition_id: str) -> int:
        """Update comment_count on discussions from the actual stored comments."""
        self._conn.execute(
            """
            UPDATE discussions SET comment_count = (
                SELECT COUNT(*) FROM discussion_comments c
                WHERE c.competition_id = discussions.competition_id
                  AND c.discussion_id = discussions.discussion_id
            ) WHERE competition_id = ?
            """,
            (competition_id,),
        )
        self._conn.commit()
        changed = self._conn.execute("SELECT changes()").fetchone()[0]
        return changed

    # ── Aggregates ──────────────────────────────────────────────────

    def list_competitions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT competition_id, COUNT(*) AS cnt,
                   MIN(ingested_at) AS first_ingest, MAX(ingested_at) AS last_ingest
            FROM discussions
            GROUP BY competition_id
            ORDER BY cnt DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def competition_summary(self, competition_id: str) -> CompetitionSummary:
        disc_count = self._conn.execute(
            "SELECT COUNT(*) FROM discussions WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()[0]

        author_rows = self._conn.execute(
            """
            SELECT author, COUNT(*) AS cnt FROM discussions
            WHERE competition_id = ? GROUP BY author ORDER BY cnt DESC LIMIT 10
            """,
            (competition_id,),
        ).fetchall()
        top_authors = [(row["author"], row["cnt"]) for row in author_rows]

        vote_row = self._conn.execute(
            "SELECT MIN(votes) AS min_v, MAX(votes) AS max_v, AVG(votes) AS avg_v FROM discussions WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        vote_stats = {
            "min": vote_row["min_v"] or 0,
            "max": vote_row["max_v"] or 0,
            "avg": round(vote_row["avg_v"] or 0, 1),
        }

        date_row = self._conn.execute(
            "SELECT MIN(created_at) AS earliest, MAX(created_at) AS latest FROM discussions WHERE competition_id = ? AND created_at IS NOT NULL",
            (competition_id,),
        ).fetchone()
        date_range = None
        if date_row["earliest"] and date_row["latest"]:
            date_range = (date_row["earliest"], date_row["latest"])

        comp_info = self.get_competition_info(competition_id)

        return CompetitionSummary(
            competition_id=competition_id,
            discussion_count=disc_count,
            top_authors=top_authors,
            vote_stats=vote_stats,
            date_range=date_range,
            competition_info=comp_info,
        )

    # ── Competition info ────────────────────────────────────────────

    def upsert_competition_info(self, info: CompetitionInfo) -> None:
        self._conn.execute(
            """
            INSERT INTO competition_info
                (competition_id, title, description, evaluation_metric, url, deadline, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(competition_id) DO UPDATE SET
                title = excluded.title, description = excluded.description,
                evaluation_metric = excluded.evaluation_metric, url = excluded.url,
                deadline = excluded.deadline, updated_at = excluded.updated_at
            """,
            (
                info.competition_id, info.title, info.description,
                info.evaluation_metric, info.url, info.deadline,
                info.updated_at.isoformat() if hasattr(info.updated_at, "isoformat") else str(info.updated_at),
            ),
        )
        self._conn.commit()

    def get_competition_info(self, competition_id: str) -> Optional[CompetitionInfo]:
        row = self._conn.execute(
            "SELECT * FROM competition_info WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        if not row:
            return None
        return CompetitionInfo(
            competition_id=row["competition_id"],
            title=row["title"], description=row["description"],
            evaluation_metric=row["evaluation_metric"], url=row["url"],
            deadline=row["deadline"], updated_at=row["updated_at"],
        )

    # ── Row helpers ─────────────────────────────────────────────────

    @staticmethod
    def _row_to_discussion(row: sqlite3.Row) -> DiscussionRecord:
        tags_raw = row["tags"]
        try:
            tags = json.loads(tags_raw) if tags_raw else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        keys = row.keys()
        return DiscussionRecord(
            competition_id=row["competition_id"],
            discussion_id=row["discussion_id"],
            title=row["title"], author=row["author"],
            author_username=row["author_username"] if "author_username" in keys else "",
            author_tier=row["author_tier"] if "author_tier" in keys else "",
            votes=row["votes"], comment_count=row["comment_count"],
            body_markdown=row["body_markdown"], url=row["url"],
            tags=tags,
            created_at=row["created_at"], updated_at=row["updated_at"],
            last_fetched_at=row["last_fetched_at"], ingested_at=row["ingested_at"],
        )

    @staticmethod
    def _row_to_comment(row: sqlite3.Row) -> DiscussionComment:
        keys = row.keys()
        return DiscussionComment(
            id=row["id"], discussion_id=row["discussion_id"],
            competition_id=row["competition_id"], author=row["author"],
            author_username=row["author_username"] if "author_username" in keys else "",
            author_tier=row["author_tier"] if "author_tier" in keys else "",
            votes=row["votes"], body_markdown=row["body_markdown"],
            created_at=row["created_at"],
        )
