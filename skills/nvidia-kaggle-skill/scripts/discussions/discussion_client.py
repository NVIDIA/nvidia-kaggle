"""Kaggle API client for fetching competition discussions and comments.

Uses two APIs:
  - Search API (api.kaggle.com) for listing discussions
  - Kaggle web service endpoints for fetching per-discussion comments
    (requires XSRF token obtained from an authenticated Kaggle session)
"""

import time
from typing import Any

import httpx

from runtime import require_kaggle_token

from .models import CompetitionInfo, DiscussionComment, DiscussionRecord

KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"
KAGGLE_SEARCH_API = "https://api.kaggle.com/v1/search.SearchApiService/ListEntities"
KAGGLE_WEB_SERVICE_API = "https://www.kaggle.com/api/i"


class DiscussionClient:
    """Fetch discussions and comments from the Kaggle API."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self._token = require_kaggle_token()
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._xsrf_token: str | None = None
        self._session: httpx.Client | None = None

    def _reset_session(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._xsrf_token = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with httpx.Client(
                    headers=self._headers(), follow_redirects=True, timeout=30.0,
                ) as client:
                    resp = client.get(f"{KAGGLE_API_BASE}{path}", params=params)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
        raise RuntimeError(f"Kaggle API failed after {self._max_retries} retries: {last_exc}") from last_exc

    def _post_search(self, body: dict[str, Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with httpx.Client(
                    headers=self._headers(), follow_redirects=True, timeout=30.0,
                ) as client:
                    resp = client.post(KAGGLE_SEARCH_API, json=body)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
        raise RuntimeError(f"Search API failed after {self._max_retries} retries: {last_exc}") from last_exc

    # ── Discussion detail endpoint (requires XSRF session) ─────────

    def _ensure_session(self) -> httpx.Client:
        """Create a persistent session with XSRF token for discussion-detail calls."""
        if self._session is not None and self._xsrf_token:
            return self._session
        session = httpx.Client(
            follow_redirects=True, timeout=30.0,
            headers={"Authorization": f"Bearer {self._token}"},
        )
        session.get("https://www.kaggle.com")
        cookies = dict(session.cookies)
        self._xsrf_token = cookies.get("XSRF-TOKEN", "")
        if not self._xsrf_token:
            session.close()
            raise RuntimeError("Failed to obtain XSRF token from Kaggle session")
        self._session = session
        return session

    def _post_discussion_service(self, service_method: str, body: dict[str, Any]) -> Any:
        """POST to a Kaggle web service method."""
        session = self._ensure_session()
        url = f"{KAGGLE_WEB_SERVICE_API}/{service_method}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "X-XSRF-TOKEN": self._xsrf_token or "",
        }
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = session.post(url, json=body, headers=headers)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    self._reset_session()
                    session = self._ensure_session()
                    headers["X-XSRF-TOKEN"] = self._xsrf_token or ""
                    time.sleep(self._retry_delay * (attempt + 1))
        raise RuntimeError(f"Kaggle discussion service call failed after {self._max_retries} retries: {last_exc}") from last_exc

    def close(self) -> None:
        self._reset_session()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Slug-to-title resolution ────────────────────────────────────

    def _resolve_competition_title(self, slug: str) -> str | None:
        """Resolve a competition slug to its title via the competitions list API.

        The Search API matches on competition titles, not slugs, so we need
        to look up the title first for reliable discussion search.
        """
        try:
            results = self._get("/competitions/list", params={"search": slug})
        except RuntimeError:
            return None
        if not isinstance(results, list):
            return None
        for item in results:
            ref = item.get("ref", "") or ""
            if ref.rstrip("/").endswith(f"/{slug}"):
                return item.get("title")
        return None

    # ── Discussion listing (search API) ─────────────────────────────

    def list_discussions(
        self,
        competition: str,
        *,
        sort_by: str = "hotness",
        page_size: int = 20,
        max_pages: int | None = None,
    ) -> list[DiscussionRecord]:
        """Fetch discussions for a competition via the search API."""
        sort_map = {
            "hotness": "LIST_SEARCH_CONTENT_ORDER_BY_HOTNESS",
            "votes": "LIST_SEARCH_CONTENT_ORDER_BY_VOTES",
            "comments": "LIST_SEARCH_CONTENT_ORDER_BY_TOTAL_COMMENTS",
            "created": "LIST_SEARCH_CONTENT_ORDER_BY_DATE_CREATED",
            "updated": "LIST_SEARCH_CONTENT_ORDER_BY_DATE_UPDATED",
        }

        # The Search API matches on competition title, not slug.
        # Resolve the slug to a title first; fall back to the slug if
        # resolution fails (works for competitions where slug ≈ title).
        search_query = self._resolve_competition_title(competition) or competition

        all_discussions: list[DiscussionRecord] = []
        page_token: str | None = None
        page = 0

        while True:
            if max_pages and page >= max_pages:
                break

            body: dict[str, Any] = {
                "filters": {
                    "query": search_query,
                    "documentTypes": ["DOCUMENT_TYPE_TOPIC"],
                    "discussionFilters": {
                        "sourceType": "SEARCH_DISCUSSIONS_SOURCE_TYPE_COMPETITION",
                    },
                },
                "pageSize": page_size,
            }

            order = sort_map.get(sort_by)
            if order:
                body["canonicalOrderBy"] = order

            if page_token:
                body["pageToken"] = page_token

            try:
                data = self._post_search(body)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to fetch discussions for '{competition}' on page {page + 1}; "
                    f"results may be incomplete."
                ) from exc

            documents = data.get("documents", [])
            if not documents:
                break

            for doc in documents:
                disc = self._parse_discussion_doc(doc, competition)
                if disc:
                    all_discussions.append(disc)

            page_token = data.get("nextPageToken")
            if not page_token:
                break
            page += 1

        return all_discussions

    def _parse_discussion_doc(self, doc: dict, competition_id: str) -> DiscussionRecord | None:
        disc_id = doc.get("id")
        if not disc_id:
            return None
        try:
            disc_id = int(disc_id)
        except (ValueError, TypeError):
            return None

        title = doc.get("title", "")
        owner = doc.get("ownerUser", {}) if isinstance(doc.get("ownerUser"), dict) else {}
        author = owner.get("displayName", "")
        author_username = (owner.get("userName", "") or owner.get("url", "")).lstrip("/")
        author_tier = owner.get("tier", "")
        votes = doc.get("votes", 0) or 0
        created = doc.get("createTime", "")
        updated = doc.get("updateTime", "")

        disc_doc = doc.get("discussionDocument", {}) or {}
        body = disc_doc.get("messageMarkdown", "") or disc_doc.get("messageStripped", "")

        tags_raw = doc.get("tags", [])
        tags = [t.get("name", "") for t in tags_raw if isinstance(t, dict)] if isinstance(tags_raw, list) else []

        return DiscussionRecord(
            competition_id=competition_id,
            discussion_id=disc_id,
            title=title,
            author=author,
            author_username=author_username,
            author_tier=author_tier,
            votes=votes,
            comment_count=0,
            body_markdown=body,
            url=f"https://www.kaggle.com/competitions/{competition_id}/discussion/{disc_id}",
            tags=tags,
            created_at=created or None,
            updated_at=updated or None,
        )

    # ── Per-discussion comments ────────────────────────────────────

    def get_discussion_detail(
        self, discussion_id: int
    ) -> tuple[int, list[DiscussionComment], str, dict]:
        """Fetch full discussion detail including ALL comments.

        Returns:
            (total_messages, comments_list, first_message_markdown, op_author_info)
            op_author_info is a dict with keys: displayName, url (username), tier
        """
        data = self._post_discussion_service(
            "discussions.DiscussionsService/GetForumTopicById",
            {"forumTopicId": discussion_id, "includeComments": True},
        )

        topic = data.get("forumTopic", {})
        total_messages = topic.get("totalMessages", 0) or 0
        competition_id = topic.get("parentName", "")

        first_msg = topic.get("firstMessage", {}) or {}
        first_md = ""
        op_author: dict = {}
        if first_msg:
            first_content = first_msg.get("content", "")
            first_md = first_content if first_content else ""
            author_obj = first_msg.get("author", {}) or {}
            if isinstance(author_obj, dict):
                op_author = {
                    "displayName": author_obj.get("displayName", ""),
                    "username": (author_obj.get("userName", "") or author_obj.get("url", "")).lstrip("/"),
                    "tier": author_obj.get("tier", ""),
                }

        raw_comments = topic.get("comments", [])
        comments: list[DiscussionComment] = []

        def _extract_comments(nodes: list[dict]) -> None:
            """Recursively extract top-level comments and nested replies."""
            for c in nodes:
                author_obj = c.get("author", {}) or {}
                if not isinstance(author_obj, dict):
                    author_obj = {}
                author_name = author_obj.get("displayName", "")
                author_username = (author_obj.get("userName", "") or author_obj.get("url", "")).lstrip("/")
                author_tier = author_obj.get("tier", "")
                body_text = c.get("rawMarkdown", "") or c.get("content", "")
                votes_obj = c.get("votes", {})
                vote_count = votes_obj.get("totalVotes", 0) if isinstance(votes_obj, dict) else 0
                created = c.get("postDate", "")

                comments.append(DiscussionComment(
                    discussion_id=discussion_id,
                    competition_id=competition_id,
                    author=author_name,
                    author_username=author_username,
                    author_tier=author_tier,
                    votes=vote_count,
                    body_markdown=body_text,
                    created_at=created or None,
                ))

                replies = c.get("replies", [])
                if replies:
                    _extract_comments(replies)

        _extract_comments(raw_comments)

        return total_messages, comments, first_md, op_author

    # ── Competition info ────────────────────────────────────────────

    def get_competition_info(self, competition_slug: str) -> CompetitionInfo:
        results = self._get("/competitions/list", params={"search": competition_slug})

        match = {}
        if isinstance(results, list):
            for item in results:
                ref = item.get("ref", "") or ""
                if ref.rstrip("/").endswith(f"/{competition_slug}"):
                    match = item
                    break

        return CompetitionInfo(
            competition_id=competition_slug,
            title=match.get("title") or competition_slug,
            description=(match.get("description") or "")[:2000],
            evaluation_metric=match.get("evaluationMetric") or "",
            url=match.get("url") or f"https://www.kaggle.com/competitions/{competition_slug}",
            deadline=match.get("deadline"),
        )
