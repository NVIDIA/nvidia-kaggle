"""Kaggle SDK search helpers for public kernel score lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.search.services.search_api_service import SearchApiClient
from kagglesdk.search.types.search_api_service import (
    ApiSearchKernelsFilters,
    DocumentType,
    ListEntitiesDocument,
    ListEntitiesFilters,
    ListEntitiesRequest,
    ListSearchContentOrderBy,
)


@dataclass(frozen=True)
class KernelScore:
    """Public score metadata returned by Kaggle search."""

    ref: str
    title: str
    score: float | None
    votes: int | None = None
    has_linked_submission: bool = False


def parse_competition_slug(slug_or_url: str) -> str:
    """Extract a competition slug from a Kaggle URL or return the input unchanged."""
    match = re.search(r"kaggle\.com/competitions/([^/?#]+)", slug_or_url)
    return match.group(1) if match else slug_or_url


def parse_kernel_ref(slug_or_url: str) -> str:
    """Extract owner/slug from a Kaggle kernel URL or return the input unchanged."""
    match = re.search(r"kaggle\.com/code/([^/]+/[^/?#]+)", slug_or_url)
    return match.group(1) if match else slug_or_url


def _document_ref(document: ListEntitiesDocument) -> str | None:
    owner = document.owner_user
    username = getattr(owner, "user_name", "") if owner else ""
    if not username or not document.slug:
        return None
    return f"{username}/{document.slug}"


def _kernel_score(document: ListEntitiesDocument) -> KernelScore | None:
    kernel_document = document.kernel_document
    ref = _document_ref(document)
    if not kernel_document or not ref:
        return None

    has_submission = bool(kernel_document.has_linked_submission)
    raw_score = kernel_document.best_public_score
    score = float(raw_score) if has_submission or raw_score else None
    return KernelScore(
        ref=ref,
        title=document.title or document.slug,
        score=score,
        votes=document.votes,
        has_linked_submission=has_submission,
    )


class KaggleKernelSearchClient:
    """Search public Kaggle kernels through the supported Kaggle SDK."""

    def __init__(self, client: SearchApiClient | None = None):
        self._client = client or SearchApiClient(KaggleHttpClient())

    def _list_kernel_documents(
        self,
        query: str,
        *,
        order_by: ListSearchContentOrderBy = ListSearchContentOrderBy.LIST_SEARCH_CONTENT_ORDER_BY_VOTES,
        page_size: int = 50,
        max_pages: int | None = 3,
    ) -> list[ListEntitiesDocument]:
        documents: list[ListEntitiesDocument] = []
        page_token = ""
        page = 0

        while True:
            if max_pages is not None and page >= max_pages:
                break
            filters = ListEntitiesFilters()
            filters.query = query
            filters.document_types = [DocumentType.KERNEL]
            filters.kernel_filters = ApiSearchKernelsFilters()

            request = ListEntitiesRequest()
            request.filters = filters
            request.canonical_order_by = order_by
            request.page_size = page_size
            request.page_token = page_token

            response = self._client.list_entities(request)
            documents.extend(response.documents or [])
            page_token = response.next_page_token
            if not page_token:
                break
            page += 1

        return documents

    def list_kernel_scores(
        self,
        competition_slug_or_url: str,
        *,
        sort: str = "descending",
        page_size: int = 50,
        max_pages: int | None = 3,
    ) -> list[KernelScore]:
        """Return public kernel scores for a competition search query."""
        competition_slug = parse_competition_slug(competition_slug_or_url)
        order_by = (
            ListSearchContentOrderBy.LIST_SEARCH_CONTENT_ORDER_BY_HOTNESS
            if sort == "hotness"
            else ListSearchContentOrderBy.LIST_SEARCH_CONTENT_ORDER_BY_VOTES
        )
        documents = self._list_kernel_documents(
            competition_slug,
            order_by=order_by,
            page_size=page_size,
            max_pages=max_pages,
        )
        scores = _dedupe_scores(filter(None, (_kernel_score(d) for d in documents)))

        if sort == "ascending":
            return sorted(
                scores,
                key=lambda item: (item.score is None, item.score if item.score is not None else float("inf")),
            )
        if sort == "descending":
            return sorted(
                scores,
                key=lambda item: (item.score is None, -(item.score if item.score is not None else float("-inf"))),
            )
        return scores

    def get_kernel_score(self, kernel_ref_or_url: str) -> KernelScore | None:
        """Return public score metadata for an exact owner/slug kernel reference."""
        ref = parse_kernel_ref(kernel_ref_or_url)
        if "/" not in ref:
            raise RuntimeError(f"Invalid kernel ref '{kernel_ref_or_url}'. Expected 'owner/kernel-slug'.")

        expected_owner, expected_slug = ref.lower().split("/", 1)
        documents = self._list_kernel_documents(ref, page_size=50, max_pages=None)
        for document in documents:
            candidate_ref = _document_ref(document)
            if not candidate_ref:
                continue
            owner, slug = candidate_ref.lower().split("/", 1)
            if owner == expected_owner and slug == expected_slug:
                return _kernel_score(document)
        return None

    def get_kernel_scores(self, kernel_refs_or_urls: Iterable[str]) -> dict[str, KernelScore | None]:
        """Return score metadata for exact kernel refs without dropping missing scores."""
        results: dict[str, KernelScore | None] = {}
        for kernel_ref in kernel_refs_or_urls:
            ref = parse_kernel_ref(kernel_ref)
            results[ref] = self.get_kernel_score(ref)
        return results


def _dedupe_scores(scores: Iterable[KernelScore]) -> list[KernelScore]:
    seen: set[str] = set()
    unique: list[KernelScore] = []
    for score in scores:
        if score.ref in seen:
            continue
        seen.add(score.ref)
        unique.append(score)
    return unique
