# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for kaggle-discussions discussion_client (mocked HTTP)."""

import os
from unittest.mock import patch, MagicMock

import pytest

from discussions.discussion_client import DiscussionClient
from discussions.models import DiscussionRecord


@pytest.fixture
def client():
    with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "fake-token"}):
        c = DiscussionClient(max_retries=1, retry_delay=0.0)
        yield c
        c.close()


# ── _parse_discussion_doc ───────────────────────────────────────────────


class TestParseDiscussionDoc:
    def test_valid_doc(self, client):
        doc = {
            "id": "123",
            "title": "My Thread",
            "ownerUser": {"displayName": "alice", "userName": "alice99", "tier": "expert"},
            "votes": 42,
            "createTime": "2026-01-15T00:00:00Z",
            "updateTime": "2026-01-16T00:00:00Z",
            "discussionDocument": {"messageMarkdown": "Hello world"},
            "tags": [{"name": "tip"}, {"name": "data"}],
        }
        rec = client._parse_discussion_doc(doc, "comp")
        assert isinstance(rec, DiscussionRecord)
        assert rec.discussion_id == 123
        assert rec.title == "My Thread"
        assert rec.author == "alice"
        assert rec.author_username == "alice99"
        assert rec.votes == 42
        assert rec.tags == ["tip", "data"]
        assert "comp/discussion/123" in rec.url

    def test_missing_id(self, client):
        assert client._parse_discussion_doc({}, "comp") is None

    def test_non_numeric_id(self, client):
        assert client._parse_discussion_doc({"id": "abc"}, "comp") is None

    def test_missing_owner(self, client):
        doc = {"id": "1", "title": "T"}
        rec = client._parse_discussion_doc(doc, "comp")
        assert rec is not None
        assert rec.author == ""

    def test_empty_tags(self, client):
        doc = {"id": "1", "title": "T", "tags": []}
        rec = client._parse_discussion_doc(doc, "comp")
        assert rec.tags == []

    def test_non_dict_tags(self, client):
        doc = {"id": "1", "title": "T", "tags": "not-a-list"}
        rec = client._parse_discussion_doc(doc, "comp")
        assert rec.tags == []


# ── _resolve_competition_title ──────────────────────────────────────────


class TestResolveCompetitionTitle:
    def test_resolves_slug_to_title(self, client):
        response = [
            {"ref": "https://www.kaggle.com/competitions/playground-series-s6e5",
             "title": "Predicting F1 Pit Stops"},
        ]
        with patch.object(client, "_get", return_value=response):
            title = client._resolve_competition_title("playground-series-s6e5")
        assert title == "Predicting F1 Pit Stops"

    def test_returns_none_when_no_match(self, client):
        response = [{"ref": "https://www.kaggle.com/competitions/other-comp", "title": "Other"}]
        with patch.object(client, "_get", return_value=response):
            assert client._resolve_competition_title("playground-series-s6e5") is None

    def test_returns_none_on_api_error(self, client):
        with patch.object(client, "_get", side_effect=RuntimeError("fail")):
            assert client._resolve_competition_title("comp") is None


# ── list_discussions ────────────────────────────────────────────────────


class TestListDiscussions:
    def test_basic(self, client):
        response = {
            "documents": [
                {"id": "1", "title": "Thread 1", "votes": 10},
                {"id": "2", "title": "Thread 2", "votes": 5},
            ],
        }
        with patch.object(client, "_post_search", return_value=response):
            results = client.list_discussions("comp", max_pages=1)
        assert len(results) == 2
        assert results[0].discussion_id == 1

    def test_searches_by_title_not_slug(self, client):
        """list_discussions should resolve slug to title for the search query."""
        comp_list = [
            {"ref": "https://www.kaggle.com/competitions/my-comp",
             "title": "My Competition Title"},
        ]
        search_response = {"documents": [{"id": "1", "title": "Thread"}]}
        with patch.object(client, "_get", return_value=comp_list), \
             patch.object(client, "_post_search", return_value=search_response) as mock_search:
            client.list_discussions("my-comp", max_pages=1)
        call_body = mock_search.call_args[0][0]
        assert call_body["filters"]["query"] == "My Competition Title"

    def test_falls_back_to_slug_when_resolution_fails(self, client):
        search_response = {"documents": []}
        with patch.object(client, "_get", side_effect=RuntimeError("fail")), \
             patch.object(client, "_post_search", return_value=search_response) as mock_search:
            client.list_discussions("titanic", max_pages=1)
        call_body = mock_search.call_args[0][0]
        assert call_body["filters"]["query"] == "titanic"

    def test_pagination_stops_on_empty(self, client):
        page1 = {"documents": [{"id": "1", "title": "T"}], "nextPageToken": "tok"}
        page2 = {"documents": []}
        with patch.object(client, "_post_search", side_effect=[page1, page2]):
            results = client.list_discussions("comp")
        assert len(results) == 1

    def test_max_pages(self, client):
        page = {"documents": [{"id": "1", "title": "T"}], "nextPageToken": "tok"}
        with patch.object(client, "_post_search", return_value=page) as mock:
            results = client.list_discussions("comp", max_pages=2)
        assert mock.call_count == 2

    def test_empty_result(self, client):
        with patch.object(client, "_post_search", return_value={"documents": []}):
            results = client.list_discussions("comp")
        assert results == []

    def test_search_failure_raises(self, client):
        with patch.object(client, "_post_search", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="results may be incomplete"):
                client.list_discussions("comp")


# ── get_discussion_detail ───────────────────────────────────────────────


class TestGetDiscussionDetail:
    def test_with_nested_comments(self, client):
        response = {
            "forumTopic": {
                "totalMessages": 4,
                "parentName": "comp",
                "firstMessage": {
                    "content": "OP body",
                    "author": {"displayName": "alice", "userName": "alice99", "tier": "expert"},
                },
                "comments": [
                    {
                        "author": {"displayName": "bob"},
                        "rawMarkdown": "Comment 1",
                        "votes": {"totalVotes": 3},
                        "postDate": "2026-01-15",
                        "replies": [
                            {
                                "author": {"displayName": "carol"},
                                "rawMarkdown": "Reply 1",
                                "votes": {"totalVotes": 1},
                                "postDate": "2026-01-16",
                                "replies": [],
                            },
                        ],
                    },
                    {
                        "author": {"displayName": "dave"},
                        "content": "Comment 2",
                        "votes": {},
                        "postDate": "",
                        "replies": [],
                    },
                ],
            }
        }
        with patch.object(client, "_post_discussion_service", return_value=response):
            total, comments, first_md, op_author = client.get_discussion_detail(100)

        assert total == 4
        assert first_md == "OP body"
        assert op_author["displayName"] == "alice"
        assert op_author["username"] == "alice99"
        assert len(comments) == 3
        assert comments[0].author == "bob"
        assert comments[0].votes == 3
        assert comments[1].author == "carol"
        assert comments[2].author == "dave"

    def test_empty_comments(self, client):
        response = {"forumTopic": {"totalMessages": 0, "comments": []}}
        with patch.object(client, "_post_discussion_service", return_value=response):
            total, comments, first_md, op_author = client.get_discussion_detail(1)
        assert total == 0
        assert comments == []
        assert first_md == ""


# ── get_competition_info ────────────────────────────────────────────────


class TestGetCompetitionInfo:
    def test_found(self, client):
        response = [
            {
                "ref": "https://kaggle.com/competitions/titanic",
                "title": "Titanic",
                "description": "Predict survival",
                "evaluationMetric": "accuracy",
                "url": "https://kaggle.com/competitions/titanic",
                "deadline": "2026-12-31",
            }
        ]
        with patch.object(client, "_get", return_value=response):
            info = client.get_competition_info("titanic")
        assert info.title == "Titanic"
        assert info.evaluation_metric == "accuracy"

    def test_no_match_falls_back_to_slug_only(self, client):
        response = [{"ref": "other/comp", "title": "Other"}]
        with patch.object(client, "_get", return_value=response):
            info = client.get_competition_info("titanic")
        assert info.title == "titanic"
        assert info.url == "https://www.kaggle.com/competitions/titanic"

    def test_empty_list(self, client):
        with patch.object(client, "_get", return_value=[]):
            info = client.get_competition_info("titanic")
        assert info.title == "titanic"


# ── error handling ──────────────────────────────────────────────────────


class TestErrorHandling:
    def test_get_raises_after_retries(self, client):
        with patch.object(client, "_get", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError):
                client.get_competition_info("comp")

    def test_token_required(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("KAGGLE_API_TOKEN", None)
            with pytest.raises(RuntimeError, match="KAGGLE_API_TOKEN"):
                DiscussionClient()
