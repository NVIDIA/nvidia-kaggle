# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

from kernels.kaggle_client import KaggleKernelClient


class TestSplitKernelRef:
    def test_split_kernel_ref_accepts_code_url(self):
        owner, slug = KaggleKernelClient._split_kernel_ref(
            "https://www.kaggle.com/code/alice/my-kernel?scriptVersionId=1"
        )
        assert owner == "alice"
        assert slug == "my-kernel"

    def test_split_kernel_ref_accepts_owner_slug(self):
        owner, slug = KaggleKernelClient._split_kernel_ref("alice/my-kernel")
        assert owner == "alice"
        assert slug == "my-kernel"

    def test_split_kernel_ref_rejects_invalid_ref(self):
        with pytest.raises(RuntimeError, match="Invalid kernel ref"):
            KaggleKernelClient._split_kernel_ref("invalid-ref")


class TestKaggleKernelClient:
    @pytest.fixture
    def client(self):
        with patch(
            "kernels.kaggle_client.require_kaggle_token",
            return_value="test-token",
        ):
            return KaggleKernelClient(max_retries=1)

    def test_get_returns_json_response(self, client):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"kernels": []}

        mock_http_client = Mock()
        mock_http_client.__enter__ = Mock(return_value=mock_http_client)
        mock_http_client.__exit__ = Mock(return_value=None)
        mock_http_client.get.return_value = response

        with patch(
            "kernels.kaggle_client.httpx.Client",
            return_value=mock_http_client,
        ):
            result = client._get("/kernels/list")

        assert result == {"kernels": []}
        mock_http_client.get.assert_called_once()

    def test_get_raises_runtime_error_after_retries(self, client):
        response = Mock()
        response.status_code = 500

        error = httpx.HTTPStatusError(
            "Server error",
            request=Mock(),
            response=response,
        )

        mock_http_client = Mock()
        mock_http_client.__enter__ = Mock(return_value=mock_http_client)
        mock_http_client.__exit__ = Mock(return_value=None)
        mock_http_client.get.side_effect = error

        with patch(
            "kernels.kaggle_client.httpx.Client",
            return_value=mock_http_client,
        ):
            with pytest.raises(RuntimeError, match="Kaggle API call failed"):
                client._get("/kernels/list")

    def test_pull_kernel_writes_source_and_metadata(self, client, tmp_path):
        response = {
            "blob": {
                "id": "123",
                "idNo": 456,
                "source": '{"cells": []}',
                "kernelType": "notebook",
                "language": "python",
            },
            "metadata": {
                "kernelDataSources": ["dataset/example"],
            },
        }

        with patch.object(client, "_get", return_value=response):
            result = client.pull_kernel("alice/my-kernel", tmp_path)

        assert result == Path(tmp_path)

        notebook = tmp_path / "my-kernel.ipynb"
        metadata_file = tmp_path / "kernel-metadata.json"

        assert notebook.read_text(encoding="utf-8") == '{"cells": []}'

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        assert metadata["id"] == "123"
        assert metadata["id_no"] == 456
        assert metadata["slug"] == "my-kernel"
        assert metadata["kernel_type"] == "notebook"
        assert metadata["language"] == "python"
        assert metadata["kernelDataSources"] == ["dataset/example"]