# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import sys

import kernel_read


def test_kernel_read_warns_when_script_version_id_is_ignored(capsys, monkeypatch):
    calls = []

    def fake_read_kernel(kernel_ref, **kwargs):
        calls.append((kernel_ref, kwargs))

    monkeypatch.setattr(kernel_read, "read_kernel", fake_read_kernel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kernel_read.py",
            "https://www.kaggle.com/code/alice/my-kernel?scriptVersionId=1",
            "--raw",
        ],
    )

    kernel_read.main()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Note: scriptVersionId is ignored" in captured.err
    assert "kernel_read always fetches the latest version" in captured.err
    assert calls == [
        (
            "alice/my-kernel",
            {"competition_id": "__unscoped__", "raw": True, "force": False},
        )
    ]
