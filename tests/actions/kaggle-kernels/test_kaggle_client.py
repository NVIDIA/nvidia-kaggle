# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

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
