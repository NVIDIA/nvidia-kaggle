#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fetch the public LB score for a single Kaggle kernel through the Kaggle SDK.

Usage: python fetch_kernel_score.py <kernel-slug-or-url>
"""

import sys

from kernels.kaggle_search import KaggleKernelSearchClient, parse_kernel_ref

def fetch_kernel_score(kernel_ref: str) -> float | None:
    """Fetch the public LB score from Kaggle SDK search.

    Args:
        kernel_ref: owner/slug or full Kaggle URL.

    Returns the public score as a float, or None if not found.
    """
    ref = parse_kernel_ref(kernel_ref)
    result = KaggleKernelSearchClient().get_kernel_score(ref)
    return result.score if result else None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <kernel-slug-or-url>", file=sys.stderr)
        sys.exit(1)

    ref = parse_kernel_ref(sys.argv[1])
    score = fetch_kernel_score(ref)
    print(f"{ref}: {score}")
