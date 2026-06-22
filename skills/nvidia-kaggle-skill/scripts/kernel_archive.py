#!/usr/bin/env python3
"""Archive the best public-LB version of a Kaggle kernel.

Lists every version of a kernel, reads each version's public leaderboard score,
selects the best one, and downloads that version's source via Kaggle's internal
web service (no browser). Use --list to inspect versions without downloading.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernels.archive import (
    archive_best_kernel_source,
    resolve_kernel_versions,
    select_best_public_lb_version,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive the best public-LB version of a Kaggle kernel")
    parser.add_argument("kernel_ref", help="Kernel reference (owner/kernel-slug) or a Kaggle /code/ URL")
    parser.add_argument("output_dir", nargs="?", help="Directory to write the archived version into")
    parser.add_argument(
        "--score-direction",
        choices=["auto", "minimize", "maximize"],
        default="auto",
        help="Whether lower or higher LB is better (default: auto-infer from Kaggle metadata)",
    )
    parser.add_argument("--include-outputs", action="store_true", help="Include cell outputs in the source download")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing source file")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all versions with scores as JSON instead of downloading",
    )
    args = parser.parse_args()

    try:
        if args.list:
            rows = resolve_kernel_versions(args.kernel_ref)
            print(json.dumps(rows, indent=2))
            return

        if not args.output_dir:
            parser.error("output_dir is required unless --list is given")

        metadata = archive_best_kernel_source(
            args.kernel_ref,
            args.output_dir,
            score_direction=args.score_direction,
            include_outputs=args.include_outputs,
            force=args.force,
        )
        print(json.dumps(metadata, indent=2))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
