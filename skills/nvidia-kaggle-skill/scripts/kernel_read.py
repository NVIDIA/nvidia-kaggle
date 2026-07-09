#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Download and display the source code of a Kaggle kernel."""

import argparse

from rich.console import Console
from rich.markdown import Markdown

from kernels.notebook_reader import NotebookReader
from kernels.paths import default_notebook_cache_dir
from runtime import load_project_env

load_project_env()

def read_kernel(
    kernel_ref: str,
    competition_id: str = "__unscoped__",
    raw: bool = False,
    force: bool = False,
):
    console = Console()
    cache_dir = default_notebook_cache_dir()

    console.print(f"[bold]Reading kernel:[/bold] {kernel_ref}")

    reader = NotebookReader(cache_dir=cache_dir)
    content = reader.read_kernel(
        kernel_ref,
        competition_id=competition_id,
        force_download=force,
    )

    if raw:
        raw_path = reader.get_raw_path(kernel_ref, competition_id)
        if raw_path and raw_path.exists():
            print(raw_path.read_text(errors="replace"))
        else:
            console.print("[yellow]Raw file not found[/yellow]")
        return

    rendered = content.render_readable()
    code_cells = sum(1 for c in content.cells if c.cell_type == "code")
    md_cells = sum(1 for c in content.cells if c.cell_type == "markdown")
    console.print(f"[dim]Cells: {len(content.cells)} ({code_cells} code, {md_cells} markdown) | {len(rendered)} chars[/dim]\n")

    console.print(Markdown(rendered))

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and display a Kaggle kernel")
    parser.add_argument("kernel_ref", help="Kernel reference (e.g. 'username/kernel-slug')")
    parser.add_argument("--competition-id", default="__unscoped__", help="Competition to scope cache under")
    parser.add_argument("--raw", action="store_true", help="Output raw file content instead of rendered markdown")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    if "/" not in args.kernel_ref:
        parser.error("kernel_ref must use Kaggle owner/slug format, for example 'username/kernel-slug'")

    try:
        read_kernel(
            args.kernel_ref,
            competition_id=args.competition_id,
            raw=args.raw,
            force=args.force,
        )
    except Exception as exc:
        Console(stderr=True).print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()
