#!/usr/bin/env python3
"""Classify a kernel's version history as real changes vs identical reruns.

A per-version public-LB score list is misleading on its own: many Kaggle
kernels are stochastic (random seeds, particle filters), so the *same code*
can post different scores across versions, and a notebook's title often
advertises its luckiest rerun. This tool downloads the versions, diffs their
code byte-for-byte, and tells you which score moves were real edits and which
were pure run-to-run noise.

For each version it fetches the verified public-LB score, then extracts code
only (code cells, with outputs/execution-counts stripped; `.ipynb` or `.py`),
normalizes trivial whitespace churn, hashes it, and unified-diffs consecutive
versions:

  * identical normalized hash  -> IDENTICAL rerun (score delta is noise)
  * non-empty diff             -> CHANGED (+added / -removed lines reported)

It also flags when two versions across the whole set share one code hash, and
prints the SHA so you can spot a fork/copy that `kernel-metadata.json` will not
show (kernel_sources usually lists only auto-generated packagemanager entries).

Usage:
    python diff_kernel_versions.py <owner/slug|code-url> [output_dir]
        [--versions 1,3,6] [--all] [--keep] [--include-outputs] [--json]

Without --versions/--all it downloads every *scored* version (the default,
matching how you'd audit a leaderboard notebook). Requires KAGGLE_API_TOKEN.
"""

import argparse
import difflib
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from kernels.archive import archive_kernel_version, kernel_version_scores


def code_of_dir(d: Path) -> str:
    """Return code-only text for an archived version dir (ipynb or py)."""
    nb, py = d / "source.ipynb", d / "source.py"
    if nb.exists():
        doc = json.loads(nb.read_text())
        cells = doc.get("cells")
        if cells is None:  # very old nbformat
            wks = doc.get("worksheets") or [{}]
            cells = wks[0].get("cells", [])
        parts = []
        for c in cells:
            if c.get("cell_type") != "code":
                continue
            src = c.get("source", c.get("input", ""))
            if isinstance(src, list):
                src = "".join(src)
            parts.append(src.rstrip())
        return ("\n\n".join(parts)).strip() + "\n"
    if py.exists():
        return py.read_text().rstrip() + "\n"
    raise FileNotFoundError(f"no source.ipynb or source.py in {d}")


def norm(code: str) -> list[str]:
    """Normalize away whitespace-only churn: strip trailing ws, drop blank lines."""
    return [ln.rstrip() for ln in code.splitlines() if ln.strip()]


def classify(kernel_ref: str, output_dir: Path, versions: list[int] | None,
             want_all: bool, include_outputs: bool) -> dict:
    info = kernel_version_scores(kernel_ref)
    all_versions = info.get("versions", [])
    by_num = {v["version_number"]: v for v in all_versions}

    if versions:
        targets = versions
    elif want_all:
        targets = [v["version_number"] for v in all_versions]
    else:  # default: every version that has a numeric score
        targets = [v["version_number"] for v in all_versions
                   if v.get("public_lb_numeric") is not None]
    targets = sorted(set(targets))
    if not targets:
        raise SystemExit("No versions to compare (none scored?). Use --all to force.")

    rows = []
    prev_norm = None
    hashes: dict[str, list[int]] = {}
    for vnum in targets:
        vmeta = by_num.get(vnum, {})
        try:
            archive_kernel_version(kernel_ref, str(output_dir), vnum,
                                   include_outputs=include_outputs, force=True)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(version=vnum, score=vmeta.get("public_lb"),
                             vs_prev="download failed", note=str(exc)[:120]))
            continue
        vdir = next(output_dir.glob(f"v{vnum:03d}__scriptVersionId-*"), None)
        if vdir is None:
            vdir = next((p for p in output_dir.glob("v*__scriptVersionId-*")
                         if int(re.match(r"v(\d+)", p.name).group(1)) == vnum), None)
        code = code_of_dir(vdir)
        cur = norm(code)
        h = hashlib.sha256("\n".join(cur).encode()).hexdigest()
        hashes.setdefault(h[:12], []).append(vnum)
        kind = "ipynb" if (vdir / "source.ipynb").exists() else "py"
        if prev_norm is None:
            status, added, removed = "first", 0, 0
        elif cur == prev_norm:
            status, added, removed = "IDENTICAL rerun", 0, 0
        else:
            diff = list(difflib.unified_diff(prev_norm, cur, lineterm=""))
            added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
            status = "CHANGED"
        rows.append(dict(version=vnum, score=vmeta.get("public_lb"), kind=kind,
                         vs_prev=status, added=added, removed=removed, codehash=h[:12]))
        prev_norm = cur

    dupes = {h: vs for h, vs in hashes.items() if len(vs) > 1}
    return dict(kernel_ref=kernel_ref, owner_slug=info.get("owner_slug"),
                kernel_slug=info.get("kernel_slug"), rows=rows,
                identical_code_groups=dupes)


def print_table(result: dict) -> None:
    print(f"\n{result['owner_slug']}/{result['kernel_slug']}")
    print(f"{'ver':>4}  {'public_lb':>10}  {'kind':>5}  {'vs previous':<16}  {'diff':>10}  code_sha")
    print("-" * 74)
    for r in result["rows"]:
        if r["vs_prev"] == "download failed":
            print(f"{r['version']:>4}  {str(r['score']):>10}  {'':>5}  download failed: {r.get('note','')}")
            continue
        diff = "" if r["vs_prev"] in ("first", "IDENTICAL rerun") else f"+{r['added']}/-{r['removed']}"
        print(f"{r['version']:>4}  {str(r['score']):>10}  {r['kind']:>5}  {r['vs_prev']:<16}  {diff:>10}  {r['codehash']}")
    groups = result["identical_code_groups"]
    if groups:
        print("\nidentical-code groups (same SHA across versions -> reruns; same SHA across kernels -> fork):")
        for h, vs in groups.items():
            print(f"  {h}  versions {vs}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kernel_ref", help="owner/kernel-slug or a Kaggle /code/ URL")
    p.add_argument("output_dir", nargs="?", help="where to write downloaded versions "
                   "(default: a temp dir removed on exit unless --keep)")
    p.add_argument("--versions", help="comma-separated version numbers to compare (e.g. 1,3,6)")
    p.add_argument("--all", action="store_true", dest="want_all",
                   help="compare every version, including unscored ones")
    p.add_argument("--include-outputs", action="store_true", help="keep cell outputs in downloads")
    p.add_argument("--keep", action="store_true", help="do not delete a temp output dir")
    p.add_argument("--json", action="store_true", dest="as_json", help="print JSON instead of a table")
    args = p.parse_args()

    versions = [int(x) for x in args.versions.split(",")] if args.versions else None
    tmp = None
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.mkdtemp(prefix="kver_diff_")
        out = Path(tmp)

    try:
        result = classify(args.kernel_ref, out, versions, args.want_all, args.include_outputs)
        if args.as_json:
            print(json.dumps(result, indent=2))
        else:
            print_table(result)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if tmp and not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
