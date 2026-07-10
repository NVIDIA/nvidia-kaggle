#!/usr/bin/env python3
"""Rewrite local image paths in a markdown file to absolute URLs, so the
result can be pasted directly into a Kaggle discussion/competition post.

Kaggle's discussion editor renders standard Markdown plus a couple of
extensions (GFM-style tables, and `![alt](url =WxH)` image sizing) - see
https://www.kaggle.com/product-feedback/82853. Headings, bold/italic, code
blocks, links, and tables paste through unchanged. The one thing that never
works unmodified is a locally-generated report's `![alt](plots/foo.png)`
image reference: Kaggle's editor (like any browser) fetches whatever the
`![]()` target resolves to, so it must already be a public URL that serves
raw image bytes - not a local path, and not an HTML share/viewer page (a
Google Drive share link or a GitHub *blob* URL will not work; a
raw.githubusercontent.com link or a GitHub Pages URL will).

This script does not host anything itself - it assumes the same folder
structure referenced by the markdown's local image paths has already been
published somewhere public (e.g. committed to a repo), and rewrites each
local `![]()` target to `<base-url>/<same-relative-path>`. `--verify` then
fetches every resulting image URL to confirm it actually resolves to image
bytes before you trust the output enough to paste it.

Usage:
    python kaggle_markdown_export.py <input.md> --base-url <public-folder-url> [-o OUT.md] [--verify]
"""

import argparse
import re
import sys
import urllib.error
import urllib.request

# Matches `![alt](path)` / `![alt](path =WxH)` where `path` is NOT already an
# absolute http(s) or data URL - i.e. exactly the references that need rewriting.
LOCAL_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(\s*(?!(?:https?://|data:))([^)\s]+)(\s+=\S+)?\s*\)")

# Matches every `![alt](target)` regardless of scheme, for the post-rewrite verify pass.
ANY_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(\s*([^)\s]+)(\s+=\S+)?\s*\)")


def rewrite_local_images(markdown_text: str, base_url: str) -> tuple[str, list[str]]:
    """Rewrite local image paths to `<base_url>/<path>`.

    Returns (rewritten_markdown, relative_paths_that_were_rewritten).
    """
    base = base_url.rstrip("/")
    rewritten: list[str] = []

    def _sub(match: re.Match) -> str:
        alt, path, dims = match.group(1), match.group(2), match.group(3) or ""
        if path.startswith("./"):
            path = path[2:]
        rewritten.append(path)
        return f"![{alt}]({base}/{path}{dims})"

    return LOCAL_IMAGE_RE.sub(_sub, markdown_text), rewritten


def find_image_urls(markdown_text: str) -> list[str]:
    """Return every `![]()` target in document order, any scheme."""
    return [m.group(2) for m in ANY_IMAGE_RE.finditer(markdown_text)]


def verify_image_url(url: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Confirm a URL resolves to image bytes. Returns (ok, detail)."""
    headers = {"User-Agent": "kaggle-markdown-export/1.0"}
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if resp.status == 200 and ctype.startswith("image/"):
                return True, f"200 {ctype}"
            return False, f"{resp.status} {ctype or '(no content-type)'}"
    except urllib.error.HTTPError as exc:
        if exc.code != 405:  # some hosts don't allow HEAD; fall through to a ranged GET below
            return False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - report any failure, don't crash the verify pass
        return False, f"error: {exc}"

    try:
        req = urllib.request.Request(url, method="GET", headers={**headers, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if resp.status in (200, 206) and ctype.startswith("image/"):
                return True, f"{resp.status} {ctype}"
            return False, f"{resp.status} {ctype or '(no content-type)'}"
    except Exception as exc:  # noqa: BLE001
        return False, f"error: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the source markdown file (e.g. brief.md)")
    parser.add_argument(
        "--base-url",
        required=True,
        help="Public URL of the folder that mirrors the markdown file's own location "
        "(e.g. a raw.githubusercontent.com/<owner>/<repo>/<branch>/<subdir> path)",
    )
    parser.add_argument("-o", "--output", help="Output path (default: <input>.kaggle.md next to the input)")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Fetch every resulting image URL and confirm it resolves to image bytes",
    )
    args = parser.parse_args()

    try:
        text = open(args.input).read()
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    new_text, rewritten = rewrite_local_images(text, args.base_url)

    out_path = args.output or re.sub(r"\.md$", "", args.input, flags=re.IGNORECASE) + ".kaggle.md"
    with open(out_path, "w") as f:
        f.write(new_text)

    base = args.base_url.rstrip("/")
    print(f"Rewrote {len(rewritten)} local image reference(s) to {base}/...")
    for p in rewritten:
        print(f"  {p}")
    print(f"Wrote {out_path}")

    if args.verify:
        urls = find_image_urls(new_text)
        print(f"\nVerifying {len(urls)} image URL(s)...")
        failures = 0
        for url in urls:
            ok, detail = verify_image_url(url)
            print(f"  [{'OK' if ok else 'FAIL'}] {url} ({detail})")
            failures += 0 if ok else 1
        if failures:
            print(f"\n{failures} image(s) failed verification - fix before pasting into Kaggle.", file=sys.stderr)
            raise SystemExit(1)
        print("\nAll image links verified.")


if __name__ == "__main__":
    main()
