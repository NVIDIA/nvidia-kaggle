#!/usr/bin/env python3
"""Runtime helpers shared by the unified Kaggle skill scripts."""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from pathlib import Path


def competition_slug(value: str) -> str:
    """Extract a Kaggle competition slug from a slug or competition URL."""
    match = re.search(r"kaggle\.com/competitions/([^/?#]+)", value)
    return match.group(1) if match else value.strip().strip("/")


def kernel_ref(value: str) -> str:
    """Extract owner/kernel-slug from a Kaggle code URL or return a ref as-is."""
    match = re.search(r"kaggle\.com/code/([^/?#]+)/([^/?#]+)", value)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return value.strip()


def require_kaggle_token() -> str:
    """Return KAGGLE_API_TOKEN or raise a clear runtime error."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("Kaggle credentials not found. Set KAGGLE_API_TOKEN in the environment.")
    return token


def kaggle_api():
    """Return an authenticated KaggleApi instance.

    Loads .env (so KAGGLE_API_TOKEN is available), verifies the token is
    present, then authenticates the official Kaggle client.
    """
    load_project_env()
    require_kaggle_token()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


class _MarkdownTextParser(HTMLParser):
    """Render the subset of HTML Kaggle uses in competition pages to markdown.

    Handles headings, paragraphs, line breaks, list items, and table cells.
    Anchor hrefs are preserved as markdown links so source URLs survive.
    Unknown tags are dropped but their text content is kept.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._href: str | None = None

    def handle_starttag(self, tag, attrs):
        if re.fullmatch(r"h[1-6]", tag):
            self._parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag == "br":
            self._parts.append("\n")
        elif tag in ("p", "div", "tr"):
            self._parts.append("\n\n")
        elif tag in ("td", "th"):
            self._parts.append(" | ")
        elif tag == "a":
            self._href = dict(attrs).get("href")
            self._parts.append("[")

    def handle_endtag(self, tag):
        if re.fullmatch(r"h[1-6]", tag):
            self._parts.append("\n")
        elif tag == "a":
            self._parts.append(f"]({self._href})" if self._href else "]")
            self._href = None

    def handle_data(self, data):
        self._parts.append(data)

    def get_markdown(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_markdown(content: str) -> str:
    """Convert Kaggle page HTML (or markdown-with-inline-HTML) to markdown text."""
    if not content or not content.strip():
        return ""
    if "<" not in content:
        return content.strip()
    parser = _MarkdownTextParser()
    parser.feed(content)
    return parser.get_markdown()


def competition_pages(slug: str) -> dict[str, str]:
    """Return a competition's content pages as {lowercased-name: markdown}.

    Uses the Kaggle API (`competition_list_pages`) instead of scraping the
    public web page. Page names include 'description', 'evaluation', 'rules',
    and 'data-description'.
    """
    api = kaggle_api()
    pages = api.competition_list_pages(slug) or []
    result: dict[str, str] = {}
    for page in pages:
        data = page.to_dict() if hasattr(page, "to_dict") else dict(page)
        name = (data.get("name") or "").strip().lower()
        content = data.get("content") or ""
        if name:
            result[name] = content
    return result


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing pyproject.toml, or fall back to cwd."""
    explicit = os.environ.get("PROJECT_ROOT")
    if explicit:
        return Path(explicit).resolve()

    candidates = [Path.cwd().resolve()]
    if start:
        candidates.insert(0, start.resolve())
    candidates.extend(Path(__file__).resolve().parents)

    for candidate in candidates:
        probe = candidate if candidate.is_dir() else candidate.parent
        for parent in (probe, *probe.parents):
            if (parent / "pyproject.toml").exists():
                return parent

    return Path.cwd().resolve()


def load_project_env(root: Path | None = None) -> Path:
    """Load .env from the project root and return that root."""
    project_root = root or find_project_root()
    env_path = project_root / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)
    return project_root
