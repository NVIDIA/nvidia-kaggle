#!/usr/bin/env python3
"""Small Playwright helpers for public Kaggle page scraping."""

from __future__ import annotations

from support.constants import (
    CONTENT_POLL_INTERVAL_MS,
    CONTENT_SETTLE_TIMEOUT_MS,
    CONTENT_STABLE_POLLS,
    DEFAULT_BROWSER_TIMEOUT_MS,
    LONG_BROWSER_TIMEOUT_MS,
    MIN_RENDERED_TEXT_LEN,
)


def _validated_url(url: str) -> str:
    if not url or not url.strip():
        raise ValueError("A non-empty URL is required.")
    return url


def _goto_and_settle(
    page,
    url: str,
    timeout: int,
    wait_selector: str | None,
    wait_for_text: tuple[str, ...] = (),
) -> None:
    """Navigate, then wait for real content instead of network idle.

    Kaggle is a single-page app that polls in the background, so
    ``wait_until="networkidle"`` frequently never settles and the call hangs
    until it times out. Instead we wait for the DOM to load, then for either an
    explicit selector, specific text, or the body to contain a meaningful
    amount of text.

    ``wait_for_text`` is a tuple of strings; if any appears in the body we
    proceed. This lets a caller wait for an end-of-region boundary to render
    (e.g. "Data Explorer" on the dataset page) so extraction sees a fully
    bounded region instead of a half-hydrated one. It is best-effort and
    bounded by ``settle`` — if the text never renders we fall through rather
    than hang.
    """
    page.goto(url, wait_until="domcontentloaded", timeout=timeout)

    settle = min(CONTENT_SETTLE_TIMEOUT_MS, timeout)

    if wait_selector:
        try:
            page.wait_for_selector(wait_selector, timeout=settle)
            return
        except Exception:
            # Fall through to the generic settle wait below.
            pass

    # First, wait for *some* meaningful text to appear (cheap, single shot).
    try:
        page.wait_for_function(
            "(minLen) => (document.body && document.body.innerText "
            "&& document.body.innerText.trim().length > minLen)",
            arg=MIN_RENDERED_TEXT_LEN,
            timeout=settle,
        )
    except Exception:
        # Best effort: return whatever rendered. The caller handles thin pages
        # (e.g. "We can't find that page.") gracefully.
        return

    # If the caller named boundary text, wait (bounded) for any of it to render
    # so the region we extract is reliably bounded at read time.
    if wait_for_text:
        try:
            page.wait_for_function(
                "(needles) => { const t = (document.body && document.body.innerText) || '';"
                " return needles.some(n => t.includes(n)); }",
                arg=list(wait_for_text),
                timeout=settle,
            )
        except Exception:
            # Boundary never rendered in time; the cleaner remains the
            # safety net. Fall through to the stability poll.
            pass

    # Then poll until the body text stops growing, so we don't capture a
    # half-hydrated page (e.g. header rendered but the description still
    # streaming in). Bounded by `settle` so it can never hang.
    deadline_polls = max(1, settle // max(1, CONTENT_POLL_INTERVAL_MS))
    last_len = -1
    stable = 0
    for _ in range(deadline_polls):
        try:
            cur_len = page.evaluate(
                "() => (document.body && document.body.innerText "
                "? document.body.innerText.trim().length : 0)"
            )
        except Exception:
            break
        if cur_len == last_len:
            stable += 1
            if stable >= CONTENT_STABLE_POLLS:
                break
        else:
            stable = 0
            last_len = cur_len
        page.wait_for_timeout(CONTENT_POLL_INTERVAL_MS)


def read_page_text(
    url: str,
    *,
    timeout: int = DEFAULT_BROWSER_TIMEOUT_MS,
    wait_selector: str | None = None,
    wait_for_text: tuple[str, ...] = (),
) -> str:
    """Load a page with stealth Playwright and return body text."""
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    url = _validated_url(url)
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            _goto_and_settle(page, url, timeout, wait_selector, wait_for_text)
            text = page.inner_text("body")
        finally:
            browser.close()
    return text


def evaluate_page(
    url: str,
    script: str,
    *,
    timeout: int = LONG_BROWSER_TIMEOUT_MS,
    wait_selector: str | None = None,
    wait_for_text: tuple[str, ...] = (),
):
    """Load a page with stealth Playwright and evaluate JavaScript."""
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    url = _validated_url(url)
    if not script or not script.strip():
        raise ValueError("A non-empty JavaScript snippet is required.")

    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            _goto_and_settle(page, url, timeout, wait_selector, wait_for_text)
            result = page.evaluate(script)
        finally:
            browser.close()
    return result
