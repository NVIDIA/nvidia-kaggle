#!/usr/bin/env python3
"""Small Playwright helpers for public Kaggle page scraping."""

from __future__ import annotations

from support.constants import DEFAULT_BROWSER_TIMEOUT_MS, LONG_BROWSER_TIMEOUT_MS


def _validated_url(url: str) -> str:
    if not url or not url.strip():
        raise ValueError("A non-empty URL is required.")
    return url


def read_page_text(url: str, *, timeout: int = DEFAULT_BROWSER_TIMEOUT_MS) -> str:
    """Load a page with stealth Playwright and return body text."""
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    url = _validated_url(url)
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout)
        text = page.inner_text("body")
        browser.close()
    return text


def evaluate_page(url: str, script: str, *, timeout: int = LONG_BROWSER_TIMEOUT_MS):
    """Load a page with stealth Playwright and evaluate JavaScript."""
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    url = _validated_url(url)
    if not script or not script.strip():
        raise ValueError("A non-empty JavaScript snippet is required.")

    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout)
        result = page.evaluate(script)
        browser.close()
    return result
