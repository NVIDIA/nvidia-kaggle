"""Shared cleanup for scraped Kaggle page text.

Both ``fetch_competition_info`` and ``fetch_dataset_info`` extract a slice of
``inner_text`` from a Kaggle page, which carries UI chrome: Material-icon
ligatures rendered as bare words, nav-tab labels, timeline widgets, and section
headers. Centralizing the line cleaner here keeps the two fetchers from drifting
apart (one getting a fix the other misses).

All matching is *line-anchored* on fully-stripped standalone lines — we never
substring-strip, so prose that legitimately contains a word like "link" or
"Description" is left untouched.
"""

from __future__ import annotations

import re

# Nav-bar tab labels that leak in from the competition page chrome.
NAV_ITEMS = {"Data", "Code", "Models", "Discussion", "Leaderboard", "Rules", "Overview"}

# Material Symbols icon ligatures render as bare lowercase tokens in the scraped
# text. Two shapes:
#   - multi-part with underscores: "keyboard_arrow_up", "all_inclusive",
#     "chevron_right", "emoji_events" -> caught by the regex below.
#   - single words: "link", "fullscreen", "search" -> exact-match set.
ICON_LIGATURE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
STANDALONE_ICON_WORDS = {
    "link",
    "search",
    "menu",
    "fullscreen",
    "download",
    "bookmark",
    "share",
    "expand_more",
    "expand_less",
    "chevron_right",
    "chevron_left",
    "more_vert",
    "content_copy",
}

# Anchors / collapse hints that carry no signal.
NOISE_LINES = {
    "Learn more",
    "keyboard_arrow_up",
    "keyboard_arrow_down",
    "This directory is empty.",
}

# File-browser listing entries like "gender_submission.csv(3.26 kB)" or
# "train.csv (61.19 MB)" — file-name followed by a parenthesized size. These
# are the data-page file browser, not description prose.
FILE_LISTING_RE = re.compile(
    r"^[\w.\-/ ]+\.\w+\s*\(\s*[\d.,]+\s*[kKmMgGtT]?i?[bB]\s*\)$"
)

# Data-source summary lines like "llm_20_questions(4 files)" or
# "titanic (3 files)" — a name followed by "(N file[s])", no extension or size.
# These are the file-browser header, not description prose.
FILE_COUNT_RE = re.compile(
    r"^[\w.\-/ ]+\(\s*\d+\s+files?\s*\)$"
)

# Section-label lines. Promoting them to subheadings makes the output read as
# structured prose instead of stray one-word lines.
SECTION_LABELS = {
    "Description",
    "Evaluation",
    "Timeline",
    "Prizes",
    "Citation",
    "Code Requirements",
    "Submission File",
    "Files",
    "Columns",
    "Frequently Asked Questions",
    "Acknowledgements",
    "Acknowledgments",
}

# A date-ish line such as "May 15, 2024" or "Aug 29, 2024".
DATE_LINE_RE = re.compile(r"^[A-Z][a-z]{2,8}\.?\s+\d{1,2},\s+\d{4}$")
# Short timeline labels that precede a date on the overview page.
TIMELINE_LABELS = {"Start", "Close", "End", "Deadline", "Merger & Entry", "Final Submission"}


def clean_page_lines(raw_lines: list[str]) -> list[str]:
    """Strip page chrome and tidy section/timeline labels in scraped text.

    - Drops nav-tab labels, Material-icon ligatures, and "Learn more" anchors.
    - Folds a timeline label followed by its date ("Start" + "May 15, 2024")
      into "**Start:** May 15, 2024"; drops bare timeline labels with no date.
    - Promotes known section labels to "### " subheadings.
    """
    cleaned: list[str] = []
    # A timeline label is held until we see whether the next content line is its
    # date. Date follows -> fold; otherwise the bare label is chrome -> drop.
    pending_label: str | None = None

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        if line in NAV_ITEMS:
            continue
        if line in STANDALONE_ICON_WORDS or ICON_LIGATURE_RE.match(line):
            continue
        if line in NOISE_LINES:
            continue
        if FILE_LISTING_RE.match(line) or FILE_COUNT_RE.match(line):
            continue

        # Resolve any held timeline label first.
        if pending_label is not None:
            if DATE_LINE_RE.match(line):
                cleaned.append(f"**{pending_label}:** {line}")
                pending_label = None
                continue
            # No date followed → bare chrome; drop the label, process this line.
            pending_label = None

        if line in TIMELINE_LABELS:
            pending_label = line
            continue
        if line in SECTION_LABELS:
            cleaned.append(f"### {line}")
            continue
        cleaned.append(line)

    # A trailing unresolved timeline label is chrome; nothing to append.
    return cleaned
