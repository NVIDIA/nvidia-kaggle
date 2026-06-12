"""Shared numeric defaults for Kaggle skill script entry points."""

DATE_PREVIEW_CHARS = 10

DEFAULT_BROWSER_TIMEOUT_MS = 30_000
LONG_BROWSER_TIMEOUT_MS = 60_000

# Bounded wait for the SPA to render meaningful content after the initial DOM
# load. We never block on "networkidle" (Kaggle polls in the background and it
# may never settle); instead we wait up to this long for body text to settle.
CONTENT_SETTLE_TIMEOUT_MS = 15_000
# Minimum body text length treated as "page has rendered something useful".
MIN_RENDERED_TEXT_LEN = 200
# Poll interval while waiting for body text to stop growing (SPA hydration).
CONTENT_POLL_INTERVAL_MS = 400
# Number of consecutive stable polls (no text growth) that mark the page as
# settled. Stops us returning half-hydrated pages without waiting on networkidle.
CONTENT_STABLE_POLLS = 2

DEFAULT_DISCUSSION_MAX_PAGES = 5
DEFAULT_PAGE_SIZE = 20
DEFAULT_KERNEL_SCORE_PAGE_SIZE = 100

DEFAULT_QUERY_LIMIT = 50
DEFAULT_AUTHOR_COLUMN_WIDTH = 20
DEFAULT_REF_COLUMN_WIDTH = 55
DEFAULT_TITLE_COLUMN_WIDTH = 50

SECONDS_PER_MINUTE = 60
DEFAULT_POLL_INTERVAL_SECONDS = 30
DEFAULT_KERNEL_TIMEOUT_SECONDS = 86_400
OUTPUT_SEPARATOR_WIDTH = 50

MIN_DATASET_SLUG_LENGTH = 6
MAX_DATASET_SLUG_LENGTH = 50
