#!/usr/bin/env python3
"""Fetch Kaggle writeup with UTF-8 output to file."""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from browser import evaluate_page
from support.constants import LONG_BROWSER_TIMEOUT_MS

JS_CONVERT = r"""
() => {
    function nodeToMd(node, listDepth) {
        if (node.nodeType === Node.TEXT_NODE) {
            return node.textContent;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return '';
        const tag = node.tagName.toLowerCase();
        if (['nav', 'footer', 'header', 'svg', 'button', 'script', 'style', 'noscript'].includes(tag)) return '';
        if (node.getAttribute('role') === 'navigation') return '';
        let children = Array.from(node.childNodes).map(c => nodeToMd(c, listDepth)).join('');
        switch (tag) {
            case 'h1': return '\n# ' + children.trim() + '\n';
            case 'h2': return '\n## ' + children.trim() + '\n';
            case 'h3': return '\n### ' + children.trim() + '\n';
            case 'h4': return '\n#### ' + children.trim() + '\n';
            case 'h5': return '\n##### ' + children.trim() + '\n';
            case 'h6': return '\n###### ' + children.trim() + '\n';
            case 'p': return '\n' + children.trim() + '\n';
            case 'br': return '\n';
            case 'strong': case 'b': return '**' + children.trim() + '**';
            case 'em': case 'i': return '*' + children.trim() + '*';
            case 'code':
                if (node.parentElement && node.parentElement.tagName.toLowerCase() === 'pre') {
                    return children;
                }
                return '`' + children.trim() + '`';
            case 'pre': {
                let lang = '';
                const codeEl = node.querySelector('code');
                if (codeEl) {
                    const cls = codeEl.className || '';
                    const m = cls.match(/language-(\w+)/);
                    if (m) lang = m[1];
                }
                return '\n```' + lang + '\n' + children.trim() + '\n```\n';
            }
            case 'a': {
                const href = node.getAttribute('href') || '';
                const text = children.trim();
                if (!text) return '';
                return '[' + text + '](' + href + ')';
            }
            case 'img': {
                const alt = node.getAttribute('alt') || '';
                const src = node.getAttribute('src') || '';
                return '![' + alt + '](' + src + ')';
            }
            case 'ul': case 'ol':
                return '\n' + children + '\n';
            case 'li': {
                const prefix = (node.parentElement && node.parentElement.tagName.toLowerCase() === 'ol')
                    ? '1. ' : '- ';
                const indent = '  '.repeat(listDepth);
                return indent + prefix + children.trim() + '\n';
            }
            case 'blockquote':
                return '\n' + children.trim().split('\n').map(l => '> ' + l).join('\n') + '\n';
            case 'table': {
                const rows = Array.from(node.querySelectorAll('tr'));
                if (rows.length === 0) return children;
                let table = '\n';
                rows.forEach((row, i) => {
                    const cells = Array.from(row.querySelectorAll('th, td'));
                    const line = '| ' + cells.map(c => c.textContent.trim()).join(' | ') + ' |';
                    table += line + '\n';
                    if (i === 0) {
                        table += '| ' + cells.map(() => '---').join(' | ') + ' |\n';
                    }
                });
                return table + '\n';
            }
            case 'hr': return '\n---\n';
            default:
                return children;
        }
    }
    let container = document.querySelector('[class*="writeup"]')
        || document.querySelector('article')
        || document.querySelector('[class*="discussion-detail"]')
        || document.querySelector('[class*="comment-body"]')
        || document.querySelector('main')
        || document.body;
    return nodeToMd(container, 0);
}
"""

NOISE_PATTERNS = [
    r'^menu\s*$', r'^Skip to\s*$', r'^content\s*$', r'^Create\s*$',
    r'^explore\s*$', r'^Home\s*$', r'^Sign In\s*$', r'^Register\s*$',
    r'^Kaggle uses cookies.*$', r'^Learn more\s*$', r'^OK, Got it\.\s*$',
    r'^emoji_events\s*$', r'^table_chart\s*$', r'^tenancy\s*$',
    r'^leaderboard\s*$', r'^smart_toy\s*$', r'^code\s*$', r'^comment\s*$',
    r'^school\s*$', r'^expand_more\s*$', r'^auto_awesome_motion\s*$',
    r'^View Active Events\s*$', r'^more_horiz\s*$', r'^more_vert\s*$',
    r'^arrow_drop_up\s*$', r'^arrow_drop_down\s*$', r'^Competitions\s*$',
    r'^Datasets\s*$', r'^Models\s*$', r'^Benchmarks\s*$',
    r'^Game Arena\s*$', r'^Code\s*$', r'^Discussions\s*$',
    r'^Learn\s*$', r'^More\s*$',
]

def fetch_writeup(url: str) -> str:
    md = evaluate_page(url, JS_CONVERT, timeout=LONG_BROWSER_TIMEOUT_MS)

    lines = md.split('\n')
    cleaned = []
    blank_count = 0
    for line in lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                cleaned.append('')
        else:
            blank_count = 0
            cleaned.append(line)

    result = '\n'.join(cleaned).strip()

    noise_re = re.compile('|'.join(NOISE_PATTERNS), re.MULTILINE)
    result = noise_re.sub('', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a Kaggle writeup as markdown")
    parser.add_argument("url", help="Kaggle writeup URL")
    parser.add_argument("output_path", nargs="?", help="Optional markdown output path")
    args = parser.parse_args()

    if not re.match(r"^https?://", args.url):
        parser.error("url must be an absolute http(s) Kaggle writeup URL")

    try:
        content = fetch_writeup(args.url)
        if args.output_path:
            out_path = Path(args.output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"Saved to {out_path} ({len(content)} chars)", file=sys.stderr)
        else:
            sys.stdout.buffer.write(content.encode("utf-8"))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
