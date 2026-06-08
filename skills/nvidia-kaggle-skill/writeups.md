# Kaggle Writeups

Use this workflow to fetch Kaggle competition writeups and convert them to markdown. It supports a single writeup URL or a competition/leaderboard plus top-k count.

## Inputs

| Input | Required | Description |
|---|---|---|
| Writeup URL | Yes for single mode | Full URL containing `/writeups/`. |
| Competition slug/URL/leaderboard URL | Yes for top-k mode | Used to find linked leaderboard writeups. |
| Top-k count | No | Defaults to 3 in top-k mode. |
| Save folder | No | Defaults to current working directory. |

## Single Writeup

```bash
python ./scripts/fetch_writeup.py <writeup-url>
```

Save the result to a markdown file named after the writeup slug. Post-process the output:

- Ensure URLs are readable and valid.
- Do not retrieve comments.
- After the title, add: `Fetched using the nvidia-kaggle-skill writeups workflow. Original write-up link: <link>`.
- If the source shows the competition name with an image, replace the non-rendered image with a link to the competition.

## Top-k Writeups

If the input is a competition slug, competition URL, or leaderboard URL, build/use the leaderboard URL and fetch linked writeups:

```bash
python ./scripts/fetch_leaderboard_writeups.py <leaderboard-url>
```

The script returns JSON records with `{rank, team, writeup_url}`. Take the first k records, skip writeups already present in the output directory, and fetch the rest in parallel when the agent runtime supports it:

```bash
python ./scripts/fetch_writeup.py <writeup-url>
```

Save each file as `<rank>th_place_writeup.md` unless a clearer rank suffix is needed. Write `summary.md` with:

- rank, team, filename, and one-line approach summary
- per-writeup summaries
- competition takeaways after the first summary table

## Outputs

- One markdown file per fetched writeup.
- Optional `summary.md` for top-k mode.
- Markdown should preserve headings, tables, code blocks, lists, links, and images where possible.

## Workflow-Specific Troubleshooting

See [SKILL.md](SKILL.md#troubleshooting) for common browser, access, rate-limit, and page-structure failures.

| Symptom | Action |
|---|---|
| Kaggle markup changed | Preserve raw script output and note any manual cleanup needed. |
