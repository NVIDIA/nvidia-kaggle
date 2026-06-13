#!/usr/bin/env python3
"""Analyze an agentic demo run from its STRUCTURED TRACE — the single source of
truth for both verification jobs (per the ratified design):

  1. Agentic-behavior evidence: which skill workflows the agent chose & chained,
     and whether it read SKILL.md — proven from the trace, not asserted.
  2. Grounding allow-list: the entity set is reconstructed from the agent's
     ACTUAL tool outputs in the trace (never a fresh regen). `verify_synthesis`
     runs the synthesis against THAT set.

Honesty rule: if the trace does not expose tool outputs cleanly enough to
reconstruct the gathered set, grounding-eval DEGRADES to header/chrome/length
invariants only + an explicit "hallucination check skipped" note — we never
silently verify against a superset ("everything available").

Supports both runtimes:
  - codex: `codex exec --json` JSONL — `command_execution` items carry
    `command` + `aggregated_output`.
  - claude: `claude -p --output-format stream-json` JSONL — tool_use /
    tool_result events; granular workflow output may live in sub-agent traces.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SKILL_SCRIPTS = Path(__file__).resolve().parents[2] / "skills/nvidia-kaggle-skill/scripts"
sys.path.insert(0, str(_SKILL_SCRIPTS))

from verify_synthesis import KERNEL_REF_BACKTICK_RE, KERNEL_REF_URL_RE, verify_synthesis  # noqa: E402

# The skill's individual workflow scripts (the "tools" an agentic run chains).
WORKFLOW_SCRIPTS = [
    "fetch_competition_info",
    "fetch_dataset_info",
    "fetch_leaderboard_writeups",
    "fetch_writeup",
    "discussion_ingest",
    "discussion_query",
    "discussion_read",
    "kernel_ingest",
    "kernel_query",
    "kernel_read",
    "fetch_top_kernel_scores",
    "fetch_kernel_score",
    "generate_briefing",
]
# Match an ACTUAL invocation — the script run as a command — not a mere mention
# (e.g. `cat SKILL.md` prints "generate_briefing.py" in docs, which must NOT
# count). Require a python/path execution context immediately before the script.
_WORKFLOW_RE = re.compile(
    r"(?:python[0-9.]*\s+|\./|/|scripts/)\S*?\b(" + "|".join(WORKFLOW_SCRIPTS) + r")\.py\b"
)
# owner/slug refs as they appear in gathered tool output (CSV rows, URLs, etc.)
_REF_IN_OUTPUT_RE = re.compile(r"\b([a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*)\b")
_NON_REF_PREFIXES = {"competition", "competitions", "code", "datasets", "dataset",
                     "models", "model", "discussion", "discussions",
                     "www.kaggle.com", "kaggle.com", "api", "scripts", "bin"}


# Raw notebook-path segments from the trace: `/notebooks/<comp>/<seg>/` where
# <seg> is the cached-notebook dir name (owner_slug underscore form). We keep
# these as opaque strings and NEVER split them into owner/slug (that split is
# ambiguous). Instead, to VERIFY a cited ref we form its OWN unambiguous
# path-form (owner + "_" + slug, from the known-good cited owner/slug) and check
# membership here. A fabricated ref's path-form is absent → still flagged.
_NOTEBOOK_PATH_SEG_RE = re.compile(r"/notebooks/[^/]+/([^/\"\\]+)")


@dataclass
class RunAnalysis:
    runtime: str
    read_skill_md: bool = False
    workflows_invoked: dict = field(default_factory=dict)   # script -> count
    distinct_workflows: int = 0
    gathered_refs: set = field(default_factory=set)
    gathered_artifact_text: str = ""  # concatenated research/ + raw/ gathering output
    notebook_path_segs: set = field(default_factory=set)  # cached-notebook dir names (owner_slug)
    tool_output_chars: int = 0
    reconstructable: bool = False  # could we rebuild the gathered set?
    subagent_dispatches: int = 0   # Task/Agent tool_use calls (Claude delegation)
    subagent_outputs_visible: bool = True  # did granular outputs surface in-trace?


def _iter_command_outputs(trace_path: Path, runtime: str):
    """Yield (command_str, output_str) for each tool execution in the trace."""
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue

        if runtime == "codex":
            if o.get("type") == "item.completed":
                it = o.get("item", {})
                if it.get("type") == "command_execution":
                    yield it.get("command", ""), it.get("aggregated_output", "")
        else:  # claude stream-json
            if o.get("type") == "assistant":
                for blk in o.get("message", {}).get("content", []):
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        inp = blk.get("input", {})
                        cmd = inp.get("command", "") if isinstance(inp, dict) else ""
                        yield cmd, ""
            elif o.get("type") == "user":
                # tool_result blocks carry the output of the prior tool_use
                for blk in o.get("message", {}).get("content", []):
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        content = blk.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                c.get("text", "") for c in content if isinstance(c, dict)
                            )
                        yield "", str(content)


# A kernel ref the agent passed to a workflow command (e.g.
# `kernel_read.py owner/slug`) OR that appears in a cached-notebook path as the
# underscore form `.../notebooks/<comp>/owner_slug/...`. Running a workflow
# against a ref — or having its notebook cached — is proof the agent gathered
# it, even if the slash form never appears in stdout. Harvesting these closes a
# false-positive gap where a genuinely-gathered ref would look hallucinated.
_CMD_REF_RE = re.compile(
    r"(?:kernel_read|kernel_query|fetch_kernel_score|kernels?\s+pull)\S*\s+"
    r"([a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*)"
)
_CACHED_NOTEBOOK_RE = re.compile(r"/notebooks/[^/]+/([a-z0-9][a-z0-9._-]*)_([a-z0-9][a-z0-9._-]*)/")


def _extract_cmd_refs(cmd: str) -> set:
    """Refs the agent demonstrably gathered, harvested from the COMMAND text."""
    refs = set()
    for m in _CMD_REF_RE.finditer(cmd):
        ref = m.group(1)
        if ref.split("/", 1)[0] not in _NON_REF_PREFIXES:
            refs.add(ref)
    return refs


# Kaggle profile/code URLs: kaggle.com/<owner>/<slug> (note: NOT /code/ — a
# kernel's public URL is often kaggle.com/owner/slug directly). We harvest the
# owner/slug AFTER the host so the host prefix doesn't shadow the real ref
# (the word-boundary regex would otherwise match "www.kaggle.com/owner" first).
_KAGGLE_URL_REF_RE = re.compile(
    r"(?:www\.)?kaggle\.com/(?:code/)?([a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*)"
)


def _extract_refs(text: str) -> set:
    """Unambiguous gathered refs from tool output: kaggle URLs (slash form) and
    bare owner/slug tokens. Does NOT include cached-notebook path refs — those
    are ambiguous (owner_slug split) and handled separately with corroboration.
    """
    refs = set()
    # 1) owner/slug inside a kaggle URL (strip host first so it isn't shadowed).
    for m in _KAGGLE_URL_REF_RE.finditer(text):
        ref = m.group(1)
        if ref.split("/", 1)[0] not in _NON_REF_PREFIXES \
                and not ref.endswith((".py", ".md", ".csv", ".json", ".txt")):
            refs.add(ref)
    # 2) bare owner/slug tokens elsewhere in the output.
    for m in _REF_IN_OUTPUT_RE.finditer(text):
        ref = m.group(1)
        if ref.count("/") == 1 and ref.split("/", 1)[0] not in _NON_REF_PREFIXES:
            if not ref.endswith((".py", ".md", ".csv", ".json", ".txt")):
                refs.add(ref)
    return refs


def _extract_path_candidates(text: str) -> set:
    """Cached-notebook path refs (owner_slug underscore form) — AMBIGUOUS, since
    Kaggle owners AND slugs may contain underscores, so the split point is a
    guess. Returned SEPARATELY as *candidates*: a candidate is only admitted to
    the gathered set if corroborated by a command-arg invocation of the same
    ref (see analyze_trace). Never minted into the allow-list on its own — an
    over-wide list silently trades away the anti-hallucination guarantee.
    """
    cands = set()
    for m in _CACHED_NOTEBOOK_RE.finditer(text):
        owner, slug = m.group(1), m.group(2)
        if owner not in _NON_REF_PREFIXES:
            cands.add(f"{owner}/{slug}")
    return cands


def _count_subagent_dispatches(trace_path: Path, runtime: str) -> int:
    """Claude may delegate research to sub-agents via the Task/Agent tool. Count
    those dispatches — if granular fetch_* calls happened inside sub-agents and
    their outputs don't surface in this trace, the gathered-set is only partly
    recoverable, which we must report honestly (never paper over)."""
    if runtime != "claude":
        return 0
    n = 0
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") == "assistant":
            for blk in o.get("message", {}).get("content", []):
                if isinstance(blk, dict) and blk.get("type") == "tool_use" \
                        and blk.get("name") in ("Task", "Agent"):
                    n += 1
    return n


# Run-dir subdirs that hold the agent's *gathered output* (skill workflow
# results it saved), as opposed to its *authored output* (brief.md, plots/).
# A value present in one of these is gathered-this-run evidence, the same as a
# value in the trace — codex --json doesn't echo every workflow's stdout into
# the trace, so a real gathered number (e.g. a kernel's vote count written to
# research/kernels_top.json) can be absent from trace.jsonl yet genuinely
# fetched. We harvest refs from these too so we stop false-flagging correct
# gathered data. GUARDRAIL: only these gathering-output subdirs are counted —
# NOT brief.md or plots/ (the agent's own output, which is what we're checking),
# and NOT data/ (multi-GB downloaded CSVs, handled via recompute-and-match). A
# fabricated ref absent from trace + these dirs still fails (verified on
# _004/_008), so this widening doesn't weaken fabrication detection.
_GATHERED_ARTIFACT_DIRS = ("research", "raw")
_GATHERED_ARTIFACT_EXTS = (".json", ".txt", ".tsv", ".csv", ".md")


def _harvest_gathered_artifacts(run_dir: Path) -> tuple[str, set]:
    """Return (concatenated text, refs) from the run's gathering-output files.

    Reads only the allow-listed gathering subdirs (research/, raw/) — never the
    agent-authored brief.md / plots/, and never data/. The text is returned so
    callers can also do per-value provenance against gathered artifacts (not
    just trace.jsonl); refs feed the gathered-ref allow-list.
    """
    text_parts: list = []
    refs: set = set()
    for sub in _GATHERED_ARTIFACT_DIRS:
        d = run_dir / sub
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix.lower() in _GATHERED_ARTIFACT_EXTS:
                try:
                    txt = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                text_parts.append(txt)
                refs |= _extract_refs(txt)
    return "\n".join(text_parts), refs


def analyze_trace(trace_path: Path, runtime: str) -> RunAnalysis:
    a = RunAnalysis(runtime=runtime)
    cmd_refs: set = set()          # refs with clean command-arg provenance
    path_candidates: set = set()   # ambiguous cached-path owner_slug splits
    for cmd, out in _iter_command_outputs(trace_path, runtime):
        blob = f"{cmd}\n{out}"
        if "SKILL.md" in blob:
            a.read_skill_md = True
        for m in _WORKFLOW_RE.finditer(cmd):
            a.workflows_invoked[m.group(1)] = a.workflows_invoked.get(m.group(1), 0) + 1
        # Refs the agent passed to a workflow command are demonstrably gathered.
        cmd_refs |= _extract_cmd_refs(cmd)
        path_candidates |= _extract_path_candidates(cmd)
        a.notebook_path_segs |= set(_NOTEBOOK_PATH_SEG_RE.findall(cmd))
        if out:
            a.tool_output_chars += len(out)
            a.gathered_refs |= _extract_refs(out)
            path_candidates |= _extract_path_candidates(out)
            a.notebook_path_segs |= set(_NOTEBOOK_PATH_SEG_RE.findall(out))
    # Command-arg refs are sound provenance → always admitted.
    a.gathered_refs |= cmd_refs
    # Widen the gathered set with refs from the run's gathering-output files
    # (research/, raw/) — these are skill-workflow results the agent saved that
    # codex --json may not have echoed into trace.jsonl. Without this, a real
    # gathered value (e.g. a kernel vote in research/kernels_top.json) gets
    # false-flagged as fabricated. Guardrailed to gathering-output dirs only.
    _gathered_text, _gathered_refs = _harvest_gathered_artifacts(trace_path.parent)
    a.gathered_refs |= _gathered_refs
    a.gathered_artifact_text = _gathered_text
    # NOTE: cached-notebook provenance is NOT applied here by blindly splitting
    # `owner_slug` path segments (that reverse-split is ambiguous — owners and
    # slugs both contain underscores). Instead, cached fetches are matched at
    # VERIFY time against each *cited* ref's own forward-normalized path-form
    # (owner/slug → owner_slug) vs `notebook_path_segs` — see main(). That keys
    # off the cached `.ipynb`'s existence as proof-of-fetch (the agreed rule),
    # admits cached-only refs that have NO command-arg, and can't over-widen
    # because a fabricated ref's path-form is absent from the trace.
    # (`path_candidates` is retained only for diagnostics; not unioned in here.)
    a.distinct_workflows = len(a.workflows_invoked)
    # We can reconstruct a grounding set if we actually captured tool output.
    a.reconstructable = a.tool_output_chars > 0
    a.subagent_dispatches = _count_subagent_dispatches(trace_path, runtime)
    # If the agent delegated to sub-agents BUT we still see no workflow calls /
    # no tool output at the parent level, the granular invocations likely live
    # in sub-agent traces this stream did not expose → flag as not visible.
    if a.subagent_dispatches > 0 and (a.distinct_workflows == 0 or a.tool_output_chars == 0):
        a.subagent_outputs_visible = False
    return a


def build_allow_list_from_trace(analysis: RunAnalysis, fallback_json: dict | None) -> dict:
    """Build a `verify_synthesis`-compatible entities block from the gathered
    refs. Other closed classes (teams/authors/titles) are left empty unless a
    captured briefing JSON is provided — they degrade to 'unverified', honestly.
    """
    entities = {
        "competition_slug": (fallback_json or {}).get("entities", {}).get("competition_slug", ""),
        "kernel_refs": sorted(analysis.gathered_refs),
        "teams": [],
        "authors": [],
        "discussion_titles": [],
    }
    return {"entities": entities}


def _plot_value_in_gathered(value, gathered_text: str) -> bool:
    """True if a plotted value appears in the run's gathered text (trace +
    research + raw), matched at token boundaries so a small integer can't
    coincidentally substring-match inside a larger number.

    For numeric values we also accept NUMERIC EQUIVALENCE, not just the exact
    string: a plotted `6.72` must match a gathered `6.720` (trailing zero) and
    vice-versa — the gathered CSV and the agent's sidecar can format the same
    number differently. We find every number token in the text near the value's
    integer part and compare as floats. This avoids false-flagging a clean run
    on pure formatting (caught _013: gathered `6.720` vs plotted `6.72`)."""
    s = str(value).strip()
    if not s:
        return False
    # 1) exact token match (covers strings/labels and exactly-formatted numbers)
    if re.search(r"(?<![\w.])" + re.escape(s) + r"(?![\w.])", gathered_text):
        return True
    # 2) numeric equivalence for numbers (handles trailing-zero / formatting)
    try:
        fv = float(s)
    except ValueError:
        return False
    # scan number tokens in the text and compare as floats (tight tolerance:
    # this is formatting equivalence, NOT fuzzy matching — 6.72 == 6.720, but
    # 6.72 != 6.73). Pre-filter by leading digits to keep the scan cheap.
    lead = re.escape(s.lstrip("-").split(".")[0][:3])
    for m in re.finditer(r"-?\d+(?:\.\d+)?", gathered_text):
        tok = m.group(0)
        if lead and lead not in tok:
            continue
        try:
            if abs(float(tok) - fv) < 1e-9:
                return True
        except ValueError:
            continue
    return False


def check_plot_provenance(run_dir: Path, gathered_text: str) -> list:
    """Per-plot, per-value provenance over the run's plot sidecars — the check
    that actually catches plot fabrications (which live in the sidecar JSON, not
    the brief text the kernel-ref grounding inspects).

    For each `plots/*.json` sidecar, every `series[].value` must appear in the
    run's gathered text (trace + research/ + raw/). A `source` that names a
    downloaded dataset is NOT a free pass — but small-integer aggregates
    (histogram bucket counts) coincidentally appear everywhere, so a plot whose
    `source` marks it a downloaded/derived dataset metric AND whose values don't
    all token-match is reported as UNVERIFIED-derived (caller decides), not a
    hard fabrication. A plot with NON-dataset source and untraceable values is a
    fabrication (the `_004`/`_008` mode: invented leaderboard/discussion rows).

    Returns a list of (plot_name, kind, detail) where kind ∈
    {"fabrication","derived-unverified"}. Empty list = all plots clean.
    """
    findings = []
    plots_dir = run_dir / "plots"
    if not plots_dir.is_dir():
        return findings
    for f in sorted(plots_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        series = d.get("series") if isinstance(d, dict) else (d if isinstance(d, list) else [])
        if not isinstance(series, list) or not series:
            continue
        source = str(d.get("source", "")).lower() if isinstance(d, dict) else ""
        is_dataset = any(k in source for k in ("dataset", "download", "competition_dataset", "csv"))
        untraced = []
        for e in series:
            if not isinstance(e, dict):
                continue
            val = e.get("value")
            if val is None:
                continue
            if not _plot_value_in_gathered(val, gathered_text):
                untraced.append((e.get("label"), val))
        if not untraced:
            continue
        if is_dataset:
            # Derived/dataset-computed: values are computed over downloaded CSVs
            # (recompute-and-match is the proper rebuttal; token-match can't
            # verify a computed aggregate). Report as unverified, not fabrication.
            findings.append((f.name, "derived-unverified", untraced))
        else:
            # Gathered-entity plot (votes/comments/leaderboard) with values that
            # appear in NO gathered artifact = fabricated rows. The _004/_008 mode.
            findings.append((f.name, "fabrication", untraced))
    return findings


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Analyze an agentic demo run from its structured trace")
    ap.add_argument("trace", help="Path to the structured trace (JSONL)")
    ap.add_argument("--runtime", choices=["claude", "codex"], required=True)
    ap.add_argument("--synthesis", help="Path to the agent's strategy brief (for grounding-eval)")
    ap.add_argument("--briefing-json", help="Optional captured briefing JSON for richer allow-list")
    args = ap.parse_args()

    analysis = analyze_trace(Path(args.trace), args.runtime)

    print("=" * 62)
    print(f" Agentic-behavior evidence — runtime: {args.runtime}")
    print("=" * 62)
    print(f"  read SKILL.md:        {'YES' if analysis.read_skill_md else 'NO'}")
    print(f"  distinct workflows:   {analysis.distinct_workflows}")
    for name, n in sorted(analysis.workflows_invoked.items(), key=lambda kv: -kv[1]):
        print(f"    - {name} (x{n})")
    print(f"  tool output captured: {analysis.tool_output_chars:,} chars")
    print(f"  gathered kernel refs: {len(analysis.gathered_refs)}")
    if analysis.subagent_dispatches:
        vis = "outputs visible in trace" if analysis.subagent_outputs_visible \
            else "OUTPUTS NOT EXPOSED — gathered-set only partly recoverable"
        print(f"  sub-agent dispatches: {analysis.subagent_dispatches} ({vis})")

    if not args.synthesis or not Path(args.synthesis).exists():
        if args.synthesis:
            print()
            print(f"  NOTE: synthesis not found at {args.synthesis} "
                  "(run may have been cut off before writing it) — "
                  "agentic-evidence above still stands.")
        return

    print()
    print("=" * 62)
    print(" Grounding-eval (against what the agent ACTUALLY gathered)")
    print("=" * 62)

    synth = Path(args.synthesis).read_text(encoding="utf-8")
    fallback = None
    if args.briefing_json and Path(args.briefing_json).exists():
        fallback = json.loads(Path(args.briefing_json).read_text(encoding="utf-8"))

    from verify_synthesis import _extract_kernel_refs

    gathered_unrecoverable = (not analysis.reconstructable) or (not analysis.subagent_outputs_visible)
    if gathered_unrecoverable:
        # Honesty rule: cannot rebuild the gathered set from the INDEPENDENT
        # trace → we CANNOT run the anti-hallucination check at all (an empty
        # allow-list would vacuously "pass" any cited ref — unsound). Report
        # DEGRADED and stop here; do NOT fall through to a PASS verdict.
        why = ("trace did not expose tool outputs"
               if not analysis.reconstructable
               else "granular workflow outputs ran inside sub-agents not exposed "
                    "in this trace")
        n_cited = len(_extract_kernel_refs(synth))
        print(f"  NOTE: {why} — gathered-set UNRECOVERABLE for this runtime.")
        print(f"        Brief cites {n_cited} checkable ref(s), but they CANNOT be")
        print("        checked against an unrecoverable gathered-set. Anti-")
        print("        hallucination NOT verifiable here; chrome + eyeball only.")
        # Chrome is still checkable (format-independent). Report it, then DEGRADE.
        chrome = [f for f in verify_synthesis(synth, {"entities": {}}).hard_failures
                  if "chrome" in f]
        for f in chrome:
            print(f"  FAIL (chrome): {f}")
        verdict = "FAIL" if chrome else "DEGRADED — grounding unverifiable (gathered-set unrecoverable)"
        print(f"  GROUNDING RESULT: {verdict}")
        raise SystemExit(1 if chrome else 3)

    briefing = build_allow_list_from_trace(analysis, fallback)

    # Cached-notebook provenance, done SOUNDLY: a cited ref is grounded if its
    # OWN unambiguous path-form (owner + "_" + slug, formed FROM the cited
    # owner/slug — never by splitting a path) matches a notebook dir the agent
    # actually fetched. This recovers refs gathered only via `kernel pull` /
    # cached notebooks (not in stdout), WITHOUT the over-widening risk: a
    # fabricated ref's path-form is absent from the trace, so it still fails.
    allow = {r.lower() for r in briefing["entities"]["kernel_refs"]}
    segs = analysis.notebook_path_segs
    cited = _extract_kernel_refs(synth)
    recovered = []
    for ref in cited:
        if ref.lower() in allow or "/" not in ref:
            continue
        owner, slug = ref.split("/", 1)
        pathform = f"{owner}_{slug}"
        # EXACT path-form match only. A bare startswith() would over-recover:
        # a fabricated ref whose path-form is merely a PREFIX of a real fetched
        # dir (e.g. cite `foo/bar` when only `foo_bar-something-else` was
        # fetched) would falsely clear. We allow exact, or exact + a dotted
        # file-suffix (`<pathform>.ipynb`), but never a bare prefix.
        if any(seg == pathform or seg.startswith(pathform + ".") for seg in segs):
            briefing["entities"]["kernel_refs"].append(ref)
            recovered.append(ref)
    print(f"  allow-list reconstructed from trace: "
          f"{len(briefing['entities']['kernel_refs'])} kernel refs"
          + (f" ({len(recovered)} via cached-notebook provenance)" if recovered else ""))

    result = verify_synthesis(synth, briefing)

    # In the AGENTIC shape the agent writes a free-form brief, not the
    # constrained "## Key Strategies" synthesis. So the header/length invariants
    # (which encode the old pipeline's prompt contract) are INFORMATIONAL here —
    # the soundness checks that actually matter for an agentic run are:
    #   * hallucinated kernel refs  (cited but never gathered)  -> hard
    #   * leaked page chrome                                     -> hard
    # We split the hard_failures accordingly and report honestly.
    HEADER_LENGTH_MARKERS = ("required header", "synthesis is empty")
    grounding_fails = [f for f in result.hard_failures
                       if not any(m in f for m in HEADER_LENGTH_MARKERS)]
    format_notes = [f for f in result.hard_failures
                    if any(m in f for m in HEADER_LENGTH_MARKERS)]

    # PLOT-VALUE PROVENANCE — the check that actually catches plot fabrications
    # (_004/_008-style invented rows live in the plot sidecar JSON, which the
    # kernel-ref grounding above never inspects). Build the run's gathered text
    # (trace outputs + research/ + raw/) and confirm every plotted value traces.
    run_dir = Path(args.trace).parent
    trace_out = "\n".join(out for _, out in _iter_command_outputs(Path(args.trace), args.runtime))
    gathered_text = trace_out + "\n" + analysis.gathered_artifact_text
    plot_findings = check_plot_provenance(run_dir, gathered_text)
    plot_fabrications = [pf for pf in plot_findings if pf[1] == "fabrication"]
    plot_derived = [pf for pf in plot_findings if pf[1] == "derived-unverified"]
    for name, _kind, untraced in plot_fabrications:
        ex = ", ".join(f"{lab}={val}" for lab, val in untraced[:3])
        grounding_fails.append(
            f"plot '{name}' has {len(untraced)} value(s) in NO gathered artifact "
            f"(fabricated rows): {ex}")
    for name, _kind, untraced in plot_derived:
        print(f"  NOTE (derived/dataset plot — needs recompute-vs-CSV, not token-trace): "
              f"'{name}' {len(untraced)} computed value(s)")

    for f in grounding_fails:
        print(f"  FAIL (grounding/chrome): {f}")
    for f in format_notes:
        print(f"  NOTE (free-form brief, not constrained synthesis): {f}")
    for w in result.warnings:
        print(f"  WARN: {w}")
    if result.unverified:
        print(f"  UNVERIFIED (no allow-list data): {', '.join(result.unverified)}")

    # Honesty: the anti-hallucination gate only MEANS something if the brief
    # actually cites refs in a checkable form (backtick `owner/slug` or a kaggle
    # code URL). If it cites zero, "no hallucinated refs" is VACUOUSLY true — the
    # gate never fired. Report that explicitly instead of a green PASS.
    n_checkable = len(_extract_kernel_refs(synth))

    if grounding_fails:
        print(f"  GROUNDING RESULT: FAIL ({len(grounding_fails)} grounding/chrome violation(s))")
        raise SystemExit(1)
    if n_checkable == 0:
        # No checkable refs → anti-hallucination gate UNEXERCISED. Not a pass.
        print("  GROUNDING RESULT: NOT EXERCISED — brief cites no checkable "
              "owner/slug refs; anti-hallucination gate did not fire. "
              "Degraded to chrome-clean + eyeball.")
        # Distinct exit code so callers don't read this as a green PASS.
        raise SystemExit(2)
    print(f"  GROUNDING RESULT: PASS — {n_checkable} cited ref(s) all in gathered set, no chrome")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
