"""Archive the best public-LB version of a Kaggle kernel.

Kaggle's REST API only exposes a kernel's latest version. To find and download
the *best-scoring historical* version, this module calls Kaggle's internal web
service (``/api/i/...``): it lists every version, reads each version's public
leaderboard score, selects the best one (auto-inferring whether lower or higher
is better), and downloads that version's source.

HTTP goes through ``runtime.KaggleWebServiceClient`` (token + XSRF), the same
internal-endpoint client used elsewhere in the skill.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime import KaggleWebServiceClient, kaggle_web_service

VIEW_MODEL = "kernels.LegacyKernelsService/GetKernelViewModel"
LIST_VERSIONS = "kernels.KernelsService/ListKernelVersions"
SOURCE = "kernels.KernelsService/GetKernelSessionSource"


class KernelArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class KernelRef:
    owner_slug: str
    kernel_slug: str


def parse_kernel_ref(kernel_ref: str) -> KernelRef:
    """Parse ``owner/kernel-slug`` or a Kaggle ``/code/owner/slug`` URL."""
    kernel_ref = kernel_ref.strip()
    if not kernel_ref:
        raise KernelArchiveError("Kernel reference is empty")

    if kernel_ref.startswith(("http://", "https://")):
        from urllib.parse import urlparse

        parts = [p for p in urlparse(kernel_ref).path.split("/") if p]
        if len(parts) >= 3 and parts[0] == "code":
            return KernelRef(parts[1], parts[2])
        raise KernelArchiveError(f"Could not parse Kaggle code URL: {kernel_ref}")

    parts = [p for p in kernel_ref.split("/") if p]
    if len(parts) != 2:
        raise KernelArchiveError(
            "Kernel reference must be owner/kernel-slug or a Kaggle code URL"
        )
    return KernelRef(parts[0], parts[1])


def parse_public_score(value: Any) -> float | None:
    """Parse a leaderboard score string into a float, or None if not numeric."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"-", "na", "n/a", "nan", "none", "null"}:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def close_score(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _get_view_model(
    client: KaggleWebServiceClient,
    ref: KernelRef,
    *,
    version_number: int | None = None,
    tab: str = "output",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "authorUserName": ref.owner_slug,
        "kernelSlug": ref.kernel_slug,
        "tab": tab,
    }
    if version_number is not None:
        body["versionNumber"] = version_number
    return client.post(VIEW_MODEL, body)


def _version_record(item: dict[str, Any]) -> dict[str, Any] | None:
    version = item.get("version") or {}
    run = item.get("run") or {}
    version_number = version.get("versionNumber")
    kernel_session_id = run.get("id")
    if not version_number or not kernel_session_id:
        return None
    return {
        "version_number": int(version_number),
        "version_id": version.get("id"),
        "version_name": version.get("versionName"),
        "kernel_session_id": int(kernel_session_id),
        "date_created": run.get("dateCreated"),
        "status": run.get("status"),
        "title": run.get("title"),
    }


def resolve_kernel_versions(
    kernel_ref: str, client: KaggleWebServiceClient | None = None
) -> list[dict[str, Any]]:
    """Return every version of a kernel, enriched with its public LB score."""
    ref = parse_kernel_ref(kernel_ref)
    client = client or kaggle_web_service()

    initial = _get_view_model(client, ref, tab="output")
    kernel_id = (initial.get("kernel") or {}).get("id")
    if not kernel_id:
        raise KernelArchiveError("Could not resolve kernel id from Kaggle view model")

    total = int(initial.get("totalVersionCount") or 0)
    data = client.post(
        LIST_VERSIONS,
        {"kernelId": int(kernel_id), "sortOption": "VERSION_ID", "pageSize": max(total, 200)},
    )
    items = data.get("items") or []
    if not isinstance(items, list):
        raise KernelArchiveError("Unexpected ListKernelVersions response: items is not a list")

    versions = [rec for item in items if (rec := _version_record(item))]
    versions.sort(key=lambda row: row["version_number"])

    records: list[dict[str, Any]] = []
    for version in versions:
        view = _get_view_model(client, ref, version_number=version["version_number"], tab="output")
        submission = view.get("submission") or {}
        kernel_run = view.get("kernelRun") or {}
        public_lb = submission.get("scoreFormatted")
        records.append(
            {
                "owner_slug": ref.owner_slug,
                "kernel_slug": ref.kernel_slug,
                "kernel_id": int(kernel_id),
                **version,
                "public_lb": public_lb,
                "public_lb_numeric": parse_public_score(public_lb),
                "best_submission_score_for_kernel": view.get("bestSubmissionScore"),
                "language": kernel_run.get("language"),
            }
        )
    return records


def _infer_direction(scored_rows: list[dict[str, Any]]) -> str | None:
    scores = [r["public_lb_numeric"] for r in scored_rows if r.get("public_lb_numeric") is not None]
    if len(scores) < 2 or close_score(min(scores), max(scores)):
        return None
    best_score = None
    for row in scored_rows:
        best = row.get("best_submission_score_for_kernel")
        if isinstance(best, dict):
            best_score = parse_public_score(best.get("scoreFormatted"))
        if best_score is not None:
            break
    if best_score is None:
        return None
    if close_score(best_score, min(scores)):
        return "minimize"
    if close_score(best_score, max(scores)):
        return "maximize"
    return None


def select_best_public_lb_version(
    version_rows: list[dict[str, Any]], *, score_direction: str = "auto"
) -> dict[str, Any]:
    """Pick the version with the best public LB score.

    ``score_direction`` is 'auto' (infer from Kaggle's bestSubmissionScore),
    'minimize', or 'maximize'.
    """
    scored = [r for r in version_rows if r.get("public_lb_numeric") is not None]
    if not scored:
        raise KernelArchiveError("No versions have a valid numeric public LB score")
    if score_direction not in {"auto", "minimize", "maximize"}:
        raise KernelArchiveError("score_direction must be 'auto', 'minimize', or 'maximize'")

    if score_direction == "auto":
        inferred = _infer_direction(scored)
        if inferred:
            score_direction = inferred
        elif len(scored) == 1:
            return {**scored[0], "score_direction": "auto", "selection_reason": "only one scored version"}
        else:
            raise KernelArchiveError(
                "Could not determine score direction from Kaggle metadata; "
                "pass score_direction='minimize' or 'maximize'."
            )

    reverse = score_direction == "maximize"
    best = sorted(scored, key=lambda r: r["public_lb_numeric"], reverse=reverse)[0]
    return {**best, "score_direction": score_direction, "selection_reason": f"best public LB ({score_direction})"}


def _source_extension(source_text: str, language: str | None) -> str:
    try:
        obj = json.loads(source_text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("cells"), list):
        return ".ipynb"
    language = (language or "").lower()
    if "language_r" in language or language == "r":
        return ".r"
    if "julia" in language:
        return ".jl"
    if "sql" in language:
        return ".sql"
    if "python" in language:
        return ".py"
    return ".txt"


def _download_version(
    ref: KernelRef,
    selected: dict[str, Any],
    output_dir: str | Path,
    client: KaggleWebServiceClient,
    *,
    include_outputs: bool,
    force: bool,
) -> dict[str, Any]:
    """Download a resolved version's source + write metadata.json under output_dir."""
    version_dir = Path(output_dir) / (
        f"v{selected['version_number']:03d}__scriptVersionId-{selected['kernel_session_id']}"
    )
    version_dir.mkdir(parents=True, exist_ok=True)

    source_text = client.post_text(
        SOURCE,
        {"kernelSessionId": selected["kernel_session_id"], "includeOutputIfAvailable": include_outputs},
    )
    source_path = version_dir / f"source{_source_extension(source_text, selected.get('language'))}"
    if force or not source_path.exists():
        source_path.write_text(source_text, encoding="utf-8")

    metadata = {
        "owner_slug": ref.owner_slug,
        "kernel_slug": ref.kernel_slug,
        "selected_version": selected,
        "source": {"path": str(source_path), "bytes": source_path.stat().st_size},
    }
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def archive_best_kernel_source(
    kernel_ref: str,
    output_dir: str | Path,
    *,
    score_direction: str = "auto",
    include_outputs: bool = False,
    force: bool = False,
    client: KaggleWebServiceClient | None = None,
) -> dict[str, Any]:
    """Find the best public-LB version of a kernel and save its source + metadata."""
    ref = parse_kernel_ref(kernel_ref)
    client = client or kaggle_web_service()

    rows = resolve_kernel_versions(kernel_ref, client=client)
    selected = select_best_public_lb_version(rows, score_direction=score_direction)
    return _download_version(
        ref, selected, output_dir, client, include_outputs=include_outputs, force=force
    )


def archive_kernel_version(
    kernel_ref: str,
    output_dir: str | Path,
    version_number: int,
    *,
    include_outputs: bool = False,
    force: bool = False,
    client: KaggleWebServiceClient | None = None,
) -> dict[str, Any]:
    """Archive a specific kernel version's source by version number."""
    ref = parse_kernel_ref(kernel_ref)
    client = client or kaggle_web_service()

    rows = resolve_kernel_versions(kernel_ref, client=client)
    selected = next((r for r in rows if r["version_number"] == version_number), None)
    if selected is None:
        available = ", ".join(str(r["version_number"]) for r in rows)
        raise KernelArchiveError(
            f"Version {version_number} not found for {kernel_ref}. Available versions: {available}"
        )
    selected = {**selected, "selection_reason": f"explicit version {version_number}"}
    return _download_version(
        ref, selected, output_dir, client, include_outputs=include_outputs, force=force
    )
