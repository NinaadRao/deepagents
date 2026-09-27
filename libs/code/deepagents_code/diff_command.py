"""Build and render the `/diff` command's report of uncommitted changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from deepagents_code._git import find_git_root
from deepagents_code.diff_utils import (
    DIFF_TRUNCATION_MARKER,
    DiffStats,
    count_diff_change_lines,
)

_MAX_DIFF_LINES: Final[int] = 400
"""Cap on rendered diff lines before truncating with `DIFF_TRUNCATION_MARKER`."""

_GIT_TIMEOUT_SECONDS: Final[float] = 10.0
"""Timeout for each `git` subprocess call, generous for large working trees."""


@dataclass(frozen=True, kw_only=True)
class DiffReport:
    """Result of building the `/diff` command's report."""

    diff_text: str | None
    """Unified diff body against `HEAD`, `None` when there is nothing to show."""

    stats: DiffStats | None
    """Line-count stats for `diff_text`, taken before any truncation."""

    untracked_files: tuple[str, ...]
    """Untracked file paths, listed but not diffed (no prior content to compare)."""

    error: str | None
    """User-facing message when the report could not be built."""


def _run_git(args: list[str], *, cwd: str) -> tuple[int, str]:
    """Run a `git` subcommand, swallowing environment failures.

    Returns:
        The process exit code (`1` when git could not even be started or timed
            out) and its captured stdout.
    """
    import subprocess  # noqa: S404  # stdlib subprocess fallback

    try:
        result = subprocess.run(  # noqa: S603  # trusted argv, no shell
            ["git", *args],  # noqa: S607  # trusted argv, no shell
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
            cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return result.returncode, result.stdout


def _list_untracked_files(cwd: str) -> tuple[str, ...]:
    """Return untracked file paths, sorted for stable display."""
    status_code, status_out = _run_git(["status", "--porcelain", "-z"], cwd=cwd)
    if status_code != 0:
        return ()
    return tuple(
        sorted(entry[3:] for entry in status_out.split("\0") if entry.startswith("?? "))
    )


def build_diff_report(cwd: str) -> DiffReport:
    """Build a report of uncommitted working-tree changes for `/diff`.

    Args:
        cwd: Directory to resolve the repository from.

    Returns:
        A `DiffReport` describing tracked changes since `HEAD` plus any
            untracked files, or one carrying `error` when git or a repository
            is unavailable.
    """
    if find_git_root(cwd) is None:
        return DiffReport(
            diff_text=None,
            stats=None,
            untracked_files=(),
            error="Not inside a git repository.",
        )

    untracked = _list_untracked_files(cwd)

    diff_code, diff_out = _run_git(
        ["diff", "--no-color", "--no-ext-diff", "HEAD"], cwd=cwd
    )
    if diff_code != 0:
        # No commits yet, so `HEAD` doesn't resolve -- fall back to comparing
        # the working tree against the index.
        diff_code, diff_out = _run_git(["diff", "--no-color", "--no-ext-diff"], cwd=cwd)
        if diff_code != 0:
            return DiffReport(
                diff_text=None,
                stats=None,
                untracked_files=untracked,
                error="Could not read the git diff.",
            )

    if not diff_out.strip():
        return DiffReport(
            diff_text=None, stats=None, untracked_files=untracked, error=None
        )

    lines = diff_out.splitlines()
    stats = count_diff_change_lines(lines)
    if len(lines) > _MAX_DIFF_LINES:
        lines = [*lines[: _MAX_DIFF_LINES - 1], DIFF_TRUNCATION_MARKER]
    return DiffReport(
        diff_text="\n".join(lines), stats=stats, untracked_files=untracked, error=None
    )


def render_diff_report(report: DiffReport) -> str:
    """Render a `DiffReport` as chat-message text for `/diff`.

    Args:
        report: Report built by `build_diff_report`.

    Returns:
        User-facing text summarizing and showing the diff.
    """
    if report.error is not None:
        return report.error

    sections: list[str] = []
    if report.diff_text is None:
        sections.append("No uncommitted changes.")
    else:
        stats = report.stats
        summary = f" (+{stats.additions}/-{stats.deletions})" if stats else ""
        sections.append(
            f"Uncommitted changes{summary}:\n```diff\n{report.diff_text}\n```"
        )

    if report.untracked_files:
        file_list = "\n".join(f"- {path}" for path in report.untracked_files)
        sections.append(f"Untracked files:\n{file_list}")

    return "\n\n".join(sections)
