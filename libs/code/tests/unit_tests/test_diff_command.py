"""Tests for `deepagents_code.diff_command`."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from deepagents_code.diff_command import (
    _MAX_DIFF_LINES,
    build_diff_report,
    render_diff_report,
)
from deepagents_code.diff_utils import DIFF_TRUNCATION_MARKER


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_git_repo(root: Path) -> None:
    root.mkdir()
    _run_git(root, "init")
    _run_git(root, "config", "user.email", "test@example.com")
    _run_git(root, "config", "user.name", "Test")
    _run_git(root, "config", "commit.gpgsign", "false")
    (root / "tracked.txt").write_text("line one\nline two\n")
    _run_git(root, "add", "tracked.txt")
    _run_git(root, "commit", "-m", "initial")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _init_git_repo(root)
    return root


class TestBuildDiffReport:
    """`/diff` reads real `git` state, so these run against a throwaway repo."""

    def test_not_a_git_repository(self, tmp_path: Path) -> None:
        report = build_diff_report(str(tmp_path))

        assert report.error == "Not inside a git repository."
        assert report.diff_text is None
        assert report.untracked_files == ()

    def test_no_changes(self, git_repo: Path) -> None:
        report = build_diff_report(str(git_repo))

        assert report.error is None
        assert report.diff_text is None
        assert report.stats is None
        assert report.untracked_files == ()

    def test_tracked_modification_is_diffed(self, git_repo: Path) -> None:
        (git_repo / "tracked.txt").write_text("line one\nchanged\n")

        report = build_diff_report(str(git_repo))

        assert report.error is None
        assert report.diff_text is not None
        assert "-line two" in report.diff_text
        assert "+changed" in report.diff_text
        assert report.stats is not None
        assert report.stats.additions == 1
        assert report.stats.deletions == 1

    def test_untracked_files_are_listed_not_diffed(self, git_repo: Path) -> None:
        (git_repo / "new_file.txt").write_text("hello\n")

        report = build_diff_report(str(git_repo))

        assert report.error is None
        assert report.diff_text is None
        assert report.untracked_files == ("new_file.txt",)

    def test_large_diff_is_truncated(self, git_repo: Path) -> None:
        (git_repo / "tracked.txt").write_text(
            "\n".join(f"line {i}" for i in range(1000)) + "\n"
        )

        report = build_diff_report(str(git_repo))

        assert report.diff_text is not None
        lines = report.diff_text.splitlines()
        assert len(lines) == _MAX_DIFF_LINES
        assert lines[-1] == DIFF_TRUNCATION_MARKER
        # Stats are computed before truncation, so they reflect the real change.
        assert report.stats is not None
        assert report.stats.deletions == 2  # noqa: PLR2004


class TestRenderDiffReport:
    """Rendering only needs a `DiffReport`, no real git repository."""

    def test_no_git_repository_shows_error(self, git_repo: Path) -> None:
        report = build_diff_report(str(git_repo.parent))

        assert render_diff_report(report) == "Not inside a git repository."

    def test_no_changes_message(self, git_repo: Path) -> None:
        report = build_diff_report(str(git_repo))

        assert render_diff_report(report) == "No uncommitted changes."

    def test_diff_and_untracked_files_both_rendered(self, git_repo: Path) -> None:
        (git_repo / "tracked.txt").write_text("line one\nchanged\n")
        (git_repo / "new_file.txt").write_text("hello\n")

        report = build_diff_report(str(git_repo))
        rendered = render_diff_report(report)

        assert "Uncommitted changes (+1/-1):" in rendered
        assert "```diff" in rendered
        assert "Untracked files:" in rendered
        assert "- new_file.txt" in rendered
