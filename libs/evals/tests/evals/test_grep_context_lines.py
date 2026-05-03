"""Evals for grep context_lines parameter — surrounding-line retrieval efficiency.

Tests whether agents correctly use `context_lines` to surface surrounding code in
a single grep call, avoiding follow-up `read_file` round-trips.

Three scenarios:
  1. Prompted: agent is told to use context_lines=3. Verifies correctness and
     that no read_file call was needed.
  2. Multi-file prompted: agent is told to show 2 lines of context around
     every match across multiple files using a single grep call.
  3. Unprompted: agent is asked to show surrounding lines without being told
     about context_lines; benchmarks whether the model discovers the parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from tests.evals.utils import (
    TrajectoryScorer,
    final_text_contains,
    run_agent,
    tool_call,
)


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_grep_context_lines_prompted_shows_surrounding_lines(
    model: BaseChatModel,
) -> None:
    """Agent uses context_lines=3 as instructed and surfaces the surrounding code."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/src/validators.py": (
                "import re\n"
                "\n"
                'EMAIL_REGEX = re.compile(r"[^@]+@[^@]+\\.[^@]+")\n'
                "\n"
                "\n"
                "def validate_email(address: str) -> bool:\n"
                '    """Return True if address looks like a valid email."""\n'
                "    return bool(EMAIL_REGEX.match(address))\n"
                "\n"
                "\n"
                "def validate_phone(number: str) -> bool:\n"
                "    return number.isdigit() and len(number) == 10\n"
            ),
        },
        query=(
            "Use grep with output_mode='content' and context_lines=3 to find "
            "'validate_email' in /src/validators.py. "
            "Report the matching line and all surrounding lines returned by the tool."
        ),
        # 1st step: grep with context_lines=3.
        # 2nd step: report result.
        # 1 tool call: single grep, no read_file needed.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[
                tool_call(
                    name="grep",
                    step=1,
                    args_contains={"pattern": "validate_email", "context_lines": 3},
                ),
            ],
        )
        .success(
            # EMAIL_REGEX is 3 lines before the match — only present if context was returned.
            final_text_contains("EMAIL_REGEX"),
            # The docstring is 1 line after the match.
            final_text_contains("Return True"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_grep_context_lines_avoids_read_file_for_multi_file_call_sites(
    model: BaseChatModel,
) -> None:
    """Agent surfaces 2 lines of context around matches across 3 files in one grep call."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/services/auth.py": (
                "def authenticate(token: str) -> str:\n"
                '    if not token.startswith("Bearer "):\n'
                '        raise ValueError("token must use Bearer scheme")\n'
                "    return token[7:]\n"
            ),
            "/services/payment.py": (
                "def charge(amount: float) -> None:\n"
                "    if amount <= 0:\n"
                '        raise ValueError("amount must be positive")\n'
                "    _process_charge(amount)\n"
            ),
            "/services/shipping.py": (
                "def calculate_fee(weight: float) -> float:\n"
                "    if weight < 0:\n"
                '        raise ValueError("weight cannot be negative")\n'
                "    return weight * 0.5\n"
            ),
        },
        query=(
            "Find every place that raises a ValueError across the /services directory "
            "and show 2 lines of context around each match. "
            "Use grep with output_mode='content' and context_lines=2 in a single call — "
            "do not use read_file. "
            "Output the raw code lines returned by the tool verbatim, do not summarize."
        ),
        # 1st step: grep with context_lines=2.
        # 2nd step: report all matches with context.
        # 1 tool call: a single grep across all 3 files, no read_file needed.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[
                tool_call(
                    name="grep",
                    step=1,
                    args_contains={"pattern": "raise ValueError", "context_lines": 2},
                ),
            ],
        )
        .success(
            # Each of these strings is a context line (not the match line itself),
            # so they only appear if the agent quoted the surrounding code verbatim.
            final_text_contains('token.startswith("Bearer ")'),  # before auth match
            final_text_contains("_process_charge"),  # after payment match
            final_text_contains("return weight * 0.5"),  # after shipping match
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_grep_context_lines_unprompted_reduces_round_trips(
    model: BaseChatModel,
) -> None:
    """Agent discovers context_lines unprompted to show surrounding lines in one shot.

    Benchmarks whether the model reaches for context_lines on its own when asked
    to show lines surrounding a match, rather than issuing a bare grep followed
    by a read_file to gather context.

    Success is correct surrounding-line output regardless of strategy.
    The expect tier records whether the model achieved it in a single tool call
    (optimal) vs. two (grep + read_file fallback).
    """
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/config.py": (
                "# Retry and timeout settings\n"
                "DEFAULT_TIMEOUT = 30\n"
                "MAX_RETRIES = 5\n"
                "BACKOFF_FACTOR = 1.5\n"
                "\n"
                "# Connection pool settings\n"
                "MAX_CONNECTIONS = 100\n"
            ),
        },
        query=(
            "Show me where MAX_RETRIES is defined along with the "
            "2 lines immediately before it and the 2 lines immediately after it. "
            "Use grep."
        ),
        # Optimal path: grep with context_lines=2 → 1 tool call, 2 steps.
        # Fallback path: grep then read_file → 2 tool calls, 3 steps.
        # The expect tier captures which path the model took.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[
                tool_call(
                    name="grep",
                    step=1,
                    args_contains={"pattern": "MAX_RETRIES"},
                ),
            ],
        )
        .success(
            # The match line itself.
            final_text_contains("MAX_RETRIES = 5"),
            # Lines before the match — only present if context was retrieved.
            final_text_contains("DEFAULT_TIMEOUT"),
            # Line after the match — only present if context was retrieved.
            final_text_contains("BACKOFF_FACTOR"),
        ),
    )
