"""analytics/environment.py's two Manual QA Phase 13 additions - the About
dialog's own "Build Timestamp" and "Python Version" fields.
"""

from __future__ import annotations

import platform
import subprocess

from picklikeme.analytics.environment import resolve_git_commit_timestamp, resolve_python_version


def test_resolve_python_version_matches_the_actual_interpreter() -> None:
    assert resolve_python_version() == platform.python_version()


def test_resolve_git_commit_timestamp_matches_the_real_commit_date() -> None:
    """This repo IS a git checkout during tests, so the real path (not the
    "not a git repo" fallback) is what actually runs here."""
    actual = subprocess.run(
        ["git", "log", "-1", "--format=%cI"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert resolve_git_commit_timestamp() == actual


def test_resolve_git_commit_timestamp_is_none_when_git_is_unavailable(monkeypatch) -> None:
    """Never fabricated as "now" - a packaged/non-git install must get None,
    the same "explicit unknown" contract resolve_git_commit already has."""
    import picklikeme.analytics.environment as environment_module

    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(environment_module.subprocess, "run", _raise)

    assert resolve_git_commit_timestamp() is None


def test_resolve_git_commit_timestamp_is_none_on_a_nonzero_git_exit(monkeypatch) -> None:
    import picklikeme.analytics.environment as environment_module

    class _FakeResult:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(environment_module.subprocess, "run", lambda *a, **k: _FakeResult())

    assert resolve_git_commit_timestamp() is None
