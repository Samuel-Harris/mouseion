from __future__ import annotations

from pathlib import Path

import pytest

from mouseion.config import Settings
from mouseion.errors import MouseionError
from mouseion.ingest import repo as repo_module
from mouseion.ingest.repo import RepoService


async def test_existing_repo_refresh_does_not_force_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(MOUSEION_DATA_DIR=tmp_path / "data", MOUSEION_REPOS_DIR=tmp_path / "repos")
    repo_path = settings.mouseion_repos_dir / "project"
    (repo_path / ".git").mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run_git(args: list[str], cwd: Path) -> str:
        calls.append(args)
        if args[:2] == ["git", "rev-parse"] and "@{u}" in args:
            return "origin/main\n"
        if args[:3] == ["git", "merge", "--ff-only"]:
            raise MouseionError("local branch diverged")
        if args[:3] == ["git", "reset", "--hard"]:
            raise AssertionError("repo refresh must not discard local work")
        return "ok\n"

    monkeypatch.setattr(repo_module, "_run_git", fake_run_git)

    with pytest.raises(MouseionError, match="local branch diverged"):
        await RepoService(settings).add_repo("https://example.com/project.git")

    assert ["git", "reset", "--hard", "origin/main"] not in calls
