from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from mouseion.config import Settings
from mouseion.errors import MouseionError
from mouseion.support.utils import safe_filename


@dataclass(slots=True)
class RepoService:
    settings: Settings

    async def add_repo(self, repo_url: str, name: str | None = None) -> dict[str, str]:
        repo_name = name or _repo_name_from_url(repo_url)
        repo_path = (self.settings.mouseion_repos_dir / repo_name).resolve()
        self.settings.mouseion_repos_dir.mkdir(parents=True, exist_ok=True)
        if not repo_path.exists():
            _run_git(
                ["git", "clone", repo_url, str(repo_path)],
                cwd=self.settings.mouseion_repos_dir,
            )
            status = "cloned"
        else:
            if not (repo_path / ".git").exists():
                raise MouseionError(f"Repository path exists but is not a git repo: {repo_path}")
            _run_git(["git", "fetch", "--prune"], cwd=repo_path)
            upstream = _run_git(
                ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
                cwd=repo_path,
            ).strip()
            _run_git(["git", "merge", "--ff-only", upstream], cwd=repo_path)
            status = "pulled"
        commit_sha = _run_git(["git", "rev-parse", "HEAD"], cwd=repo_path).strip()
        return {"repo_path": str(repo_path), "status": status, "commit_sha": commit_sha}


def _repo_name_from_url(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    candidate = Path(parsed.path).name or repo_url.rsplit("/", maxsplit=1)[-1]
    candidate = re.sub(r"\.git$", "", candidate)
    return safe_filename(candidate, "repo")


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"git failed: {' '.join(args)}"
        raise MouseionError(message)
    return result.stdout
