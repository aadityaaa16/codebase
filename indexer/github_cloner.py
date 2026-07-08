"""
github_cloner.py

Clones a public GitHub repository to a local temp directory so it can
be indexed by the existing AST chunker pipeline. This is what lets a
publicly deployed demo accept "paste any GitHub URL" instead of being
locked to a repo path that only exists on the developer's own machine.

Security notes:
- Only accepts github.com HTTPS URLs (not git://, ssh://, or arbitrary
  hosts) to avoid this becoming an open proxy for cloning arbitrary or
  internal network resources (a real SSRF-style risk if left open).
- Uses a shallow clone (depth=1) to keep clone time and disk usage
  reasonable, since a public demo has no control over how large a
  visitor-supplied repo might be.
- Only one repo is kept cloned at a time (replacing the previous one),
  consistent with the rest of the app's single-repo-at-a-time design.
"""

import os
import re
import shutil
import tempfile
import git  # GitPython

_CLONE_DIR_NAME = "codebase_navigator_clone"

# Matches https://github.com/<user>/<repo> or .../<repo>.git, nothing else
_GITHUB_URL_PATTERN = re.compile(
    r"^https://github\.com/[\w.-]+/[\w.-]+(?:\.git)?/?$"
)


def is_valid_github_url(url: str) -> bool:
    """
    Validates that the URL is a plain github.com HTTPS repo URL.
    Rejects anything else (other hosts, git://, ssh://, query strings
    with unexpected content, etc.) as a basic safety guard.
    """
    return bool(_GITHUB_URL_PATTERN.match(url.strip()))


def clone_repo(github_url: str) -> str:
    """
    Clones the given public GitHub repo (shallow, depth=1) into a fixed
    temp directory, replacing any previously cloned repo there.

    Returns the local filesystem path to the cloned repo.

    Raises:
        ValueError: if the URL isn't a valid github.com HTTPS URL.
        RuntimeError: if the clone itself fails (repo doesn't exist,
                      is private, network issue, etc.)
    """
    if not is_valid_github_url(github_url):
        raise ValueError(
            "Only public GitHub HTTPS URLs are supported, e.g. "
            "https://github.com/username/repository"
        )

    clone_path = os.path.join(tempfile.gettempdir(), _CLONE_DIR_NAME)

    if os.path.exists(clone_path):
        shutil.rmtree(clone_path, ignore_errors=True)

    try:
        git.Repo.clone_from(github_url, clone_path, depth=1)
    except git.exc.GitCommandError as e:
        raise RuntimeError(
            f"Could not clone repository. Check the URL is correct and the "
            f"repo is public. ({str(e)[:200]})"
        )

    return clone_path


if __name__ == "__main__":
    # Quick manual test against a small real public repo
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/octocat/Hello-World"
    path = clone_repo(url)
    print(f"Cloned to: {path}")
    print("Contents:", os.listdir(path))
