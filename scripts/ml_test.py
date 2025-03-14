from deriva_ml.demo_catalog import DemoML
from importlib.metadata import version, PackageNotFoundError
from setuptools_git_versioning import version_from_git, get_tag, get_branch, get_latest_file_commit
import subprocess
from pathlib import Path
import inspect

def get_top_stack_filename():
    stack = inspect.stack()
    if len(stack) > 1:
        return Path(stack[1].filename)  # Get the caller's filename
    return None  # Stack is too shallow

def file_is_dirty(file_path):
    """Check if a file has been modified but not committed in a Git repository."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        return bool(result.stdout.strip())  # Returns True if output is non-empty (file is modified)
    except subprocess.CalledProcessError:
        return False  # If Git command fails, assume no changes

def repo_root(repo_root):
    while repo_root != repo_root.root:
        if (repo_root / '.git').exists():
            break
        else:
            repo_root = repo_root.parent
    return repo_root

def get_version(repo_root) -> str:
    return version_from_git(
        root=repo_root,
        count_commits_from_version_file=True,
        version_file='VERSION',
        template="{tag}+{branch}.{sha}",
        dev_template="{tag}.post{ccount}+{branch}.{sha}",
        dirty_template="{tag}.post{ccount}+{branch}.{sha}.dirty"
    )

def github_url():
    filename = get_top_stack_filename()
    result = subprocess.run(['git', 'remote', 'get-url', 'origin'], capture_output=True, text=True)
    github_url = result.stdout.strip().removesuffix('.git')
    sha = get_latest_file_commit(filename)
    url = f'{github_url}/blob/{sha}/{filename.relative_to(repo_root(filename))}'
    print(file_is_dirty(filename))
    return url

if __name__ == "__main__":
    github_url = github_url()
    print(github_url)
    print(get_tag())
    print(get_branch())
