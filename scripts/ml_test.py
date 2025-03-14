from deriva_ml.demo_catalog import DemoML
from importlib.metadata import version, PackageNotFoundError
from setuptools_git_versioning import version_from_git, get_tag, get_branch, get_latest_file_commit, get_sha
import subprocess
from pathlib import Path
import inspect

def get_top_stack_filename():
    stack = inspect.stack()
    if len(stack) > 1:
        return Path(stack[1].filename)  # Get the caller's filename
    return None  # Stack is too shallow


def get_version() -> str:
    repo_root = Path(__file__)
    while repo_root != repo_root.root:
        if (repo_root / '.git').exists():
            break
        else:
            repo_root = repo_root.parent

    return version_from_git(
        root=repo_root,
        count_commits_from_version_file=True,
        version_file='VERSION',
        template="{tag}+{branch}.{sha}",
        dev_template="{tag}.post{ccount}+{branch}.{sha}",
        dirty_template="{tag}.post{ccount}+{branch}.{sha}.dirty"
    )

def github_url(file):
    # Running a shell command and capturing its output
    result = subprocess.run(['git', 'remote', 'get-url', 'origin'], capture_output=True, text=True)
    github_url = result.stdout.strip().removesuffix('.git')
    result = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True)
    current_branch = result.stdout.strip()
    print(get_latest_file_commit(file))
    url = f'{github_url}/blob/{get_sha()}/{filename}'
    return url

if __name__ == "__main__":
    filename = get_top_stack_filename()
    github_url = github_url(filename)
    print(filename)
    print(github_url)
    print('version', get_version())
    print(get_tag())
    print(get_branch())
    print(get_latest_file_commit(filename))
