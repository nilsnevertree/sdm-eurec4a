"""Use Super-Droplet Model and EUREC4A data to simulate rain evaporation."""

from __future__ import annotations

import subprocess

from pathlib import Path
from typing import Any


class RepositoryPath:
    """Path to the repository root."""

    _known_development_regimes = dict(
        levante=dict(
            repo_dir=Path("/home/m/m301096/repositories/sdm-eurec4a/"),
            data_dir=Path("/work/mh1126/m301096/softlinks/sdm-eurec4a/data"),
        ),
        nils_levante=dict(
            repo_dir=Path("/home/m/m301096/repositories/sdm-eurec4a/"),
            data_dir=Path("/work/mh1126/m301096/softlinks/sdm-eurec4a/data"),
        ),
    )

    def __init__(self, development_regime, *args, **kwargs):
        if development_regime not in RepositoryPath._known_development_regimes:
            raise ValueError(
                f"Unknown development regime: {development_regime}. "
                f"Known development regimes: {RepositoryPath._known_development_regimes}"
            )
        else:
            self.repo_dir = RepositoryPath._known_development_regimes[development_regime]["repo_dir"]
            self.data_dir = RepositoryPath._known_development_regimes[development_regime]["data_dir"]

    def set_repo_dir(self, repo_dir):
        self.repo_dir = repo_dir

    def set_data_dir(self, data_dir):
        self.data_dir = data_dir

    def get_repo_dir(self):
        return self.repo_dir

    def get_data_dir(self):
        return self.data_dir

    def __call__(self) -> Path:
        return self.repo_dir

    def __str__(self) -> str:
        return str(self.repo_dir)


def get_git_revision_hash() -> str:
    """
    Get the git revision hash.

    Parameters
    ----------
    None

    Returns
    -------
    str
        The full git revision hash.
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
