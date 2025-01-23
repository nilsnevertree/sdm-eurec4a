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
            data_dir=Path("/home/m/m301096/repositories/sdm-eurec4a/data/"),
            fig_dir=Path("/home/m/m301096/repositories/sdm-eurec4a/results/"),
            CLEO_dir=Path("/home/m/m301096/CLEO/"),
            CLEO_data_dir=Path("/home/m/m301096/CLEO/data/"),
        ),
    )

    def __init__(self, development_regime, *args, **kwargs):
        if development_regime not in RepositoryPath._known_development_regimes:
            raise ValueError(
                f"Unknown development regime: {development_regime}. "
                f"Known development regimes: {RepositoryPath._known_development_regimes}"
            )
        else:

            self._development_regime = development_regime
            self._repo_dict = RepositoryPath._known_development_regimes[development_regime]

    @property
    def repo_dir(self) -> Path:
        return self._repo_dict["repo_dir"]

    @property
    def data_dir(self) -> Path:
        return self._repo_dict["data_dir"]

    @property
    def fig_dir(self) -> Path:
        return self._repo_dict["fig_dir"]

    @property
    def CLEO_dir(self) -> Path:
        return self._repo_dict["CLEO_dir"]

    @property
    def CLEO_data_dir(self) -> Path:
        return self._repo_dict["CLEO_data_dir"]

    def __call__(self) -> dict:
        return self._repo_dict

    def __str__(self) -> str:
        return f"{self._development_regime}\n" + str(self._repo_dict)


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
