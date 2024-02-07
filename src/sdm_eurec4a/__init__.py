"""Use Super-Droplet Model and EUREC4A data to simulate rain evaporation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

class RepositoryPath():
    """Path to the repository root."""
    def __init__(self, development_regime, *args, **kwargs):
        if development_regime == 'levante':
            self.repo_dir = Path('/home/m/m301096/repositories/sdm-eurec4a/')
            self.data_dir = Path('/work/mh1126/m301096/softlinks/sdm-eurec4a/data')

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
    