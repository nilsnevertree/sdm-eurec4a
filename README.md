# SDM-EUREC4A
Using Super Droplet Model and EUREC4A data to simulate rain evaporation.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![codecov](https://codecov.io/gh/nilsnevertree/sdm-eurec4a/branch/main/graph/badge.svg)](https://codecov.io/gh/nilsnevertree/sdm-eurec4a)
[![Documentation Status](https://readthedocs.org/projects/sdm-eurec4a/badge/?version=latest)](https://sdm-eurec4a.readthedocs.io/en/latest/?badge=latest)

<!-- [![PyPI](https://img.shields.io/pypi/v/sdm-euerc4a.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/sdm-euerc4a.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/sdm-euerc4a)][pypi status] -->
<!-- [![Read the documentation at https://sdm-euerc4a.readthedocs.io/](https://img.shields.io/readthedocs/sdm-euerc4a/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/yoctoyotta1024/sdm-euerc4a/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/nilsnevertree/sdm-euerc4a/branch/main/graph/badge.svg)][codecov] -->


[pypi status]: https://pypi.org/project/sdm-euerc4a/
[read the docs]: https://sdm-euerc4a.readthedocs.io/
[tests]: https://github.com/yoctoyotta1024/sdm-euerc4a/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/nilsnevertree/sdm-euerc4a
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

In this project, the rain evaporation below cloud base shall be simulated using the Super Droplet Model [CLEO].

To understand how to use this project, please refer to [docs/source/pipeline_howto.md](https://github.com/yoctoyotta1024/sdm-eurec4a/blob/main/docs/source/pipeline_howto.md).
Additionally, you must source the ``data`` folder (contact authors) and create two new (empty) folders for ``logs`` and ``results``.

# Installing mpi4py Levante

On Levante, you may have trouble using ``mpi4py`` within a micromamba/conda environment. E.g. the
following Python script will fail:

``` python
from mpi4py import MPI

comm = MPI.COMM_WORLD
print(f"Rank: {comm.Get_rank()}, Size: {comm.Get_size()}")
```

with the error ``RuntimeError: cannot load MPI library``.

If so, you need to re-install ``mpi4py`` with the correct links to Levante's openmpi modules:

``` bash
### activate your environment
$ mamba activate sdm_eurec4a_env312

### load relevant packages on Levante
$ module load python3 gcc/11.2.0-gcc-11.2.0 openmpi/4.1.2-gcc-11.2.0
$ export MPI4PY_BUILD_MPICC=/sw/spack-levante/openmpi-4.1.2-mnmady/bin/mpicc
$ export MPI4PY_BUILD_MPILD=/sw/spack-levante/openmpi-4.1.2-mnmady/lib

### uninstall and re-install mpi4py
$ mamba install mpi=*=*
$ python -m pip uninstall mpi4py
$ python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py

### (optional but good to remove if they've been installed)
$ rm  /path/to/your/env/sdm_eurec4a_env312/lib/libmpi.so
$ rm  /path/to/your/env/sdm_eurec4a_env312/lib/libmpi.so.40

### check the installation worked
$ python -c 'import ctypes.util; print(ctypes.util.find_library("mpi"))'
```

## Commiting.
The pre-commit hook is already installed in the ``.git`` folder. To diable this, if you wish to run it manually, please make sure to diable the file ``pre-commit``. Do not remove the ``pre-commit.sample`` file, as it is just a sample file.
<!--
## Features

- TODO

## Requirements

- TODO

## Installation

You can install _sdm-euerc4a_ via [pip] from [PyPI]:

```console
$ pip install sdm-euerc4a
```

## Usage

Please see the [Command-line Reference] for details. -->

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [GPL 3.0 license][license],
_sdm-euerc4a_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

<!-- ## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template. -->


<!-- LINKS -->
[CLEO]: https://github.com/yoctoyotta1024/CLEO
[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/yoctoyotta1024/sdm-eurec4a/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/yoctoyotta1024/sdm-euerc4a/blob/main/LICENSE
[contributor guide]: https://github.com/yoctoyotta1024/sdm-euerc4a/blob/main/CONTRIBUTING.md
[command-line reference]: https://sdm-euerc4a.readthedocs.io/en/latest/usage.html
