# How to run the pipeline

## Structure of the repos and data directories

Firstly, the whole project is split into 2 main repositories.

- S: script path
- ID: Input directory
- OD: Output directory

- [sdm-eurec4a](https://github.com/nilsnevertree/sdm-eurec4a)
    - Data preprocessing
        - Observational data preprocessing
            - S: ``./scripts/preprocessing``
            - ID: ``./data/observation/``
        - Cloud identification
            - S: ``./scripts/preprocessing/cluster_identification_general.py``
            - ID: ``./data/observation/cloud_composite/processed``
            - OD: ``./data/observation/cloud_composite/processed``
    - Fitting of DSDs and thermodynamics as INPUTS
        - S:
    - Visulization of Simulations by CLEO
- [CLEO-sdm-eurec4a](https://github.com/nilsnevertree/sdm-eurec4a)
    - Usage of INPUTS to run 1D-rainshaft instances
    - Run CLEO:
        - For each one of the 4 microphysical setups:
            RUN CLEO for all clouds
    - RAW CLEO output post processing
        - (Transform lagragian view 2 eulerian view for each cloud)
        - Selection of important data.
        - Combination of individual clouds to a merged dataset for each microphysical setup.

**BUT how to point to directories in a script or a notebook?**

For this purpose, please add the following line of code to your script or notebook:
````python
from sdm_eurec4a import RepositoryPath
````
To get the location of the ``sdm-eurec4a`` repo, specify in which development regime you are. E.g. ``levante``. Then run:

````python
REPO_PATH = RepositoryPath("levante").repo_dir
````

You can add your own development regime or change the existing ``levante`` regime to your individual locations in the ``RepositoryPath`` class in ``.src/sdm_eurec4a/__init__.py``

````python
_known_development_regimes = dict(
        levante=dict(
            repo_dir=Path("/home/m/m301096/repositories/sdm-eurec4a/"),
            data_dir=Path("/home/m/m301096/repositories/sdm-eurec4a/data/"),
            fig_dir=Path("/home/m/m301096/repositories/sdm-eurec4a/results/"),
            CLEO_dir=Path("/home/m/m301096/CLEO/"),
            CLEO_data_dir=Path("/home/m/m301096/CLEO/data/"),
        ),
    )
````

## 1. Observational data preprocessing

### 1.0 Download observational datasets

**Cloud composite**
There is a yaml-file describing the download procedure, and time of download.
``./data/observation/cloud_composite/download_info.yaml``

**Drop sondes**
There is a yaml-file describing the download procedure, and time of download.
``./data/observation/dropsonde/download_info.yaml``

### 1.1 Prepare the observational dataset to have consistent units

#### 1.1.1 Prepare the cloud composite dataset

S : ``./scripts/preprocessing/cloud_composite_si_units.py``
ID: ``./data/observation/cloud_composite/raw`` (location of the downloaded cloud composite files)
OD: specified by you, e.g. ``data/observation/cloud_composite/processed/cloud_composite_SI_units_20241025.nc``

The script  combines all individual cloud composite files, converts them into SI units and stores the output netcdf file in the ``DESTINATION_FILEPATH`` location.

Please change the following lines in the script before running.
````python
ORIGIN_DIRECTORY = REPO_PATH / Path("data/observation/cloud_composite/raw")
DESTINATION_DIRECTORY = REPO_PATH / Path("data/observation/cloud_composite/processed")
DESTINATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
DESTINATION_FILENAME = "cloud_composite_SI_units_20241025.nc"
DESTINATION_FILEPATH = DESTINATION_DIRECTORY / DESTINATION_FILENAME

log_file_path = DESTINATION_DIRECTORY / "cloud_composite_preprocessing.log"
````
