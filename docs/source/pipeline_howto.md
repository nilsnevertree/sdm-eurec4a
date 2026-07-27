# How to run the pipeline

## Overview
This document explains the steps required to run the EURCE4A-SDM pipeline with CLEO, the sections are as follows:
- [Structure of the repos and data directories](#structure-of-the-repos-and-data-directories)
- [Observational data and fittings](#observational-data-and-fittings)
- [Running CLEO based on Observational Fits](#running-cleo-based-on-observational-fits)
- [Post Processing of CLEOs Raw Output](#post-processing-of-cleos-raw-output)
- [Plot and Use the Conservation and Eulerian Views](#plot-and-use-the-conservation-and-eulerian-views) (currently work-in-progress)

---

# Structure of the repos and data directories

Firstly, the whole project is split into 2 main repositories.

- S: Script path
- A : File with Arguments for the script
- ID: Input directory
- OD: Output directory

- [sdm-eurec4a](https://github.com/yoctoyotta1024/sdm-eurec4a)
    - Data preprocessing
        - Observational data preprocessing
            - S: ``./scripts/preprocessing``
            - ID: ``./data/observation/``
        - Cloud identification
            - S: ``./scripts/preprocessing/cluster_identification_general.py``
            - ID: ``./data/observation/cloud_composite/processed``
            - OD: ``./data/observation/cloud_composite/processed``
    - Fitting of DSDs and thermodynamics as INPUTS
        - S: ``.ipynb`` notebooks under issues ``107`` and ``114``
    - Visulization of Simulations by CLEO
- [sdm-eurec4a-CLEO](https://github.com/yoctoyotta1024/sdm-eurec4a-CLEO)
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
REPO_PATH = RepositoryPath("levante_m300950").repo_dir
````

You can add your own development regime or change the existing ``levante`` regime to your individual locations in the ``RepositoryPath`` class in ``.src/sdm_eurec4a/__init__.py``

````python
_known_development_regimes = dict(
        levante_m301096=dict(
            repo_dir=Path("/home/m/m301096/repositories/sdm-eurec4a/"),
            data_dir=Path("/home/m/m301096/repositories/sdm-eurec4a/data/"),
            fig_dir=Path("/home/m/m301096/repositories/sdm-eurec4a/results/"),
            CLEO_dir=Path("/home/m/m301096/CLEO/"),
            CLEO_data_dir=Path("/home/m/m301096/CLEO/data/"),
        ),
    )
````

**Before you begin with sdm_eurec4a**
1) create a mamba/conda environment from the environment.yml file (e.g. ``mamba create -f environment.yml``).
2) After activating your new environment (e.g. ``mamba activate sdm_eurec4a_env312``), install the ``sdm_eurec4a`` package locally with ``python -m pip install .``
3) replace all mentions of ``m301096`` and ``m300950``, and ``um1487`` and ``mh1126``, with your DKRZ account and project IDs.

---

# Observational data and fittings

## 1. Observational data preprocessing

The whole idea is to:
1. Prepare observational datasets to have meaningful variable names, SI units
2. Identify individual clouds
3. Create a distance dataset which can be used as a lookup table. It countains the spatial and temporal distance between drop sondes and individual clouds

### 1.0 Download observational datasets

**Cloud Composite**
There is a yaml-file describing the download procedure, and time of download.
``./docs/source/download_info/cloud_composite_download_info.yaml``

**Dropsondes**
There is a yaml-file describing the download procedure, and time of download.
``./docs/source/download_info/dropsonde_download_info.yaml``

**Safire Cores**
There is a yaml-file describing the download procedure, and time of download.
``./docs/source/download_info/safire_core_download_info.yaml``

*NOTE:* you will need to replace ``SPECIFYTHIS`` with the name you want for the directory of the data you download. Good ideas are to save them under ``raw`` in seperate folders of a directory called ``/path/to/your/data_dir/observation/``, i.e. in:
- ``/path/to/your/data_dir/observation/cloud_composite/raw``,
- ``/path/to/your/data_dir/observation/dropsonde/raw``, and
- ``/path/to/your/data_dir/observation/safire_core/raw``

respectively. For the dropsonde data you also need to replace ``SPECIFYLEVEL`` with the level you want, it's best to download ``Level_3``, ``Level_4`` and ``QC`` within seperate directories of ``[...]/observation/dropsonde/raw``, e.g. ``[...]/observation/dropsonde/raw/Level_3``. (Although probably you will only use ``Level_3``.)

*NOTE:* For the scripts to run automatically, you will then need to move the ``*.nc`` files in ``[...]/observation/dropsonde/raw/[Level_3 or Level_4 or QC]/`` out of their nested directories and into ``[...]/observation/dropsonde/raw/[Level_3 or Level_4 or QC]/``, e.g.

``mv ./data/observation/dropsonde/raw/Level_3/eurec4a-data/PRODUCTS/MERGED-MEASUREMENTS/JOANNE/v2.0.0/Level_3/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc ./data/observation/dropsonde/raw/Level_3/``

(and to clean-up: ``rm -rf ./data/observation/dropsonde/raw/Level_3/eurec4a-data``). Likewise you have to move the ``*.nc`` in ``./data/observation/safire_core/raw/eurec4a-data/`` into ``./data/observation/safire_core/raw/`` (and ``rm -rf eurec4a-data``).

*NOTE:* while you're at it, it's advisable to make to two further directories in ``data``, ``./data/model`` and ``./data/sharing``.

### 1.1 Prepare the observational dataset to have consistent units

#### A) Prepare the cloud composite dataset

S : ``./scripts/preprocessing/cloud_composite_si_units.py``
ID: ``/path/to/your/data_dir/observation/cloud_composite/raw`` (location of the downloaded cloud composite files)
OD: specified by you, e.g. ``/path/to/your/data_dir/observation/cloud_composite/processed/cloud_composite_SI_units_20241025.nc``

The script  combines all individual cloud composite files, converts them into SI units and stores the output netcdf file in the ``DESTINATION_FILEPATH`` location.

Please change the following lines in the script before running.
````python
ORIGIN_DIRECTORY = DATA_PATH / Path("observation/cloud_composite/raw")
DESTINATION_DIRECTORY = DATA_PATH / Path("observation/cloud_composite/processed")
DESTINATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
DESTINATION_FILENAME = "cloud_composite_SI_units_20241025.nc"
DESTINATION_FILEPATH = DESTINATION_DIRECTORY / DESTINATION_FILENAME

log_file_path = DESTINATION_DIRECTORY / "cloud_composite_preprocessing.log"
````

#### B) Prepare Drop sonde dataset

S: ``./scripts/preprocessing/drop_sondes.py``
ID: ``/path/to/your/data_dir/observation/dropsonde/raw/Level_3/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc``
OD: ``/path/to/your/data_dir/observation/dropsonde/processed/drop_sondes.nc``

The script mainly provides more meaningful variable names and uses  "time" (the launch time of the dropsonde) as the leading dimension instead of the dropsonde ID.

#### C) Prepare the Safire Core dataset

S: ``./scripts/preprocessing/safire_core.py``
ID: ``/path/to/your/data_dir/observation/safire_core/raw/*.nc``
OD: ``/path/to/your/data_dir/observation/safire_core/processed/safire_core.nc``


### 1.2 Identify individual rain clouds

To identify rain clouds in the cloud composite dataset, there are two scripts available.
Please use the cluster identification script:

S: ``./scripts/preprocessing/cluster_identification_general.py``
A: ``./scripts/preprocessing/settings/cluster_identification.yaml``
ID: ``/path/to/your/data_dir/specified/in/A``
OD: ``/path/to/your/data_dir/specified/in/A``

The script identifies all individual rain clouds based on the ``rain_mask`` given in the cloud composite dataset.
The output netcdf file contains the new dimension ``cloud_id``

The settings file looks like this:
````yaml

# The settings for the

paths:
  # The paths need to be relative to the root directory of the project.
  # The path to the input file
  input_filepath: observation/cloud_composite/processed/cloud_composite_SI_units_20241025.nc
  # The path to the directory where the output data will be stored
  output_directory: observation/cloud_composite/processed/identified_clusters/
  # The output file name. If NULL is provided, the input file will be automatically made:
  # f"identified_clouds_{mask_name}.nc"
  output_file_name : NULL
# The settings for the cloud identification
setup:
  # The name of the mask that shall be used
  mask_name : 'rain_mask'
  # The minimum duration of the cloud holes in timesteps, for them to be considered as cloud holes
  # this defines how far each cloud in a cluster can be apart from each other
  min_duration_cloud_holes : 5 # in timesteps
````

### 1.3 Create cloud and drop sonde distance dataset

S: ``./scripts/preprocessing/distance_relation_IC_DS.py``
A: ``./scripts/preprocessing/settings/distance_calculation.yaml``
ID: ``/path/to/your/data_dir/specified/in/A``
OD: ``/path/to/your/data_dir/specified/in/A``

Within the settings file, you can specify the input and output files

````yaml
# The settings for the distance calculation between identified clouds and the dropsondes
paths:
  # The paths need to be relative to the root directory of the project.
  # The path to the input file
  input_filepath_clouds: observation/cloud_composite/processed/identified_clusters/identified_clusters_rain_mask_5.nc
  input_filepath_dropsondes: observation/dropsonde/processed/drop_sondes.nc
  # The path to the directory where the output data will be stored
  output_directory: data/observation/combined/distance/
  # The output file name. If NULL is provided, the input file will be automatically made:
  # f"identified_clouds_{mask_name}.nc"
  output_file_name : NULL
````

## 2. How to select individual clouds and dropsondes

````python

from sdm_eurec4a.identifications import match_clouds_and_dropsondes, match_clouds_and_cloudcomposite
import xarray as xr
RP = RepositoryPath("levante_m300950")
repo_dir = RP.repo_dir
data_dir = RP.data_dir

drop_sondes = xr.open_dataset(repo_dir / "data/observation/dropsonde/processed/drop_sondes.nc")
distance = xr.open_dataset(
    repo_dir
    / "data/observation/combined/distance/distance_dropsondes_identified_clusters_rain_mask_5.nc"
)
cloud_composite = xr.open_dataset(
    repo_dir / "data/observation/cloud_composite/processed/cloud_composite_SI_units_20241025.nc"
)
identified_clusters = xr.open_dataset(
    repo_dir
    / "data/observation/cloud_composite/processed/identified_clusters/identified_clusters_rain_mask_5.nc"
)
````

To select an individual cloud, you can simply select the cloud based on the ``time`` or swap the dimensions ``time`` and ``cloud_id`` and select by ``cloud_id``.

To select all data of the cloud composite dataset for one cloud, use this code:

````python
cloud_composite_selected = match_clouds_and_cloudcomposite(
    ds_clouds=your_cloud_id,
    ds_cloudcomposite=cloud_composite,
)
````

To select an individual drop sonde, you can simply select in based on the launch time ``time``.

To select data for all drop sondes which were release within spatial distance (100km) and temporal distance (3h), you can use the following code:

````python
your_cloud_id = 42
# swap ``time`` and ``cloud_id`` to select by your cloud id
ic = identified_clusters.swap_dims({"time": "cloud_id"}).sel(cloud_id=your_cloud_id)

# to select the subset of dropsondes within 3h hours and 100km distance, use this code:
# if cloud was at 13:00, all dropsondes within 10:00-16:00 will be used.

ds = match_clouds_and_dropsondes(
    ds_clouds=ic,
    ds_sonde=drop_sondes,
    ds_distance=distance,
    max_temporal_distance=np.timedelta64(3, "h"),
    max_spatial_distance=1e2,
)
````


## 3. Fit the DSDs and thermodynamics

The scripts to run CLEO for all clouds in the ``sdm-eurec4a-CLEO`` repo need input files for the DSDs and thermodynamic fits.

The files are netcdf files with the ``cloud_id`` as leading dimension.
Four input files are needed:
- ``particle_size_distribution_parameters.nc``
- ``potential_temperature_parameters.nc``
- ``relative_humidity_parameters.nc``
- ``pressure_parameters.nc``

To create these files, there are two notebooks that you can use, as described below:


### 3.1. Fit the DSDs for all clouds

S: ``./notebooks/issues/107/107-different-fit-options-good-setup.ipynb``
ID: defined in the script
OD: defined in the script

In the notebook, the cloud composite data is coarsened to have more information per radius bin.

There are multiple double Log-Normal fits produced within the notebook.
- No weight
- Weighted by the square of the radius
- Weighted by the cube of the radius

We use the weighting by the cube of the radius.
This gives more weight to the larger radii, which have much higher mass but very little number concentration and would otherwise be underestimated.


The output will be multiple files in the output dir (e.g. ``./data/model/input_v4.2``)

**particle_size_distribution_parameters_linear_space.nc**
File containing the parameters for the bimodal Log-normal distributions for all DSDs. It is a netcdf file, with teh leading dimension being the ``cloud_id``.
The parameters for the bimodal Log-Normal fits are individual variables.

````python
<xarray.Dataset> Size: 9kB
Dimensions:             (cloud_id: 154)
Coordinates:
  * cloud_id            (cloud_id) int64 1kB 0 1 2 5 6 7 ... 561 563 565 569 571
Data variables:
    geometric_mean1     (cloud_id) float64 1kB ...
    geometric_std_dev1  (cloud_id) float64 1kB ...
    scale_factor1       (cloud_id) float64 1kB ...
    geometric_mean2     (cloud_id) float64 1kB ...
    geometric_std_dev2  (cloud_id) float64 1kB ...
    scale_factor2       (cloud_id) float64 1kB ...
Attributes:
    description:        parameters of a double log-normal distribution fitted...
    details:            Parameters of the double log-normal distribution fitt...
    creation_time:      2024-12-02 12:52:20
    author:             Nils Niebaum
    email:              nils-ole.niebaum@mpimet.mpg.de
    institution:        Max Planck Institute for Meteorology
    github_repository:  https://github.com/yoctoyotta1024/sdm-eurec4a
    git_commit:         ecf2fec509d23640392c7c40b7e7522d4639ce67
    parameter_space:    geometric
    independent_space:  linear
````

### 3.2. Fit the Thermodynamics for all clouds

S: ``./notebooks/issues/114/114-enhance-thermodynamic-fit.ipynb``
ID: defined in the script
OD: defined in the script

Will produce thermodynamic fits in the output dir (e.g. ``./data/model/input_v4.2``).

There are fits for the
- potential_temperature : constant in sub cloud layer with linear fit above cloud base
- relative_humidity : linear fit with saturation of 100% at cloud base
- pressure : linear fit

Example in the relative humidity file ``./data/model/input_v4.2/relative_humidity_parameters.nc``

````python
<xarray.Dataset> Size: 10kB
Dimensions:   (cloud_id: 260)
Coordinates:
  * cloud_id  (cloud_id) int64 2kB 80 81 82 83 84 85 ... 567 568 569 570 571 572
Data variables:
    f_0       (cloud_id) float64 2kB ...
    slope_1   (cloud_id) float64 2kB ...
    x_split   (cloud_id) float64 2kB ...
    slope_2   (cloud_id) float64 2kB ...
````

---

# Running CLEO based on Observational Fits

*NOTE:* If you have created all the input data in your ``${HOME}`` (in ``/home/[...]/sdm-eurec4a/data``), it is
likely you will eventually exceed your Levante disk quota. It's therefore advisable to now more that
data directory elsewhere, e.g. to ``work`` (``/work/[...]/sdm-eurec4a/data``).

To run CLEO, we need the input files which we created in (3).

We use the example directory in ``./examples/eurec4a1d``.

For each microphysical setup, steps 4.1 and 4.2 need to be performed.

You can perform step 4.1 for all setups and then perform 4.2 afterwards.

Post-processing CLEO's output data is explained in 4.3.

Before you begin it is reccomended to create a folder called ``data`` somewhere where you have a large
disk-space allocation e.g. ``/work/[...]/sdm-eurec4a-CLEO/data`` directory. (Do not overwrite ``/work/[...]/sdm-eurec4a/data``!)

Additionally it is advisable to make the logfiles folder, e.g. with
``cd /work/[...]/sdm-eurec4a/data && mkdir logfiles && cd logfiles && mkdir create_init_files  full_workflow  run_CLEO  run_CLEO_single  run_CLEO_singledebug  update_config``.

As above for the ``sdm-eurec4a`` repository, before you can begin, first replace all mentions of ``m301096`` and ``m300950``, and ``um1487``, ``mh1126`` and ``bm1183``, with your DKRZ account and project IDs. You will probably also want to edit the email address in SLURM jobs, ``#SBATCH --mail-user=[...]```


## 4.1 Prepare the input files for CLEO

S: ``./examples/eurec4a1d/create_model_input_files.sh``
ID : defined as ``path2input``
OD: defined as ``path2output``

This bash script invokes an MPI task with a number of workers to parallel create the input files for CLEO.

It uses the script ``examples/eurec4a1d/scripts/create_model_input_mpi4py.py``.
The python script uses the input files created in (3).

A log file for each mpi task is created within ``/your/path2logfiles/logfiles/create_init_files/mpi4py/yyyymmdd-hhMMss``, given ``path2logfiles`` defined in ``create_model_input_mpi4py.py``. (This is as well as the logfile from the SLURM job defined in ``create_model_input_files.sh``.)

To select a microphysical setup, comment/uncomment within ``./examples/eurec4a1d/create_model_input_files.sh`` these lines:

````bash
### ------------------ Input Parameters ---------------- ###
# microphysics="null_microphysics"
microphysics="condensation"
# microphysics="collision_condensation"
# microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"
````

Output directory naming convention is ``/your/path2data/output_[version_output]/[choice_microphysics]/[cluster_XXX]/``, and
within each of these output directories, i.e. for each cluster, you will find the ``config``, ``figures``, and ``share`` directories which contain and plot the initial condition, grid-information, and configuration files for CLEO.

For more details, see the python script.

## 4.2 Simulate all clouds for 1 microphysical setup.

The main script is: ``./examples/eurec4a1d/build_compile_run_eurec4a1d.sh``

Here you define:
- your source directory, e.g. ``path2sdmeurec4aCLEO=/home/[...]/sdm-eurec4a-CLEO``
- your build directory, e.g. ``path2build=/work/[...]/sdm-eurec4a-CLEO/build/``
- your data directory (for output of runs), e.g. ``path2data=/work/[...]/sdm-eurec4a-CLEO/data/[...]``

It can be used to build CLEO, compile CLEO, run CLEO simply toggle the lines:

``` bash
build=true
compile=true
run=false
```

After build and compile (via SLURM or on your command line), you can run CLEO for one
microphysics setup (via SLURM). Select which microphysics by commenting/uncommenting the lines:

``` bash
# microphysics="null_microphysics"
microphysics="condensation"
# microphysics="collision_condensation"
# microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"
# microphysics="coalbure_condensation_cke"
```

By running CLEO using this script, a SLURM Array will be spawned, which will run CLEO for all clouds
for which input files were created in 4.1

As above, the output directory naming convention is ``/your/path2data/output_[version_output]/[choice_microphysics]/[cluster_XXX]/``. Successfully running CLEO will add ``eurec4a1d_sol.zarr`` and ``./config/eurec4a1d_setup.txt`` to each cluster's directory.

Logfiles are created from the SLURM scripts ``build_compile_run_eurec4a1d.sh`` and ``run_job_array_eurec4a1d.sh``.

###### Extra Info. / Side-Note / Maybe Wrong:
###### For individual runs, the output directory naming convention may be ``/your/path2data/output_YOUR-CHOICE-CLEO_VERIONS-OF-CLEO-input_NETCDF_INPUT-VERSION`` as defined in``create_model_input_files.sh``, e.g. ``/work/[...]/sdm-eurec4a-CLEO/data/[...]``. The output is stored in ``/your/path2data/output_v[version_output]-CLEO_v[version_cleo]-input_v[version_netcdf_input]/[choice_microphysics]``, e.g. for:
###### - your output version: v4.4
###### - microphysics: null_microphysics
###### - NETCDF input version from (3):  v4.2
###### - CLEO verion: v0.39.7
###### the output would be stored in: ``/your/path2data/output_v4.4-CLEO_v0.39.7-input_v4.2/null_microphysics``.

---

## Post Processing of CLEOs Raw Output

This is done is the ``sdm-eurec4a`` repo.

### Eulerian Views

We create the eulerian views with the bash script ``./scripts/CLEO/output_processing/eulerian_views.sh``.
The underlying python script is: ``./scripts/CLEO/output_processing/create_eulerian_views_mpi4py.py``

To create a single, combined netcdf file, you can toggle two options in  ``eulerian_views.sh``:
``` bash
create_eulerian_view=true # calls ``./scripts/CLEO/output_processing/create_eulerian_views_mpi4py.py``
concatenate_eulerian_view=true # calls ``./scripts/CLEO/output_processing/concatenate_eulerian_views.py``
```

Each cluster has it's own dataset ``eulerian_dataset.nc`` written in ``/your/path2data/output_[version_output]/[choice_microphysics]/[cluster_XXX]/``,
and the combined dataset is stored in ``/your/path2data/output_[version_output]/[choice_microphysics]/combined/eulerian_dataset_combined.nc``.

### Conservation View

We create the conservation view (Inflow, Outflow, Reservoir change, Evaporation) with the ``./scripts/CLEO/output_processing/conservation_views.sh`` bash script. It invokes a MPI parallel run for all clouds. The underlying python script is: ``./scripts/CLEO/output_processing/create_inflow_outflow_mpi4py.py``

Like for eulerian views, to create a single, combined netcdf file, you can toggle two options in  ``conservation_views.sh``:
``` bash
create_inflow_outflow=true # calls ``./scripts/CLEO/output_processing/create_inflow_outflow_mpi4py.py``
concatenate_inflow_outflow=true # calls ``./scripts/CLEO/output_processing/concatenate_inflow_outflow.py``
```

Each cluster has it's own dataset ``conservation_dataset.nc`` written in ``/your/path2data/output_[version_output]/[choice_microphysics]/[cluster_XXX]/``,
and the combined dataset is stored in ``/your/path2data/output_[version_output]/[choice_microphysics]/combined/conservation_dataset_combined.nc``.

---

# Plot and Use the Conservation and Eulerian Views.

To plot specific figures for paper:
- ``./notebooks/paper/15-NN-compare-thermodynamic-profiles.ipynb`` produces Fig.1 ``compare_observations_cloud_396``
- ``./notebooks/paper/14-NN-paper-figures.ipynb`` plots lots of the paper's figures.
