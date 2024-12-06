# %%
import argparse
import datetime
import logging
import sys
import traceback

from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import Tuple, TypedDict, Union

import lmfit
import numpy as np
import xarray as xr
import yaml


develpoment = True


__epilog__ = """

    NOTE:
    -----
    !!! TODO: add explanation of these steps in the docs !!!
    The IC needs to be created from the CC before executing this script.
    The DR needs to be created from the IC and DS before executing this script.


    IC needs the following variables:
    - cloud_id: the id of the cloud
    - time: the time of the cloud
    - start: the start time of the cloud
    - end: the end time of the cloud
    - alt: the altitude of the cloud
    - vertical_extent: the vertical extent of the cloud
    - duration: the duration of the cloud
    - liquid_water_content: the liquid water content of the cloud

    CC needs the following variables:
    - time: the time of the cloud
    - radius: the radius of the cloud
    - particle_size_distribution: the particle size distribution of the cloud

    DS needs the following variables:
    - time: the time of the dropsonde
    - alt: the altitude of the dropsonde
    - air_temperature: the air temperature of the dropsonde
    - specific_humidity: the specific humidity of the dropsonde
    - potential_temperature: the potential temperature of the dropsonde
    - pressure: the pressure of the dropsonde

    DR needs the following variables:
    - time: the time of the dropsonde
    - time_identified_clouds: the time of the identified cloud as given in the IC
    - index_ds_dropsonde: the index of the dropsonde as given in the DS
    - temporal_distance: the temporal distance between the dropsonde and the identified cloud
    - spatial_distance : the spatial distance between the dropsonde and the identified cloud
"""

__short_doc__ = """
With this script the input yaml files are created to execute CLEO in a EUREC4A1D Rainshaft experiment in a stationary state.

The input paths to the datasets can be relative!
Then the script uses the repository path to find the datasets.
This is done by using ``sdm_eurec4a.RepositoryPath(envciroment)`` where enviroment is by default "nils_levante".

OUTPUT:
-------
- For each cloud in the IC, a yaml is created with the following information regarding the cloud:
    - cloud specific information
    - fit of the particle size distribution of the cloud as bimodal Lognormal distribution
    - fit of the thermodynamic profiles of the dropsondes as linear fits
- The yaml is saved in a sub diretory of the defined output directory ``--output-dir``.
- The sub directory is named as: ``[identification_type]_[cloud_id]``.
"""


# do the argparsing
parser = argparse.ArgumentParser(
    description=__short_doc__, epilog="DETAILS: \n" + __epilog__, formatter_class=RawTextHelpFormatter
)
parser.add_argument(
    "--output-dir",
    "-o",
    type=Path,
    help="path or relative path to output directory",
    default="data/model/input/output_vX.X",
)

parser.add_argument(
    "--identified_cloud_path",
    "-i",
    type=Path,
    help="path or relative path to identified clouds",
    default="data/observation/cloud_composite/processed/identified_clouds/identified_clusters_rain_mask_5.nc",
)
parser.add_argument(
    "--distance_relation_path",
    "-r",
    type=Path,
    help="path or relative path to distance relations",
    default="data/observation/combined/distance_relations/distance_dropsondes_identified_clusters_rain_mask_5.nc",
)
parser.add_argument(
    "--cloud_composite_path",
    "-c",
    type=Path,
    help="path or relative path to cloud composite",
    default="data/observation/cloud_composite/processed/cloud_composite_si_units.nc",
)
parser.add_argument(
    "--drop_sonde_path",
    "-d",
    type=Path,
    help="path or relative path to dropsondes",
    default="data/observation/dropsonde/processed/drop_sondes.nc",
)
parser.add_argument(
    "--identification_type",
    "-t",
    type=str,
    help="type of identification",
    default="clusters",
)
parser.add_argument(
    "--environment",
    "-e",
    type=str,
    help="environment e.g. 'levante'",
    default="nils_levante",
)


if develpoment == True:
    args = parser.parse_args(["--output-dir", "data/model/input/output_test_new/"])
else:
    args = parser.parse_args()

from sdm_eurec4a import RepositoryPath, conversions, get_git_revision_hash
from sdm_eurec4a.identifications import (
    match_clouds_and_cloudcomposite,
    match_clouds_and_dropsondes,
    select_individual_cloud_by_id,
)
from sdm_eurec4a.input_processing import transfer
from sdm_eurec4a.reductions import shape_dim_as_dataarray


repo_path = RepositoryPath("nils_levante").get_repo_dir()
print(repo_path)


version_control = dict(
    git_commit=get_git_revision_hash(),
    date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
)


def optional_prepend_repo_path(path: Path, repo_path: Path = repo_path):
    """Make relative paths absolute by prepending the repo path."""
    if path.is_absolute():
        return path
    else:
        return repo_path / path


# ======================================
# PATHS
# ======================================

identified_clouds_path = optional_prepend_repo_path(
    Path(args.identified_cloud_path), repo_path=repo_path
)
distance_relation_path = optional_prepend_repo_path(
    Path(args.distance_relation_path), repo_path=repo_path
)
cloud_composite_path = optional_prepend_repo_path(Path(args.cloud_composite_path), repo_path=repo_path)
drop_sonde_path = optional_prepend_repo_path(Path(args.drop_sonde_path), repo_path=repo_path)
output_dir = optional_prepend_repo_path(Path(args.output_dir), repo_path=repo_path)
identification_type = args.identification_type

# make sure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)


# ======================================
# LOGGING SETUP
# ======================================
log_dir = output_dir / "logs"
log_file_path = log_dir / "create_input.log"

log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(log_file_path)
handler.setLevel(logging.INFO)

# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(console_handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Execution terminated due to an Exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


sys.excepthook = handle_exception

logging.info("============================================================")
logging.info("Start preprocessing of identified clouds to create input for CLEO")
logging.info("Git hash: %s", get_git_revision_hash())
logging.info("Date: %s", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
logging.info("Output directory: %s", output_dir)
logging.info("============================================================")


# ======================================
# LOAD AND SELECT THE DATASETS
# ======================================

identified_clouds = xr.open_dataset(identified_clouds_path)
distance_relation = xr.open_dataset(distance_relation_path)
cloud_composite = xr.open_dataset(cloud_composite_path)
drop_sondes = xr.open_dataset(drop_sonde_path)

# confine the datasets

# set selection criteria for the identified clouds
identified_clouds = identified_clouds.where(
    (identified_clouds["alt"] <= 1200)
    & (identified_clouds["alt"] >= 500)
    & (identified_clouds["vertical_extent"] <= 150)
    & (identified_clouds["duration"] >= np.timedelta64(3, "s"))
    # & (identified_clouds["liquid_water_content"] / identified_clouds["duration"].dt.seconds >=0.1)
    ,
    drop=True,
)

# set selection criteria for the dropsondes
drop_sondes = drop_sondes.where(drop_sondes.alt <= 1600, drop=True)

# add relative humidity to the dataset
drop_sondes["relative_humidity"] = conversions.relative_humidity_from_tps(
    temperature=drop_sondes["air_temperature"],
    pressure=drop_sondes["pressure"],
    specific_humidity=drop_sondes["specific_humidity"],
)


# ======================================
# FIT FUNCTION
# ======================================


def select_subdatasets_from_cloud_id(
    chosen_id: int,
    identified_clouds: xr.Dataset,
    cloud_composite: xr.Dataset,
    drop_sondes: xr.Dataset,
    distance_relation: xr.Dataset,
    identification_type: str,
    max_spatial_distance: int = 100,
    max_temporal_distance: np.timedelta64 = np.timedelta64(3, "h"),
    particle_split_radius: float = 45e-6,  # 45 micrometer
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, dict]:
    """
    Selects subdatasets related to a specific cloud ID.

    Parameters
    ----------
    chosen_id : int
        The ID of the cloud to select subdatasets for.
    identified_clouds : xr.Dataset
        Dataset containing identified clouds.
    cloud_composite : xr.Dataset
        Dataset containing cloud composite data.
    drop_sondes : xr.Dataset
        Dataset containing drop sondes data.
    distance_relation : xr.Dataset
        Dataset containing distance relation data.
    identification_type : str
        Type of cloud identification.
    max_spatial_distance : int, optional
        Maximum spatial distance for matching clouds and dropsondes. Defaults to 100.
    max_temporal_distance : np.timedelta64, optional
        Maximum temporal distance for matching clouds and dropsondes. Defaults to np.timedelta64(3, "h").
    particle_split_radius : float, optional
        Particle split radius. Defaults to 45e-6.

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset, xr.Dataset, dict]
        A tuple containing the selected subdatasets and cloud information.

    Raises
    ------
    ValueError
        If no cloudcomposite data is found for the cloud.
    """

    cloud_information = dict(
        cloud_id=chosen_id,
        split_radius=particle_split_radius,
        dropsonde_distance=dict(
            max_spatial_distance=max_spatial_distance,
            max_temporal_distance=str(max_temporal_distance),
        ),
        identification_type=identification_type,
    )

    # select a single cloud
    ds_cloud = select_individual_cloud_by_id(identified_clouds, chosen_id)

    cloud_information["altitude"] = ds_cloud["alt"].data
    cloud_information["duration"] = ds_cloud["duration"].dt.seconds.data
    cloud_information["time"] = ds_cloud["time"].dt.strftime("%Y-%m-%d %H:%M:%S").data

    ds_cloudcomposite_with_zeros = match_clouds_and_cloudcomposite(
        ds_clouds=ds_cloud,
        ds_cloudcomposite=cloud_composite,
    )

    ds_dropsondes = match_clouds_and_dropsondes(
        ds_clouds=ds_cloud,
        ds_sonde=drop_sondes,
        ds_distance=distance_relation,
        max_spatial_distance=max_spatial_distance,
        max_temporal_distance=max_temporal_distance,
    )

    # remove 0s
    ds_cloudcomposite = ds_cloudcomposite_with_zeros.where(
        ds_cloudcomposite_with_zeros["particle_size_distribution"] != 0
    )

    # Make sure the data is not empty
    logging.info(f"Number of dropsondes: {ds_dropsondes['time'].shape}")
    logging.info(f"Number of cloudcomposite: {ds_cloudcomposite['time'].shape}")

    if ds_cloudcomposite["time"].shape[0] == 0:
        raise ValueError("No cloudcomposite data found for cloud")
    if ds_dropsondes["time"].shape[0] == 0:
        raise ValueError("No dropsonde data found for cloud")

    return ds_cloud, ds_cloudcomposite, ds_dropsondes, cloud_information


def fit_particle_size_distribution(
    ds_cloudcomposite: xr.Dataset,
    particle_split_radius: float = 45e-6,  # 45 micrometer
) -> transfer.PSD_LnNormal:
    """
    Fits the particle size distribution (PSD) of cloud and rain droplets idependently.

    Note
    ----
    The PSD is fitted with a bimodal Lognormal distribution.
    For the cloud droplets, the PSD is fitted with
    - geometric mean between 0.1 micrometer and the split radius.
    - geometric sigma between 0 and 1.7.
    For the rain droplets, the PSD is fitted with
    - geometric mean within the range of radius values provided.

    Parameters
    ----------
    ds_cloudcomposite : xr.Dataset
        Dataset containing the cloud composite data.
    particle_split_radius : float, optional
        The radius at which to split the data into cloud and rain droplets. Default is 45 micrometers.

    Returns
    -------
    psd_fit : transfer.PSD_LnNormal
        The fitted particle size distribution.
    """

    # Split data into cloud and rain
    ds_small_droplets = ds_cloudcomposite.sel(radius=slice(None, particle_split_radius))
    ds_rain_droplets = ds_cloudcomposite.sel(radius=slice(particle_split_radius, None))

    # ======================================
    # Fit the PSDs
    # ======================================

    # Use the PSD_LnNormal model
    psd_rain_fit = transfer.PSD_LnNormal()
    psd_cloud_fit = transfer.PSD_LnNormal()

    # ---------
    # Rain
    # ---------
    data = ds_rain_droplets["particle_size_distribution"]
    radi2d = shape_dim_as_dataarray(da=data, output_dim="radius")
    psd_model = psd_rain_fit.get_model()

    # update geometric mean to be within range of the data
    psd_rain_fit.update_individual_model_parameters(
        lmfit.Parameter(
            name="geometric_means",
            min=data["radius"].min().data,
            max=data["radius"].max().data,
        )
    )

    # fit model parameters and update them
    model_result = psd_model.fit(
        data=data.data, radii=radi2d.data, params=psd_rain_fit.get_model_parameters(), nan_policy="omit"
    )
    psd_rain_fit.lmfitParameterValues_to_dict(model_result.params)

    # ---------
    # Small cloud and drizzle
    # ---------
    # For this, the parameters need to be updated

    # update geometric mean to be within range of 0.1 micrometer and the split radius
    psd_cloud_fit.update_individual_model_parameters(
        lmfit.Parameter(
            name="geometric_means",
            value=1e-5,
            min=0.1e-6,  # at least 0.1 micrometer
            max=particle_split_radius,  # at most the split radius (default 45 micrometer)
        )
    )
    # update geometric sigma to be within range of 0 and 1.7.
    # NOTE: No real physical meaning, but it is a good range for the fit
    psd_cloud_fit.update_individual_model_parameters(
        lmfit.Parameter(
            name="geometric_sigmas",
            value=1.1,
            min=0,
            max=1.7,
        )
    )

    data = ds_small_droplets["particle_size_distribution"]
    radi2d = shape_dim_as_dataarray(da=data, output_dim="radius")
    psd_model = psd_cloud_fit.get_model()

    # fit model parameters and update them
    model_result = psd_model.fit(
        data=data.data, radii=radi2d.data, params=psd_cloud_fit.get_model_parameters(), nan_policy="omit"
    )
    psd_cloud_fit.lmfitParameterValues_to_dict(model_result.params)

    # --------
    # Combine the fits
    # --------

    psd_fit = psd_rain_fit + psd_cloud_fit

    return psd_fit


thermo_fit = dict(
    air_temperature=transfer.ThermodynamicLinear(),
    potential_temperature=transfer.ThermodynamicLinear(),
    specific_humidity=transfer.ThermodynamicLinear(),
    relative_humidity=transfer.ThermodynamicLinear(),
    pressure=transfer.ThermodynamicLinear(),
)
for var in thermo_fit:
    logging.info(f"Fitting {var}")
    thermo_fit[var] = transfer.fit_thermodynamics(
        da_thermo=ds_dropsondes[var].sel(alt=slice(200, 500)),
        thermo_fit=thermo_fit[var],
        dim="alt",
        x_split=None,
        f0_boundaries=False,
    )


def fit_thermodynamics_default(
    ds_dropsondes: xr.Dataset,
) -> dict:
    thermo_dict = dict(
        air_temperature=transfer.ThermodynamicSplitLinear(),
        specific_humidity=transfer.ThermodynamicSplitLinear(),
        potential_temperature=transfer.ThermodynamicSplitLinear(),
        relative_humidity=transfer.ThermodynamicSplitLinear(),
    )
    for var in thermo_dict:
        logging.info(f"Fitting {var}")
        thermo_dict[var] = transfer.fit_thermodynamics(
            da_thermo=ds_dropsondes[var],
            thermo_fit=thermo_dict[var],
            dim="alt",
            x_split=None,
        )

    mean_x_split = float(np.mean([thermo_dict[key].get_x_split() for key in thermo_dict]))
    logging.info(f"Split level: {mean_x_split} m")

    # Fit the ThermodynamicLinear models
    thermo_dict = dict(
        air_temperature=transfer.ThermodynamicSplitLinear(),
        specific_humidity=transfer.ThermodynamicSplitLinear(),
        potential_temperature=transfer.ThermodynamicSplitLinear(),
        relative_humidity=transfer.ThermodynamicSplitLinear(),
        pressure=transfer.ThermodynamicLinear(),
    )
    for var in thermo_dict:
        logging.info(f"Fitting {var}")
        thermo_dict[var] = transfer.fit_thermodynamics(
            da_thermo=ds_dropsondes[var],
            thermo_fit=thermo_dict[var],
            dim="alt",
            x_split=mean_x_split,
        )

    return thermo_dict


def fit_thermodynamics_linear(
    ds_dropsondes: xr.Dataset,
) -> dict:
    thermo_dict = dict(
        air_temperature=transfer.ThermodynamicSplitLinear(),
        specific_humidity=transfer.ThermodynamicSplitLinear(),
    )
    for var in thermo_dict:
        logging.info(f"Fitting {var}")
        thermo_dict[var] = transfer.fit_thermodynamics(
            da_thermo=ds_dropsondes[var],
            thermo_fit=thermo_dict[var],
            dim="alt",
            x_split=None,
        )
    return thermo_dict


# ======================================
# FIT FUNCTION
# ======================================


def save_config_yaml(
    particle_size_distribution_fit: transfer.PSD_LnNormal,
    thermodynamic_fits: dict,
    cloud_information: dict,
    version_control: dict,
    output_dir: Path,
    identification_type: str,
    chosen_id: int,
):
    # Add the representers
    add_representer()

    output_dictonary = dict(
        particle_size_distribution=particle_size_distribution_fit,
        thermodynamics=thermodynamic_fits,
        cloud=cloud_information,
        version_control=version_control,
    )

    # write config to yaml file
    output_filepath = output_dir / f"{identification_type}_{chosen_id}.yaml"
    with open(output_filepath, "w") as file:
        s = yaml.dump(output_dictonary)
        file.write(s)


def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    """
    Converts a numpy ndarray to a list and represents it for YAML dumping.

    Parameters:
    dumper (yaml.Dumper): The YAML dumper instance.
    array (np.ndarray): The numpy ndarray to represent.

    Returns:
    yaml.Node: The represented list.
    """
    return dumper.represent_list(array.tolist())


def float64_representer(dumper: yaml.Dumper, array: np.float64) -> yaml.Node:
    """
    Represents a numpy float64 for YAML dumping.

    Parameters:
    dumper (yaml.Dumper): The YAML dumper instance.
    array (np.float64): The numpy float64 to represent.

    Returns:
    yaml.Node: The represented float.
    """
    return dumper.represent_float(array)


def thermofit_split_linear_representer(
    dumper: yaml.Dumper, obj: transfer.ThermodynamicSplitLinear
) -> yaml.Node:
    """
    Represents a ThermodynamicSplitLinear object for YAML dumping.

    Parameters:
    dumper (yaml.Dumper): The YAML dumper instance.
    obj (transfer.ThermodynamicSplitLinear): The ThermodynamicSplitLinear object to represent.

    Returns:
    yaml.Node: The represented dictionary of parameters.
    """
    parameters = {"slopes": obj.get_slopes()[0]}
    parameters.update(obj.get_parameters())

    data = {
        "parameters": parameters,
        "type": obj.type,
    }
    return dumper.represent_dict(data)


def thermofit_linear_representer(dumper: yaml.Dumper, obj: transfer.ThermodynamicLinear) -> yaml.Node:
    """
    Represents a ThermodynamicSplitLinear object for YAML dumping.

    Parameters:
    dumper (yaml.Dumper): The YAML dumper instance.
    obj (transfer.ThermodynamicSplitLinear): The ThermodynamicSplitLinear object to represent.

    Returns:
    yaml.Node: The represented dictionary of parameters.
    """
    parameters = obj.get_parameters()

    data = {
        "parameters": parameters,
        "type": obj.type,
    }
    return dumper.represent_dict(data)


def psdfit_representer(dumper: yaml.Dumper, obj: transfer.PSD_LnNormal) -> yaml.Node:
    """
    Represents a PSD_LnNormal object for YAML dumping.

    Parameters:
    dumper (yaml.Dumper): The YAML dumper instance.
    obj (transfer.PSD_LnNormal): The PSD_LnNormal object to represent.

    Returns:
    yaml.Node: The represented dictionary of parameters.
    """
    data = {
        "parameters": obj.get_parameters(),
        "type": obj.type,
    }
    return dumper.represent_dict(data)


def add_representer() -> None:
    """Adds custom representers for numpy ndarrays, numpy float64s,
    ThermodynamicSplitLinear objects, and PSD_LnNormal objects to the YAML dumper."""
    yaml.add_representer(np.ndarray, ndarray_representer)
    yaml.add_representer(np.float64, float64_representer)
    yaml.add_representer(transfer.ThermodynamicSplitLinear, thermofit_split_linear_representer)
    yaml.add_representer(transfer.ThermodynamicLinear, thermofit_linear_representer)
    yaml.add_representer(transfer.PSD_LnNormal, psdfit_representer)


# %%

if __name__ == "__main__":
    for cloud_id in identified_clouds.cloud_id:
        cloud_id = int(cloud_id)

        # select individual datasets

        try:
            logger.info(f"Cloud {cloud_id} - select sub datasets")

            (
                ds_cloud,
                ds_cloudcomposite,
                ds_dropsondes,
                cloud_information,
            ) = select_subdatasets_from_cloud_id(
                chosen_id=cloud_id,
                identified_clouds=identified_clouds,
                cloud_composite=cloud_composite,
                drop_sondes=drop_sondes,
                distance_relation=distance_relation,
                identification_type=identification_type,
            )

            # fit the particle size distribution
            logger.info(f"Cloud {id} - fit particle size distribution")
            particle_size_distribution_fit = fit_particle_size_distribution(
                ds_cloudcomposite=ds_cloudcomposite,
                particle_split_radius=cloud_information["split_radius"],
            )

            # fit the thermodynamics
            logger.info(f"Cloud {cloud_id} - fit thermodynamics")
            thermodynamic_fits = fit_thermodynamics_default(ds_dropsondes=ds_dropsondes)

            # fit the linear thermodynamics for subcloud layer only

            # save the config to yaml
            logger.info(f"Cloud {id} - save config to yaml")
            save_config_yaml(
                particle_size_distribution_fit=particle_size_distribution_fit,
                thermodynamic_fits=thermodynamic_fits,
                cloud_information=cloud_information,
                version_control=version_control,
                output_dir=output_dir,
                identification_type=identification_type,
                chosen_id=cloud_id,
            )

            logger.info(f"Created input for cloud {cloud_id}")
        except Exception as e:
            message = traceback.format_exception(None, e, e.__traceback__)
            logger.error(
                f"Failed to create input for cloud {cloud_id}: " + repr(e) + "\n" + "".join(message)
            )
            continue
