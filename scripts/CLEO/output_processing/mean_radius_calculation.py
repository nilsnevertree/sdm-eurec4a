# %%

import xarray as xr
import numpy as np
import logging
from pathlib import Path
import sys
import argparse

from typing import Tuple

from sdm_eurec4a import slurm_cluster as scluster
from sdm_eurec4a import RepositoryPath
from sdm_eurec4a.constants import TimeSlices
import datetime
import secrets

RP = RepositoryPath("levante")

repo_dir = RP.repo_dir
sdm_data_dir = RP.data_dir

log_dir = repo_dir / "logs" / "mean_radius_calculation"
log_dir.mkdir(parents=False, exist_ok=True)

time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
random_hex = secrets.token_hex(4)

log_dir.mkdir(exist_ok=True, parents=True)
log_file_path = log_dir / f"{time_str}-{random_hex}.log"


# === logging ===
# create log file


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(log_file_path)
handler.setLevel(logging.INFO)

# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

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


logging.info(f"====================")


def add_mean_and_stddev_radius(da: xr.DataArray, radius_name: str) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    This function calculates the mean and standard deviation of the radius for each gridbox and time.
    It is weighted by the attribute with ``attribute_name``.

    Parameters
    ----------
    xr.DataArray
        A DataArray with the attribute that shall be used for weighting the radius.
        Needs to contain the radius as a coordinate.
    radius_name : str
        The name of the radius coordinate in the DataArray.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        A tuple containing the mean and standard deviation of the radius.
    """

    mass_mean = da.mean("time", keep_attrs=True).compute()
    probability = mass_mean / mass_mean.sum(radius_name, keep_attrs=True)
    probability = probability.fillna(0)

    # mean radius is then the sum of the radius weighted by the probability
    # m = sum_i (r_i * p_i)
    mean_radius = (da[radius_name] * probability).sum(radius_name, keep_attrs=True)
    # standard deviation is then the square root of the sum of the squared difference between the radius and the mean radius weighted by the probability
    # s = sqrt(sum_i ( (r_i - m)^2 * p_i))
    std_dev_radius: xr.DataArray = ((probability * (da[radius_name] - mean_radius) ** 2) ** (0.5)).sum(
        radius_name, keep_attrs=True
    )

    units = da[radius_name].attrs.get("units", "")
    attribute_name = da.name

    mean_radius.attrs.update(
        long_name="Mean radius",
        description=f"Mean radius weighted by {attribute_name}.",
        units=units,
    )
    std_dev_radius.attrs.update(
        long_name="Standard deviation of radius",
        description=f"Standard deviation radius weighted by {attribute_name}.",
        units=units,
    )

    return mean_radius, std_dev_radius


# %%
client, cluster = scluster.init_dask_slurm_cluster(
    scale=1, processes=16, walltime="00:30:00", memory="16GiB"
)

# %%
microphysics = (
    "null_microphysics",
    "condensation",
    "collision_condensation",
    "coalbure_condensation_small",
    "coalbure_condensation_large",
)
# %%
parser = argparse.ArgumentParser(
    description="Create eulerian view for data_dir which contains all subfolders of cloud data it should contain the subfolder cluster_*/processed/eulerian_dataset.nc"
)

# Add arguments
parser.add_argument("-d", "--data_dir", type=str, help="Path to data directory", required=True)
# Parse arguments
args = parser.parse_args()

data_dir = Path(args.data_dir)

eulerian_data_path = lambda microphysics: data_dir / Path(
    f"{microphysics}/combined/eulerian_dataset_combined.nc"
)
mean_radius_data_path = lambda microphysics: data_dir / Path(
    f"{microphysics}/combined/mean_radius_combined.nc"
)
chunks = dict(
    cloud_id=2,
)
time_slice = TimeSlices.quasi_stationary_state


radius_split = 200  # um

for microphysic in microphysics:
    logging.info("-----------")
    logging.info(f"Processing {microphysic}")
    ds_input = xr.open_dataset(eulerian_data_path(microphysic), chunks=chunks)
    # It is imporant to omit using aerosols, which we define as droplets below 10 um
    ds_input = ds_input.sel(radius_bins=slice(1, None))
    ds_input["radius_bins"].attrs["units"] = "µm"
    # Calculate the mean and standard deviation of the radius

    ds = xr.Dataset()

    # ====================
    # xi
    # ====================
    logging.info("Multiplicities")
    try:
        m, s = add_mean_and_stddev_radius(
            da=ds_input["xi"].sel(time=time_slice), radius_name="radius_bins"
        )
        ds["xi_radius_mean"] = m
        ds["xi_radius_std"] = s
    except Exception as e:
        logging.error(
            f"Error in calculating the mean and standard deviation of the radius for {microphysic}."
        )
        logging.error(e)

    # Weight by Multiplicity for small droplets
    logging.info("Multiplicities small")
    try:
        m, s = add_mean_and_stddev_radius(
            da=ds_input["xi"].sel(time=time_slice).sel(radius_bins=slice(0, radius_split)),
            radius_name="radius_bins",
        )
        m.attrs["description"] += "For small droplets (radius < 200 um)"
        s.attrs["description"] += "For small droplets (radius < 200 um)"
        ds["small_xi_radius_mean"] = m
        ds["small_xi_radius_std"] = s
    except Exception as e:
        logging.error(
            f"Error in calculating the mean and standard deviation of the radius for {microphysic}."
        )
        logging.error(e)

    # ====================
    # MASS
    # ====================
    logging.info("Mass")
    try:
        m, s = add_mean_and_stddev_radius(
            da=ds_input["mass_represented"].sel(time=time_slice), radius_name="radius_bins"
        )
        ds["mass_radius_mean"] = m
        ds["mass_radius_std"] = s
    except Exception as e:
        logging.error(
            f"Error in calculating the mean and standard deviation of the radius for {microphysic}."
        )
        logging.error(e)

    logging.info("Mass small")
    try:
        # Weight by Mass for small droplets
        m, s = add_mean_and_stddev_radius(
            da=ds_input["mass_represented"].sel(time=time_slice).sel(radius_bins=slice(0, radius_split)),
            radius_name="radius_bins",
        )
        m.attrs["description"] += "For small droplets (radius < 200 um)"
        s.attrs["description"] += "For small droplets (radius < 200 um)"
        ds["small_mass_radius_mean"] = m
        ds["small_mass_radius_std"] = s
    except Exception as e:
        logging.error(
            f"Error in calculating the mean and standard deviation of the radius for {microphysic}."
        )
        logging.error(e)

    # ====================
    # STORE
    # ====================

    ds.to_netcdf(mean_radius_data_path(microphysic))
# %%
