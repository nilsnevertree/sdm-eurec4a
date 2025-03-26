# %%

import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

# import dask.distributed

from sdm_eurec4a import RepositoryPath
from sdm_eurec4a.constants import TimeSlices

import datetime

import secrets


RP = RepositoryPath("levante")
repo_dir = RP.repo_dir

# %%

# === logging ===
# create log file

time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

random_hex = secrets.token_hex(4)

log_file_dir = repo_dir / "logs" / f"concatenate_eulerian_views/{time_str}-{random_hex}"
log_file_dir.mkdir(exist_ok=True, parents=True)
log_file_path = log_file_dir / "main.log"

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


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Execution terminated due to an Exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


# %%
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

logging.info(f"====================")

# Ensure we only start the client once

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(console_handler)

parser = argparse.ArgumentParser(
    description="Create eulerian view for data_dir which contains all subfolders of cloud data it should contain the subfolder cluster_*/processed/eulerian_dataset.nc"
)

# Add arguments
parser.add_argument("-d", "--data_dir", type=str, help="Path to data directory", required=True)
# Parse arguments
args = parser.parse_args()

master_data_dir = Path(args.data_dir)

output_dir = master_data_dir / "combined"
output_dir.mkdir(exist_ok=True, parents=False)
output_file_path = output_dir / "eulerian_dataset_combined.nc"

relative_path_to_eulerian_dataset = Path("processed/eulerian_dataset.nc")
pattern = "cluster_*/"

logging.info(f"Enviroment: {sys.prefix}")
logging.info("Concatenate eulerian views in:")
logging.info(master_data_dir)
logging.info(f"Subfolder pattern: {pattern}")
logging.info(f"Relative path to eulerian dataset: {relative_path_to_eulerian_dataset}")
logging.info(f"Save the combined eulerian view to: {output_file_path}")

data_dir_list = np.array(sorted(list(master_data_dir.glob(pattern))))
eulerian_dataset_path_list = data_dir_list / relative_path_to_eulerian_dataset

# sublist_data_dirs = np.array_split(np.array(data_dir_list), number_ranks)[rank]
# sublist_eulerian_dataset_paths = np.array_split(np.array(eulerian_dataset_path_list), number_ranks)[rank]
# total_npro = len(sublist_data_dirs)

sucessful = []

max_gridbox_list = []
file_path_list = []
cloud_id_list = []

for step, (data_dir, eulerian_dataset_path) in enumerate(
    zip(
        data_dir_list,
        eulerian_dataset_path_list,
    )
):
    # logging.info(f"Rank {rank+1} {step+1}/{total_npro}")
    cloud_id = int(data_dir.name.split("_")[1])
    logging.info(f"Start {cloud_id}")
    try:
        logging.info(f"Open {eulerian_dataset_path}")
        euler_dataset = xr.open_dataset(eulerian_dataset_path)

        logging.info(f"Find max gridbox")
        max_gridbox_list.append(euler_dataset["gridbox"].max())

        logging.info(f"Add cloud to combination list")
        file_path_list.append(eulerian_dataset_path)
        cloud_id_list.append(cloud_id)

        logging.info(f"Processed {cloud_id}")

    except FileNotFoundError as e:
        logger.error(e)

number_sucessful = len(cloud_id_list)
number_total = len(data_dir_list)

# %%
logging.info(f"All processes finished with {number_sucessful}/{number_total} sucessful")
logging.info(f"Sucessful clouds are: {list(cloud_id_list)}")


# %%

logging.info("Attempt to open all sucessful clouds and combine them with xr.open_mfdataset")


# create the concatenation index
cloud_id_index = pd.Index(cloud_id_list, name="cloud_id")
ds = xr.open_mfdataset(file_path_list, combine="nested", concat_dim=[cloud_id_index], chunks={})
logging.info("Add cloud_id and max_gridbox to the dataset")
ds["cloud_id"].attrs.update(dict(long_name="Cloud identification number", units=""))
max_gridbox = xr.concat(max_gridbox_list, dim=cloud_id_index)
ds["max_gridbox"] = max_gridbox.compute()
ds["max_gridbox"].attrs.update(
    dict(
        long_name="Maximum gridbox value for each cloud. Above this value, the cloud has no data.",
        units="",
    )
)

logging.info("Dimension radius_bins: replace -inf values with 0.1µm")
ds["radius_bins"] = ds["radius_bins"].where(ds["radius_bins"] > 0, 0.1)

ds_4d = ds[["radius", "xi", "mass_represented"]]

# do not use the 4D arrays to save memory
drop_vars = [
    "radius",
    "xi",
    "number_superdroplets",
    "mass_represented",
    "mass_left",
    "number_superdroplets_left",
    "mass_difference",
]
ds = ds.drop_vars(drop_vars)


# %%


# WORK ON THE 4D ARRAYS
logging.info("WORK ON THE 4D ARRAYS")

logging.info("Add mean radius and standard deviation")

time_slice = TimeSlices.quasi_stationary_state

radius_split = 200  # um

# ====================
# xi
# ====================
logging.info("Radius - Multiplicities")
try:
    m, s = add_mean_and_stddev_radius(
        da=ds_4d["xi"].sel(time=time_slice).load(), radius_name="radius_bins"
    )
    ds["xi_radius_mean"] = m
    ds["xi_radius_std"] = s
except Exception as e:
    logging.error(f"Error in calculating the mean and standard deviation of the radius.")
    logging.error(e)

# # Weight by Multiplicity for small droplets
# logging.info("Radius - Multiplicities small")
# try:
#     m, s = add_mean_and_stddev_radius(
#         da=ds_4d["xi"].sel(time=time_slice).sel(radius_bins=slice(0, radius_split)).load(),
#         radius_name="radius_bins",
#     )
#     m.attrs["description"] += "For small droplets (radius < 200 um)"
#     s.attrs["description"] += "For small droplets (radius < 200 um)"
#     ds["small_xi_radius_mean"] = m
#     ds["small_xi_radius_std"] = s
# except Exception as e:
#     logging.error(f"Error in calculating the mean and standard deviation of the radius.")
#     logging.error(e)

# ====================
# MASS
# ====================
logging.info("Radius - Mass")
try:
    m, s = add_mean_and_stddev_radius(
        da=ds_4d["mass_represented"].sel(time=time_slice).load(), radius_name="radius_bins"
    )
    ds["mass_radius_mean"] = m
    ds["mass_radius_std"] = s
except Exception as e:
    logging.error(f"Error in calculating the mean and standard deviation of the radius.")
    logging.error(e)

# logging.info("Radius - Mass small")
# try:
#     # Weight by Mass for small droplets
#     m, s = add_mean_and_stddev_radius(
#         da=ds_4d["mass_represented"].sel(time=time_slice).sel(radius_bins=slice(0, radius_split)).load(),
#         radius_name="radius_bins",
#     )
#     m.attrs["description"] += "For small droplets (radius < 200 um)"
#     s.attrs["description"] += "For small droplets (radius < 200 um)"
#     ds["small_mass_radius_mean"] = m
#     ds["small_mass_radius_std"] = s
# except Exception as e:
#     logging.error(f"Error in calculating the mean and standard deviation of the radius.")
#     logging.error(e)


# Add mean xi and mass distributions

ds["xi_temporal_mean"] = ds_4d["xi"].sel(time=time_slice).mean("time", keep_attrs=True).compute()
ds["mass_represented_temporal_mean"] = (
    ds_4d["mass_represented"].sel(time=time_slice).mean("time", keep_attrs=True).compute()
)

# %%

logging.info("Add domain masks")
# create domain mask and sub cloud layer mask
ds["domain_mask"] = ds["gridbox"] <= ds["max_gridbox"]
ds["domain_mask"].compute()
ds["domain_mask"].attrs.update(
    long_name="Domain Mask",
    description="Boolean mask indicating valid gridbox in the domain for each cloud",
    units="1",
)

ds["cloud_layer_mask"] = ds["gridbox"] == ds["max_gridbox"]
ds["cloud_layer_mask"].compute()
ds["cloud_layer_mask"].attrs.update(
    long_name="Cloud Layer Mask",
    description="Boolean mask indicating if the gridbox is part of the cloud layer",
    units="1",
)

ds["sub_cloud_layer_mask"] = ds["gridbox"] < ds["max_gridbox"]
ds["sub_cloud_layer_mask"] = ds["sub_cloud_layer_mask"].compute()
ds["sub_cloud_layer_mask"].attrs.update(
    long_name="Sub Cloud Layer Mask",
    description="Boolean mask indicating if the gridbox is part of the sub cloud layer",
    units="1",
)

# %%
ds = ds.sortby("cloud_id")
ds.attrs.update(
    dict(
        description="Combined eulerian view for all clouds.",
        author="Nils Niebaum",
        date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
)


# %%
logging.info(f"Attempt to save combined dataset to: {output_file_path}")
ds.to_netcdf(output_file_path, compute=True)
logging.info(f"Closing dataset")
logging.info(f"Done")
