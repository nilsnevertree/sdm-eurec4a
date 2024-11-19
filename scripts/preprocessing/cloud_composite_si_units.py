# %%
"""
Script to pre-process cloud composite data from Coutris, P. (2021).  SAFIRE
ATR42: PMA/Cloud composite dataset.  [Dataset].  Aeris.
https://doi.org/10.25326/237 which is stored locally in
../data/observation/cloud_composite/raw The pre-processing includes:

    - rename variables to longer names
    - convert time to datetime object
    - creates radius variable and uses it as coordinate for size bins
    - modify attributes
    - save the produced dataset to netcdf file
The produced dataset is stored in ../data/observation/cloud_composite/processed
"""

import datetime
import logging
import sys

from pathlib import Path

import cftime
import dask
import numpy as np
import pandas as pd
import xarray as xr

from sdm_eurec4a import get_git_revision_hash
from sdm_eurec4a.conversions import msd_from_psd_dataarray  # , lwc_from_psd, msd_from_psd, vsd_from_psd
from sdm_eurec4a.reductions import validate_datasets_same_attrs

from sdm_eurec4a import RepositoryPath

# %%

USER_CONSENT_NEEDED = False

REPO_PATH = Path(__file__).resolve().parent.parent.parent
# REPO_PATH = RepositoryPath("levante").repo_dir
# REPO_PATH = RepositoryPath("nils_private").repo_dir

ORIGIN_DIRECTORY = REPO_PATH / Path("data/observation/cloud_composite/raw")
DESTINATION_DIRECTORY = REPO_PATH / Path("data/observation/cloud_composite/processed")
DESTINATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
DESTINATION_FILENAME = "cloud_composite_SI_units_20241025.nc"
DESTINATION_FILEPATH = DESTINATION_DIRECTORY / DESTINATION_FILENAME

log_file_path = DESTINATION_DIRECTORY / "cloud_composite_preprocessing.log"

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
logging.info("Start cloud composite pre-processing")
logging.info("Git hash: %s", get_git_revision_hash())
logging.info("Origin directory: %s", ORIGIN_DIRECTORY.relative_to(REPO_PATH))
logging.info("Destination directory: %s", DESTINATION_DIRECTORY.relative_to(REPO_PATH))
logging.info("Destination filename: %s", DESTINATION_FILENAME)

if USER_CONSENT_NEEDED == True:
    print(f"Save the produced dataset to netcdf file?\n{DESTINATION_FILEPATH}")
    user_input = input("Do you want to continue running the script? (y/n): ")
    if user_input.lower() == "y":
        print("Saving dataset\nPlease wait...")
    else:
        logging.error("User denied proceeding with saving the dataset")
        raise KeyboardInterrupt


def add_flight_number(ds: xr.Dataset) -> xr.Dataset:
    """
    Add flight number to dataset based on filename ending. This function is used as a
    pre-processing step when opening the files of the cloud composite dataset.

    Coutris, P. (2021).  SAFIRE
    ATR42: PMA/Cloud composite dataset.  [Dataset].  Aeris.
    https://doi.org/10.25326/237

    Args:
        ds (xr.Dataset): dataset to add flight number to

    Returns:
        xr.Dataset: dataset with flight number added

    Example flight number : 2
    Example filename ending : EUREC4A_ATR_PMA_Composite-CDP-2DS_20200125_F02_v1.nc
    """
    filename = ds.encoding["source"]
    flight_number = int(str(filename.split("_")[-2]).replace("F", ""))
    ds["flight_number"] = flight_number
    ds["flight_id"] = ds.attrs["flight_id"]
    return ds


files = sorted(ORIGIN_DIRECTORY.glob("*.nc"))
if len(files) == 0:
    logging.error("No files found in %s", ORIGIN_DIRECTORY)
    raise FileNotFoundError("No files found in %s", ORIGIN_DIRECTORY)
else:
    logging.info("Number of files to combine: %s", len(files))

# --- Validate that all datasets have the same attributes except for fligth_id and creation_date ---
datasets_list = []
for file in files:
    datasets_list.append(xr.open_dataset(file))
try:
    validate_datasets_same_attrs(datasets_list, skip_attrs=["flight_id", "creation_date"])
    logging.info("All datasets have the same attributes except for flight_id and creation_date")
except AssertionError:
    logging.error("Not all datasets have the same attributes")
    raise AssertionError("Not all datasets have the same attributes")


# --- Load datasets into chunks ---
# Also make sure to add the flight number as a new variable
try:
    datas = xr.open_mfdataset(
        paths=files,
        combine="by_coords",
        preprocess=add_flight_number,
        chunks="auto",
        parallel=True,
    )
except Exception as e:
    logging.exception("Error while opening files")
    raise e

# --- Reorganize dataset ---

logging.info("Rename variables")
# Use longer names for variable to make it more readable
# do not rename the dimension time and size
VARNAME_MAPPING = {
    "lon": "longitude",
    "lat": "latitude",
    "alt": "altitude",
    "PSD": "particle_size_distribution",
    "MSD": "mass_size_distribution",
    "LWC": "liquid_water_content",
    "NT": "total_concentration",
    "MVD": "median_volume_diameter",
    "M6": "radar_reflectivity_factor",
    "diameter": "diameter",
    "bw": "bin_width",
    "compo_index": "composition_index",
    "CDP_flag": "CDP_flag",
    "2DS_flag": "2DS_flag",
    "CLOUD_mask": "cloud_mask",
    "DZ_mask": "drizzle_mask",
    "RA_mask": "rain_mask",
    "flight_number": "flight_number",
}
datas = datas.rename(VARNAME_MAPPING)

# add long names to liquid water content, diameter and median volume diameter and bin_width

logging.info("Add long names to variables")
long_name_mapping = {
    "liquid_water_content": "Liquid water content",
    "diameter": "Diameter",
    "median_volume_diameter": "Median volume diameter",
    "bin_width": "Bin width",
}
for key, value in long_name_mapping.items():
    datas[key].attrs.update(long_name=value)

logging.info("Use latex compatible unit for variables")
unit_mapping = {
    "liquid_water_content": "g m^{-3}",
}
for key, value in unit_mapping.items():
    datas[key].attrs.update(unit=value)


# Convert time to datetime object
# Note, that the time is in seconds since 2020-01-01 00:00:00
logging.info("Convert UTC time to datetime object")
attrs = datas["time"].attrs
datas["time"] = cftime.num2date(
    datas["time"], units="seconds since 2020-01-01 00:00:00", calendar="standard"
)
# make sure to use the more simple datetime object
datas["time"] = datas["time"].indexes["time"].to_datetimeindex()
attrs.update(
    long_name="Time",
    unit="UTC",
    calender="standard",
    comment="UTC time. Use cftime.num2date to convert to datetime nanoseconds object from seconds since 2020-01-01 00:00:00.",
)
datas["time"].attrs.update(attrs)

logging.info("Validate that diameter and bin_width do not vary along time axis")
assert np.all(datas.diameter == datas.diameter.isel(time=0))
assert np.all(datas.bin_width == datas.bin_width.isel(time=0))

logging.info("Convert diameter and bin_width to meters")
# Convert from µm to m -> 1e-6
# Diameter
datas["diameter"] = datas["diameter"].mean("time", keep_attrs=True)
attrs = datas["diameter"].attrs
datas["diameter"] = datas["diameter"] * 1e-6
attrs.update(
    long_name="Diameter",
    unit="m",
)
datas["diameter"].attrs.update(attrs)

# Bin width
datas["bin_width"] = datas["bin_width"].mean("time", keep_attrs=True)
attrs = datas["bin_width"].attrs
datas["bin_width"] = datas["bin_width"] * 1e-6
attrs.update(
    long_name="Bin width",
    unit="m",
)
datas["bin_width"].attrs.update(attrs)

# Create radius variable and use it as leading dimension
logging.info("Create radius variable and use it as leading dimension")
datas["radius"] = datas["diameter"] / 2
datas["radius"].attrs.update(long_name="Radius", unit="m", comment="Radius of the bin.")

logging.info("Use radius as leading dimension for size bins")
datas = datas.swap_dims({"size": "radius"})

# Add major identifiers for the dataset
logging.info("Modify and add attributes")
datas.assign_attrs(
    {
        "flight_id": "varying, see also flight_number",
        "Modified_by": "Nils Niebaum",
        "Modification_date_UTC": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "GitHub Repository": "https://github.com/nilsnevertree/sdm-eurec4a",
        "GitHub Commit": get_git_revision_hash(),
    }
)

logging.info("Convert the particle size distribution from #/L/µm to SI units m^{-3} m^{-1}")
# Convert from #/l to #/m^3 ->  * 1e3
# Convert from µm to m -> 1e6
datas["particle_size_distribution"] = datas["particle_size_distribution"] * 1e3 * 1e6
attrs = datas["particle_size_distribution"].attrs
comment = "Number of droplets per cubic meter of air per meter bin width"
comment += (
    "The number concentration is normalized by the bin width, so it is the particle size distribution."
)
datas["particle_size_distribution"].attrs.update(
    long_name="Number concentration",
    unit="m^{-3} m^{-1}",
    comment=comment,
)


logging.info("Add mass size distribution to dataset")
# Calculate mass size distribution in g m^-3 m^-1
datas["mass_size_distribution"] = msd_from_psd_dataarray(
    datas["particle_size_distribution"],
    radius_name="radius",
    radius_scale_factor=1,  # one because radius is given in meters
    rho_water=1000,  # in kg/m^3
)
# Make sure to have the correct units!
# The mass size distribution is in kg/m^3/m
comment = "Mass of droplets per cubic meter of air assuming water density of 1000 kg/m3."
comment += "\nNormalized by the bin width."
datas["mass_size_distribution"].attrs.update(
    long_name="Mass concentration",
    unit="kg m^{-3} m^{-1}",
    comment=comment,
)

# Add non normalized particle size distribution
logging.info("Add non normalized particle size distribution")
# Multiply the particle size distribution by the bin width to get the total number of particles in #/m^3
datas["particle_size_distribution_non_normalized"] = (
    datas["particle_size_distribution"] * datas["bin_width"]
)
attrs = datas["particle_size_distribution"].attrs
datas["particle_size_distribution_non_normalized"].attrs.update(
    long_name="Number concentration",
    unit="m^{-3}",
    comment="Number of droplets per cubic meter of air, NOT normalized by the bin width. To normalize, divide by the bin width.",
)
# %%

# validate the values of the lwc given by the mass size distribution and the original lwc
# are equal to computer precision 1e-12
try:
    # liquid water content is given in g/m^3
    desired = 1e-3 * datas["liquid_water_content"].compute()
    actual = (datas["bin_width"] * datas["mass_size_distribution"]).sum("radius", skipna=True).compute()
    # get a mask of values which are not the same for both
    wrong_values = actual != desired
    # In the actual LWC (reconstructed from msd), some values are NaN due to instrument failure.
    # We need to exclude these values from the comparison.
    # They are by definition not equal to the desired values.
    nan_in_actual = np.isnan(actual.where(wrong_values, other=0))
    # nan_in_desired = np.isnan(desired.where(wrong_values, other = 0))
    # nan_in_wrong_values = nan_in_actual + nan_in_desired

    # Check if the actual and desired values are equal to computer precision 1e-12 while omiting the values which are
    # NOT wrong AND NaN in the actual values due to measurement failure.
    # And fill nan values with 0 to make sure the comparison works.
    np.testing.assert_allclose(
        actual=actual[~nan_in_actual].fillna(0),
        desired=desired[~nan_in_actual].fillna(0),
        rtol=1e-12,
        err_msg="The sum of the mass size distribution does not match the liquid water content.",
    )
except AssertionError as e:
    logging.error("The sum of the mass size distribution does not match the liquid water content.")
    raise e

# --- Add more mask to the dataset ---
# Make sure the masks are boolean
logging.info("Make sure the masks are boolean")

datas["cloud_mask"] = datas["cloud_mask"].astype(bool)
datas["drizzle_mask"] = datas["drizzle_mask"].astype(bool)
datas["rain_mask"] = datas["rain_mask"].astype(bool)

# Create mask if drizzle and rain are present
logging.info("Create mask if drizzle and rain are present")
datas["drizzle_rain_mask"] = ((datas["drizzle_mask"] == 1) + (datas["rain_mask"] == 1)).astype(bool)

comment = "based on the liquid water content of drizzle and rain size particles (diameters ranging from 100+/-5 to 2550+/-50 micro meter) ; 0: NT = 0/cm3  ; 1: NT > 0/cm3."
comment += "\nThe mask is simply the logical OR of the drizzle and rain mask."

datas["drizzle_rain_mask"].attrs.update(
    long_name="Drizzle and rain mask",
    unit="",
)


logging.info("Start storing produced dataset to netcdf file")
datas.to_netcdf(DESTINATION_FILEPATH)
logging.info("Finished cloud composite pre-processing")

# %%
