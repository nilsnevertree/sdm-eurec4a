"""
Script to pre-process cloud composite data from Coutris, P. (2021).  SAFIRE
ATR42: PMA/Cloud composite dataset.  [Dataset].  Aeris.
https://doi.org/10.25326/237 which is stored locally in
../data/observation/cloud_composite/raw The pre-processing includes:

    - rename variables to longer names
    - convert time to datetime object
    - creates radius variable and uses it as coordinate for size bins
    - modify attributes
    - save the produced datset to netcdf file
The produced dataset is stored in ../data/observation/cloud_composite/processed
"""

import datetime
import logging

from pathlib import Path

import cftime
import dask
import numpy as np
import pandas as pd
import xarray as xr

from sdm_eurec4a import get_git_revision_hash
from sdm_eurec4a.reductions import validate_datasets_same_attrs


REPO_PATH = Path(__file__).resolve().parent.parent.parent

ORIGIN_DIRECTORY = REPO_PATH / Path("data/observation/cloud_composite/raw")
DESTINATION_DIRECTORY = REPO_PATH / Path("data/observation/cloud_composite/processed")
DESTINATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
DESTINATION_FILENAME = "cloud_composite_si_units.nc"


logging.basicConfig(
    filename=DESTINATION_DIRECTORY / "cloud_composite_preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("============================================================")
logging.info("Start cloud composite pre-processing")
logging.info("Git hash: %s", get_git_revision_hash())
logging.info("Origin directory: %s", ORIGIN_DIRECTORY.relative_to(REPO_PATH))
logging.info("Destination directory: %s", DESTINATION_DIRECTORY.relative_to(REPO_PATH))
logging.info("Destination filename: %s", DESTINATION_FILENAME)


def add_flight_number(ds: xr.Dataset) -> xr.Dataset:
    """
    Add flight number to dataset based on filename ending. This function is
    used as a pre-processing step when opening the files of the cloud composite
    dataset.

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

logging.info("Number of files: %s", len(files))

# --- Validate that all datasets have the same attributes except for fligth_id and creation_date ---

datasets_list = []
for file in files:
    datasets_list.append(xr.open_dataset(file))

logging.info(
    "Validate that all datasets have the same attributes except for fligth_id and creation_date"
)
try:
    validate_datasets_same_attrs(datasets_list, skip_attrs=["flight_id", "creation_date"])
except AssertionError:
    logging.error("Not all datasets have the same attributes")
    raise AssertionError("Not all datasets have the same attributes")


# --- Load data ---
try:
    datas = xr.open_mfdataset(
        paths=files,
        combine="by_coords",
        parallel=False,
        chunks={"time": 1000},
        preprocess=add_flight_number,
    )
except Exception as e:
    logging.exception("Error while opening files")
    raise e

# --- Reorganize dataset ---

try:
    logging.info("Rename variables")
    # Use longer names for variable to make it more readable
    # do not rename the dimension time and size
    VARNAME_MAPPING = {
        "lon": "lon",
        "lat": "lat",
        "alt": "alt",
        "PSD": "particle_size_distribution",
        "MSD": "mass_size_distribution",
        "LWC": "liquid_water_content",
        "NT": "total_concentration",
        "MVD": "median_volume_diameter",
        "M6": "radar_reflectivity_factor",
        "diameter": "diameter",
        "bw": "bin_width",
        "compo_index": "composition_index",
        "CDP_flag": "flag_CDP",
        "2DS_flag": "flag_2DS",
        "CLOUD_mask": "cloud_mask",
        "DZ_mask": "drizzle_mask",
        "RA_mask": "rain_mask",
        "flight_number": "flight_number",
    }
    datas = datas.rename(VARNAME_MAPPING)

    # Convert time to datetime object
    # Note, that the time is in seconds since 2020-01-01 00:00:00
    logging.info("Convert UTC time to datetime object")
    datas["time"] = cftime.num2date(
        datas.time, units="seconds since 2020-01-01 00:00:00", calendar="standard"
    )

    logging.info("Validate that diameter and bin_width do not vary along time axis")
    assert np.all(datas.diameter == datas.diameter.isel(time=0))
    assert np.all(datas.bin_width == datas.bin_width.isel(time=0))
    datas["diameter"] = datas["diameter"].mean("time", keep_attrs=True)
    datas["bin_width"] = datas["bin_width"].mean("time", keep_attrs=True)

    logging.info("Renormalize the particle size distribution from #/L/µm to #/m^3")
    # Multiply the particle size distribution by the bin width to get the total number of particles in #/L
    datas["particle_size_distribution"] = datas["particle_size_distribution"] * datas["bin_width"]
    # Convert from #/l to #/m^3 ->  * 1e3
    datas["particle_size_distribution"] = datas["particle_size_distribution"] * 1e3
    # Update attributes
    attrs = datas["particle_size_distribution"].attrs
    datas["particle_size_distribution"].attrs.update(
        unit="#/m^3",
        comment="histogram: each bin gives the number of droplets per cubic meter of air, NOT normalized by the bin width. To normalize, divide by the bin width.",
    )
    logging.info("Convert diameter and bin_width to meters")
    # Convert from µm to m -> 1e-6
    # Diameter
    attrs = datas["diameter"].attrs
    datas["diameter"] = datas["diameter"] * 1e-6
    attrs.update(
        unit="meter",
    )
    datas["diameter"].attrs.update(attrs)
    # Bin width
    attrs = datas["bin_width"].attrs
    datas["bin_width"] = datas["bin_width"] * 1e-6
    attrs.update(
        unit="meter",
    )
    datas["bin_width"].attrs.update(attrs)

    logging.info("Create radius variable and use it as leading dimension")
    datas["radius"] = datas["diameter"] / 2
    datas["radius"].attrs.update(long_name="Radius", unit="m", comment="radius of the droplets")

    logging.info("Use radius as leading dimension for size bins")
    datas = datas.swap_dims({"size": "radius"})

    logging.info("Modify and add attributes")
    datas.assign_attrs(
        {
            "flight_id": "varying, see also flight_number",
            "Modified_by": "Nils Niebaum",
            "Modification_date_UTC": str(datetime.datetime.now(datetime.UTC)) + " GMT",
            "GitHub Repository": "https://github.com/nilsnevertree/sdm-eurec4a",
            "GitHub Commit": get_git_revision_hash(),
        }
    )

except Exception as e:
    logging.exception("Error while organizing dataset")
    raise e


print(f"Save the produced datset to netcdf file?\n{DESTINATION_DIRECTORY / DESTINATION_FILENAME}")
user_input = input("Do you want to continue running the script? (y/n): ")
if user_input.lower() == "y":
    print("Saving dataset\nPlease wait...")
    datas.to_netcdf(DESTINATION_DIRECTORY / DESTINATION_FILENAME)
else:
    logging.error("User denied proceeding with saving the dataset")
    raise KeyboardInterrupt

logging.info("Finished cloud composite pre-processing")

# VERY SLOW
# comp = dict(zlib=True, complevel=5)
# encoding = {var: comp for var in datas.data_vars}
# datas.to_netcdf(DESTINATION_DIRECTORY / 'cloud_composite_compressed.nc', encoding=encoding)
