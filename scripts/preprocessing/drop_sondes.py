"""
Script to pre-process drop sonde data from JOANNE dataset: references
https://doi.org/10.25326/246, George et al. 2021.

The dataset storage location in this Repository is assumed to be at
    ../data/observation/cloud_composite/raw/Level_3/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc

The script will:
    - rename variables to longer names
    - rename "launch_time" to "time" and make it the leading dimension
    - modify attributes
    - save the produced dataset to netcdf file
The produced dataset is stored in ../data/observation/dropsonde/processed/drop_sondes.nc
"""

# %%
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


REPO_PATH = Path(__file__).resolve().parent.parent.parent

ORIGIN_DIRECTORY = REPO_PATH / Path("data/observation/dropsonde/raw/Level_3")
ORIGIN_FILENAME = "EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc"
DESTINATION_DIRECTORY = REPO_PATH / Path("data/observation/dropsonde/processed")
DESTINATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
DESTINATION_FILENAME = "drop_sondes.nc"

log_file_path = DESTINATION_DIRECTORY / "drop_sondes_preprocessing.log"

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


logging.info("============================================================")
logging.info("Start drop_sondes pre-processing")
logging.info("Git hash: %s", get_git_revision_hash())
logging.info("Origin directory: %s", ORIGIN_DIRECTORY.relative_to(REPO_PATH))
logging.info("Origin filename: %s", ORIGIN_FILENAME)
logging.info("Destination directory: %s", DESTINATION_DIRECTORY.relative_to(REPO_PATH))
logging.info("Destination filename: %s", DESTINATION_FILENAME)

# --- Load data ---
try:
    original = xr.open_dataset(ORIGIN_DIRECTORY / ORIGIN_FILENAME)
except Exception as e:
    logging.exception("Error while opening files")
    raise e

# --- Reorganize dataset ---

# %%

try:
    logging.info("Rename variables")
    # Use longer names for variable to make it more readable
    # do not rename the dimension time and size
    VARNAME_MAPPING = {
        # make sure theses are uniform throughout the datasets
        "lat": "latitude",
        "lon": "longitude",
        "alt": "altitude",
        # rename the rest
        "launch_time": "time",
        "p": "pressure",
        "ta": "air_temperature",
        "rh": "relative_humidity",
        "wspd": "wind_speed",
        "wdir": "wind_direction",
        "u": "u",
        "v": "v",
        "theta": "potential_temperature",
        "q": "specific_humidity",
        "low_height_flag": "low_height_flag",
        "platform_id": "platform_id",
        "flight_altitude": "flight_alt",
        "flight_lat": "flight_latitude",
        "flight_lon": "flight_longitude",
        "N_ta": "N_air_temperature",
        "N_rh": "N_relative_humidity",
        "N_gps": "N_gps",
        "m_p": "method_pressure",
        "m_ta": "method_air_temperature",
        "m_rh": "method_relative_humidity",
        "m_gps": "method_gps",
        "alt_bnds": "altitude_bounds",
    }
    logging.info("Rename variables.")
    data = original.rename(VARNAME_MAPPING)

    logging.info("Make time the leading dimension. Swap with sonde_id and sort by time.")
    data = data.swap_dims({"sonde_id": "time"})
    data = data.sortby("time")
    logging.info("Modify and add attributes")
    data.assign_attrs(
        {
            "modified_by": "Nils Niebaum",
            "modification_date_UTC": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "GitHub Repository": "https://github.com/nilsnevertree/sdm-eurec4a",
            "GitHub Commit": get_git_revision_hash(),
        }
    )

    logging.info("Convert relative humidity to percentage")
    attrs = data["relative_humidity"].attrs
    data["relative_humidity"] = data["relative_humidity"] * 100
    data["relative_humidity"].attrs = attrs
    data["relative_humidity"].attrs["units"] = "\%"

except Exception as e:
    logging.exception("Error while organizing dataset")
    raise e


# print(f"Save the produced dataset to netcdf file?\n{DESTINATION_DIRECTORY / 'cloud_composite.nc'}")
# user_input = input("Do you want to continue running the script? (y/n): ")
# if user_input.lower() == "y":
# print("Saving dataset\nPlease wait...")
data.to_netcdf(DESTINATION_DIRECTORY / DESTINATION_FILENAME)
# else:
#     logging.error("User denied proceeding with saving the dataset.")
#     raise KeyboardInterrupt

logging.info("Finished drop sonde pre-processing")

# VERY SLOW
# comp = dict(zlib=True, complevel=5)
# encoding = {var: comp for var in datas.data_vars}
# datas.to_netcdf(DESTINATION_DIRECTORY / 'cloud_composite_compressed.nc', encoding=encoding)

# %%
