"""
This script computes a dataset which stores the spatial and temporal distance between
the identified clouds and the dropsondes.

The identified clouds dataset path is hardcoded in the script.
    'data/observation/cloud_composite/processed/identified_clouds_more.nc'
The dropsondes dataset path is hardcoded in the script.
    'data/observation/dropsonde/Level_3/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc'
"""

import datetime
import logging
import os
import sys

from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from dask.diagnostics import ProgressBar
from sdm_eurec4a import get_git_revision_hash
from sdm_eurec4a.calculations import great_circle_distance_np


# %%
# Example dataset
script_path = Path(os.path.abspath(__file__))
print(f"Script path is\n\t{script_path}")
SCRIPT_DIR = script_path.parent

SETTINGS_PATH = SCRIPT_DIR / "settings" / "distance_calculation.yaml"

# open settings path
with open(SETTINGS_PATH, "r") as stream:
    try:
        SETTINGS = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise exc


REPO_PATH = Path(script_path).parent.parent.parent
print(f"Repository root is\n\t{REPO_PATH}")

OUTPUT_DIR = REPO_PATH / Path(SETTINGS["paths"]["output_directory"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILEPATH_CLOUDS = REPO_PATH / Path(SETTINGS["paths"]["input_filepath_clouds"])
INPUT_FILENAME_CLOUDS = INPUT_FILEPATH_CLOUDS.stem

INPUT_FILEPATH_DROPSONDES = REPO_PATH / Path(SETTINGS["paths"]["input_filepath_dropsondes"])

print(SETTINGS["paths"]["output_file_name"])

if SETTINGS["paths"]["output_file_name"] is None:
    OUTPUT_FILE_NAME = f"distance_dropsondes_{INPUT_FILENAME_CLOUDS}.nc"
    settings_output_name = f"distance_dropsondes_{INPUT_FILENAME_CLOUDS}.yaml"
else:
    OUTPUT_FILE_NAME = SETTINGS["paths"]["output_file_name"]
    settings_output_name = SETTINGS["paths"]["output_file_name"].split(".")[0] + ".yaml"

SETTINGS["paths"]["output_file_name"] = OUTPUT_FILE_NAME

# prepare logger

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(OUTPUT_DIR / "distance.log")
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


# input("Check paths and press Enter to continue...")

# %%

logging.info("============================================================")
logging.info("Start distance calculation")
logging.info("Git hash: %s", get_git_revision_hash())
logging.info("Input clouds file: %s", INPUT_FILEPATH_CLOUDS.relative_to(REPO_PATH))
logging.info("Input dropsondes file: %s", INPUT_FILEPATH_DROPSONDES.relative_to(REPO_PATH))
logging.info("Destination directory: %s", OUTPUT_DIR.relative_to(REPO_PATH))
logging.info("Destination filename: %s", OUTPUT_FILE_NAME)

logging.info("Save settings to output directory")
with open(OUTPUT_DIR / settings_output_name, "w") as file:
    yaml.dump(SETTINGS, file)


def main():
    identified_clouds = xr.open_dataset(INPUT_FILEPATH_CLOUDS)
    # display(identified_clouds)

    drop_sondes = xr.open_dataset(INPUT_FILEPATH_DROPSONDES)

    # 1. Create combined dataset
    # 2. Compute the distances

    ds_combined = xr.Dataset(
        data_vars={
            "latitude_cloud_composite": identified_clouds.rename({"time": "time_identified_clouds"})[
                "latitude"
            ],
            "longitude_cloud_composite": identified_clouds.rename({"time": "time_identified_clouds"})[
                "longitude"
            ],
            "latitude_drop_sondes": drop_sondes.rename({"time": "time_drop_sondes"})["flight_latitude"],
            "longitude_drop_sondes": drop_sondes.rename({"time": "time_drop_sondes"})[
                "flight_longitude"
            ],
        },
        attrs={
            "title": "Distance between identified clouds and dropsondes datasets",
            "created with": "scripts/preprocessing/distance_relation_IC_DS.py",
            "author": "Nils Niebaum",
            "author email": "nils-ole.niebaum@mpimet.mpg.de",
            "featureType": "trajectory",
            "creation_time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    ds_combined = ds_combined.drop_vars(["sonde_id"])
    ds_combined["time_identified_clouds"].attrs = dict(
        long_name="time of identified clouds measurments",
        standard_name="time_cc",
        axis="T",
    )

    # Create dummy arrays for time to be chunkable and store them in the combined dataset
    ds_combined["t_ic"] = (("time_identified_clouds"), ds_combined["time_identified_clouds"].data)
    ds_combined["t_ds"] = (("time_drop_sondes"), ds_combined["time_drop_sondes"].data)
    ds_combined["t_ic"] = ds_combined["t_ic"].chunk({"time_identified_clouds": 1000})
    ds_combined["t_ds"] = ds_combined["t_ds"].chunk({"time_drop_sondes": -1})

    # %%
    with ProgressBar():
        # Calculate the distance between the ART measurements and the dropsondes release locations
        logging.info("Compute spatial distance")
        hdistance = xr.apply_ufunc(
            great_circle_distance_np,
            ds_combined["latitude_cloud_composite"],
            ds_combined["longitude_cloud_composite"],
            ds_combined["latitude_drop_sondes"],
            ds_combined["longitude_drop_sondes"],
            # input_core_dims=[['time_cc'], ['time_cc'], ["time_ds"], ["time_ds"]],
            # output_core_dims=[["time_cc", "time_ds"]],
            vectorize=True,
            dask="allowed",
            output_dtypes=[float],
            # output_sizes={"time": 1000},
        )

        hdistance.name = "spatial_distance"
        hdistance.attrs = dict(
            long_name="spatial distance",
            units="km",
            comment="spatial distance calculated with great_circle_distance_np function between identified clouds and dropsonde measurements",
        )
        hdistance
        logging.info("Store spatial distance")
        hdistance.to_netcdf(OUTPUT_DIR / "temp_spatial.nc")

        # Calculate the temporal distance between the ART measurements and the dropsondes release times
        logging.info("Compute temporal distance")
        tdistance = ds_combined.t_ic - ds_combined.t_ds
        tdistance.name = "temporal_distance"
        tdistance.attrs = dict(
            long_name="temporal distance",
            comment="temporal distance between identified clouds (IC) and dropsonde launch time (DS). The values are IC - DS.",
        )
        logging.info("Store temporal distance")
        tdistance.to_netcdf(OUTPUT_DIR / "temp_temporal.nc")
    # %%
    # Load the dataarrays and store them in the combined dataset
    tdistance = xr.open_dataarray(
        OUTPUT_DIR / "temp_temporal.nc"  # , chunks={"time_cc": 1000, "time_ds": -1}
    )
    hdistance = xr.open_dataarray(
        OUTPUT_DIR / "temp_spatial.nc"  # , chunks={"time_cc": 1000, "time_ds": -1}
    )

    logging.info("Create combined dataset")
    result = ds_combined
    result = result.drop_vars(["t_ic", "t_ds"])

    result["temporal_distance"] = tdistance
    result["spatial_distance"] = hdistance

    logging.info("The combined dataset looks like this:")
    # print(result)
    logging.info("Store combined dataset")
    with ProgressBar():
        result.to_netcdf(OUTPUT_DIR / OUTPUT_FILE_NAME)
    logging.info("Successfully finished distance calculation pre-processing")
    logging.info(f"File written to {OUTPUT_DIR / OUTPUT_FILE_NAME}")


# %%


if __name__ == "__main__":
    main()
