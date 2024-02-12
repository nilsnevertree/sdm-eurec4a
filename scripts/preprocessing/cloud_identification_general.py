"""
This script identifies clouds in the cloud composite dataset.
It uses one of the masks from the original dataset to identify clouds.
This mask can be selected by setting the variable ``mask_name``.
All np.nan values in the rain_mask are set to 0/False.

A new dataset is created which looks like this:
<xarray.Dataset>
Dimensions:               (time: 100)
Coordinates:
        cloud_id              (time) int64 0 1 2 ...
        * time                  (time) datetime64[ns] 2017-01-01T00:30:00 ...
Data variables:
        start                 (time) datetime64[ns] 2017-01-01T00:30:00 ...
        end                   (time) datetime64[ns] 2017-01-01T00:34:00 ...
        duration              (time) timedelta64[ns] 00:00:00 ...
        mid_time              (time) datetime64[ns] 2017-01-01T00:32:00 ...
        liquid_water_content  (time) float64 ...
        alt                   (time) float64 ...
        lat                   (time) float64 ...
        lon                   (time) float64 ...

The variables are defined as follows:
        - start: start time of cloud event
        - end: end time of cloud event
        - duration: duration of cloud event
        - mid_time: mid time of cloud event
        - liquid_water_content: total LWC of cloud event
        - alt: mean altitude of cloud event
        - lat: mean latitude of cloud event
        - lon: mean longitude of cloud event

The dataset is stored in the folder data/observation/cloud_composite/processed/

The identification is simply done by using the diff function of xarray.
Thus the script can not identify cloud clusters.

author: Nils Niebaum
email: nils-ole.niebaum@mpimet.mpg.de
github_username: nilsnevertree
"""
# %%

import datetime
import logging
import os
import secrets
import sys

from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from dask.diagnostics import ProgressBar
from sdm_eurec4a import get_git_revision_hash
from sdm_eurec4a.calculations import horizontal_extent_func, vertical_extent_func


# %%
# Example dataset
script_path = Path(os.path.abspath(__file__))
print(f"Script path is\n\t{script_path}")
SCRIPT_DIR = script_path.parent

SETTINGS_PATH = SCRIPT_DIR / "settings" / "cloud_identification.yaml"

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
print(f"Output directory is\n\t{OUTPUT_DIR}")

INPUT_FILEPATH = REPO_PATH / Path(SETTINGS["paths"]["input_filepath"])
print(f"Input file path is\n\t{INPUT_FILEPATH}")

# specify the mask to use for cloud identification
mask_name = SETTINGS["setup"]["mask_name"]
print(f"Use mask '{mask_name}' to identify clouds")

if SETTINGS["paths"]["output_file_name"] is None:
    OUTPUT_FILE_NAME = f"identified_clouds_{mask_name}.nc"
    settings_output_name = f"identified_clouds_{mask_name}.yaml"
else:
    OUTPUT_FILE_NAME = SETTINGS["paths"]["output_file_name"]
    settings_output_name = SETTINGS["paths"]["output_file_name"].split(".")[0] + ".yaml"

SETTINGS["paths"]["output_file_name"] = OUTPUT_FILE_NAME
print(f"Output file name is\n\t'{OUTPUT_FILE_NAME}'")

temp_file_name = f"{secrets.token_hex(nbytes=4)}_temporary.nc"
TEMPORARY_FILEPATH = OUTPUT_DIR / temp_file_name
SETTINGS["paths"]["temporary_filepath"] = TEMPORARY_FILEPATH.relative_to(REPO_PATH).as_posix()

# %%
# prepare logger

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(OUTPUT_DIR / "cloud_identification.log")
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


# %%

logging.info("============================================================")
logging.info("Start cloud identification pre-processing")
logging.info("Git hash: %s", get_git_revision_hash())
logging.info("Input file: %s", INPUT_FILEPATH.relative_to(REPO_PATH))
logging.info("Destination directory: %s", OUTPUT_DIR.relative_to(REPO_PATH))
logging.info("Destination filename: %s", OUTPUT_FILE_NAME)
logging.info("Temporary file path: %s", TEMPORARY_FILEPATH)
logging.info("Mask name: %s", mask_name)

logging.info("Save settings to output directory")
with open(OUTPUT_DIR / settings_output_name, "w") as file:
    yaml.dump(SETTINGS, file)


def main():
    cloud_composite = xr.open_dataset(INPUT_FILEPATH, chunks={"time": 1000})

    with ProgressBar():
        logging.info("Identify clouds using xr.diff")
        cloud_diff = cloud_composite[mask_name].fillna(0).astype(int).diff(dim="time")
        cloud_diff = cloud_diff.compute()
        cloud_start = cloud_diff.time.where(cloud_diff == 1, drop=True)
        cloud_end = cloud_diff.time.where(cloud_diff == -1, drop=True)
        logging.info(f"{cloud_start.shape} number of clouds were identified")

        logging.info("Create cloud identification dataset")
        clouds = xr.Dataset(
            coords={"cloud_id": np.arange(0, cloud_start.size)},
        )
        details = (
            f"Cloud identification is done by using the {mask_name} from the original dataset.\n"
            f"All np.nan values in the {mask_name} are set to 0/False.\n"
            f"The original {mask_name} is then used to identify individual clouds.\n"
        )

        clouds.attrs = {
            "description": "cloud identification dataset",
            "creation_time": str(datetime.datetime.now()),
            "details": details,
            "author": "Nils Niebaum",
            "email": "nils-ole.niebaum@mpimet.mpg.de",
            "institution": "Max Planck Institute for Meteorology",
        }

        clouds["start"] = ("cloud_id", cloud_start.data)
        clouds["start"].attrs = {"long_name": "start time of cloud event"}
        clouds["end"] = ("cloud_id", cloud_end.data)
        clouds["end"].attrs = {"long_name": "end time of cloud event"}
        clouds["duration"] = clouds.end - clouds.start
        clouds["duration"].attrs = {"long_name": "duration of cloud event"}
        clouds["mid_time"] = clouds.start + clouds.duration / 2
        clouds["mid_time"].attrs = {"long_name": "mid time of cloud event"}

        # Define
        clouds = clouds.assign_coords({"time": clouds.mid_time})
        clouds = clouds.swap_dims({"cloud_id": "time"})
        logging.info("Store cloud identification dataset")
        clouds.to_netcdf(TEMPORARY_FILEPATH)

    with ProgressBar():
        clouds = xr.open_dataset(TEMPORARY_FILEPATH)

        logging.info("Calculate mean altitude of cloud events")
        clouds["alt"] = (
            "time",
            [
                cloud_composite["alt"].sel(time=slice(start, end)).mean()
                for start, end in zip(clouds.start.data, clouds.end.data)
            ],
        )
        clouds["alt"].attrs = {
            "long_name": "mean altitude of cloud event",
            "units": "m",
            "comment": "This is the mean altitude of all pixels in the cloud event.\nFrom SAFIRE ATR42 Inertial/GPS System",
        }

        logging.info("Calculate mean latitude of cloud events")
        clouds["lat"] = (
            "time",
            [
                cloud_composite["lat"].sel(time=slice(start, end)).mean()
                for start, end in zip(clouds.start.data, clouds.end.data)
            ],
        )
        clouds["lat"].attrs = {
            "long_name": "mean latitude of cloud event",
            "units": "degree",
            "comment": "This is the mean latitude of all pixels in the cloud event.\nFrom SAFIRE ATR42 Inertial/GPS System.",
        }

        logging.info("Calculate mean longitude of cloud events")
        clouds["lon"] = (
            "time",
            [
                cloud_composite["lon"].sel(time=slice(start, end)).mean()
                for start, end in zip(clouds.start.data, clouds.end.data)
            ],
        )
        clouds["lon"].attrs = {
            "long_name": "mean longitude of cloud event",
            "units": "degree",
            "comment": "This is the mean longitude of all pixels in the cloud event.\nFrom SAFIRE ATR42 Inertial/GPS System.",
        }

        logging.info("Calculate spatial extent of cloud events")

        clouds["horizontal_extent"] = xr.DataArray(
            [
                horizontal_extent_func(cloud_composite.sel(time=slice(start, end)))
                for start, end in zip(clouds.start.data, clouds.end.data)
            ],
            dims="time",
            coords={"time": clouds.time},
            attrs={
                "long_name": "horizontal extent of cloud",
                "units": "km",
                "description": "The horizontal extent of the cloud in m. Calculated as the great circle distance based on the minimum and maximum of both latitude and longitude.",
            },
        )
        clouds["vertical_extent"] = xr.DataArray(
            [
                vertical_extent_func(cloud_composite.sel(time=slice(start, end)))
                for start, end in zip(clouds.start.data, clouds.end.data)
            ],
            dims="time",
            coords={"time": clouds.time},
            attrs={
                "long_name": "vertical extent of cloud",
                "units": "km",
                "description": "The vertical extent of the cloud in km. Calculated as the difference between the maximum and minimum altitude.",
            },
        )

        logging.info("Calculate total LWC of cloud events")
        clouds["liquid_water_content"] = (
            "time",
            [
                cloud_composite["liquid_water_content"].sel(time=slice(start, end)).sum()
                for start, end in zip(clouds.start.data, clouds.end.data)
            ],
        )
        clouds["liquid_water_content"].attrs = {
            "long_name": "total LWC of cloud event",
            "units": "g/m3",
            "comment": "This is the sum of the LWC of all pixels in the cloud event.\nMass of all droplets per cubic meter of air, assuming water spheres with density = 1g/cm3",
        }

    with ProgressBar():
        clouds.to_netcdf(OUTPUT_DIR / OUTPUT_FILE_NAME)
    logging.info(f"File written to {OUTPUT_DIR / OUTPUT_FILE_NAME}")
    logging.info("Remove temporary file")
    Path(TEMPORARY_FILEPATH).unlink()
    logging.info("Successfully finished cloud identification pre-processing")


if __name__ == "__main__":
    main()
# %%
