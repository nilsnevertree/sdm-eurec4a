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
import sys

from pathlib import Path

import numpy as np
import xarray as xr

from dask.diagnostics import ProgressBar


# %%
# Example dataset
script_path = os.path.abspath(__file__)
print(f"Script path is\n\t{script_path}")

REPO_PATH = Path(script_path).parent.parent.parent
print(f"Repository root is\n\t{REPO_PATH}")

OUTPUT_DIR = REPO_PATH / Path("data/observation/cloud_composite/processed/identified_clouds/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory is\n\t{OUTPUT_DIR}")

INPUT_FILEPATH = REPO_PATH / Path("data/observation/cloud_composite/processed/cloud_composite.nc")
print(f"Input file path is\n\t{INPUT_FILEPATH}")

# specify the mask to use for cloud identification
mask_name = "rain_mask"
print(f"Use mask '{mask_name}' to identify clouds")

OUTPUT_FILE_NAME = f"identified_clouds_{mask_name}.nc"
print(f"Output file name is\n\t'{OUTPUT_FILE_NAME}'")


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
logging.info("Input file: %s", INPUT_FILEPATH.relative_to(REPO_PATH))
logging.info("Destination directory: %s", OUTPUT_DIR.relative_to(REPO_PATH))
logging.info("Destination filename: %s", OUTPUT_FILE_NAME)
logging.info("Mask name: %s", mask_name)


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
        # # The following code is omitted because storing objects is not a good idea.
        # clouds['selection'] = ('cloud_id', [(start, end) for start, end in zip(clouds.start.data, clouds.end.data)])
        # clouds['selection'].attrs = {
        #         'long_name': 'time selection of cloud event',
        #         'description': 'tupel of (start, end) for the cloud event. This can help to select the cloud event from the original dataset'
        #         }

        # Define
        clouds = clouds.assign_coords({"time": clouds.mid_time})
        clouds = clouds.swap_dims({"cloud_id": "time"})
        logging.info("Store cloud identification dataset")
        clouds.to_netcdf(OUTPUT_DIR / "temporary.nc")

    clouds = xr.open_dataset(OUTPUT_DIR / "temporary.nc")

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

    with ProgressBar():
        clouds.to_netcdf(OUTPUT_DIR / OUTPUT_FILE_NAME)
    logging.info("Successfully finished cloud identification pre-processing")
    logging.info(f"File written to {OUTPUT_DIR / OUTPUT_FILE_NAME}")


if __name__ == "__main__":
    main()
# %%
