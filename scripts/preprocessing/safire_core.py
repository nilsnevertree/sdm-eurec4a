"""
This script preprocesses raw SAFIRE ATR42 core measurement data from the EUREC4A campaign.
The main steps include:

1. Setup and Logging: Initializes logging to track the preprocessing steps and outputs log messages to a file.
2. Flight Number Addition: Adds a flight number to each dataset based on the given attribute.
3. Variable Renaming: Renames variables to more readable names.
4. Unit Conversion: Converts units to SI units where necessary.
5. Dataset Validation: Validates that all datasets have the same attributes, except for specific attributes like `flight_id` and `creation_date`.
6. Data Loading and Preprocessing: Loads and preprocesses each dataset individually.
7. Data Combination: Combines the preprocessed datasets by the coordinate time.
8. Time Rounding: Rounds the time variable to the nearest second.
9. Attribute Modification: Modifies and adds attributes to the dataset.
10. Saving the Dataset: Saves the processed dataset to a NetCDF file.

The output dataset is an xarray.Dataset with dimensions for time and level, coordinates for time, latitude, longitude, and altitude, and various data variables including trajectory, time bounds, heading, course, roll, pitch, platform speed, pressure, temperature, humidity, wind parameters, and flight information.

Differences between input files and output NetCDF file:
- The output dataset has standardized variable names and units.
- Unused attributes are removed.
- Time is rounded to the nearest second.
- Units are converted to SI units where necessary.
- Additional attributes are added to the dataset for better metadata documentation.

Variable Mapping:
- 'LONGITUDE': 'longitude'
- 'LATITUDE': 'latitude'
- 'ALTITUDE': 'altitude'
- 'HEADING': 'heading'
- 'COURSE': 'course'
- 'ROLL': 'roll'
- 'PITCH': 'pitch'
- 'TAS': 'platform_speed_wrt_air'
- 'GS': 'platform_speed_wrt_ground'
- 'PRESSURE': 'pressure'
- 'TTEMP': 'total_temperature'
- 'TEMPERATURE': 'temperature'
- 'THETA': 'potential_temperature'
- 'DEW_POINT1': 'dew_point_temperature'
- 'ABS_HU1': 'absolute_humidity_1'
- 'MR1': 'humidity_mixing_ratio'
- 'ABS_HU2': 'absolute_humidity_2'
- 'LWC': 'liquid_water_content'
- 'EASTWARD_WIND': 'eastward_wind'
- 'NORTHWARD_WIND': 'northward_wind'
- 'VERTICAL_WIND': 'vertical_wind'
- 'WIND_DD': 'wind_from_direction'
- 'WIND_FF': 'wind_speed'
- 'flight_number': 'flight_number'
- 'flight_id': 'flight_id'

Unit Conversions:
- Temperature: Celsius to Kelvin (°C to K)
- Pressure: hPa to Pa
- Absolute Humidity: g/m³ to kg/m³
- Humidity Mixing Ratio: g/kg to kg/kg
- Wind Speeds: m/s to m s⁻¹
"""

# %%

import datetime
import logging

from pathlib import Path

import numpy as np
import xarray as xr

from sdm_eurec4a import get_git_revision_hash
from sdm_eurec4a.reductions import validate_datasets_same_attrs


REPO_PATH = Path(__file__).resolve().parent.parent.parent

ORIGIN_DIRECTORY = REPO_PATH / Path("data/observation/safire_core/raw")
DESTINATION_DIRECTORY = REPO_PATH / Path("data/observation/safire_core/processed")
DESTINATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
DESTINATION_FILENAME = "safire_core.nc"


logging.basicConfig(
    filename=DESTINATION_DIRECTORY / "safire_core.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("============================================================")
logging.info("Start SAFIRE CORE pre-processing")
logging.info("Git hash: %s", get_git_revision_hash())
logging.info("Origin directory: %s", ORIGIN_DIRECTORY.relative_to(REPO_PATH))
logging.info("Destination directory: %s", DESTINATION_DIRECTORY.relative_to(REPO_PATH))
logging.info("Destination filename: %s", DESTINATION_FILENAME)


def add_flight_number(ds: xr.Dataset) -> xr.Dataset:
    """
    Add flight number to dataset based on the given attribute.

    Args:
        ds (xr.Dataset): dataset to add flight number to which contains the
            attribute 'flight_id' with the flight number in a form of 'as0000XX'
            for flight number XX.

    Returns:
        xr.Dataset: dataset with variable 'flight_number' and 'flight_id' added.
    """
    flight_id = ds.attrs["flight_id"]
    flight_number = int(flight_id[-2:])
    ds["flight_number"] = flight_number
    ds["flight_number"].attrs.update(
        long_name="Flight number",
        comment="Flight number of the measurement campaign.",
    )
    ds["flight_id"] = flight_id
    return ds


def variable_mapping(ds: xr.Dataset) -> xr.Dataset:
    """
    Rename variables to more readable names.

    Args:
        ds (xr.Dataset): dataset with original variable names.

    Returns:
        xr.Dataset: dataset with renamed variables.
    """
    VARNAME_MAPPING = {
        "LONGITUDE": "longitude",
        "LATITUDE": "latitude",
        "ALTITUDE": "altitude",
        "HEADING": "heading",
        "COURSE": "course",
        "ROLL": "roll",
        "PITCH": "pitch",
        "TAS": "platform_speed_wrt_air",
        "GS": "platform_speed_wrt_ground",
        "PRESSURE": "pressure",
        "TTEMP": "total_temperature",
        "TEMPERATURE": "temperature",
        "THETA": "potential_temperature",
        "DEW_POINT1": "dew_point_temperature",
        "ABS_HU1": "absolute_humidity_1",
        "MR1": "humidity_mixing_ratio",
        "ABS_HU2": "absolute_humidity_2",
        "LWC": "liquid_water_content",
        "EASTWARD_WIND": "eastward_wind",
        "NORTHWARD_WIND": "northward_wind",
        "VERTICAL_WIND": "vertical_wind",
        "WIND_DD": "wind_from_direction",
        "WIND_FF": "wind_speed",
        "flight_number": "flight_number",
        "flight_id": "flight_id",
    }
    return ds.rename(VARNAME_MAPPING)


def make_si_conform(ds: xr.Dataset) -> xr.Dataset:

    # Convert temperature to Kelvin
    ds["temperature"] += 273.15
    ds["temperature"].attrs.update(units="K")

    # Convert dew point temperature to Kelvin
    ds["dew_point_temperature"] += 273.15
    ds["dew_point_temperature"].attrs.update(units="K")

    # Convert total temperature to Kelvin
    ds["total_temperature"] += 273.15
    ds["total_temperature"].attrs.update(units="K")

    # Convert potential temperature to Kelvin
    ds["potential_temperature"] += 273.15
    ds["potential_temperature"].attrs.update(units="K")

    # Convert pressure to Pa
    ds["pressure"] *= 100
    ds["pressure"].attrs.update(units="Pa")

    # Convert absolute humidity to kg/kg
    ds["absolute_humidity_1"] /= 1000
    ds["absolute_humidity_1"].attrs.update(units="kg m^{-3}")
    ds["absolute_humidity_2"] /= 1000
    ds["absolute_humidity_2"].attrs.update(units="kg m^{-3}")

    # Convert humidity mixing ratio to kg/kg
    ds["humidity_mixing_ratio"] /= 1000
    ds["humidity_mixing_ratio"].attrs.update(units="kg kg^{-1}")

    # liquid water content
    ds["liquid_water_content"].attrs.update(units="g m^{-3}")

    # wind speeds
    for k in list(ds.variables.keys()):
        units = ds[k].attrs.get("units")
        if units == "m/s":
            ds[k].attrs.update(units="m s^{-1}")

    return ds


def preprocess_func(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess the dataset by adding flight number, renaming variables, and converting units to SI units.

    Args:
        ds (xr.Dataset): dataset to preprocess.

    Returns:
        xr.Dataset: preprocessed dataset.
    """
    ds = add_flight_number(ds)
    ds = variable_mapping(ds)
    ds = make_si_conform(ds)
    # ds.load()
    return ds


files = sorted(ORIGIN_DIRECTORY.glob("*.nc"))
if len(files) == 0:
    logging.error("No files found in %s", ORIGIN_DIRECTORY)
    raise FileNotFoundError("No files found in %s", ORIGIN_DIRECTORY)

logging.info("Number of files: %s", len(files))

# --- Validate that all datasets have the same attributes except for fligth_id and creation_date ---

skip_attrs = [
    # 'creator_name',
    # 'creator_email',
    "flight_id",
    # 'processing_level',
    # 'featureType',
    "flight_date",
    "time_take_off",
    "time_landing",
    "comment",
    # 'acqid',
    # 'project',
    # 'platform',
    # 'doi',
    # 'date_modified',
    # 'date_created',
    # 'creator_institution',
    # 'contributor_name',
    # 'contributor_role',
    # 'summary',
    # 'history',
    # 'source',
    # 'title',
    # 'institution',
    # 'id',
    # 'product_version',
    # 'Conventions',
    "time_coverage_start",
    "time_coverage_end",
    "time_coverage_resolution",
    "geospatial_lon_min",
    "geospatial_lon_max",
    "geospatial_lat_min",
    "geospatial_lat_max",
    # 'geospatial_vertical_min ',
]

logging.info("Begin loading data")
logging.info("Preprocess each individual dataset")
logging.info("Add flight number")
logging.info("Rename variables")
logging.info("Convert units to SI units")

datasets_list = []
for file in files:
    ds = xr.open_dataset(file)
    ds = preprocess_func(ds)
    for attrs in skip_attrs:
        ds.attrs.pop(attrs)

    datasets_list.append(ds)

logging.info(
    "Validate that all datasets have the same attributes except for fligth_id and creation_date"
)
try:
    validate_datasets_same_attrs(
        datasets_list,
    )
except AssertionError:
    logging.error("Not all datasets have the same attributes")
    raise AssertionError("Not all datasets have the same attributes")

logging.info("Combine datasets by coordinate time")
datas = xr.concat(datasets_list, dim="time")

logging.info("Modify and add attributes")
datas = datas.assign_attrs(
    {
        "flight_id": "varying, see also flight_number",
        "modified_by": "Nils Niebaum",
        "date_combined": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "git commit": get_git_revision_hash(),
        "GitHub Repository": "https://github.com/nilsnevertree/sdm-eurec4a",
    }
)

# %%
# Round the time to the nearest second
logging.info("Round UTC time to seconds accuracy")
attrs = datas["time"].attrs
encoding = datas["time"].encoding
datas["time"] = datas["time"].dt.round("s")
datas["time"].attrs.update(attrs)
datas["time"].attrs.update(comment="Time rounded to the nearest second.")
datas["time"].encoding = encoding
# %%

logging.info("Save the produced dataset to netcdf file")
datas.to_netcdf(DESTINATION_DIRECTORY / DESTINATION_FILENAME)
logging.info("Finished SAFIR-CORE pre-processing")
