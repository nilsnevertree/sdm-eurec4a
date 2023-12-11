"""
Script to pre-process cloud composite data from Coutris, P. (2021).  SAFIRE
ATR42: PMA/Cloud composite dataset.  [Dataset].  Aeris.
https://doi.org/10.25326/237 which is stored locally in
../data/observation/cloud_composite/raw The pre-processing includes:

    - rename variables to longer names
    - convert time to datetime object
    - use diameter as coordinate for size bins
    - modify attributes
    - save the produced datset to netcdf file
The produced dataset is stored in ../data/observation/cloud_composite/processed
"""

import logging

from datetime import datetime
from pathlib import Path

import cftime
import dask
import numpy as np
import pandas as pd
import xarray as xr


ORIGIN_DIRECTORY = Path("../../data/observation/cloud_composite/raw")
DESTINATION_DIRECTORY = Path("../../data/observation/cloud_composite/processed")
DESTINATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
DESTINATION_FILENAME = "cloud_composite.nc"


logging.basicConfig(
    filename=DESTINATION_DIRECTORY / "cloud_composite_preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("============================================================")
logging.info("Start cloud composite pre-processing")
logging.info("Origin directory: %s", ORIGIN_DIRECTORY)
logging.info("Destination directory: %s", DESTINATION_DIRECTORY)
logging.info("Destination filename: %s", DESTINATION_FILENAME)


def add_flight_number(ds: xr.Dataset) -> xr.Dataset:
    """
    Add flight number to dataset based on filename ending.

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


def validate_datasets_same_attrs(datasets: list, skip_attrs: list = []) -> bool:
    """
    Check if all datasets have the same attributes except for the ones in
    skip_attrs.

    Args:
        datasets (list): list of datasets
        skip_attrs (list): list of attributes to skip, default is empty list

    Returns:
        bool: True if all attributes are the same, False otherwise
    """
    attrs = []
    for ds in datasets:
        attrs.append(ds.attrs)
    attrs = pd.DataFrame(attrs)
    attrs = attrs.drop(columns=skip_attrs)
    nunique_attrs = attrs.nunique()
    return np.all(nunique_attrs == 1)


empty_ds1 = xr.Dataset(
    coords={},
    data_vars={},
    attrs={
        "Conventions": "abc",
        "history": "2021-08-12 14:23:22 GMT",
        "edition": 2,
        "random_number": 1,
        "random_string": "first random",
    },
)

empty_ds2 = xr.Dataset(
    coords={},
    data_vars={},
    attrs={
        "Conventions": "abc",
        "history": "2021-08-12 14:23:22 GMT",
        "edition": 2,
        "random_number": 2,
        "random_string": "first random",
    },
)


empty_ds3 = xr.Dataset(
    coords={},
    data_vars={},
    attrs={
        "Conventions": "abc",
        "history": "2021-08-12 14:23:22 GMT",
        "edition": 2,
        "random_number": 1,
        "random_string": "second random",
    },
)


# test to check using all combinations of the three empty datasets to validate if the attributes are the same or not
# The test function is parametrized with the three empty datasets
# and it usees the validate_datasets_same_attrs function to check if the attributes are the same or not
# \
def test_validate_datasets_same_attrs():
    # same dataset
    assert (
        validate_datasets_same_attrs(
            [empty_ds1, empty_ds1],
        )
        == True
    )
    # different number
    assert (
        validate_datasets_same_attrs(
            [empty_ds1, empty_ds2],
        )
        == False
    )
    # different string
    assert (
        validate_datasets_same_attrs(
            [empty_ds1, empty_ds3],
        )
        == False
    )
    # different number but skip string
    assert validate_datasets_same_attrs([empty_ds1, empty_ds2], skip_attrs=["random_string"]) == False
    # different number and skip number
    assert validate_datasets_same_attrs([empty_ds1, empty_ds2], skip_attrs=["random_number"]) == True
    # different string but skip number
    assert validate_datasets_same_attrs([empty_ds2, empty_ds3], skip_attrs=["random_number"]) == False
    # different string and number - skip both
    assert (
        validate_datasets_same_attrs(
            [empty_ds2, empty_ds3], skip_attrs=["random_number", "random_string"]
        )
        == True
    )


test_validate_datasets_same_attrs()


files = sorted(ORIGIN_DIRECTORY.glob("*.nc"))
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
    datas["diameter"] = datas["diameter"].mean("time", keep_attrs=True)  # convert from m to um
    datas["bin_width"] = datas["bin_width"].mean("time", keep_attrs=True)  # convert from m to um

    logging.info("Use diameter as coordinate for size bins")
    datas = datas.swap_dims({"size": "diameter"})

    logging.info("Modify and add attributes")
    datas.assign_attrs(
        {
            "creation_date": "varying, see single source files",
            "flight_id": "varying, see also flight_number",
            "Modified_by": "Nils Niebaum",
            "Modification_date_UTC": str(datetime.utcnow()) + " GMT",
        }
    )

except Exception as e:
    logging.exception("Error while organizing dataset")
    raise e


print(f"Save the produced datset to netcdf file?\n{DESTINATION_DIRECTORY / 'cloud_composite.nc'}")
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
