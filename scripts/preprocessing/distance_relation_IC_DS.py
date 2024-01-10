"""
This script computes a dataset which stores the spatial and temporal distance
between the identified clouds and the dropsondes.

The identified clouds dataset path is hardcoded in the script.
    'data/observation/cloud_composite/processed/identified_clouds_more.nc'
The dropsondes dataset path is hardcoded in the script.
    'data/observation/dropsonde/Level_3/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc'
"""


# %%
import datetime
import os

from pathlib import Path

import dask
import numpy as np
import xarray as xr

from dask.diagnostics import ProgressBar
from sdm_eurec4a.calculations import great_circle_distance_np


script_path = os.path.abspath(__file__)
print("Script path", script_path)

REPOSITORY_ROOT = Path(script_path).parent.parent.parent
print("Repository Path", REPOSITORY_ROOT)

output_path = REPOSITORY_ROOT / Path("data/observation/combined/distance")
print("Output Path", output_path)

input("Check paths and press Enter to continue...")

# %%
# create output directory
output_path.mkdir(parents=True, exist_ok=True)


identified_clouds = xr.open_dataset(
    REPOSITORY_ROOT / Path("data/observation/cloud_composite/processed/identified_clouds_more.nc")
)
# display(identified_clouds)

drop_sondes = xr.open_dataset(
    REPOSITORY_ROOT
    / Path("data/observation/dropsonde/Level_3/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc")
)
drop_sondes = drop_sondes.rename({"launch_time": "time"})
drop_sondes = drop_sondes.swap_dims({"sonde_id": "time"})
drop_sondes = drop_sondes.sortby("time")
drop_sondes = drop_sondes.chunk({"time": -1})

# 1. Create combined dataset
# 2. Compute the distances

ds_combined = xr.Dataset(
    data_vars={
        "lat_cloud_composite": identified_clouds.rename({"time": "time_identified_clouds"}).lat,
        "lon_cloud_composite": identified_clouds.rename({"time": "time_identified_clouds"}).lon,
        "lat_drop_sondes": drop_sondes.rename({"time": "time_drop_sondes"}).flight_lat,
        "lon_drop_sondes": drop_sondes.rename({"time": "time_drop_sondes"}).flight_lon,
    },
    attrs={
        "title": "Distance between identified clouds and dropsondes datasets",
        "created with": "scripts/preprocessing/distance_relation_IC_DS.py",
        "author": "Nils Niebaum",
        "author email": "nils-ole.niebaum@mpimet.mpg.de",
        "featureType": "trajectory",
        "creation_time": str(datetime.datetime.now()),
    },
)
ds_combined = ds_combined.drop_vars(["sonde_id"])
ds_combined["time_identified_clouds"].attrs = dict(
    long_name="time of identified clouds measurments",
    standard_name="time_cc",
    axis="T",
)

# Create dummy arrays for time to be chunkable and store them in the combined dataset
ds_combined["t_ic"] = (("time_identified_clouds"), ds_combined.time_identified_clouds.data)
ds_combined["t_ds"] = (("time_drop_sondes"), ds_combined.time_drop_sondes.data)
ds_combined["t_ic"] = ds_combined["t_ic"].chunk({"time_identified_clouds": 1000})
ds_combined["t_ds"] = ds_combined["t_ds"].chunk({"time_drop_sondes": -1})

# %%
with ProgressBar():
    # Calculate the distance between the ART measurements and the dropsondes release locations
    print("Compute spatial distance")
    hdistance = xr.apply_ufunc(
        great_circle_distance_np,
        ds_combined.lat_cloud_composite,
        ds_combined.lon_cloud_composite,
        ds_combined.lat_drop_sondes,
        ds_combined.lon_drop_sondes,
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
    print("Store spatial distance")
    hdistance.to_netcdf(output_path / "spatial_distance_IC_DS.nc")

    # Calculate the temporal distance between the ART measurements and the dropsondes release times
    print("Compute temporal distance")
    tdistance = ds_combined.t_ic - ds_combined.t_ds
    tdistance.name = "temporal_distance"
    tdistance.attrs = dict(
        long_name="temporal distance",
        comment="temporal distance between identified clouds (IC) and dropsonde launch time (DS). The values are IC - DS.",
    )
    print("Store temporal distance")
    tdistance.to_netcdf(output_path / "temporal_distance_IC_DS.nc")
# %%
# Load the dataarrays and store them in the combined dataset
tdistance = xr.open_dataarray(
    output_path / "temporal_distance_IC_DS.nc", chunks={"time_cc": 1000, "time_ds": -1}
)
hdistance = xr.open_dataarray(
    output_path / "spatial_distance_IC_DS.nc", chunks={"time_cc": 1000, "time_ds": -1}
)

print("Create combined dataset")
result = ds_combined
result = result.drop_vars(["t_ic", "t_ds"])

result["temporal_distance"] = tdistance
result["spatial_distance"] = hdistance

print("The combined dataset looks like this:")
print(result)
print("Store combined dataset")
with ProgressBar():
    result.to_netcdf(output_path / "distances_IC_DS.nc")
# %%
