"""
This script identifies clouds in the ``cloud_mask`` of the cloud composite dataset.
All np.nan values in the cloud_mask are set to 0/False.

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
import os

from pathlib import Path

import numpy as np
import xarray as xr

from dask.diagnostics import ProgressBar


# %%
# Example dataset
script_path = os.path.abspath(__file__)
print(script_path)

REPOSITORY_ROOT = Path(script_path).parent.parent.parent
print(REPOSITORY_ROOT)

output_path = REPOSITORY_ROOT / Path("data/observation/cloud_composite/processed/")
output_path.mkdir(parents=True, exist_ok=True)

FILEPATH = REPOSITORY_ROOT / Path("data/observation/cloud_composite/processed/cloud_composite.nc")
print(FILEPATH)
cloud_composite = xr.open_dataset(FILEPATH, chunks={"time": 1000})
cloud_composite = cloud_composite

# min_duration_new = 5
# cm_new = cm_org.fillna(0).astype(bool)
# cm_new = consecutive_events_xr(cm_new, min_duration = min_duration_new, axis="time")
# with ProgressBar():
#         cm_new = cm_new.compute()
input("Press Enter to continue...")
# %%
with ProgressBar():
    cloud_diff = cloud_composite.cloud_mask.fillna(0).astype(int).diff(dim="time")
    cloud_start = cloud_diff.time.where(cloud_diff == 1, drop=True)
    cloud_end = cloud_diff.time.where(cloud_diff == -1, drop=True)
    print(f"{cloud_start.shape} number of clouds were identified")

    # %%
    print("Create cloud identification dataset")
    clouds = xr.Dataset(
        coords={"cloud_id": np.arange(0, cloud_start.size)},
    )
    details = (
        "cloud identification is done by using the cloud_mask from the original dataset.\n"
        "All np.nan values in the cloud_mask are set to 0/False.\n"
        "The original cloud_mask is then used to identify individual clouds.\n"
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
    print("Store cloud identification dataset.")
    clouds.to_netcdf(output_path / Path("identified_clouds.nc"))

# %%
clouds = xr.open_dataset(output_path / Path("identified_clouds.nc"))

print("Calculate total LWC of cloud events.")
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

print("Calculate mean altitude of cloud events.")
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

print("Calculate mean latitude of cloud events.")
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

print("Calculate mean longitude of cloud events.")
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

print("Save new dataset to disk.")
with ProgressBar():
    clouds.to_netcdf(output_path / Path("identified_clouds_more.nc"))
