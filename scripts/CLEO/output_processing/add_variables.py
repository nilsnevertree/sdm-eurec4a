# %%
import argparse

from pathlib import Path

import numpy as np
import xarray as xr


def add_domain_masks(ds: xr.Dataset) -> None:
    # create domain mask and sub cloud layer mask
    ds["domain_mask"] = ds["gridbox"] <= ds["max_gridbox"]
    ds["domain_mask"].attrs.update(
        long_name="Domain Mask",
        description="Boolean mask indicating valid gridbox in the domain for each cloud",
        units="1",
    )

    ds["cloud_layer_mask"] = ds["gridbox"] == ds["max_gridbox"]
    ds["cloud_layer_mask"].attrs.update(
        long_name="Cloud Layer Mask",
        description="Boolean mask indicating if the gridbox is part of the cloud layer",
        units="1",
    )

    ds["sub_cloud_layer_mask"] = ds["gridbox"] < ds["max_gridbox"]
    ds["sub_cloud_layer_mask"].attrs.update(
        long_name="Sub Cloud Layer Mask",
        description="Boolean mask indicating if the gridbox is part of the sub cloud layer",
        units="1",
    )


def add_gridbox_properties(
    ds: xr.Dataset, d_coord3=20, d_coord2=20, d_coord1=20, cloud_resolution=100
) -> None:
    # create gridboc properties and pseudo coordinates

    dz_subcloud = ds["sub_cloud_layer_mask"] * d_coord3
    dz_cloud = ds["cloud_layer_mask"] * cloud_resolution
    # combine the two masked arrays
    dz = dz_subcloud + dz_cloud

    # calculate the upper and lower bounds of the gridbox
    z_upper = dz.cumsum("gridbox").where(ds["domain_mask"])
    z_lower = dz.shift(gridbox=1, fill_value=0).cumsum("gridbox").where(ds["domain_mask"])

    ds["gridbox_top"] = z_upper
    ds["gridbox_top"].attrs.update(
        long_name="Gridbox top",
        description=f"Gridbox top. Which is the upper bound of the gridbox for each gridbox.",
        units="$m$",
    )
    ds["gridbox_bottom"] = z_lower
    ds["gridbox_bottom"].attrs.update(
        long_name="Gridbox bottom",
        description=f"Gridbox bottom. Which is the lower bound of the gridbox for each gridbox.",
        units="$m$",
    )

    ds["gridbx_coord3"] = (ds["gridbox_top"] + ds["gridbox_bottom"]) / 2
    ds["gridbx_coord3"] = ds["gridbx_coord3"].where(ds["domain_mask"])
    ds["gridbx_coord3"].attrs.update(
        long_name="Gridbox center coordinate 3",
        description=f"Each gridbox below cloud layer is {d_coord3} m thick and cloud layer is {cloud_resolution} m thick.",
        units="$m$",
    )

    ds["gridbox_volume"] = (ds["gridbox_top"] - ds["gridbox_bottom"]) * d_coord2 * d_coord1
    ds["gridbox_volume"].attrs.update(
        long_name="Gridbox Volume",
        description=f"Gridbox Volume. V = (gridbox_top - gridbox_bottom) * {d_coord2} m * {d_coord3} m",
        units="$m^3$",
    )


def add_liquid_water_content(ds: xr.Dataset) -> None:
    ds["liquid_water_content"] = 1e3 * (ds["mass_represented"] / ds["gridbox_volume"])
    ds["liquid_water_content"].attrs.update(
        long_name="Liquid Water Content",
        description="Liquid Water Content per gridbox",
        units=r"$g m^{-3}$",
    )


def add_vertical_profiles(ds: xr.Dataset, time_slice=slice(1500, None)):
    ds["vertical_liquid_water_content"] = (
        ds["liquid_water_content"].sel(time=time_slice).sum("radius_bins").mean("time")
    ).where(ds["domain_mask"])

    ds["vertical_liquid_water_content"].attrs.update(
        long_name="Vertical Liquid Water Content",
        description=f"Vertical Profile of LWC. Sum over all radius bins and mean over time of the stationary state ({time_slice.start}-{time_slice.stop}).",
        units=ds["liquid_water_content"].attrs["units"],
    )

    ds["mass_difference_per_volume"] = ds["mass_difference_timestep"] / ds["gridbox_volume"]
    ds["mass_difference_per_volume"].attrs.update(
        long_name="Mass Difference per volume",
        description="Mass Difference per Volume. The mass difference is divided by the gridbox volume.",
        units=r"$g m^{-3} s^{-1}$",
    )

    ds["vertical_mass_difference_per_volume"] = (
        ds["mass_difference_per_volume"].sel(time=time_slice).sum("radius_bins").mean("time")
    )
    ds["vertical_mass_difference_per_volume"].attrs.update(
        long_name="Vertical mass difference per voluem",
        description=f"Vertical Profile of mass difference per voluem. Sum over all radius bins and mean over time of the stationary state ({time_slice.start}-{time_slice.stop}).",
        units=r"$g m^{-3} s^{-1}$",
    )


subdata_dir = "output_v3.5"
data_path = Path("/home/m/m301096/CLEO/data/") / subdata_dir

microphysics = (
    "null_microphysics",
    "condensation",
    "collision_condensation",
    "coalbure_condensation_large",
)


# %%

data_dict = dict(
    null_microphysics=dict(
        microphysics="Null microphysics (2048 SDs)",
        path=Path(),
    ),
    condensation=dict(
        microphysics="Condensation/Evaporation (2048 SDs)",
        path=Path(),
    ),
    collision_condensation=dict(
        microphysics="Collision and cond./evap. (2048 SDs)",
        path=Path(),
    ),
    coalbure_condensation_large=dict(
        microphysics="Coalbure condensation (2048 SDs)",
        path=Path(),
    ),
)

for mp in data_dict:
    data_dict[mp]["path"] = data_path / f"{mp}" / "combined/eulerian_dataset_combined_v2.nc"
    data_dict[mp]["output_path"] = data_path / f"{mp}" / "combined/eulerian_dataset_combined_v3.nc"

# %%
for key in data_dict:
    ds = xr.open_dataset(data_dict[key]["path"])
    # ds = ds.sel(cloud_id = [18, 301])
    ds.attrs.update(microphysics=data_dict[key]["microphysics"])
    data_dict[key]["combined_output"] = ds

# %%
from functools import reduce


print("all cloud ids which are present in all simulations")
in_all = reduce(
    np.intersect1d, [data_dict[key]["combined_output"]["cloud_id"].data for key in data_dict]
)
print(len(in_all))
print(in_all)


# %%
for key in data_dict:
    print(key)
    combined_output: xr.Dataset = data_dict[key]["combined_output"]
    add_domain_masks(combined_output)
    add_gridbox_properties(combined_output)
    add_liquid_water_content(combined_output)
    add_vertical_profiles(combined_output)
    print("save to", data_dict[key]["output_path"])
    combined_output.to_netcdf(data_dict[key]["output_path"])
    print(f"{key} saved")
