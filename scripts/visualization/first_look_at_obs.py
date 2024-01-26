"""
First look at the observations
==============================
This notebook is a first look at the observations.
It is used to get a first idea of the data and to get a feeling for the data.
It is not used for the actual analysis.

The following data sets are used:
- ATR measurements
- Dropsondes

It shall be a first draft on how to select related data from the two data sets.
The selection criteria can be changed.
A first idea is to use the following criteria:
- Temporal contraint :
    RF15 as Raphaela told us
- Spatial constraint :
    - Altitude
    - Position
- Physical constraints
    - rain existence
    - high liquid water content
"""

import os

# %%
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sdm_eurec4a.reductions import (
    latlon_dict_to_polygon,
    polygon2mask,
    rectangle_spatial_mask,
)
from sdm_eurec4a.visulization import (
    adjust_lightness,
    gen_color,
    plot_colors,
    set_custom_rcParams,
)
from shapely.geometry import Polygon


# %%
# Set custom colors

plt.style.use("dark_background")
colors = set_custom_rcParams()

# %% [markdown]
# ## 0. Define paths

# %%

RFs = np.array([6, 7, 8])

script_path = os.path.abspath(__file__)
print(script_path)

REPOSITORY_ROOT = Path(script_path).parent.parent
print(REPOSITORY_ROOT)

figure_dir = REPOSITORY_ROOT / Path(f"results/correct_name/RF{np.min(RFs)}-{np.max(RFs)}")
print(figure_dir)
# try :
# print(f"REPO dir.: {REPOSITORY_ROOT}")
# print(f"Figure dir.: {figure_dir}")
#     user_input = input("Is this correct?\ny/n\n")
#     assert user_input == "y"
# except AssertionError:
#     raise AssertionError("Please check the paths above.")

figure_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## 1. Load datasets

# %%
FILEPATH = REPOSITORY_ROOT / Path("data/observation/cloud_composite/processed/cloud_composite.nc")
print(FILEPATH)
cloud_composite = xr.open_dataset(FILEPATH)
# display(cloud_composite)

FILEPATH = REPOSITORY_ROOT / Path(
    "data/observation/dropsonde/Level_3/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc"
)
drop_sondes = xr.open_dataset(FILEPATH)
drop_sondes = drop_sondes.rename({"launch_time": "time"})
drop_sondes = drop_sondes.swap_dims({"sonde_id": "time"})
drop_sondes = drop_sondes.sortby("time")
# display(drop_sondes)

# # %%

# psd_all = cloud_composite["particle_size_distribution"]# * 1e9
# msd_all = cloud_composite["mass_size_distribution"] #* 1e12
# mask = ""
# psd = psd_all.where(psd_all != 0, drop=True)
# msd = msd_all.where(msd_all != 0, drop=True)


# %% [markdown]
# # 2. Selection criteria

# %% [markdown]
# Contraints on how to choose individual profiles
#
# - Temporal contraint :
#     RF15 as Raphaela told us
# - Spatial constraint :
#     - Altitude
#     - Position
# - Physical constraints
#     - rain existence
#     - high liquid water content

# %% [markdown]
# ### First idea on some constraints

# %% [markdown]
# For an ATR measurement to be considered,
# all the following criteria need to be met:
# - flight number 14, 15, 16
# - altitude below 1000 m
# - small region given by
#     - -58.75 E <> -58.25 E
#     -  13.5 N <> 14 N
# - provided cloud mask applied
#

# %%
#  constraints
print("Applying constraints on the datasets")
# %%
area_cloud_composite = dict(
    lon_min=-58.75,
    lon_max=-58.25,
    lat_min=13.5,
    lat_max=14,
)

area_drop_sondes = dict(
    lon_min=-59,
    lon_max=-58,
    lat_min=13,
    lat_max=14.5,
)

flight_constraint = cloud_composite.flight_number.isin(RFs)
altitude_constraint = cloud_composite.alt < 1100
spatial_constraint = rectangle_spatial_mask(
    ds=cloud_composite,
    area=area_cloud_composite,
    lat_name="lat",
    lon_name="lon",
)
drop_sondes_spatial_constraint = rectangle_spatial_mask(
    ds=drop_sondes,
    area=area_drop_sondes,
    lat_name="flight_lat",
    lon_name="flight_lon",
)


# mask_constraint = cloud_composite.rain_mask == 1
# liquid_water_content_constraint = cloud_composite.liquid_water_content > cloud_composite.liquid_water_content.quantile(dim="time", q=0.9)
full_constraint = (
    flight_constraint
    & altitude_constraint
    & spatial_constraint
    # & mask_constraint
)
time_values_of_constraint = cloud_composite.time.where(full_constraint, drop=True)


# For the cloud composite data set it is sufficient to use the time values of the constraint to select the data
cc_constraint = cloud_composite.sel(time=time_values_of_constraint)

# For the dropsonde data set we need select:
# 1. Match the spatial constraint
# 2. All time values that match the time constraint.
#    For this, the for each ATR measurement the nearest dropsonde is selected.
# 3. Only use unqiue timevalus to avoid double counting
ds_constraint = drop_sondes.where(drop_sondes_spatial_constraint, drop=True)
drop_sondes_time_constraint_all = ds_constraint.time.sel(
    time=time_values_of_constraint, method="nearest"
)
drop_sondes_time_constraint = np.unique(drop_sondes_time_constraint_all)
ds_constraint = ds_constraint.sel(time=drop_sondes_time_constraint)

print(f"{len(time_values_of_constraint)} ATR measurments are selected")
print(f"{len(drop_sondes_time_constraint)} drop sondes are selected")


# # %%
# sonde_ids = drop_sondes.sonde_id.values
# halo_ids = sonde_ids[["HALO" in v for v in sonde_ids]]
# p3_ids = sonde_ids[["P3" in v for v in sonde_ids]]
# # make sure no flight is used twice
# assert sorted(np.concatenate([halo_ids, p3_ids])) == sorted(sonde_ids)

# halo_launches = drop_sondes.time.where(drop_sondes.sonde_id.isin(halo_ids), drop=True)
# p3_launches = drop_sondes.time.where(drop_sondes.sonde_id.isin(p3_ids), drop=True)

# # %%

# # Draw a map of all drop sondes released during the campaign
# print("Plotting all drop sondes and ATR locations")

# fig = plt.figure(figsize=(10, 10), layout="constrained")
# ax = plt.axes(projection=ccrs.PlateCarree())
# # ax.coastlines()
# # ax.add_feature(cfeature.LAND)
# # ax.add_feature(cfeature.OCEAN, color="navy", alpha=0.2)
# # ax.add_feature(cfeature.BORDERS, linestyle=":")
# ax.gridlines(draw_labels=True, linestyle=":", alpha=0.4)
# ax.set_extent([-60, -56, 12, 15])

# xx, yy = latlon_dict_to_polygon(area_cloud_composite).exterior.xy
# ax.plot(xx, yy, transform=ccrs.PlateCarree(), color="red", linewidth=2, label="ATR selected area")

# xx, yy = latlon_dict_to_polygon(area_drop_sondes).exterior.xy
# ax.plot(
#     xx,
#     yy,
#     transform=ccrs.PlateCarree(),
#     color="red",
#     linestyle="--",
#     linewidth=2,
#     label="Dropsondes selected area",
# )


# ax.scatter(
#     cloud_composite.lon,
#     cloud_composite.lat,
#     transform=ccrs.PlateCarree(),
#     # color = 'b',
#     marker=".",
#     alpha=0.7,
#     label="ATR",
# )

# ax.scatter(
#     drop_sondes.sel(time=halo_launches).flight_lon,
#     drop_sondes.sel(time=halo_launches).flight_lat,
#     transform=ccrs.PlateCarree(),
#     label="HALO",
#     marker="+",
#     color=colors[1],
#     alpha=0.7,
# )

# ax.scatter(
#     drop_sondes.sel(time=p3_launches).flight_lon,
#     drop_sondes.sel(time=p3_launches).flight_lat,
#     transform=ccrs.PlateCarree(),
#     label="P3",
#     marker="x",
#     color=colors[1],
#     alpha=0.7,
# )


# ax.legend()
# ax.set_title("Dropsonde and ATR locations overview")
# fig.savefig(figure_dir / "sonde_art_locations_overview.png", dpi=300, transparent=False)

# %%
# Draw a map of all drop sondes released during the campaign
print("Plotting selected drop sondes and ATR locations")
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.OCEAN, color="navy", alpha=0.2)
# ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True)
ax.set_extent([-60, -56, 12, 15])
cm = plt.cm.get_cmap("RdYlBu")

xx, yy = latlon_dict_to_polygon(area_cloud_composite).exterior.xy
ax.plot(xx, yy, transform=ccrs.PlateCarree(), color="red", linewidth=2, label="ATR selected area")

xx, yy = latlon_dict_to_polygon(area_drop_sondes).exterior.xy
ax.plot(
    xx,
    yy,
    transform=ccrs.PlateCarree(),
    color="red",
    linestyle="--",
    linewidth=2,
    label="Dropsondes selected area",
)

ax.scatter(
    cc_constraint.lon,
    cc_constraint.lat,
    transform=ccrs.PlateCarree(),
    # c = cc_constraint.flight_number.values,
    marker="+",
    # alpha= 0.1,
    label="ATR",
    # cmap="jet"
)

ax.scatter(
    ds_constraint.flight_lon,
    ds_constraint.flight_lat,
    transform=ccrs.PlateCarree(),
    label="Dropsondes",
    marker="x",
)

ax.legend()
ax.set_title("ATR locations fitting conditions and related drop sondes")
fig.savefig(figure_dir / "art_sonde_locations_condition.png", dpi=300, transparent=False)

# %%
print("Plotting selected drop sondes and ATR locations")
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(
    time_values_of_constraint,
    time_values_of_constraint.astype(int) * 0 + 0.1,
    marker="+",
    linestyle="",
    label=f"{len(time_values_of_constraint)} ATR measurments",
)

ax.plot(
    drop_sondes_time_constraint,
    drop_sondes_time_constraint.astype(int) * 0,
    marker="x",
    linestyle="",
    label=f"{len(drop_sondes_time_constraint)} drop sondes",
)
# ax.legend(ncol=2, loc="upper center")
ax.set_ylim(-0.05, 0.15)
ax.set_yticks(
    [
        0.1,
        0,
    ],
    [
        f"ATR\n#{len(time_values_of_constraint)}",
        f"Dropsondes\n#{len(drop_sondes_time_constraint)}",
    ],
)
# ax.set_xticks(rotation=-45, ha="left")
ax.set_title(
    "Measurement times of ATR fitting the conditions.\nAnd temporal 'nearest' drop sondes in the selcted area."
)
fig.tight_layout()
fig.savefig(figure_dir / "conditions_art_sonde_times.png", dpi=300, transparent=False)

# %%
# Analysis based on the given masks

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(
    time_values_of_constraint,
    time_values_of_constraint.astype(int) * 0 + 0.1,
    marker="+",
    linestyle="",
    label=f"{len(time_values_of_constraint)} ATR",
)

ax.plot(
    drop_sondes_time_constraint,
    drop_sondes_time_constraint.astype(int) * 0.0,
    marker="x",
    linestyle="",
    label=f"{len(drop_sondes_time_constraint)} Dropsondes",
)

# plot the drizzle, cloud, rain mask

drizzle_time_values = time_values_of_constraint.where(cc_constraint.drizzle_mask == 1, drop=True)
cloud_time_values = time_values_of_constraint.where(cc_constraint.cloud_mask == 1, drop=True)
rain_time_values = time_values_of_constraint.where(cc_constraint.rain_mask == 1, drop=True)

ax.plot(
    cloud_time_values,
    cloud_time_values.astype(int) * 0 + 0.2,
    marker="+",
    linestyle="",
    label=f"{len(cloud_time_values)} ATR cloud",
)

ax.plot(
    drizzle_time_values,
    drizzle_time_values.astype(int) * 0 + 0.3,
    marker="+",
    linestyle="",
    label=f"{len(drizzle_time_values)} ATR drizzle",
)

ax.plot(
    rain_time_values,
    rain_time_values.astype(int) * 0 + 0.4,
    marker="+",
    linestyle="",
    label=f"{len(rain_time_values)} ATR rain",
)


# # Shrink current axis's height by 10% on the bottom
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

ax.set_ylim(-0.05, 0.45)
ax.set_yticks(
    [0.1, 0, 0.2, 0.3, 0.4],
    [
        f"ATR\n#{len(time_values_of_constraint)}",
        f"Dropsondes\n#{len(drop_sondes_time_constraint)}",
        f"ATR cloud\n#{len(cloud_time_values)}",
        f"ATR drizzle\n#{len(drizzle_time_values)}",
        f"ATR rain\n#{len(rain_time_values)}",
    ],
)
# ax.set_xticks(rotation=-45, ha="left")
ax.set_title(
    "Measurement times of ATR fitting the conditions.\nAnd temporal 'nearest' drop sondes in the selcted area."
)
# ax.legend(bbox_to_anchor=(1.04, 0.5))
fig.tight_layout()
fig.savefig(figure_dir / "conditions_art_sonde_times_masks.png", dpi=300, transparent=False)

# %% [markdown]
# ===========================
# Dropsonde data
# ===========================

# %%
n, a = ds_constraint.theta.shape
# Create a color dict with individual colors for each day of the year which is present in the data set
color_func = lambda ds: ds.time.dt.date

color_n = np.unique([color_func(ds_constraint)]).astype(str)
color_list = gen_color("Set3", n=color_n.size)
plot_colors(color_list)
color_dict = dict(zip(color_n, color_list))

xx = np.tile(ds_constraint.alt, (n, 1))
cc = np.tile(ds_constraint.time.dt.day, (a, 1)).T

inner_times = ds_constraint.time.where(
    rectangle_spatial_mask(
        ds=ds_constraint,
        area=area_cloud_composite,
        lat_name="flight_lat",
        lon_name="flight_lon",
    )
)

outer_times = ds_constraint.time.where(
    rectangle_spatial_mask(
        ds=ds_constraint,
        area=area_drop_sondes,
        lat_name="flight_lat",
        lon_name="flight_lon",
    )
)

# Plot the temperature profiles for the selected sondes and color them by their day of the year value
style = dict(linewidth=0.8, linestyle="-", alpha=0.8)

print("Plotting selected drop sondes")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5), sharey=True)

axs_theta = axs[0]
axs_q = axs[1]
# ds_constraint.theta.shape
old_day = None
for i, t in enumerate(ds_constraint.time):
    day = str(color_func(t).values)
    color = color_dict[day]
    if old_day != day:
        axs_theta.plot(
            ds_constraint.theta.sel(time=t), ds_constraint.alt, color=color, label=f"{day}", **style
        )
        axs_q.plot(ds_constraint.q.sel(time=t), ds_constraint.alt, color=color, label=f"{day}", **style)
    else:
        axs_theta.plot(ds_constraint.theta.sel(time=t), ds_constraint.alt, color=color, **style)
        axs_q.plot(ds_constraint.q.sel(time=t), ds_constraint.alt, color=color, **style)

    old_day = day

for ax in axs.flatten():
    ax.legend(loc="lower right")
    ax.set_ylim(0, 2000)
    ax.set_ylabel("Altitude [m]")

axs[0].set_xlim(297, 305)
axs[0].set_xlabel("Potential Temperature [K]")
axs[1].set_xlim(0, 0.025)
axs[1].set_xlabel("Relative humidity in [kg / kg]")

axs[0].set_title("Potential temperature")
axs[1].set_title("Relative humidity")
fig.suptitle("Profiles of realted drop sondes | colored by date")
fig.savefig(figure_dir / "art_sonde_temperature_profiles.png", dpi=300, transparent=False)


## difference between inner box and outer box
# %%
inner_style = dict(linewidth=0.8, linestyle="-", alpha=0.8)
outer_style = dict(linewidth=0.8, linestyle="--", alpha=0.8)

print("Plotting selected drop sondes | Inner and Outer box")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5), sharey=True)

axs_theta = axs[0]
axs_q = axs[1]
# ds_constraint.theta.shape

old_day = None
for i, t in enumerate(ds_constraint.time):
    day = str(color_func(t).values)
    color = color_dict[day]
    if np.isin(t.values, inner_times.values):
        style = inner_style
    elif np.isin(t.values, outer_times.values):
        style = outer_style
    else:
        print(t.values)
        raise ValueError("Something went wrong with the inner outer mask")
    if old_day != day:
        axs_theta.plot(
            ds_constraint.theta.sel(time=t), ds_constraint.alt, color=color, label=f"{day}", **style
        )
        axs_q.plot(ds_constraint.q.sel(time=t), ds_constraint.alt, color=color, label=f"{day}", **style)
    else:
        axs_theta.plot(ds_constraint.theta.sel(time=t), ds_constraint.alt, color=color, **style)
        axs_q.plot(ds_constraint.q.sel(time=t), ds_constraint.alt, color=color, **style)

    old_day = day

for ax in axs.flatten():
    ax.legend(loc="lower right")
    ax.set_ylim(0, 2000)
    ax.set_ylabel("Altitude [m]")

axs[0].set_xlim(297, 305)
axs[0].set_xlabel("Potential Temperature [K]")
axs[1].set_xlim(0, 0.025)
axs[1].set_xlabel("Relative humidity in [kg / kg]")

axs[0].set_title("Potential temperature")
axs[1].set_title("Relative humidity")
fig.suptitle(
    "Profiles of realted drop sondes | colored by date\nSolid for sondes in small box | Dashed for sondes in bigger box"
)
fig.savefig(figure_dir / "art_sonde_temperature_profiles_boxes.png", dpi=300, transparent=False)


# %% [markdown]
### ATR data: PSD and MSD
# Stuff

# %%
print("Plotting selected ATR measurments")
fig, axs = plt.subplots(figsize=(15, 7.5), ncols=2, sharex=True)

#  Plot the particle_size_distribution for all and for the selected sondes
axs[0].set_xscale("log")

axs[0].plot(
    cc_constraint.diameter,
    cc_constraint.particle_size_distribution,
    color=colors[0],
    alpha=0.1,
    linewidth=0,
    marker=".",
    # label = f'individual measurements {q*100:.0f}ths percentile based on LWC'
)

axs[0].plot(
    cc_constraint.diameter,
    cc_constraint.particle_size_distribution.where(cc_constraint.particle_size_distribution != 0).median(
        dim="time", skipna=True
    ),
    color="r",
    alpha=1,
    linewidth=2,
    # marker = '.',
    label=f"Median",
)

axs[0].plot(
    cc_constraint.diameter,
    cc_constraint.particle_size_distribution.where(cc_constraint.particle_size_distribution != 0).mean(
        dim="time", skipna=True
    ),
    color="r",
    alpha=1,
    linewidth=2,
    linestyle="--",
    # marker = '.',
    label=f"Mean",
)


axs[0].set_xlabel("Particle diameter [µm]")
axs[0].set_ylabel("Particle size distribution [#/L]")
axs[0].set_title("Particle size distribution")
#  Plot the particle_size_distribution for all and for the selected sondes

axs[1].plot(
    cc_constraint.diameter,
    cc_constraint.mass_size_distribution,
    color=colors[0],
    alpha=0.1,
    linewidth=0,
    marker=".",
    # label = f'individual measurements {q*100:.0f}th percentile based on LWC'
)

axs[1].set_xscale("log")
axs[1].plot(
    cc_constraint.diameter,
    cc_constraint.mass_size_distribution.where(cc_constraint.mass_size_distribution != 0).median(
        dim="time", skipna=True
    ),
    color="r",
    alpha=1,
    linewidth=2,
    label=f"Median",
)
axs[1].plot(
    cc_constraint.diameter,
    cc_constraint.mass_size_distribution.where(cc_constraint.mass_size_distribution != 0).mean(
        dim="time", skipna=True
    ),
    color="r",
    alpha=1,
    linewidth=2,
    linestyle="--",
    label=f"Mean",
)

axs[1].set_xlabel("Particle diameter [µm]")
axs[1].set_ylabel("Mass size distribution [g/L/µm]")
axs[1].set_title("Mass size distribution")

for ax in axs.flatten():
    ax.legend()
    ax.set_yscale("log")

fig.suptitle(f"Particle size distribution and mass size distribution from ATR measurements.")
fig.savefig(figure_dir / "art_sonde_psd_msd_all.png", dpi=300, transparent=False)


# %%
print("Plotting selected ATR measurments | Masks impact")
fig = plt.figure(figsize=(20, 15), layout="constrained")
subfigs = fig.subfigures(1, 2, wspace=0.1)

axs_left = subfigs[0].subplots(ncols=1, nrows=4, sharey=True)
axs_right = subfigs[1].subplots(ncols=1, nrows=4, sharey=True)

masks = [None, "cloud_mask", "drizzle_mask", "rain_mask"]

# on the left axes plot the PSD
# on the right axes plot the MSD

psd = cc_constraint["particle_size_distribution"].where(
    cc_constraint["particle_size_distribution"] != 0, drop=True
)
msd = cc_constraint["mass_size_distribution"].where(
    cc_constraint["mass_size_distribution"] != 0, drop=True
)

linthresh_psd = 10 ** np.floor(np.log10(np.abs(psd.min().values)))
linthresh_msd = 10 ** np.floor(np.log10(np.abs(msd.min().values)))

for idx in np.arange(4):
    axs_left[idx].set_xscale("log")
    symlog_psd = mpl.scale.SymmetricalLogScale(
        axs_left[idx], base=10, linthresh=linthresh_psd, subs=None, linscale=0.1
    )
    axs_left[idx].set_yscale(symlog_psd)
    axs_right[idx].set_xscale("log")
    symlog_msd = mpl.scale.SymmetricalLogScale(
        axs_right[idx], base=10, linthresh=linthresh_msd, subs=None, linscale=0.1
    )
    axs_right[idx].set_yscale(symlog_msd)


for idx in np.arange(4):
    mask = masks[idx]
    if mask is None:
        psd_all = cc_constraint["particle_size_distribution"]  # * 1e9
        msd_all = cc_constraint["mass_size_distribution"]  # * 1e12
        psd = psd_all.where(psd_all != 0, drop=True)
        msd = msd_all.where(msd_all != 0, drop=True)
        mask = ""
    else:
        try:
            psd_all = cc_constraint["particle_size_distribution"].where(
                cc_constraint[mask] == 1, drop=True
            )  # * 1e9
            msd_all = cc_constraint["mass_size_distribution"].where(
                cc_constraint[mask] == 1, drop=True
            )  # * 1e12
            psd = psd_all.where(psd_all != 0, drop=True)
            msd = msd_all.where(msd_all != 0, drop=True)
        except TypeError as e:
            print(f"TypeError: in {mask}")

    # Plot PSD
    q66 = psd.quantile(0.66, dim="time")
    q33 = psd.quantile(0.33, dim="time")
    q10 = psd.quantile(0.1, dim="time")
    q90 = psd.quantile(0.9, dim="time")

    # plot
    axs_left[idx].fill_between(
        psd["diameter"],
        q66,
        q33,
        color=colors[idx],
        alpha=0.5,
        label=r"33-66% omiting 0s",
        zorder=3,
    )
    axs_left[idx].fill_between(
        psd["diameter"],
        q90,
        q10,
        color=colors[idx],
        alpha=0.2,
        label=r"10-90% omiting 0s",
        zorder=4,
    )

    axs_left[idx].plot(
        psd["diameter"],
        psd.median(dim="time", skipna=True),
        color=colors[idx],
        alpha=0.8,
        linewidth=2,
        label="Median omiting 0s",
        zorder=5,
        # marker=".",
    )

    axs_left[idx].plot(
        psd_all["diameter"],
        psd_all,
        color=adjust_lightness(colors[idx], 0.8),
        alpha=0.01,
        linewidth=0,
        marker=".",
        # label = f'Measurements',
        zorder=1,
    )

    axs_left[idx].set_title(f"PSD {mask}")

    # axs_right[idx].plot(
    #     msd["diameter"],
    #     msd,
    #     color=colors[idx],
    #     alpha=0.1,
    #     linewidth=0,
    #     marker=".",
    # )
    # axs_right[idx].set_title(f"MSD {mask}")

    # Plot MSD
    # Plot PSD
    q66 = msd.quantile(0.66, dim="time")
    q33 = msd.quantile(0.33, dim="time")
    q10 = msd.quantile(0.1, dim="time")
    q90 = msd.quantile(0.9, dim="time")

    # plot
    axs_right[idx].fill_between(
        msd["diameter"],
        q66,
        q33,
        color=colors[idx],
        alpha=0.5,
        label=r"33-66% omiting 0s",
        zorder=3,
    )
    axs_right[idx].fill_between(
        msd["diameter"],
        q90,
        q10,
        color=colors[idx],
        alpha=0.2,
        label=r"10-90% omiting 0s",
        zorder=4,
    )

    axs_right[idx].plot(
        msd["diameter"],
        msd.median(dim="time", skipna=True),
        color=colors[idx],
        alpha=0.8,
        linewidth=2,
        label="Median omiting 0s",
        zorder=5,
        # marker=".",
    )

    axs_right[idx].plot(
        msd_all["diameter"],
        msd_all,
        color=adjust_lightness(colors[idx], 0.8),
        alpha=0.01,
        linewidth=0,
        marker=".",
        # label = f'Measurements',
        zorder=1,
    )

    axs_right[idx].set_title(f"MSD {mask}")

axs_left[-1].set_xlabel("Particle diameter [µm]")
axs_right[-1].set_xlabel("Particle diameter [µm]")

for ax in axs_left.flatten():
    ax.set_ylabel("[#/L]")
    ax.legend()
for ax in axs_right.flatten():
    ax.set_ylabel("[g/L/µm]")
    ax.legend()

subfigs[0].suptitle(f"Particle size distribution from ATR measurements.")
subfigs[1].suptitle(f"Mass size distribution from ATR measurements.")
# fig.tight_layout()
fig.savefig(figure_dir / "art_sonde_psd_msd_masks.png", dpi=300, transparent=False)


# # %%
# logbins = np.logspace(-11, -4, 40)
# diameters = cloud_composite.diameter
# diameters = cloud_composite.diameter
# color_list = gen_color("Reds_r", n=len(diameters))

# # %%
# plot_colors(color_list)


# # %%
# def histo(array, **kwargs):
#     # print(f"received {type(array)} shape: {array.shape}") #, kwargs: {kwargs}")
#     result = np.histogram(
#         array,
#         **kwargs,
#     )
#     # print(f"result.shape: {result.shape}")
#     return result


# # %%
# counts, bin_edges = xr.apply_ufunc(
#     histo,
#     cloud_composite.mass_size_distribution.where(cloud_composite.mass_size_distribution != 0),
#     input_core_dims=[
#         ["time"],
#     ],
#     output_core_dims=[
#         ["bin_center"],
#         ["bin_edge"],
#     ],
#     # exclude_dims=set(('time',)),
#     output_dtypes=[float, float],
#     vectorize=True,
#     kwargs={"bins": logbins},
# )
# # %%
# # counts = counts / counts.std('bin_center')
# # counts = counts - counts.mean('bin_center')

# counts["bin_center"] = ("bin_center", (logbins[:-1] + logbins[1:]) / 2)

# # fig, ax = plt.subplots(figsize=(7.5, 7.5))
# # ax.set_xscale("log")
# # ax.set_yscale("log")
# # for idx, selected_diameter in enumerate(diameters) :
# #     ax.step()

# fig, ax = plt.subplots(figsize=(7.5, 7.5))
# ax.set_xscale("log")
# ax.plot(counts.bin_center, counts.T, color="w", alpha=0.2)
# # plt.colorbar()

# # %%
# logbins = np.logspace(-12, -1, 40)
# diameters = cloud_composite.diameter
# color_list = gen_color("Reds_r", n=len(diameters))
# fig, ax = plt.subplots(figsize=(7.5, 7.5))
# ax.set_xscale("log")
# ax.set_yscale("log")
# for idx, selected_diameter in enumerate(diameters):
#     cloud_composite.mass_size_distribution.where(cloud_composite.mass_size_distribution != 0).sel(
#         diameter=selected_diameter
#     ).plot.hist(ax=ax, bins=logbins, histtype="step", color=color_list[idx])


# # %%

# %%
