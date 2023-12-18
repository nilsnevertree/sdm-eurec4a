"""
First look at the observations
==============================
This notebook is a first look at the observations.
It is used to get a first idea of the data and to get a feeling for the data.
It is not used for the actual analysis.

The following data sets are used:
- ART measurements
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

# %%
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sdm_eurec4a.reductions import (
    latlon_dict_to_polygon,
    polygon2mask,
    rectangle_spatial_mask,
)
from sdm_eurec4a.visulization import gen_color, plot_colors, set_custom_rcParams
from shapely.geometry import Polygon


# %%
# Set custom colors

plt.style.use("dark_background")
colors = set_custom_rcParams()
plot_colors(colors)

figure_dir = Path("../results/first-data-analysis/all")
figure_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## 1. Load datasets

# %%
FILEPATH = Path("../data/observation/cloud_composite/processed/cloud_composite.nc")
cloud_composite = xr.open_dataset(FILEPATH)
display(cloud_composite)

FILEPATH = Path(r"../data/observation/dropsonde/Level_3/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc")
drop_sondes = xr.open_dataset(FILEPATH)
drop_sondes = drop_sondes.rename({"launch_time": "time"})
drop_sondes = drop_sondes.swap_dims({"sonde_id": "time"})
drop_sondes = drop_sondes.sortby("time")
display(drop_sondes)

# %% [markdown]
# ## 2. Get a first idea of the combined datasets

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


# %%
sonde_ids = drop_sondes.sonde_id.values
halo_ids = sonde_ids[["HALO" in v for v in sonde_ids]]
p3_ids = sonde_ids[["P3" in v for v in sonde_ids]]
# make sure no flight is used twice
assert sorted(np.concatenate([halo_ids, p3_ids])) == sorted(sonde_ids)

halo_launches = drop_sondes.time.where(drop_sondes.sonde_id.isin(halo_ids), drop=True)
p3_launches = drop_sondes.time.where(drop_sondes.sonde_id.isin(p3_ids), drop=True)

# %%

# Draw a map of all dropsondes released during the campaign


fig = plt.figure(figsize=(10, 10), layout="constrained")
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.OCEAN, color="navy", alpha=0.2)
# ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True, linestyle=":", alpha=0.4)
ax.set_extent([-60, -56, 12, 15])

xx, yy = latlon_dict_to_polygon(area_cloud_composite).exterior.xy
ax.plot(xx, yy, transform=ccrs.PlateCarree(), color="red", linewidth=2, label="ART selected area")

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
    cloud_composite.lon,
    cloud_composite.lat,
    transform=ccrs.PlateCarree(),
    # color = 'b',
    marker=".",
    alpha=0.7,
    label="ART",
)

ax.scatter(
    drop_sondes.sel(time=halo_launches).flight_lon,
    drop_sondes.sel(time=halo_launches).flight_lat,
    transform=ccrs.PlateCarree(),
    label="HALO",
    marker="+",
    color=colors[1],
    alpha=0.7,
)

ax.scatter(
    drop_sondes.sel(time=p3_launches).flight_lon,
    drop_sondes.sel(time=p3_launches).flight_lat,
    transform=ccrs.PlateCarree(),
    label="P3",
    marker="x",
    color=colors[1],
    alpha=0.7,
)


ax.legend()
ax.set_title("Dropsonde and ART locations overview")
# fig.savefig(figure_dir / "sonde_art_locations_overview.png", dpi=300, transparent=False)

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
# For an ART measurement to be considered,
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
"""
CONSTRAINTS ON THE DATASETS
"""
flight_constraint = cloud_composite.flight_number.isin([14, 15, 16])
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
    # flight_constraint
    altitude_constraint
    & spatial_constraint
    # & mask_constraint
)
time_values_of_constraint = cloud_composite.time.where(full_constraint, drop=True)

# %%
# For the cloud composite data set it is sufficient to use the time values of the constraint to select the data
cc_constraint = cloud_composite.sel(time=time_values_of_constraint)

# For the dropsonde data set we need select:
# 1. Match the spatial constraint
# 2. All time values that match the time constraint.
#    For this, the for each ART measurement the nearest dropsonde is selected.
# 3. Only use unqiue timevalus to avoid double counting
ds_constraint = drop_sondes.where(drop_sondes_spatial_constraint, drop=True)
drop_sondes_time_constraint_all = ds_constraint.time.sel(
    time=time_values_of_constraint, method="nearest"
)
drop_sondes_time_constraint = np.unique(drop_sondes_time_constraint_all)
ds_constraint = ds_constraint.sel(time=drop_sondes_time_constraint)

print(f"{len(time_values_of_constraint)} ART measurments are selected")
print(f"{len(drop_sondes_time_constraint)} dropsondes are selected")

# %%
# Draw a map of all dropsondes released during the campaign

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
ax.plot(xx, yy, transform=ccrs.PlateCarree(), color="red", linewidth=2, label="ART selected area")

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

mpl = ax.scatter(
    cc_constraint.lon,
    cc_constraint.lat,
    transform=ccrs.PlateCarree(),
    # c = cc_constraint.flight_number.values,
    marker="+",
    # alpha= 0.1,
    label="ART",
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
ax.set_title("ART locations fitting conditions and related dropsondes")
fig.savefig(figure_dir / "art_sonde_locations_condition.png", dpi=300, transparent=False)

# %%
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(
    time_values_of_constraint,
    time_values_of_constraint.astype(int) * 0 + 0.1,
    marker="+",
    linestyle="",
    label=f"{len(time_values_of_constraint)} ART measurments",
)

ax.plot(
    drop_sondes_time_constraint,
    drop_sondes_time_constraint.astype(int) * 0,
    marker="x",
    linestyle="",
    label=f"{len(drop_sondes_time_constraint)} dropsondes",
)
# ax.legend(ncol=2, loc="upper center")
ax.set_ylim(-0.05, 0.15)
ax.set_yticks(
    [
        0.1,
        0,
    ],
    [
        f"ART\n#{len(time_values_of_constraint)}",
        f"Dropsondes\n#{len(drop_sondes_time_constraint)}",
    ],
)
# ax.set_xticks(rotation=-45, ha="left")
ax.set_title(
    "Measurement times of ART fitting the conditions.\nAnd temporal 'nearest' dropsondes in the selcted area."
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
    label=f"{len(time_values_of_constraint)} ART",
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
    label=f"{len(cloud_time_values)} ART cloud",
)

ax.plot(
    drizzle_time_values,
    drizzle_time_values.astype(int) * 0 + 0.3,
    marker="+",
    linestyle="",
    label=f"{len(drizzle_time_values)} ART drizzle",
)

ax.plot(
    rain_time_values,
    rain_time_values.astype(int) * 0 + 0.4,
    marker="+",
    linestyle="",
    label=f"{len(rain_time_values)} ART rain",
)


# # Shrink current axis's height by 10% on the bottom
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

ax.set_ylim(-0.05, 0.45)
ax.set_yticks(
    [0.1, 0, 0.2, 0.3, 0.4],
    [
        f"ART\n#{len(time_values_of_constraint)}",
        f"Dropsondes\n#{len(drop_sondes_time_constraint)}",
        f"ART cloud\n#{len(cloud_time_values)}",
        f"ART drizzle\n#{len(drizzle_time_values)}",
        f"ART rain\n#{len(rain_time_values)}",
    ],
)
# ax.set_xticks(rotation=-45, ha="left")
ax.set_title(
    "Measurement times of ART fitting the conditions.\nAnd temporal 'nearest' dropsondes in the selcted area."
)
# ax.legend(bbox_to_anchor=(1.04, 0.5))
fig.tight_layout()
fig.savefig(figure_dir / "conditions_art_sonde_times_masks.png", dpi=300, transparent=False)
# %%


fig, axs = plt.subplots(1, 1, figsize=(15, 5))


xx, yy = np.meshgrid(
    ds_constraint.time,
    ds_constraint.alt,
)

mpl = axs.scatter(
    xx,
    yy,
    c=ds_constraint.theta.T,
    marker=".",
    s=1,
    # alpha= 0.1,
    zorder=1,
    vmin=297,
    vmax=305,
    cmap="Greens",
    label="Dropsondes",
)

plt.colorbar(mpl, label="Potential temperature (K)", orientation="vertical")

all_measurments = len(cc_constraint.time)
axs.plot(
    cc_constraint.time,
    cc_constraint.alt,
    marker=".",
    markersize=8,
    linestyle="",
    # alpha= 0.1,
    zorder=2,
    label=f"ART all (#{all_measurments})",
)
cloud_mask_measurements = (cc_constraint.cloud_mask == 1).sum().values
axs.plot(
    cc_constraint.time.where(cc_constraint.cloud_mask == 1),
    cc_constraint.alt.where(cc_constraint.cloud_mask == 1),
    marker="x",
    markersize=8,
    linestyle="",
    alpha=1,
    zorder=3,
    label=f"ART cloud masked (#{cloud_mask_measurements})",
)

rain_mask_measurements = (cc_constraint.rain_mask == 1).sum().values
axs.plot(
    cc_constraint.time.where(cc_constraint.rain_mask == 1),
    cc_constraint.alt.where(cc_constraint.rain_mask == 1),
    marker="o",
    linestyle="",
    markersize=8,
    alpha=1,
    zorder=4,
    label=f"ART rain masked (#{rain_mask_measurements})",
)

axs.set_ylim(0, 2000)
axs.set_xlabel("Time")
axs.set_ylabel("Altitude (m)")
axs.legend()
axs.set_xticklabels(axs.get_xticklabels(), rotation=-45, ha="left")

axs.set_title("Potential temperature of dropsondes and ART measurements\n")
fig.tight_layout()
fig.savefig(figure_dir / "art_sonde_potential_temperature.png", dpi=300, transparent=False)

# %%
cc_constraint = cloud_composite.sel(time=time_values_of_constraint)

fig, axs = plt.subplots(figsize=(15, 7.5), ncols=2, sharex=True)

#  Plot the particle_size_distribution for all and for the selected sondes

axs[0].plot(
    cc_constraint.diameter,
    cc_constraint.particle_size_distribution,
    color=colors[0],
    alpha=0.1,
    linewidth=0,
    marker=".",
    # label = f'individual measurements {q*100:.0f}ths percentile based on LWC'
)

axs[0].set_xscale("log")
axs[0].plot(
    cloud_composite.diameter,
    cloud_composite.particle_size_distribution.mean(dim="time", skipna=True),
    color="r",
    alpha=1,
    linewidth=2,
    # marker = '.',
    label=f"mean",
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
    cloud_composite.diameter,
    cloud_composite.mass_size_distribution.mean(dim="time", skipna=True),
    color="r",
    alpha=1,
    linewidth=2,
    label=f"mean",
)

axs[1].set_xlabel("Particle diameter [µm]")
axs[1].set_ylabel("Mass size distribution [g/L/µm]")
axs[1].set_title("Mass size distribution")

for ax in axs.flatten():
    ax.legend()
    ax.set_yscale("log")

fig.suptitle(f"Particle size distribution and mass size distribution from ART measurements.")
fig.savefig(figure_dir / "art_sonde_particle_size_distribution.png", dpi=300, transparent=False)


# %%
n, a = ds_constraint.theta.shape
# Create a color dict with individual colors for each day of the year which is present in the data set
color_n = np.unique([ds_constraint.time.dt.date]).astype(str)
color_list = gen_color("tab10", n=color_n.size)
plot_colors(color_list)
color_dict = dict(zip(color_n, color_list))

# %%
# Set style
style = dict(linewidth=1, marker=".", s=1, linestyle="-", alpha=1, cmap="Set2")


# Plot the temperature profiles for the selected sondes and color them by their day of the year value
fig, axs = plt.subplots(1, 1, figsize=(7.5, 7.5))

xx = np.tile(ds_constraint.alt, (n, 1))
cc = np.tile(ds_constraint.time.dt.day, (a, 1)).T
# ds_constraint.theta.shape
# mpl = axs.scatter(
#     ds_constraint.theta,
#     xx,
#     c = cc,
#     **style)
old_day = None
for i, t in enumerate(ds_constraint.time):
    day = str(t.dt.date.values)
    color = color_dict[day]
    if old_day != day:
        axs.plot(
            ds_constraint.theta.sel(time=t),
            ds_constraint.alt,
            color=color,
            label=f"{day}",
        )
    else:
        axs.plot(
            ds_constraint.theta.sel(time=t),
            ds_constraint.alt,
            color=color,
        )
    old_day = day
axs.legend(loc="lower right")

axs.set_ylim(0, 2000)
axs.set_xlim(297, 305)
plt.colorbar(mpl)
axs.set_xlabel("Potential Temperature [K]")
axs.set_ylabel("Altitude [m]")
axs.set_title("Temperature profiles of realted dropsondes\nColored by date")
fig.savefig(figure_dir / "art_sonde_temperature_profiles.png", dpi=300, transparent=False)


# %%
logbins = np.logspace(-11, -4, 40)
diameters = cloud_composite.diameter
diameters = cloud_composite.diameter
color_list = gen_color("Reds_r", n=len(diameters))

# %%
plot_colors(color_list)


# %%
def histo(array, **kwargs):
    # print(f"received {type(array)} shape: {array.shape}") #, kwargs: {kwargs}")
    result = np.histogram(
        array,
        **kwargs,
    )
    # print(f"result.shape: {result.shape}")
    return result


# %%
counts, bin_edges = xr.apply_ufunc(
    histo,
    cloud_composite.mass_size_distribution.where(cloud_composite.mass_size_distribution != 0),
    input_core_dims=[
        ["time"],
    ],
    output_core_dims=[
        ["bin_center"],
        ["bin_edge"],
    ],
    # exclude_dims=set(('time',)),
    output_dtypes=[float, float],
    vectorize=True,
    kwargs={"bins": logbins},
)
# %%
# counts = counts / counts.std('bin_center')
# counts = counts - counts.mean('bin_center')

counts["bin_center"] = ("bin_center", (logbins[:-1] + logbins[1:]) / 2)

# fig, ax = plt.subplots(figsize=(7.5, 7.5))
# ax.set_xscale("log")
# ax.set_yscale("log")
# for idx, selected_diameter in enumerate(diameters) :
#     ax.step()

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.set_xscale("log")
ax.plot(counts.bin_center, counts.T, color="w", alpha=0.2)
# plt.colorbar()

# %%
logbins = np.logspace(-12, -1, 40)
diameters = cloud_composite.diameter
color_list = gen_color("Reds_r", n=len(diameters))
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.set_xscale("log")
ax.set_yscale("log")
for idx, selected_diameter in enumerate(diameters):
    cloud_composite.mass_size_distribution.where(cloud_composite.mass_size_distribution != 0).sel(
        diameter=selected_diameter
    ).plot.hist(ax=ax, bins=logbins, histtype="step", color=color_list[idx])


# %%
