# %%
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sdm_eurec4a.reductions import polygon2mask
from sdm_eurec4a.visulization import set_custom_rcParams, plot_colors, gen_color

# %%
# Set custom colors

plt.style.use("dark_background")
colors = set_custom_rcParams()
plot_colors(colors)

figure_dir = Path("../results/first-data-analysis")
figure_dir.mkdir(exist_ok=True, parents=True)

# %%
def select_region(
    ds: xr.Dataset, area: list, lon_name: str = "lon", lat_name: str = "lat"
) -> xr.Dataset:
    """
    Select a region from a xarray dataset based on a given area.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to select from.
    area : list
        List of four values [lon_min, lon_max, lat_min, lat_max].
    lon_name : str, optional
        Name of the longitude variable. The default is 'lon'.
    lat_name : str, optional
        Name of the latitude variable. The default is 'lat'.
    Returns
    -------
    ds : xarray.Dataset
        Dataset with the selected region.
    """

    return (
        (ds[lon_name] > area[0])
        & (ds[lon_name] < area[1])
        & (ds[lat_name] > area[2])
        & (ds[lat_name] < area[3])
    )

# %% [markdown]
# ## 1. Load datasets

# %%
FILEPATH = Path("../data/observation/cloud_composite/processed/cloud_composite.nc")
cloud_composite = xr.open_dataset(FILEPATH)
display(cloud_composite)

FILEPATH = Path("../data/observation/dropsonde/Level_3/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc")
drop_sondes = xr.open_dataset(FILEPATH)
drop_sondes = drop_sondes.rename({"launch_time": "time"})
drop_sondes = drop_sondes.swap_dims({"sonde_id": "time"})
drop_sondes = drop_sondes.sortby("time")
display(drop_sondes)

# %%
drop_sondes.circle_diameter.plot()

# %% [markdown]
# ## 2. Get a first idea of the combined datasets

# %%
selection_area = [-58.75, -58.25, 13.5, 14]
selection_polygon = Polygon(
    [
        [selection_area[0], selection_area[2]],
        [selection_area[0], selection_area[3]],
        [selection_area[1], selection_area[3]],
        [selection_area[1], selection_area[2]],
        [selection_area[0], selection_area[2]],
    ]
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
ax.coastlines()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN, color="navy", alpha=0.2)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True)
ax.set_extent([-60, -56, 12, 15])

xx, yy = selection_polygon.exterior.xy
ax.plot(xx, yy, transform=ccrs.PlateCarree(), color="red", linewidth=2, label="selected area")

ax.scatter(
    cloud_composite.lon,
    cloud_composite.lat,
    transform=ccrs.PlateCarree(),
    # color = 'b',
    marker=".",
    alpha=0.7,
    label="ATR",
)

ax.scatter(
    drop_sondes.sel(time=halo_launches).flight_lon,
    drop_sondes.sel(time=halo_launches).flight_lat,
    transform=ccrs.PlateCarree(),
    label="HALO",
    marker="o",
    alpha=0.7,
)

ax.scatter(
    drop_sondes.sel(time=p3_launches).flight_lon,
    drop_sondes.sel(time=p3_launches).flight_lat,
    transform=ccrs.PlateCarree(),
    label="P3",
    marker="o",
    alpha=0.7,
)


ax.legend()
ax.set_title("Dropsonde and ATR locations overview")
fig.savefig(figure_dir / "sonde_ATR_locations_overview.png", dpi=300, transparent=False)

# %% [markdown]
# # 3. Selection criteria

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
# ### Temporal constraint: Use Flight number RF15 / 15

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

flight_constraint = cloud_composite.flight_number.isin([14, 15, 16, 17])
altitude_constraint = cloud_composite.alt < 1100
spatial_constraint = select_region(cloud_composite, [-58.75, -58.25, 13.5, 14])
drop_sondes_spatial_constraint = select_region(drop_sondes, [-59, -58, 13, 15])


mask_constraint = cloud_composite.rain_mask == 1
# liquid_water_content_constraint = cloud_composite.liquid_water_content > cloud_composite.liquid_water_content.quantile(dim="time", q=0.9)
full_constraint = (
    altitude_constraint
    & spatial_constraint
    & mask_constraint
    # &     flight_constraint 

)
time_values_of_constraint = cloud_composite.time.where(full_constraint, drop=True)
# For the cloud composite data set it is sufficient to use the time values of the constraint to select the data
cc_constraint = cloud_composite.sel(time=time_values_of_constraint)


# For the dropsonde data set we need to drop all sondes that do not match the spatial constraint
ds_constraint = drop_sondes.where(drop_sondes_spatial_constraint, drop=True)
# Then we need to drop all time values that do not match the time constraint
# Only unqiue values are kept
drop_sondes_time_constraint_all = ds_constraint.time.sel(
    time=time_values_of_constraint, method="nearest"
)
drop_sondes_time_constraint = np.unique(drop_sondes_time_constraint_all)

# either use only the time values of the constraint
# ds_constraint = ds_constraint.sel(time=time_values_of_constraint)
# or use the bounds set by the constraint
ds_constraint = ds_constraint.sel(time=slice(
    drop_sondes_time_constraint.min(),
    drop_sondes_time_constraint.max()
    )
)
# ds_constraint = ds_constraint.sel(time=drop_sondes_time_constraint)

# plt.scatter(
#     cc_constraint.time,
#     cc_constraint.alt,
#     c = cc_constraint.flight_number,
# )
# plt.colorbar()

print(f"{len(time_values_of_constraint)} ATR measurments are selected by the constraint")
print(f"{len(drop_sondes_time_constraint)} dropsondes are selected by the constraint")

# %%
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
    label=f"{len(drop_sondes_time_constraint)} dropsondes",
)
ax.legend(ncol=2, loc="upper center")
ax.set_ylim(-0.1, 0.2)
ax.set_yticks([0.1, 0], ["ATR", "Dropsondes"])
# ax.set_xticks(rotation=-45, ha="left");
ax.set_title(
    "Measurement times of ATR fitting the conditions.\nAnd temporal 'nearest' dropsondes in the selcted area."
)
fig.tight_layout()
fig.savefig(figure_dir / "conditions_ATR_sonde_times.png", dpi=300, transparent=False)

# %%
# Draw a map of all dropsondes released during the campaign


fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN, color="navy", alpha=0.2)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True)
ax.set_extent([-60, -56, 12, 15])
cm = plt.cm.get_cmap("RdYlBu")

xx, yy = selection_polygon.exterior.xy
ax.plot(xx, yy, transform=ccrs.PlateCarree(), color="red", linewidth=2, label="selected area")

mpl = ax.scatter(
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
ax.set_title("ATR locations fitting conditions and related dropsondes")
fig.savefig(figure_dir / "ATR_sonde_locations_condition.png", dpi=300, transparent=False)

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
    label=f"ATR all (#{all_measurments})",
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
    label=f"ATR cloud masked (#{cloud_mask_measurements})",
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
    label=f"ATR rain masked (#{rain_mask_measurements})",
)

axs.set_ylim(0, 2000)
axs.set_xlabel("Time")
axs.set_ylabel("Altitude (m)")
axs.legend()
axs.set_xticklabels(axs.get_xticklabels(), rotation=-45, ha="left")

axs.set_title("Potential temperature of dropsondes and ATR measurements\n")
fig.tight_layout()
fig.savefig(figure_dir / "ATR_sonde_potential_temperature.png", dpi=300, transparent=False)

# %%
fig, axs = plt.subplots(figsize=(15, 7.5), ncols=2, sharex=True)

#  Plot the particle_size_distribution for all and for the selected sondes

for ax in axs.flatten():
    ax.set_xscale("log")
    ax.set_yscale("symlog")

axs[0].set_xlabel("Particle diameter [µm]")
axs[0].set_ylabel(r"Particle size distribution [$10^{-9}$#/L]")
axs[0].set_title("Particle size distribution")
#  Plot the particle_size_distribution for all and for the selected sondes

axs[1].set_xlabel("particle diameter [µm]")
axs[1].set_ylabel(r"Mass size distribution [$10^{-9}$ g/L/µm]")
axs[1].set_title("Mass size distribution")


axs[0].plot(
    cc_constraint.diameter,
    cc_constraint.particle_size_distribution* 1e9,
    color=colors[0],
    alpha=0.1,
    linestyle="",
    marker=".",
    # label = f'individual measurements {q*100:.0f}ths percentile based on LWC'
)

axs[1].plot(
    cc_constraint.diameter,
    cc_constraint.mass_size_distribution * 1e9,
    color=colors[0],
    alpha=0.1,
    linestyle="",
    marker=".",
    # label = f'individual measurements {q*100:.0f}th percentile based on LWC'
)


axs[0].plot(
    cc_constraint.diameter,
    cc_constraint.particle_size_distribution.mean(dim="time")* 1e9,
    color="r",
    alpha=1,
    linewidth=2,
    label=f"mean",
)



axs[1].plot(
    cc_constraint.diameter,
    cc_constraint.mass_size_distribution.mean(dim="time")* 1e9,
    color="r",
    alpha=1,
    linewidth=2,
    label=f"mean",
)


for ax in axs.flatten():
    ax.legend()

fig.suptitle(f"particle size distribution and mass size distribution from ATR measurements.")
fig.savefig(figure_dir / "ATR_sonde_particle_size_distribution.png", dpi=300, transparent=False)

# %%
n, a = ds_constraint.theta.shape
# Create a color dict with individual colors for each day of the year which is present in the data set
color_n = np.unique([ds_constraint.time.dt.date]).astype(str)
color_list = gen_color("tab10", n=color_n.size)
plot_colors(color_list)
color_dict = dict(zip(color_n, color_list))
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
fig.savefig(figure_dir / "ATR_sonde_temperature_profiles.png", dpi=300, transparent=False)

# %%



