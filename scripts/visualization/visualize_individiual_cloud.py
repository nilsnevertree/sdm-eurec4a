# %%
import os

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sdm_eurec4a.visulization import set_custom_rcParams


# %%
default_colors = set_custom_rcParams()
# Example dataset
script_path = os.path.abspath(__file__)
print(script_path)

REPOSITORY_ROOT = Path(script_path).parent.parent.parent
print(REPOSITORY_ROOT)

output_dir = REPOSITORY_ROOT / Path("data/model/input_examples/")
output_dir.mkdir(parents=True, exist_ok=True)

fig_path = REPOSITORY_ROOT / Path("results/individual_clouds/")
fig_path.mkdir(parents=True, exist_ok=True)

# %%

# -------
# Load data
# -------

identified_clouds = xr.open_dataset(
    REPOSITORY_ROOT / Path("data/observation/cloud_composite/processed/identified_clouds_more.nc")
)
# select only clouds which are between 800 and 1100 m
identified_clouds = identified_clouds.where(
    (identified_clouds.alt >= 800) & (identified_clouds.alt <= 1100), drop=True
)

distance_IC_DS = xr.open_dataset(
    REPOSITORY_ROOT / Path("data/observation/combined/distance/distances_IC_DS.nc")
)

cloud_composite = xr.open_dataset(
    REPOSITORY_ROOT / Path("data/observation/cloud_composite/processed/cloud_composite.nc"),
    chunks={"time": 1000},
)


drop_sondes = xr.open_dataset(
    REPOSITORY_ROOT
    / Path("data/observation/dropsonde/Level_3/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc")
)
drop_sondes = drop_sondes.rename({"launch_time": "time"})
drop_sondes = drop_sondes.swap_dims({"sonde_id": "time"})
drop_sondes = drop_sondes.sortby("time")
drop_sondes = drop_sondes.chunk({"time": -1})
# %%

# -----
# Plotting relation of duration and LWC of clouds
# -----

plt.scatter(
    identified_clouds.duration.dt.seconds.astype(int),
    identified_clouds.liquid_water_content,
)
plt.xlabel("Duration in s")
plt.ylabel("LWC in g/m3")
plt.title("LWC vs duration of all cloud events")
# %%
head_number = 20
cloud_times = identified_clouds.time.sortby(identified_clouds.duration)[-head_number - 1 :]
time_slices = [
    cloud_composite.time.sel(
        time=slice(
            identified_clouds.start.sel(time=cloud_time), identified_clouds.end.sel(time=cloud_time)
        )
    ).data
    for cloud_time in cloud_times
]
time_slice = np.concatenate(time_slices, axis=0)
cloud_selection = cloud_composite.sel(time=time_slice, drop=True)

# %%

# -----
# Plot map of selected clouds and realted ATR measurements and dropsondes
# -----

fig, axs = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10, 4),
    subplot_kw={"projection": ccrs.PlateCarree()},
    sharex=True,
    sharey=True,
)

gl0 = axs[0].gridlines(draw_labels=True)
gl0.top_labels = False
gl0.right_labels = False

gl1 = axs[1].gridlines(draw_labels=True)
gl1.top_labels = False
gl1.left_labels = False


colors = mdates.date2num(cloud_selection.time)

sc = axs[0].scatter(
    cloud_selection.lon,
    cloud_selection.lat,
    c=colors,
    cmap="plasma",
)
loc = mdates.AutoDateLocator()
cbar = plt.colorbar(
    mappable=sc,
    ax=axs[0],
    label="Time",
    ticks=loc,
    format=mdates.AutoDateFormatter(loc, defaultfmt="%y-%m-%d"),
    orientation="horizontal",
)
cbar.ax.tick_params(rotation=-35)

colors = cloud_selection.alt

sc = axs[1].scatter(
    cloud_selection.lon,
    cloud_selection.lat,
    c=colors,
    cmap="plasma",
)
plt.colorbar(mappable=sc, ax=axs[1], label="Altitude in m", orientation="horizontal")

fig.suptitle(f"{head_number} identified clouds with longest duration", fontsize=16)

fig.savefig(fig_path / Path("identified_clouds_longest_duration.svg"), dpi=300, bbox_inches="tight")

# %%

# -----
# Plot map of identified cloud and realted ATR measurements and dropsondes

fig, axs = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10, 4),
    subplot_kw={"projection": ccrs.PlateCarree()},
    sharex=True,
    sharey=True,
)

gl0 = axs[0].gridlines(draw_labels=True)
gl0.top_labels = False
gl0.right_labels = False

gl1 = axs[1].gridlines(draw_labels=True)
gl1.top_labels = False
gl1.left_labels = False

sorted_ds = identified_clouds.sortby(identified_clouds.duration)
sorted_ds = sorted_ds.isel(time=slice(-head_number, None))
colors = mdates.date2num(sorted_ds.time)

sc = axs[0].scatter(
    sorted_ds.lon,
    sorted_ds.lat,
    c=colors,
    cmap="plasma",
)
loc = mdates.AutoDateLocator()
cbar = plt.colorbar(
    mappable=sc,
    ax=axs[0],
    label="Time",
    ticks=loc,
    format=mdates.AutoDateFormatter(loc, defaultfmt="%y-%m-%d"),
    orientation="horizontal",
)
cbar.ax.tick_params(rotation=-35)

colors = sorted_ds.alt

sc = axs[1].scatter(
    sorted_ds.lon,
    sorted_ds.lat,
    c=colors,
    cmap="plasma",
)
plt.colorbar(mappable=sc, ax=axs[1], label="Altitude in m", orientation="horizontal")

fig.suptitle(f"{head_number} identified clouds with longest duration", fontsize=16)

fig.savefig(fig_path / Path("identified_clouds_longest_duration_mean.svg"), dpi=300, bbox_inches="tight")

# %%
# -----
# Chosing and individual cloud
# -----

chosen_id = 1421

# extract the cloud
single_cloud = identified_clouds.sel(time=identified_clouds.cloud_id == chosen_id)
# realted ATR measurements
chosen_cloud_composite = cloud_composite.sel(
    time=slice(single_cloud.start[0], single_cloud.end[0]), drop=True
)

# Identify dropsondes which are close to the cloud

# get distance look up table for this cloud
single_distances = distance_IC_DS.sel(time_identified_clouds=single_cloud.time.data)

# set the maximum distance for the dropsondes
max_spatial_distance = 100  # km
max_temporal_distance = np.timedelta64(1, "h")

# select the time of the dropsondes which are close to the cloud
allowed_dropsonde_times = single_distances.where(
    (np.abs(single_distances.temporal_distance) <= max_temporal_distance)
    & (single_distances.spatial_distance <= max_spatial_distance),
    drop=True,
).time_drop_sondes

# select the dropsondes which are close to the cloud
chosen_dropsondes = drop_sondes.sel(time=allowed_dropsonde_times.data, drop=True)

# lets store the example input for the model
chosen_dropsondes.to_netcdf(output_dir / Path(f"dropsondes_{chosen_id}.nc"))
chosen_cloud_composite.to_netcdf(output_dir / Path(f"cloud_composite_{chosen_id}.nc"))


# %%

# -----
# Plot map of identified cloud and realted ATR measurements and dropsondes
# -----

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.gridlines(draw_labels=True)
ax.set_extent([-60, -56, 12, 15])
cm = plt.cm.get_cmap("RdYlBu")

ax.scatter(
    single_cloud.lon,
    single_cloud.lat,
    transform=ccrs.PlateCarree(),
    c=default_colors[0],
    marker="+",
    s=100,
    label=f"Cloud {chosen_id}",
)

ax.scatter(
    chosen_cloud_composite.lon,
    chosen_cloud_composite.lat,
    transform=ccrs.PlateCarree(),
    c=default_colors[0],
    marker=".",
    alpha=0.5,
    s=10,
    label=f"ATR measurments (#{chosen_cloud_composite.time.size}))",
)

ax.scatter(
    chosen_dropsondes.flight_lon,
    chosen_dropsondes.flight_lat,
    transform=ccrs.PlateCarree(),
    marker="x",
    c=default_colors[1],
    label=f"Dropsondes (#{chosen_dropsondes.time.size})",
)

ax.legend(loc="lower right")
ax.set_title(
    f"Cloud {chosen_id} ({single_cloud.time.dt.strftime('%Y/%m/%d %H:%M:%S')[0].data})\n with corresponding ATR and dropsonde release location"
)
fig.savefig(fig_path / Path(f"selected_cloud_dropsondes_{chosen_id}.svg"), bbox_inches="tight")
fig.savefig(fig_path / Path(f"selected_cloud_dropsondes_{chosen_id}.png"), dpi=300, bbox_inches="tight")

# %%
# -----
# Plot the temperature profiles for the selected sondes and color them by their day of the year value
# -----

style = dict(linewidth=0.8, linestyle="-", alpha=0.8)

print("Plotting selected drop sondes")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5), sharey=True)

axs_theta = axs[0]
axs_q = axs[1]
# ds_constraint.theta.shape
old_day = None
for i, t in enumerate(chosen_dropsondes.time):
    axs_theta.plot(chosen_dropsondes.theta.sel(time=t), chosen_dropsondes.alt, color="r", **style)
    axs_q.plot(chosen_dropsondes.q.sel(time=t), chosen_dropsondes.alt, color="b", **style)


for ax in axs.flatten():
    ax.axhline(single_cloud.alt, color="k", linestyle="--", label="Cloud altitude")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 2000)
    ax.set_ylabel("Altitude [m]")

axs[0].set_xlim(297, 305)
axs[0].set_xlabel("Potential Temperature [K]")
axs[1].set_xlim(0, 0.025)
axs[1].set_xlabel("Specific humidity in [kg / kg]")

axs[0].set_title("Potential temperature")
axs[1].set_title("Specific humidity")

fig.suptitle(
    f"Cloud {chosen_id} ({single_cloud.time.dt.strftime('%Y/%m/%d %H:%M:%S')[0].data})\nMeasurements of related dropsonde which are close to the clouds by dx<{max_spatial_distance}km dt<{max_temporal_distance}"
)
fig.savefig(fig_path / Path(f"selected_cloud_dropsondes_profiles_{chosen_id}.svg"), bbox_inches="tight")
fig.savefig(
    fig_path / Path(f"selected_cloud_dropsondes_profiles_{chosen_id}.png"), dpi=300, bbox_inches="tight"
)

# %%

# -----
# Plot the temperature profiles for the selected sondes and color them by their day of the year value
# -----

print("Plotting selected ATR measurments")
fig, axs = plt.subplots(figsize=(10, 6), ncols=2, sharex=True)

#  Plot the particle_size_distribution for all and for the selected sondes
axs[0].set_xscale("log")
axs[0].set_xlabel("Particle diameter [µm]")
axs[0].set_ylabel("Particle size distribution [#/L]")
axs[0].set_title("Particle size distribution")
#  Plot the particle_size_distribution for all and for the selected sondes


psd = chosen_cloud_composite["particle_size_distribution"].where(
    chosen_cloud_composite["particle_size_distribution"] != 0, drop=True
)
msd = chosen_cloud_composite["mass_size_distribution"].where(
    chosen_cloud_composite["mass_size_distribution"] != 0, drop=True
)

linthresh_psd = 10 ** (np.floor(np.log10(np.abs(psd.min().values))) - 1)
linthresh_msd = 10 ** (np.floor(np.log10(np.abs(msd.min().values))) - 1)

axs[0].set_xscale("log")
symlog_psd = mpl.scale.SymmetricalLogScale(
    axs[0], base=10, linthresh=linthresh_psd, subs=None, linscale=0.2
)
axs[0].set_yscale(symlog_psd)

axs[1].set_xscale("log")
symlog_psd = mpl.scale.SymmetricalLogScale(
    axs[1], base=10, linthresh=linthresh_msd, subs=None, linscale=0.2
)
axs[1].set_yscale(symlog_psd)


axs[0].plot(
    chosen_cloud_composite.diameter,
    chosen_cloud_composite.particle_size_distribution,
    alpha=0.75,
    linewidth=0.2,
    marker=".",
    # label = f'individual measurements {q*100:.0f}th percentile based on LWC'
)

axs[1].plot(
    chosen_cloud_composite.diameter,
    chosen_cloud_composite.mass_size_distribution,
    alpha=0.75,
    linewidth=0.2,
    marker=".",
    # label = f'individual measurements {q*100:.0f}th percentile based on LWC'
)

axs[1].set_xlabel("Particle diameter [µm]")
axs[1].set_ylabel("Mass size distribution [g/L/µm]")
axs[1].set_title("Mass size distribution")

for ax in axs.flatten():
    # ax.legend()
    ax.set_ylim(0, None)


fig.suptitle(
    f"Cloud {chosen_id} ({single_cloud.time.dt.strftime('%Y/%m/%d %H:%M:%S')[0].data})\nMeasurements of ATR\n"
)
fig.tight_layout()
fig.savefig(fig_path / Path(f"selected_cloud_ATR_{chosen_id}.svg"), bbox_inches="tight")
fig.savefig(fig_path / Path(f"selected_cloud_ATR_{chosen_id}.png"), dpi=300, bbox_inches="tight")


# %% More ideas on how to identify clouds
# # %%
# # print((ds.cloud_mask == True).sum().compute())
# # print((ds.drizzle_mask == True).sum().compute())
# # print((ds.rain_mask == True).sum().compute())

# min_duration_new = 5
# cm_new = cm_org.fillna(0).astype(bool)
# cm_new = consecutive_events_xr(cm_new, min_duration = min_duration_new, axis="time")
# with ProgressBar():
#         cm_new = cm_new.compute()

# #  %%
# window_width = 10
# necessary_true = 3
# with ProgressBar():
#         cm_rolling = (cm_org.rolling(time = window_width, center = True).sum('time', center = True) >= necessary_true)
#         cm_rolling_new = (cm_new.rolling(time = window_width, center = True).sum('time', center = True) >= necessary_true)

# min_duration_rolling = 15
# cm_rolling_consecutive = consecutive_events_xr(cm_rolling, min_duration = min_duration_rolling, axis="time",)
# # %%
# print(cm_org.sum().compute())
# print(cm_rolling.sum().compute())
# print(cm_rolling_consecutive.sum().compute())
# print(cm_new.sum().compute())
# print(cm_rolling_new.sum().compute())
# # %%
# # interesting example

# temp_slice = slice("2020-01-26 18:10:10", "2020-01-26 18:11:10")

# fig, axs = plt.subplots(nrows = 5, ncols = 1, figsize=(10, 5), sharex=True, sharey=True)
# cm_org.sel(time = temp_slice).plot(ax = axs[0], marker = 'o')
# axs[0].set_title('Original cloud mask')

# for ax in axs[1:] :
#         cm_org.sel(time = temp_slice).plot(ax = ax, marker = 'o', alpha = 0.5)
# cm_rolling.sel(time = temp_slice).plot(ax = axs[1], marker = 'o')
# axs[1].set_title(f'Rolling window applied of {window_width} time steps. At least {necessary_true} time steps have to be true')

# cm_rolling_consecutive.sel(time = temp_slice).plot(ax = axs[2], marker = 'o')
# axs[2].set_title(f'Rolling selection applied. Then consecutive events of at least  15 time steps after rolling window')

# cm_new.sel(time = temp_slice).plot(ax = axs[3], marker = 'o')
# axs[3].set_title(f'Consecutive events of at least {min_duration_new} time steps')

# cm_rolling_new.sel(time = temp_slice).plot(ax = axs[4], marker = 'o')
# axs[4].set_title('consecutive events then rolling window applied')

# fig.tight_layout()
# # %%


# #  %%
# print("Plotting selected drop sondes and ATR locations")
# fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree()}, sharex=True, sharey=True)
# axs = axs.flatten()

# for ax in axs :
#     ax.gridlines(draw_labels=False)
#     ax.set_extent([-58.5, -58, 12.5, 14])


# for ax, mask, c_title in zip(
#        axs,
#        [cm_org, cm_new, cm_rolling, cm_rolling_new],
#        ["Original", "Consecutive events", "Rolling window", "Rolling window and consecutive events"]
#        ) :

#         colors = mdates.date2num(cloud_composite.time.sel(time = mask.time[mask == True]))
#         sc = ax.scatter(
#         cloud_composite.lon.sel(time = mask.time[mask == True]),
#                 cloud_composite.lat.sel(time = mask.time[mask == True]),
#                 transform=ccrs.PlateCarree(),
#                 c = colors,
#                 marker="+",
#                 cmap="plasma"
#         )
#         loc = mdates.AutoDateLocator()
#         ax.set_title(f"{c_title}\n#True: {mask.sum().compute().data}")
#         # plt.colorbar(mappable=sc, ax=ax, label="time", ticks=loc, format=mdates.AutoDateFormatter(loc), orientation="horizontal")

# fig.suptitle("Cloud identification methods", fontsize=16)
# fig.tight_layout()

# # %%
# cm = cm_org
# cloud_diff = cm.fillna(0).astype(int).diff(dim="time")
# cm.plot(marker = 'o')
# cloud_diff.plot(marker = 'o')
# cloud_start = cloud_diff.time.where(cloud_diff == 1, drop = True)
# cloud_end = cloud_diff.time.where(cloud_diff == -1, drop = True)
# # cloud_length = cloud_end cloud_start

# # %%
# identified_clouds = xr.Dataset(
#         coords = {"cloud_id": np.arange(0, cloud_start.size)},
#         data_vars = {
#                 "start": ('cloud_id', cloud_start.data),
#                 "end": ('cloud_id', cloud_end.data),
#         }
# )
# identified_clouds['duration'] = identified_clouds.end - identified_clouds.start
# identified_clouds['mid_time'] = identified_clouds.start + identified_clouds.duration/2
# identified_clouds = identified_clouds.assign_coords({'time': identified_clouds.mid_time})
# identified_clouds = identified_clouds.swap_dims({'cloud_id' : 'time'})
# identified_clouds['selection'] = ('time', [slice(start, end) for start, end in zip(identified_clouds.start.data, identified_clouds.end.data)])
# identified_clouds
# # %%

# # temp_slice = cloud_start.time.isel(time = slice(0,2))
# temp_slice = slice("2020-01-26 18:10:10", "2020-01-26 18:11:10")

# fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize=(10, 5), sharex=True, sharey=True)
# cm_org.sel(time = temp_slice).plot(ax = axs[0], marker = 'o')
# axs[0].set_title('Original cloud mask')

# for ax in axs[1:] :
#         cm_org.sel(time = temp_slice).plot(ax = ax, marker = 'o', alpha = 0.5)

# cm_rolling.sel(time = temp_slice).plot(ax = axs[1], marker = 'o')
# axs[1].set_title(f'Rolling window applied of {window_width} time steps. At least {necessary_true} time steps have to be true')

# # cm_rolling_consecutive.sel(time = temp_slice).plot(ax = axs[2], marker = 'o')
# # axs[2].set_title(f'Rolling selection applied. Then consecutive events of at least  15 time steps after rolling window')

# cm_new.sel(time = temp_slice).plot(ax = axs[2], marker = 'o')
# axs[2].plot(
#         (identified_clouds.start.sel(time = temp_slice)),
#         (identified_clouds.start.sel(time = temp_slice)).astype(int) *0 + 1,
#         marker = 'X')
# axs[2].plot(
#         (identified_clouds.end.sel(time = temp_slice)),
#         (identified_clouds.start.sel(time = temp_slice)).astype(int) *0 - 1,
#         marker = 'X')
# axs[2].set_title(f'Consecutive events of at least {min_duration_new} time steps')

# for ax in axs[:-1]:
#         ax.set_xlabel("")

# # cm_rolling_new.sel(time = temp_slice).plot(ax = axs[4], marker = 'o')
# # axs[4].set_title('consecutive events then rolling window applied')

# # axs[3].plot(cloud_start.sel(time = temp_slice), cloud_start.sel(time = temp_slice).astype(int) *0 + 1, marker = 'o', color = 'red')
# # axs[3].plot(cloud_end.sel(time = temp_slice), cloud_end.sel(time = temp_slice).astype(int) *0 + 1, marker = 'o', color = 'blue')
# # cloud_start.sel(time = temp_slice).plot(ax = axs[2], marker = 'o', color = 'green')
# # cloud_end.sel(time = temp_slice).plot(ax = axs[2], marker = 'o', color = 'green')


# # %%

# temp_slice = slice("2020-01-26 17:35:10", "2020-01-26 18:11:10")

# fig, axs = plt.subplots(nrows = 5, ncols = 1, figsize=(10, 5), sharex=True, sharey=True)
# cm_org.sel(time = temp_slice).plot(ax = axs[0], marker = 'o')
# axs[0].set_title('Original cloud mask')

# for ax in axs[1:] :
#         cm_org.sel(time = temp_slice).plot(ax = ax, marker = 'o', alpha = 0.5)
# cm_rolling.sel(time = temp_slice).plot(ax = axs[1], marker = 'o')
# axs[1].set_title(f'Rolling window applied of {window_width} time steps. At least {necessary_true} time steps have to be true')

# cm_rolling_consecutive.sel(time = temp_slice).plot(ax = axs[2], marker = 'o')
# axs[2].set_title(f'Rolling selection applied. Then consecutive events of at least  15 time steps after rolling window')

# cm_new.sel(time = temp_slice).plot(ax = axs[3], marker = 'o')
# axs[3].set_title(f'Consecutive events of at least {min_duration_new} time steps')

# cm_rolling_new.sel(time = temp_slice).plot(ax = axs[4], marker = 'o')
# axs[4].set_title('consecutive events then rolling window applied')

# axs[3].plot(cloud_start.sel(time = temp_slice), cloud_start.sel(time = temp_slice).astype(int) *0 + 2, marker = 'o', color = 'red')
# axs[3].plot(cloud_end.sel(time = temp_slice), cloud_end.sel(time = temp_slice).astype(int) *0 + 2, marker = 'o', color = 'blue')

# fig.tight_layout()
# # %%
