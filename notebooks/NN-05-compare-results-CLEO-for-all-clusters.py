# %% [markdown]
# With this Notebook, the datasets which were created by CLEO of all clusters using rain mask with 5 timestep holes removed will be compared.
# 

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

from sdm_eurec4a.visulization import set_custom_rcParams, adjust_lightness_array, handler_map_alpha

from sdm_eurec4a.identifications import (
    match_clouds_and_cloudcomposite,
    match_clouds_and_dropsondes,
    select_individual_cloud_by_id,
)

from sdm_eurec4a import RepositoryPath
from sdm_eurec4a.input_processing import transfer
from sdm_eurec4a.reductions import shape_dim_as_dataarray
from sdm_eurec4a.conversions import msd_from_psd

# %%
def adjust_spines(ax, visible_spines, position=("outward", 5)):
    ax.label_outer(remove_inner_ticks=False)

    for loc, spine in ax.spines.items():
        if loc in visible_spines:
            spine.set_position(position)  # outward by 10 points
        else:
            spine.set_visible(False)

# %%
plt.style.use("default")
default_colors = set_custom_rcParams()
darker_colors = adjust_lightness_array(default_colors, 0.75)

REPOSITORY_ROOT = RepositoryPath("levante").get_repo_dir()

output_dir = REPOSITORY_ROOT / Path("data/model/no_aerosols/")
# output_dir.mkdir(parents=True, exist_ok=True)

# %%
config_yaml_filepath = REPOSITORY_ROOT / Path("data/model/input/new/clusters_18.yaml")

with open(config_yaml_filepath, "r") as file:
    config_yaml = yaml.safe_load(file)

identification_type = config_yaml["cloud"]["identification_type"]
cloud_id = config_yaml["cloud"]["cloud_id"]

# %%
path2CLEO = Path("/home/m/m301096/CLEO")
cleo_data_dir = path2CLEO / "data/output"

# cleo_dataset_dir = cleo_data_dir / "processed/long_run/" f"{identification_type}_{cloud_id}"
cleo_dataset_dir = cleo_data_dir / "raw/no_aerosols/" f"{identification_type}_{cloud_id}"

cleo_output_path = cleo_dataset_dir / "full_dataset.nc"

# fig_path = REPOSITORY_ROOT / Path(f"results/CLEO_output/long_run/{identification_type}_{cloud_id}")
fig_path = REPOSITORY_ROOT / Path(f"results/CLEO_output/no_aerosols/{identification_type}_{cloud_id}")
fig_path.mkdir(parents=True, exist_ok=True)

# %%
clusters = xr.open_dataset(
    REPOSITORY_ROOT
    / Path(
        "data/observation/cloud_composite/processed/identified_clouds/identified_clusters_rain_mask_5.nc"
    )
)
cluster = select_individual_cloud_by_id(clusters, cloud_id)

distance_clusters = xr.open_dataset(
    REPOSITORY_ROOT
    / Path(f"data/observation/combined/distance/distance_dropsondes_identified_clusters_rain_mask_5.nc")
)

cloud_composite = xr.open_dataset(
    REPOSITORY_ROOT / Path("data/observation/cloud_composite/processed/cloud_composite_si_units.nc"),
    chunks={"time": 1000},
)

cloud_composite = match_clouds_and_cloudcomposite(
    ds_clouds=cluster,
    ds_cloudcomposite=cloud_composite,
    dim="time",
)

drop_sondes = xr.open_dataset(
    REPOSITORY_ROOT / Path("data/observation/dropsonde/processed/drop_sondes.nc")
)


dt = config_yaml["cloud"]["dropsonde_distance"]["max_temporal_distance"].split(" ")
max_temporal_distance = np.timedelta64(int(dt[0]), dt[1][0])
max_spatial_distance = config_yaml["cloud"]["dropsonde_distance"]["max_spatial_distance"]
drop_sondes = match_clouds_and_dropsondes(
    ds_clouds=cluster,
    ds_sonde=drop_sondes,
    ds_distance=distance_clusters,
    max_temporal_distance=max_temporal_distance,
    max_spatial_distance=max_spatial_distance,
)

# %% [markdown]
# 
# ### Load CLEO output and preprocess
# 
# - Convert Multiplicity $\xi$ from #/gridbox to #/m^3
# - calculate mass of each SD and mass represented in total by each SD 

# %%
ds_cleo = xr.open_dataset(cleo_output_path)
ds_cleo["radius"] = ds_cleo["radius"] * 1e-6
ds_cleo["mass"] = 4 / 3 * np.pi * ds_cleo["radius"] ** 3 * 1000  # kg/m^3

ds_cleo["xi_per_gridbox"] = ds_cleo["xi"]
ds_cleo["xi"] = ds_cleo["xi_per_gridbox"] / 20**3
# create total represented mass
ds_cleo["mass_represented"] = ds_cleo["mass"] * ds_cleo["xi"]

# %% [markdown]
# Reconstruct the fitted distribution

# %%
parameters = config_yaml["particle_size_distribution"]["parameters"]

psd = transfer.PSD_LnNormal(
    geometric_means=parameters["geometric_means"],
    geometric_sigmas=parameters["geometric_sigmas"],
    scale_factors=parameters["scale_factors"],
)

cloud_base = config_yaml["thermodynamics"]["air_temperature"]["parameters"]["x_split"][0]

# %% [markdown]
# Calculate the mass size distribution

# %%
cloud_composite["mass_size_distribution"] = msd_from_psd(cloud_composite)
cloud_composite["particle_size_distribution_fit"] = psd.eval_func(cloud_composite.radius)
cloud_composite["mass_size_distribution_fit"] = msd_from_psd(
    cloud_composite, psd_name="particle_size_distribution_fit"
)
# get the 2D radius
cloud_composite["radius_2D"] = shape_dim_as_dataarray(
    cloud_composite["particle_size_distribution"], output_dim="radius"
)

# %% [markdown]
# Plot the total number concentration

# %% [markdown]
# ## Comparing $N$ and $LWC$ of CLEO initilization to measurements and the fitted distribution

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex=True)


# Plot the total number concentration
axs[0].plot(
    cloud_composite["particle_size_distribution"].time,
    cloud_composite["particle_size_distribution"].sum(dim="radius"),
    label="ATR measurement",
)
axs[0].axhline(
    cloud_composite["particle_size_distribution_fit"].sum(dim="radius"),
    label="Fitted distribution",
    color=default_colors[1],
)
axs[0].axhline(
    ds_cleo["xi"].isel(time=0).sum(dim="sdId"), label="CLEO initialization", color=default_colors[2]
)

axs[0].set_title(f"Total number concentration")
axs[0].set_xlabel("Time of ATR measurement")
axs[0].set_ylabel(r"N #$/m^3$")

# Plot the LWC
axs[1].plot(
    cloud_composite["mass_size_distribution"].time,
    1e3 * cloud_composite["mass_size_distribution"].sum("radius"),
    label="ATR measurement",
)
axs[1].axhline(
    1e3 * cloud_composite["mass_size_distribution_fit"].sum("radius"),
    color=default_colors[1],
    label="Fitted distribution",
)
axs[1].axhline(
    1e3 * ds_cleo["mass_represented"].isel(time=0).sum("sdId"),
    label="CLEO initialization",
    color=default_colors[2],
)

axs[1].set_title(f"LWC in the cloud")
axs[1].set_ylabel(r"LWC $\left[g/kg\right]$")
axs[1].set_xlabel("Time of ATR measurement")

for ax in axs:
    ax.legend()
    # adjust_spines(ax, visible_spines = ["left", "bottom"])


fig.suptitle(
    f"Cloud {cloud_id} at {cluster.time.dt.date.astype(str).values[0]}\n Total number concentration and LWC"
)
fig.tight_layout()
fig.savefig(fig_path / "total_number_concentration_and_LWC.png", dpi=300)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex=True)

style = dict(
    s=2,
    marker="o",
    alpha=0.8,
)

# Plot the total number concentration
axs[0].scatter(
    cloud_composite["radius_2D"],
    cloud_composite["particle_size_distribution"],
    label="ATR measurement",
    **style,
)
axs[0].scatter(
    cloud_composite["particle_size_distribution_fit"].radius,
    cloud_composite["particle_size_distribution_fit"],
    label="Fitted distribution",
    color=default_colors[1],
    **style,
)
axs[0].scatter(
    ds_cleo["radius"].isel(time=0),
    ds_cleo["xi"].isel(time=0),
    label="CLEO initialization",
    color=default_colors[2],
    **style,
)

axs[0].set_title(f"PSD in the cloud")
axs[0].set_ylabel(r"PSD #$/m^3$")

# Plot the LWC
axs[1].scatter(
    cloud_composite["radius_2D"],
    1e3 * cloud_composite["mass_size_distribution"],
    label="ATR measurement",
    **style,
)
axs[1].scatter(
    cloud_composite["mass_size_distribution_fit"].radius,
    1e3 * cloud_composite["mass_size_distribution_fit"],
    label="Fitted distribution",
    color=default_colors[1],
    **style,
)
axs[1].scatter(
    ds_cleo["radius"].isel(time=0),
    1e3 * ds_cleo["mass_represented"].isel(time=0),
    label="CLEO initialization",
    color=default_colors[2],
    **style,
)


axs[1].set_title(f"MSD in the cloud")
axs[1].set_ylabel(r"MSD $\left[g/m^{3}\right]$")

for ax in axs:
    ax.legend(loc="lower right", handler_map=handler_map_alpha())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radius $[m]$")

    # adjust_spines(ax, visible_spines = ["left", "bottom"])


fig.suptitle(f"Cloud {cloud_id} at {cluster.time.dt.date.astype(str).values[0]}")
fig.tight_layout()
fig.savefig(fig_path / "PSD_MSD_cloud.png", dpi=300)

# %% [markdown]
# ### Results:
# 
# What one can see here, is that the total $LWC$ in the cloud is kind of preserved in CLEO.
# But the magnitude of the distributions differs.

# %% [markdown]
# # Lets calculate the rain evaporation

# %%
# only use drolets which reach the lowest gridbox. So where the minimum of the coord3 is smaller than 20 m.
ds_leaving_domain = ds_cleo.where(ds_cleo["coord3"].min("time") <= 20, drop=True)
# ds_leaving_domain = ds_leaving_domain.sortby(ds_leaving_domain["radius"].isel(time = 0))

ds_leaving_domain["min_coord3"] = ds_leaving_domain["coord3"].min("time")

# ds_leaving_domain["domain_leaving_time"] = ds_leaving_domain["time"].where(ds_leaving_domain["coord3"] == ds_leaving_domain["coord3"].min("time")).mean("time")
ds_leaving_domain["domain_leaving_time"] = ds_leaving_domain.isel(
    time=ds_leaving_domain["coord3"].argmin("time")
).time

ds_leaving_domain["cloud_base_time"] = (
    ds_leaving_domain["time"].where(ds_leaving_domain["coord3"] <= cloud_base).min("time")
)
ds_leaving_domain["radius_max_time"] = ds_leaving_domain.isel(
    time=ds_leaving_domain["radius"].argmax("time")
).time

ds_leaving_domain["radius_init"] = ds_leaving_domain["radius"].isel(time=0)
ds_leaving_domain["radius_cloud_base"] = ds_leaving_domain["radius"].sel(
    time=ds_leaving_domain["cloud_base_time"]
)
ds_leaving_domain["radius_final"] = ds_leaving_domain["radius"].sel(
    time=ds_leaving_domain["domain_leaving_time"]
)
ds_leaving_domain["radius_max"] = ds_leaving_domain["radius"].sel(
    time=ds_leaving_domain["radius_max_time"]
)

ds_leaving_domain["mass_init"] = ds_leaving_domain["mass"].isel(time=0)
ds_leaving_domain["mass_cloud_base"] = ds_leaving_domain["mass"].sel(
    time=ds_leaving_domain["cloud_base_time"]
)
ds_leaving_domain["mass_final"] = ds_leaving_domain["mass"].sel(
    time=ds_leaving_domain["domain_leaving_time"]
)
ds_leaving_domain["mass_max"] = ds_leaving_domain["mass"].sel(time=ds_leaving_domain["radius_max_time"])

# ds_leaving_domain["droplet_growth"] = ds_leaving_domain["radius_final"] - ds_leaving_domain["radius_init"]
# ds_leaving_domain["droplet_growth_above_cloud"] = ds_leaving_domain["radius_cloud_base"] - ds_leaving_domain["radius_init"]
# ds_leaving_domain["droplet_growth_below_cloud"] = ds_leaving_domain["radius_final"] - ds_leaving_domain["radius_cloud_base"]

# %%
sdIds_with_evaporation = (
    (ds_leaving_domain)
    .where(ds_leaving_domain["time"] >= ds_leaving_domain["cloud_base_time"], drop=True)
    .where(ds_leaving_domain["radius"].diff("time") <= 0, drop=True)["sdId"]
)

ds_rain_evaporation = ds_cleo.sel(sdId=sdIds_with_evaporation)
ds_leaving_domain_rain_evaporation = ds_leaving_domain.sel(sdId=sdIds_with_evaporation)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex=True)

style = dict(
    s=2,
    marker="o",
    alpha=0.8,
)

# Plot the total number concentration
axs[0].scatter(
    cloud_composite["radius_2D"],
    cloud_composite["particle_size_distribution"],
    label="ATR measurement",
    **style,
)
axs[0].scatter(
    cloud_composite["particle_size_distribution_fit"].radius,
    cloud_composite["particle_size_distribution_fit"],
    label="Fitted distribution",
    color=default_colors[1],
    **style,
)
axs[0].scatter(
    ds_cleo["radius"].isel(time=0),
    ds_cleo["xi"].isel(time=0),
    label="CLEO initialization",
    color=default_colors[2],
    **style,
)
axs[0].scatter(
    ds_rain_evaporation["radius"].isel(time=0),
    ds_rain_evaporation["xi"].isel(time=0),
    label="SD with rain evaporation",
    color=default_colors[3],
    marker="x",
)


axs[0].set_title(f"PSD in the cloud")
axs[0].set_ylabel(r"PSD #$/m^3$")

# Plot the LWC
axs[1].scatter(
    cloud_composite["radius_2D"],
    1e3 * cloud_composite["mass_size_distribution"],
    label="ATR measurement",
    **style,
)
axs[1].scatter(
    cloud_composite["mass_size_distribution_fit"].radius,
    1e3 * cloud_composite["mass_size_distribution_fit"],
    label="Fitted distribution",
    color=default_colors[1],
    **style,
)
axs[1].scatter(
    ds_cleo["radius"].isel(time=0),
    1e3 * ds_cleo["mass_represented"].isel(time=0),
    label="CLEO initialization",
    color=default_colors[2],
    **style,
)
axs[1].scatter(
    ds_rain_evaporation["radius"].isel(time=0),
    1e3 * ds_rain_evaporation["mass_represented"].isel(time=0),
    label="SD with rain evaporation",
    color=default_colors[3],
    marker="x",
)


axs[1].set_title(f"MSD in the cloud")
axs[1].set_ylabel(r"MSD $\left[g/m^{3}\right]$")

for ax in axs:
    ax.legend(loc="lower right", handler_map=handler_map_alpha())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radius $[m]$")

    # adjust_spines(ax, visible_spines = ["left", "bottom"])


fig.suptitle(f"Cloud {cloud_id} at {cluster.time.dt.date.astype(str).values[0]}  PSD and MSD.")
fig.tight_layout()
fig.savefig(fig_path / "PSD_MSD_cloud_evaporation.png", dpi=300)

# %% [markdown]
# ## Follow random SD and capture their evolution over time

# %%
seed = 0
size = 15

np.random.seed(seed)
ids = np.random.choice(ds_cleo.sdId, size=size, replace=False)

np.random.seed(seed)
rain_ids = np.random.choice(ds_rain_evaporation.sdId, size=size, replace=False)

style = dict(
    marker="None",
    linestyle="-",
    # markersize = 1,
    alpha=0.8,
)


fig = plt.figure(figsize=(16, 9))
outer_grid = fig.add_gridspec(ncols=25, nrows=2, wspace=0.1, hspace=0.2)

ax_top_0 = fig.add_subplot(outer_grid[0, 3:13])
ax_top_1 = fig.add_subplot(outer_grid[0, 14:24], sharey=ax_top_0)
# ax_top_2 = fig.add_subplot(outer_grid[0, 25:35], sharey = ax_top_0)
axs_top = np.array([ax_top_0, ax_top_1])  # , ax_top_2])


ax_low_0 = fig.add_subplot(outer_grid[1, 3:13], sharex=ax_top_0, sharey=ax_top_0)
ax_low_1 = fig.add_subplot(outer_grid[1, 14:24], sharex=ax_top_1, sharey=ax_top_0)
# ax_low_2 = fig.add_subplot(outer_grid[1, 25:35], sharex = ax_top_2, sharey = ax_top_0)
axs_low = np.array([ax_low_0, ax_low_1])  # , ax_low_2])

axss = np.array([axs_top, axs_low])

# Use random SD from the whole dataset
axs_top[0].plot(ds_cleo.sel(sdId=ids).time, ds_cleo.sel(sdId=ids).coord3, **style)
axs_top[0].set_title("Time")
axs_top[0].set_ylabel("ALtitude $[m]$")


axs_top[1].plot(ds_cleo.sel(sdId=ids).radius, ds_cleo.sel(sdId=ids).coord3, **style)
axs_top[1].set_xscale("log")
# axs_top[1].set_xlim(7e-5, 4e-4)
# axs_top[1].set_xlabel("Radius in m")
axs_top[1].set_title("Radius")
# axs[1].set_xlim(-0.3e-6, 0.3e-6)
# axs_top[2].plot(
#     ds_cleo.sel(sdId = ids).mass,
#     ds_cleo.sel(sdId = ids).coord3,
#     **style
# )
# axs_top[2].set_xscale("log")
# # axs_top[2].set_xlim(1e-12, 1e-9)
# # axs_top[2].set_xlabel("Mass in kg")
# axs_top[2].set_title("Mass")
# axs[2].set_xlim(-0.15e-12, 0.15e-12)

# axs[0].set_ylim(ymin = 0, ymax = cloud_base)

# Use random SD from the the rain evaporation sub dataset
axs_low[0].plot(
    ds_rain_evaporation.sel(sdId=rain_ids).time, ds_rain_evaporation.sel(sdId=rain_ids).coord3, **style
)
# axs_low[0].set_title("SD altitude")
axs_low[0].set_ylabel("Altitude $[m]$")
axs_low[0].set_xlabel("Time in s")

axs_low[1].plot(
    ds_rain_evaporation.sel(sdId=rain_ids).radius,
    ds_rain_evaporation.sel(sdId=rain_ids).coord3,
    **style,
)
axs_low[1].set_xscale("log")
# axs_low[1].set_xlim(7e-5, 4e-4)
axs_low[1].set_xlabel("Radius in m")
# axs_low[1].set_title("SD Radius")
# axs[1].set_xlim(-0.3e-6, 0.3e-6)
# axs_low[2].plot(
#     ds_rain_evaporation.sel(sdId = rain_ids).mass,
#     ds_rain_evaporation.sel(sdId = rain_ids).coord3,
#     **style
# )
# axs_low[2].set_xscale("log")
# # axs_low[2].set_xlim(1e-12, 1e-9)
# axs_low[2].set_xlabel("Mass in kg")
# # axs_low[2].set_title("SD Mass")
# # axs[2].set_xlim(-0.15e-12, 0.15e-12)


for ax in axss.flatten():
    ax.axhline(cloud_base, color="k", linestyle="--", label="cloud base")
    ax.legend(loc="lower left")
    # ax.grid(True)

fig.suptitle(
    f"Cloud {cloud_id} at {cluster.time.dt.date.astype(str).values[0]} - Random sample of SD\nTop: Whole CLEO output.   Bottom: SDs which show decrease in $r$"
)
fig.savefig(fig_path / "Droplet_evolution.png", dpi=300)

# %%
style = dict(
    marker="None",
    linestyle="-",
    # markersize = 1,
    alpha=0.8,
)


fig = plt.figure(figsize=(16, 9))
outer_grid = fig.add_gridspec(ncols=25, nrows=2, wspace=0.1, hspace=0.2)

ax_top_0 = fig.add_subplot(outer_grid[0, 3:13])
ax_top_1 = fig.add_subplot(outer_grid[0, 14:24], sharey=ax_top_0)
# ax_top_2 = fig.add_subplot(outer_grid[0, 25:35], sharey = ax_top_0)
axs_top = np.array([ax_top_0, ax_top_1])  # , ax_top_2])


ax_low_0 = fig.add_subplot(outer_grid[1, 3:13], sharex=ax_top_0, sharey=ax_top_0)
ax_low_1 = fig.add_subplot(outer_grid[1, 14:24], sharex=ax_top_1, sharey=ax_top_0)
# ax_low_2 = fig.add_subplot(outer_grid[1, 25:35], sharex = ax_top_2, sharey = ax_top_0)
axs_low = np.array([ax_low_0, ax_low_1])  # , ax_low_2])

axss = np.array([axs_top, axs_low])

# Use random SD from the whole dataset
axs_top[0].plot(ds_cleo.sel(sdId=ids).time, ds_cleo.sel(sdId=ids).coord3, **style)
axs_top[0].set_title("Time")
axs_top[0].set_ylabel("ALtitude $[m]$")


axs_top[1].plot(ds_cleo.sel(sdId=ids).radius, ds_cleo.sel(sdId=ids).coord3, **style)
axs_top[1].set_xscale("log")
# axs_top[1].set_xlim(7e-5, 4e-4)
# axs_top[1].set_xlabel("Radius in m")
axs_top[1].set_title("Radius")
# axs[1].set_xlim(-0.3e-6, 0.3e-6)
# axs_top[2].plot(
#     ds_cleo.sel(sdId = ids).mass,
#     ds_cleo.sel(sdId = ids).coord3,
#     **style
# )
# axs_top[2].set_xscale("log")
# # axs_top[2].set_xlim(1e-12, 1e-9)
# # axs_top[2].set_xlabel("Mass in kg")
# axs_top[2].set_title("Mass")
# # axs[2].set_xlim(-0.15e-12, 0.15e-12)

# axs[0].set_ylim(ymin = 0, ymax = cloud_base)

# Use random SD from the the rain evaporation sub dataset
axs_low[0].plot(
    ds_rain_evaporation.sel(sdId=rain_ids).time, ds_rain_evaporation.sel(sdId=rain_ids).coord3, **style
)
# axs_low[0].set_title("SD altitude")
axs_low[0].set_ylabel("Altitude $[m]$")
axs_low[0].set_xlabel("Time in s")

axs_low[1].plot(
    ds_rain_evaporation.sel(sdId=rain_ids).radius,
    ds_rain_evaporation.sel(sdId=rain_ids).coord3,
    **style,
)
axs_low[1].set_xscale("log")
# axs_low[1].set_xlim(7e-5, 4e-4)
axs_low[1].set_xlabel("Radius in m")
# axs_low[1].set_title("SD Radius")
# axs[1].set_xlim(-0.3e-6, 0.3e-6)
# axs_low[2].plot(
#     ds_rain_evaporation.sel(sdId = rain_ids).mass,
#     ds_rain_evaporation.sel(sdId = rain_ids).coord3,
#     **style
# )
# axs_low[2].set_xscale("log")
# # axs_low[2].set_xlim(1e-12, 1e-9)
# axs_low[2].set_xlabel("Mass in kg")
# # axs_low[2].set_title("SD Mass")
# # axs[2].set_xlim(-0.15e-12, 0.15e-12)


for ax in axss.flatten():
    ax.axhline(cloud_base, color="k", linestyle="--", label="cloud base")
    ax.legend(loc="lower left")
    # ax.grid(True)

axs_top[1].set_xlim(3e-5, 1e-3)
axs_low[1].set_xlim(3e-5, 1e-3)
# axs_top[2].set_xlim(1e-10, 1e-6)
# axs_low[2].set_xlim(1e-10, 1e-6)
fig.suptitle(
    f"Cloud {cloud_id} at {cluster.time.dt.date.astype(str).values[0]} - Random sample of SD\nTop: Whole CLEO output.   Bottom: SDs which show decrease in $r$  "
)
fig.savefig(fig_path / "Droplet_evolution_zoom.png", dpi=300)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex="row", sharey="col")

style = dict(
    s=5,
    alpha=0.7,
    marker="o",
)

# PLOT the mass loss of individual SD

mass_loss = 1e3 * (
    ds_leaving_domain_rain_evaporation["mass_max"] - ds_leaving_domain_rain_evaporation["mass_final"]
)

# plot the initial radius of the droplet against the mass loss of the droplet
# axs[0].scatter(
#     ds_leaving_domain_rain_evaporation["radius_init"],
#     mass_loss,
#     label = "$r_{init}$",
#     **style
# )
# plot the radius at cloud base of the droplet against the mass loss of the droplet
axs[0].scatter(
    ds_leaving_domain_rain_evaporation["radius_cloud_base"], mass_loss, label="$r_{cb}$", **style
)
# plot the final radius of the droplet against the mass loss of the droplet
# axs[0].scatter(
#     ds_leaving_domain_rain_evaporation["radius_final"],
#     mass_loss,
#     label = "$r_{bottom}$",
#     **style

# )


axs[0].set_xscale("log")
axs[0].set_xlabel("Radius in m")
axs[0].set_ylabel("SD Mass loss $dM_{SD}$ $[g]$")


# PLOT the represented mass loss per SD
# plot the initial radius of the droplet against the mass loss of the droplet
mass_loss_represent = 1e3 * (
    ds_leaving_domain_rain_evaporation["xi"].sel(
        time=ds_leaving_domain_rain_evaporation["radius_max_time"]
    )
    * ds_leaving_domain_rain_evaporation["mass_max"]
    - ds_leaving_domain_rain_evaporation["xi"].sel(
        time=ds_leaving_domain_rain_evaporation["domain_leaving_time"]
    )
    * ds_leaving_domain_rain_evaporation["mass_final"]
)


# axs[1].scatter(
#     ds_leaving_domain_rain_evaporation["radius_init"],
#     mass_loss_represent,
#     label = "$r_{init}$",
#     **style
# )
# plot the radius at cloud base of the droplet against the mass loss of the droplet
axs[1].scatter(
    ds_leaving_domain_rain_evaporation["radius_cloud_base"],
    mass_loss_represent,
    label="$r_{cb}$",
    **style,
)
# plot the final radius of the droplet against the mass loss of the droplet
# axs[1].scatter(
#     ds_leaving_domain_rain_evaporation["radius_final"],
#     mass_loss_represent,
#     label = "$r_{bottom}$",
#     **style
# )


axs[1].set_xscale("log")
axs[1].set_xlabel("Radius at cloud-base in m")
axs[1].set_ylabel("Mass loss $dM$ $[g]$")
axs[1].set_yscale("log")
axs[0].set_yscale("log")

axs[0].set_title("Mass loss of each SD.\n$dM_{SD} = m(t_{R_{max}}) - m(t_{bottom})$")
axs[1].set_title("Mass loss represented by each SD.\n$dM = dM_{SD} \\cdot \\xi$")

# for ax in axs:
# ax.legend(loc = "upper right")
# ax.set_xlim(7e-5, 4e-4)

fig.suptitle(
    f"Cloud {cloud_id} at {cluster.time.dt.date.astype(str).values[0]}\nMass loss of droplets with rain evaporation."
)
fig.tight_layout()
fig.savefig(fig_path / "mass_loss_rain_evaporation.png", dpi=300)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex="row", sharey="col")

style = dict(
    s=5,
    alpha=0.7,
    marker="o",
)

# PLOT the mass loss of individual SD

mass_loss = 1e3 * (
    ds_leaving_domain_rain_evaporation["mass_max"] - ds_leaving_domain_rain_evaporation["mass_final"]
)

# plot the initial radius of the droplet against the mass loss of the droplet
# axs[0].scatter(
#     ds_leaving_domain_rain_evaporation["radius_init"],
#     mass_loss,
#     label = "$r_{init}$",
#     **style
# )
# plot the radius at cloud base of the droplet against the mass loss of the droplet
axs[0].scatter(
    ds_leaving_domain_rain_evaporation["radius_cloud_base"], mass_loss, label="$r_{cb}$", **style
)
# plot the final radius of the droplet against the mass loss of the droplet
# axs[0].scatter(
#     ds_leaving_domain_rain_evaporation["radius_final"],
#     mass_loss,
#     label = "$r_{bottom}$",
#     **style

# )


axs[0].set_xscale("log")
axs[0].set_xlabel("Radius in m")
axs[0].set_ylabel("SD Mass loss $dM_{SD}$ $[g]$")


# PLOT the represented mass loss per SD
# plot the initial radius of the droplet against the mass loss of the droplet
mass_loss_represent = 1e3 * (
    ds_leaving_domain_rain_evaporation["xi"].sel(
        time=ds_leaving_domain_rain_evaporation["radius_max_time"]
    )
    * ds_leaving_domain_rain_evaporation["mass_max"]
    - ds_leaving_domain_rain_evaporation["xi"].sel(
        time=ds_leaving_domain_rain_evaporation["domain_leaving_time"]
    )
    * ds_leaving_domain_rain_evaporation["mass_final"]
)


# axs[1].scatter(
#     ds_leaving_domain_rain_evaporation["radius_init"],
#     mass_loss_represent,
#     label = "$r_{init}$",
#     **style
# )
# plot the radius at cloud base of the droplet against the mass loss of the droplet
axs[1].scatter(
    ds_leaving_domain_rain_evaporation["radius_cloud_base"],
    mass_loss_represent,
    label="$r_{cb}$",
    **style,
)
# plot the final radius of the droplet against the mass loss of the droplet
# axs[1].scatter(
#     ds_leaving_domain_rain_evaporation["radius_final"],
#     mass_loss_represent,
#     label = "$r_{bottom}$",
#     **style
# )


axs[1].set_xscale("log")
axs[1].set_xlabel("Radius at cloud-base in m")
axs[1].set_ylabel("Mass loss $dM$ $[g]$")
axs[1].set_yscale("log")

axs[0].set_title("Mass loss of each SD.\n$dM_{SD} = m(t_{R_{max}}) - m(t_{bottom})$")
axs[1].set_title("Mass loss represented by each SD.\n$dM = dM_{SD} \\cdot \\xi$")

for ax in axs:
    # ax.legend(loc = "upper right")
    ax.set_xlim(7e-5, 4e-4)

fig.suptitle(
    f"Cloud {cloud_id} at {cluster.time.dt.date.astype(str).values[0]}\nMass loss of droplets with rain evaporation."
)
fig.tight_layout()
fig.savefig(fig_path / "mass_loss_rain_evaporation_zoom.png", dpi=300)

# %%
ds_leaving_domain_rain_evaporation["coord3_r_max"] = ds_leaving_domain_rain_evaporation["coord3"].sel(
    time=ds_leaving_domain_rain_evaporation["radius_max_time"]
)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharey="row")

style = dict(
    s=10,
    alpha=0.7,
    marker="o",
)
# PLOT the mass loss of individual SD


choord3_r_max = ds_leaving_domain_rain_evaporation["coord3_r_max"]

# axs[0].plot(
#     [ds_leaving_domain_rain_evaporation["radius_init"], ds_leaving_domain_rain_evaporation["radius_cloud_base"], ds_leaving_domain_rain_evaporation["radius_final"]],
#     [choord3_r_max, choord3_r_max, choord3_r_max],
#     color = "grey",
#     alpha = 0.1,
# )


# plot the initial radius of the droplet against the mass loss of the droplet
# axs[0].scatter(
#     ds_leaving_domain_rain_evaporation["radius_init"],
#     choord3_r_max,
#     label = "$r_{init}$",
#     **style
# )
# plot the radius at cloud base of the droplet against the mass loss of the droplet
axs[0].scatter(
    ds_leaving_domain_rain_evaporation["radius_cloud_base"], choord3_r_max, label="$r_{cb}$", **style
)
# plot the final radius of the droplet against the mass loss of the droplet
# axs[0].scatter(
#     ds_leaving_domain_rain_evaporation["radius_final"],
#     choord3_r_max,
#     label = "$r_{bottom}$",
#     **style

# )


axs[0].set_xscale("log")
axs[0].set_xlabel("Radius at cloud-base in m")
axs[0].set_ylabel("Height of $r_{max}$ $[m]$")


# plot the initial radius of the droplet against the mass loss of the droplet
axs[1].scatter(mass_loss_represent, choord3_r_max, label="$r_{init}$", color=default_colors[3], **style)

axs[1].set_xscale("log")
axs[1].set_xlabel("Mass loss $dM$ $[g]$")
axs[1].set_ylabel("Height of $r_{max}$ $[m]$")
axs[1].set_xlim(2.5e-9, 7e-4)


axs[0].legend(loc="upper right")
# axs[0].set_xlim(7e-5, 4e-4)
axs[0].set_ylim(0, 800)

fig.suptitle(
    f"Cloud {cloud_id} at {cluster.time.dt.date.astype(str).values[0]}\nAltitude of maximum radius."
)
fig.tight_layout()
fig.savefig(fig_path / "alt_r_max_rain_evaporation_zoom.png", dpi=300)

# %%
ml = mass_loss.assign_coords(radius_cloud_base=ds_leaving_domain_rain_evaporation["radius_cloud_base"])
mlr = mass_loss_represent.assign_coords(
    radius_cloud_base=ds_leaving_domain_rain_evaporation["radius_cloud_base"]
)
mlr = mlr.swap_dims({"sdId": "radius_cloud_base"})
ml = ml.swap_dims({"sdId": "radius_cloud_base"})

r_bins = np.logspace(-7, -3, 100)

# %%
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex="row", sharey="col")

style = dict(
    # s = 5,
    fill=True,
    # alpha = 0.7,
    # marker = "o",
)

# PLOT the mass loss of individual SD

# plot the initial radius of the droplet against the mass loss of the droplet
# axs[0].scatter(
#     ds_leaving_domain_rain_evaporation["radius_init"],
#     mass_loss,
#     label = "$r_{init}$",
#     **style
# )
# plot the radius at cloud base of the droplet against the mass loss of the droplet
axs[0].stairs(
    ml.groupby_bins("radius_cloud_base", bins=r_bins).mean(),
    r_bins,
    label="$r_{cb}$",
    **style,
)
# plot the final radius of the droplet against the mass loss of the droplet
# axs[0].scatter(
#     ds_leaving_domain_rain_evaporation["radius_final"],
#     mass_loss,
#     label = "$r_{bottom}$",
#     **style

# )


axs[0].set_xscale("log")
axs[0].set_xlabel("Radius in m")
axs[0].set_ylabel("SD Mass loss $dM_{SD}$ $[g]$")


# PLOT the represented mass loss per SD as bins
# plot the radius at cloud base of the droplet against the mass loss of the droplet
axs[1].stairs(
    mlr.groupby_bins("radius_cloud_base", bins=r_bins).sum(), r_bins, label="$r_{cb}$", **style
)
# plot the final radius of the droplet against the mass loss of the droplet
# axs[1].scatter(
#     ds_leaving_domain_rain_evaporation["radius_final"],
#     mass_loss_represent,
#     label = "$r_{bottom}$",
#     **style
# )


axs[1].set_xscale("log")
axs[1].set_xlabel("Radius at cloud-base in m")
axs[1].set_ylabel("Mass loss $dM$ $[g]$")
# axs[1].set_yscale("log")

axs[0].set_title("Mean mass loss of individual SD.\n$dM_{SD} = m(t_{R_{max}}) - m(t_{bottom})$")
axs[1].set_title(
    "Sum of mass loss represented by each SD.\n" + r"$dM = dM_{SD} \\cdot \\xi$    $M = \sum dM =$"
    + f"{mlr.sum().data:.3f} g"
)

for ax in axs:
    # ax.legend(loc = "upper right")
    ax.set_xlim(1e-5, 1e-3)

axs[0].set_ylim(0, 5e-5)
axs[1].set_ylim(0, 4.5e-3)
fig.suptitle(
    f"Cloud {cloud_id} at {cluster.time.dt.date.astype(str).values[0]}\nMass loss of droplets with rain evaporation."
)
fig.tight_layout()
fig.savefig(fig_path / "mass_loss_rain_evaporation_hist.png", dpi=300)



# %%
