# %%
import datetime
import logging
import os
import sys
import traceback

from pathlib import Path

import lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

from sdm_eurec4a import RepositoryPath, conversions, get_git_revision_hash
from sdm_eurec4a.identifications import (
    match_clouds_and_cloudcomposite,
    match_clouds_and_dropsondes,
    select_individual_cloud_by_id,
)
from sdm_eurec4a.input_processing import transfer
from sdm_eurec4a.reductions import shape_dim_as_dataarray
from sdm_eurec4a.visulization import (
    adjust_lightness_array,
    handler_map_alpha,
    set_custom_rcParams,
)


# %%
plt.style.use("default")
default_colors = set_custom_rcParams()
dark_colors = adjust_lightness_array(default_colors, amount=0.5)
# from matplotlib import rc


# # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# # rc('font',**{'family':'serif','serif':['Palatino']})
# rc("text", usetex=False)

# THE PATH TO THE SCRIPT DIRECTORY
script_dir = Path("/home/m/m301096/repositories/sdm-eurec4a/scripts/CLEO/initalize")
print(script_dir)

# REPOSITORY_ROOT = Path(script_dir).parents[2]
REPOSITORY_ROOT = RepositoryPath("levante")()
print(REPOSITORY_ROOT)

# ======================================
# %% Define output and figure directories

output_dir = REPOSITORY_ROOT / "data/model/input/selected_clouds_lwc"
output_dir.mkdir(parents=True, exist_ok=True)


fig_path = REPOSITORY_ROOT / "results" / script_dir.relative_to(REPOSITORY_ROOT) / "create_input" / "selected_clouds_lwc"
fig_path.mkdir(parents=True, exist_ok=True)


# ======================================
#  Define logger


# %%
# create a logger which stores the log file in the script directory of the logger directory

log_dir = output_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(log_dir / 'create_input.log')
handler.setLevel(logging.INFO)

# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(console_handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Execution terminated due to an Exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


sys.excepthook = handle_exception


# %%

logging.info("============================================================")
logging.info("Start preprocessing of identified clouds to create input for CLEO")
logging.info("Git hash: %s", get_git_revision_hash())
logging.info("Date: %s", datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S"))
logging.info("Output directory: %s", output_dir)
logging.info("Figure directory: %s", fig_path)
logging.info("============================================================")
input('all correct?')


# %% [markdown]
# ======================================
# ### Load datasets
#

# %%
identified_clouds = xr.open_dataset(
    "/home/m/m301096/repositories/sdm-eurec4a/data/observation/cloud_composite/processed/identified_clouds/identified_clusters_rain_mask_5.nc"
)
# select only clouds which are
# below 1500m and have a duration of at least 10s.
identified_clouds = identified_clouds.where(
    (identified_clouds["alt"] <= 1500)
    & (identified_clouds["alt"] >= 500)
    & (identified_clouds["duration"] >= np.timedelta64(10, "s"))
    & (identified_clouds["liquid_water_content"] / identified_clouds["duration"].dt.seconds >=0.1),
    drop=True,
)


distance_IC_DS = xr.open_dataset(
    REPOSITORY_ROOT
    / Path(
        f"data/observation/combined/distance_relations/distance_dropsondes_identified_clusters_rain_mask_5.nc"
    )
)

cloud_composite = xr.open_dataset(
    REPOSITORY_ROOT / Path(f"data/observation/cloud_composite/processed/cloud_composite_si_units.nc")
)

drop_sondes = xr.open_dataset(
    REPOSITORY_ROOT / Path("data/observation/dropsonde/processed/drop_sondes.nc")
)

drop_sondes = drop_sondes.where(drop_sondes.alt <= 1600, drop=True)

# add relative humidity to the dataset

drop_sondes["relative_humidity"] = conversions.relative_humidity_from_tps(
        temperature = drop_sondes["air_temperature"],
        pressure= drop_sondes["pressure"],
        specific_humidity= drop_sondes["specific_humidity"],
    )


# Add radii and altitude arrays to datasets

cloud_composite["radius_array"] = shape_dim_as_dataarray(
    da=cloud_composite["particle_size_distribution"], output_dim="radius"
)
cloud_composite["time_array"] = shape_dim_as_dataarray(
    da=cloud_composite["particle_size_distribution"], output_dim="time"
)
drop_sondes["alt_array"] = shape_dim_as_dataarray(da=drop_sondes["air_temperature"], output_dim="alt")

# %% [markdown]
# print all the ids of clouds which are used selected by this criteria

# %%
fig, axs = plt.subplots(1, 2)
axs[0].scatter(
    identified_clouds["duration"].dt.seconds,
    identified_clouds["alt"],
    identified_clouds["liquid_water_content"],
    marker="o",
    linestyle="None",
    color=default_colors[0],
)
axs[0].set_xlabel("Duration in s")
axs[0].set_ylabel("Altitude in m")
axs[0].set_title("LWC in kg/kg")


axs[1].scatter(
    identified_clouds["horizontal_extent"],
    identified_clouds["vertical_extent"],
    identified_clouds["liquid_water_content"],
    marker="o",
    linestyle="None",
    color=default_colors[0],
)

axs[1].set_xlabel("Horizontal extent in m")
axs[1].set_ylabel("Vertical extent in m")
axs[1].set_title("LWC in kg/kg")


for t in identified_clouds.sortby("liquid_water_content").time[-10:]:
    subdata = identified_clouds.sel(time=t)
    axs[0].annotate(subdata.cloud_id.data, (subdata.duration.dt.seconds, subdata.alt))
    axs[1].annotate(subdata.cloud_id.data, (subdata["horizontal_extent"], subdata["vertical_extent"]))
fig.savefig(fig_path / Path("cloud_properties.png"))



def main(chosen_id):
    """The main function."""
    identification_type = "clusters"
    split_radius = 4.5e-5
    max_spatial_distance = 100
    max_temporal_distance = np.timedelta64(3, "h")
    subfig_path = fig_path / Path(f"{identification_type}_{chosen_id}")
    subfig_path.mkdir(parents=True, exist_ok=True)

    cloud_information = dict(
        cloud_id=chosen_id,
        split_radius=split_radius,
        dropsonde_distance=dict(
            max_spatial_distance=max_spatial_distance, max_temporal_distance=str(max_temporal_distance)
        ),
        identification_type=identification_type,
    )

    # select a single cloud
    if chosen_id is not None:
        ds_cloud = select_individual_cloud_by_id(identified_clouds, chosen_id)
    else:
        ds_cloud = identified_clouds

    cloud_information["altitude"] = ds_cloud['alt'].data
    cloud_information["duration"] = ds_cloud["duration"].dt.seconds.data
    cloud_information["time"] = ds_cloud['time'].dt.strftime("%Y-%m-%d %H:%M:%S").data

    ds_cloudcomposite_with_zeros = match_clouds_and_cloudcomposite(
        ds_clouds=ds_cloud,
        ds_cloudcomposite=cloud_composite,
    )

    ds_dropsondes = match_clouds_and_dropsondes(
        ds_clouds=ds_cloud,
        ds_sonde=drop_sondes,
        ds_distance=distance_IC_DS,
        max_spatial_distance=max_spatial_distance,
        max_temporal_distance=max_temporal_distance,
    )

    # remove 0s
    ds_cloudcomposite = ds_cloudcomposite_with_zeros.where(
        ds_cloudcomposite_with_zeros["particle_size_distribution"] != 0
    )

    logging.info(f"Number of dropsondes: {ds_dropsondes["time"].shape}")
    logging.info(f"Number of cloudcomposite: {ds_cloudcomposite["time"].shape}")

    if ds_cloudcomposite["time"].shape[0] == 0:
        raise ValueError("No cloudcomposite data found for cloud")
    if ds_dropsondes["time"].shape[0] == 0:
        raise ValueError("No dropsonde data found for cloud")


    # %% [markdown]
    # # Split data into cloud and rain

    # %%
    ds_lower = ds_cloudcomposite.sel(radius=slice(None, split_radius))
    ds_rain = ds_cloudcomposite.sel(radius=slice(split_radius, None))

    # %%
    # plot cloud PSD distribution as pcolormesh

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))

    vmin, vmax = 1, 1e8

    pc = axs.pcolormesh(
        ds_lower.radius,
        ds_lower.time,
        ds_lower["particle_size_distribution"].T,
        norm=mpl.colors.LogNorm(vmin, vmax),
        cmap="Blues",
        shading="nearest",
    )
    fig.colorbar(pc, ax=axs, label="Cloud and Drizzle drops #m$^{-3}$")
    pc = axs.pcolormesh(
        ds_rain.radius,
        ds_rain.time,
        ds_rain["particle_size_distribution"].T,
        norm=mpl.colors.LogNorm(vmin, 1e4),
        cmap="Reds",
        shading="nearest",
    )
    fig.colorbar(pc, ax=axs, label="Rain drops in #m$^{-3}$")
    axs.set_xscale("log")
    axs.set_xlabel("Radius in m")
    axs.set_ylabel("Time")

    fig.suptitle(
        f"PSD against radius and time for cloud {chosen_id}\nDate: {ds_cloud.time.dt.strftime('%Y-%m-%d').data}"
    )
    fig.tight_layout()
    fig.savefig(subfig_path / Path("PSD_radius_time.png"))

    # %%
    psd_rain_fit = transfer.PSD_LnNormal()
    psd_cloud_fit = transfer.PSD_LnNormal()

    # Drizzle and cloud droplets
    data = ds_rain.particle_size_distribution
    radi2d = shape_dim_as_dataarray(da=data, output_dim="radius")

    psd_rain_fit.update_individual_model_parameters(
        lmfit.Parameter(
            name="geometric_means",
            min=data.radius.min().data,
            max=data.radius.max().data,
        )
    )

    rain_model_result = psd_rain_fit.get_model().fit(
        data=data.data, radii=radi2d.data, params=psd_rain_fit.get_model_parameters(), nan_policy="omit"
    )
    psd_rain_fit.lmfitParameterValues_to_dict(rain_model_result.params)

    data = ds_lower.particle_size_distribution
    radi2d = shape_dim_as_dataarray(da=data, output_dim="radius")
    psd_model = psd_cloud_fit.get_model()

    psd_cloud_fit.update_individual_model_parameters(
        lmfit.Parameter(
            name="geometric_means",
            value=1e-5,
            min=data.radius.min().data * 0.2e-1,
            max=data.radius.max().data * 0.2e1,
        )
    )
    psd_cloud_fit.update_individual_model_parameters(
        lmfit.Parameter(
            name="geometric_sigmas",
            value=1.1,
            min=0,
            max=1.7,
        )
    )

    # print(psd_cloud_fit.get_model_parameters())

    cloud_model_result = psd_model.fit(
        data=data.data, radii=radi2d.data, params=psd_cloud_fit.get_model_parameters(), nan_policy="omit"
    )
    psd_cloud_fit.lmfitParameterValues_to_dict(cloud_model_result.params)

    psd_fit = psd_rain_fit + psd_cloud_fit
    # print(psd_cloud_fit.get_parameters())

    # %%
    # plt.plot(fitted_model.eval_func(ds_cloudcomposite.radius))
    style = dict(
        marker="o",
        edgecolor="none",
        s=10,
        alpha=0.5,
    )
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_ylim(1e-2, 1e8)
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.scatter(
        ds_rain.radius_array,
        ds_rain.particle_size_distribution,
        c=default_colors[0],
        label="rain drops",
        **style,
    )

    ax.plot(
        ds_cloudcomposite.radius,
        psd_rain_fit.eval_func(ds_cloudcomposite.radius),
        label="rain fit",
        linestyle="--",
        color=dark_colors[0],
    )

    ax.scatter(
        ds_lower.radius_array,
        ds_lower.particle_size_distribution,
        c=default_colors[1],
        label="cloud and drizzle",
        **style,
    )

    ax.plot(
        ds_cloudcomposite.radius,
        psd_cloud_fit.eval_func(ds_cloudcomposite.radius),
        label="cloud and drizzle fit",
        linestyle="--",
        color=dark_colors[1],
    )

    ax.plot(
        ds_cloudcomposite.radius,
        psd_fit.eval_func(ds_cloudcomposite.radius),
        label="fitted distribution",
        color="k",
    )

    ax.legend(handler_map=handler_map_alpha())

    ax.set_xlabel("radius in m")
    ax.set_ylabel("Count in #/m3")

    fig.suptitle(f"Fitting of PSD for cloud {chosen_id}")
    fig.savefig(subfig_path / Path("fitted_psd.png"))

    # %% [markdown]
    # ## Thermodynamics
    #

    #%%

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), sharey=True)

    scatter_style = dict(alpha=0.75, linewidth=0.7)

    ax_q, ax_ta  = axs[0]
    ax_rh, ax_theta = axs[1]
    ax_q.set_xlim(0, 20)
    ax_rh.set_xlim(30, 110)
    ax_ta.set_xlim(285, 305)
    ax_theta.set_xlim(295, 308)

    # First plot
    ax_q.plot(
        1e3 * ds_dropsondes["specific_humidity"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[0],
        # label="Spec. Hum.",
        **scatter_style,
    )

    ax_q.set_xlabel("Specific humidity [g/kg]")
    ax_q.set_ylabel("alt [m]")
    ax_q.xaxis.label.set_color(default_colors[0])  # Set the color of x-axis label
    ax_q.tick_params(axis="x", colors=default_colors[0])  # Set the color of x-axis ticks

    # First plot
    ax_rh.plot(
        ds_dropsondes["relative_humidity"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[3],
        # label="Spec. Hum.",
        **scatter_style,
    )
    ax_rh.axvline(100, color="grey", linestyle="-", label="Saturation")
    ax_rh.set_xlabel("Relative humidity [%]")
    ax_rh.set_ylabel("alt [m]")
    ax_rh.xaxis.label.set_color(default_colors[3])  # Set the color of x-axis label
    ax_rh.tick_params(axis="x", colors=default_colors[3])  # Set the color of x-axis ticks


    # Second plot on a new x-axis
    ax_ta.plot(
        ds_dropsondes["air_temperature"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[1],
        # label="Air Temperature",
        **scatter_style,
    )

    ax_ta.set_xlabel("Air Temperature [K]")
    ax_ta.xaxis.label.set_color(default_colors[1])  # Set the color of x-axis label
    ax_ta.tick_params(axis="x", colors=default_colors[1])  # Set the color of x-axis ticks

    # Thrid plot on a new x-axis
    ax_theta.plot(
        ds_dropsondes["potential_temperature"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[2],
        # label="Pot. Temperature",
        **scatter_style,
    )

    ax_theta.set_xlabel("Pot. Temperature [K]")
    ax_theta.xaxis.label.set_color(default_colors[2])  # Set the color of x-axis label
    ax_theta.tick_params(axis="x", colors=default_colors[2])  # Set the color of x-axis ticks

    for ax in axs.flatten():
        ax.set_ylim(0, 1500)
        ax.axhline(ds_cloud.alt, color="r", linestyle="--", label="PSD measurement")
        ax.legend(handler_map=handler_map_alpha())
        ax.tick_params(axis="x", labelrotation=-33)

    fig.suptitle(f"Dropsondes related to cloud {chosen_id}")
    fig.tight_layout()
    fig.savefig(subfig_path / Path("dropsonde_properties.png"))

    # %% [markdown]
    # force the split height to be above 150m and below the top of the domain

    # %%
    thermo_fit = dict(
        air_temperature=transfer.ThermodynamicSplitLinear(),
        specific_humidity=transfer.ThermodynamicSplitLinear(),
        potential_temperature=transfer.ThermodynamicSplitLinear(),
        relative_humidity=transfer.ThermodynamicSplitLinear(),

    )

    for var in thermo_fit:
        data = ds_dropsondes[var]
        alt2d = shape_dim_as_dataarray(da=data, output_dim="alt")

        x_split_parameter = lmfit.Parameter(
            name="x_split",
            min=150,
            max=float(data.alt.max()),
        )
        thermo_object = thermo_fit[var]
        thermo_object.update_individual_model_parameters(x_split_parameter)

        result = thermo_object.get_model().fit(
            data=data.data,
            x=alt2d.data,
            nan_policy="omit",
            params=thermo_object.get_model_parameters(),
        )
        thermo_fit[var].lmfitParameterValues_to_dict(result.params)

    # print([thermo_fit[key].get_x_split() for key in thermo_fit])
    x_split = np.mean([thermo_fit[key].get_x_split() for key in thermo_fit])
    logging.info(f"Split level: {x_split} m")

    # %%
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), sharey=True)

    scatter_style = dict(alpha=0.75, linewidth=0.7)

    ax_q, ax_ta  = axs[0]
    ax_rh, ax_theta = axs[1]
    ax_q.set_xlim(0, 20)
    ax_rh.set_xlim(30, 110)
    ax_ta.set_xlim(285, 305)
    ax_theta.set_xlim(295, 308)


    # First plot
    ax_q.plot(
        1e3 * ds_dropsondes["specific_humidity"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[0],
        # label="Spec. Hum.",
        **scatter_style,
    )
    ax_q.plot(
        1e3 * thermo_fit["specific_humidity"].eval_func(ds_dropsondes["alt"])[0],
        ds_dropsondes["alt"],
        color=dark_colors[0],
        label="fit",
    )


    ax_q.set_xlabel("Specific humidity [g/kg]")
    ax_q.set_ylabel("alt [m]")
    ax_q.xaxis.label.set_color(default_colors[0])  # Set the color of x-axis label
    ax_q.tick_params(axis="x", colors=default_colors[0])  # Set the color of x-axis ticks
    ax_q.axhline(thermo_fit["specific_humidity"].get_x_split(), color="black", linestyle="--", label="split level")


    # First plot
    ax_rh.plot(
        ds_dropsondes["relative_humidity"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[3],
        # label="Spec. Hum.",
        **scatter_style,
    )
    ax_rh.plot(
        thermo_fit["relative_humidity"].eval_func(ds_dropsondes["alt"])[0],
        ds_dropsondes["alt"],
        color=dark_colors[3],
        label="fit",
    )


    ax_rh.set_xlabel("Relative humidity [%]")
    ax_rh.set_ylabel("alt [m]")
    ax_rh.xaxis.label.set_color(default_colors[3])  # Set the color of x-axis label
    ax_rh.tick_params(axis="x", colors=default_colors[3])  # Set the color of x-axis ticks
    ax_rh.axhline(thermo_fit["relative_humidity"].get_x_split(), color="black", linestyle="--", label="split level")


    # Second plot on a new x-axis
    ax_ta.plot(
        ds_dropsondes["air_temperature"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[1],
        # label="Air Temperature",
        **scatter_style,
    )
    ax_ta.plot(
        thermo_fit["air_temperature"].eval_func(ds_dropsondes["alt"])[0],
        ds_dropsondes["alt"],
        color=dark_colors[1],
        label="fit",
    )

    ax_ta.set_xlabel("Air Temperature [K]")
    ax_ta.xaxis.label.set_color(default_colors[1])  # Set the color of x-axis label
    ax_ta.tick_params(axis="x", colors=default_colors[1])  # Set the color of x-axis ticks
    ax_ta.axhline(thermo_fit["air_temperature"].get_x_split(), color="black", linestyle="--", label="split level")


    # Thrid plot on a new x-axis
    ax_theta.plot(
        ds_dropsondes["potential_temperature"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[2],
        # label="Pot. Temperature",
        **scatter_style,
    )
    ax_theta.plot(
        thermo_fit["potential_temperature"].eval_func(ds_dropsondes["alt"])[0],
        ds_dropsondes["alt"],
        color=dark_colors[2],
        label="fit",
    )
    ax_theta.set_xlabel("Pot. Temperature [K]")
    ax_theta.xaxis.label.set_color(default_colors[2])  # Set the color of x-axis label
    ax_theta.tick_params(axis="x", colors=default_colors[2])  # Set the color of x-axis ticks
    ax_theta.axhline(thermo_fit["potential_temperature"].get_x_split(), color="black", linestyle="--", label="split level")



    for ax in axs.flatten():
        ax.axhline(ds_cloud.alt, color="red", linestyle="--", label="PSD measurement")
        ax.legend(handler_map=handler_map_alpha())
        ax.tick_params(axis="x", labelrotation=-33)

    fig.suptitle(f"Dropsondes related to cloud {chosen_id}")
    fig.tight_layout()
    fig.savefig(subfig_path / Path("dropsonde_properties_fit.png"))

    # %% [markdown]
    # ### Fix x_split

    # %%
    thermo_fit_fixed = dict(
        air_temperature=transfer.ThermodynamicSplitLinear(),
        specific_humidity=transfer.ThermodynamicSplitLinear(),
        potential_temperature=transfer.ThermodynamicSplitLinear(),
        relative_humidity=transfer.ThermodynamicSplitLinear(),
    )

    for var in thermo_fit:
        data = ds_dropsondes[var]
        alt2d = shape_dim_as_dataarray(da=data, output_dim="alt")
        thermo_fit_fixed[var].update_individual_model_parameters(
            lmfit.Parameter(name="x_split", value=x_split, vary=False)
        )
        result = (
            thermo_fit_fixed[var]
            .get_model()
            .fit(
                data=data.data,
                x=alt2d.data,
                params=thermo_fit_fixed[var].get_model_parameters(),
                nan_policy="omit",
            )
        )
        thermo_fit_fixed[var].lmfitParameterValues_to_dict(result.params)

    thermo_fit_fixed[var].get_parameters()

    # %%

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), sharey=True)

    scatter_style = dict(alpha=0.75, linewidth=0.7)

    ax_q, ax_ta  = axs[0]
    ax_rh, ax_theta = axs[1]
    ax_q.set_xlim(0, 20)
    ax_rh.set_xlim(30, 110)
    ax_ta.set_xlim(285, 305)
    ax_theta.set_xlim(295, 308)


    # First plot
    ax_q.plot(
        1e3 * ds_dropsondes["specific_humidity"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[0],
        # label="Spec. Hum.",
        **scatter_style,
    )
    ax_q.plot(
        1e3 * thermo_fit_fixed["specific_humidity"].eval_func(ds_dropsondes["alt"])[0],
        ds_dropsondes["alt"],
        color=dark_colors[0],
        label="fit",
    )


    ax_q.set_xlabel("Specific humidity [g/kg]")
    ax_q.set_ylabel("alt [m]")
    ax_q.xaxis.label.set_color(default_colors[0])  # Set the color of x-axis label
    ax_q.tick_params(axis="x", colors=default_colors[0])  # Set the color of x-axis ticks
    ax_q.axhline(thermo_fit_fixed["specific_humidity"].get_x_split(), color="black", linestyle="--", label="split level")


    # First plot
    ax_rh.plot(
        ds_dropsondes["relative_humidity"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[3],
        # label="Spec. Hum.",
        **scatter_style,
    )
    ax_rh.plot(
        thermo_fit_fixed["relative_humidity"].eval_func(ds_dropsondes["alt"])[0],
        ds_dropsondes["alt"],
        color=dark_colors[3],
        label="fit",
    )


    ax_rh.set_xlabel("Relative humidity [%]")
    ax_rh.set_ylabel("alt [m]")
    ax_rh.xaxis.label.set_color(default_colors[3])  # Set the color of x-axis label
    ax_rh.tick_params(axis="x", colors=default_colors[3])  # Set the color of x-axis ticks
    ax_rh.axhline(thermo_fit_fixed["relative_humidity"].get_x_split(), color="black", linestyle="--", label="split level")


    # Second plot on a new x-axis
    ax_ta.plot(
        ds_dropsondes["air_temperature"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[1],
        # label="Air Temperature",
        **scatter_style,
    )
    ax_ta.plot(
        thermo_fit_fixed["air_temperature"].eval_func(ds_dropsondes["alt"])[0],
        ds_dropsondes["alt"],
        color=dark_colors[1],
        label="fit",
    )

    ax_ta.set_xlabel("Air Temperature [K]")
    ax_ta.xaxis.label.set_color(default_colors[1])  # Set the color of x-axis label
    ax_ta.tick_params(axis="x", colors=default_colors[1])  # Set the color of x-axis ticks
    ax_ta.axhline(thermo_fit_fixed["air_temperature"].get_x_split(), color="black", linestyle="--", label="split level")


    # Thrid plot on a new x-axis
    ax_theta.plot(
        ds_dropsondes["potential_temperature"].T,
        ds_dropsondes["alt"].T,
        color=default_colors[2],
        # label="Pot. Temperature",
        **scatter_style,
    )
    ax_theta.plot(
        thermo_fit_fixed["potential_temperature"].eval_func(ds_dropsondes["alt"])[0],
        ds_dropsondes["alt"],
        color=dark_colors[2],
        label="fit",
    )
    ax_theta.set_xlabel("Pot. Temperature [K]")
    ax_theta.xaxis.label.set_color(default_colors[2])  # Set the color of x-axis label
    ax_theta.tick_params(axis="x", colors=default_colors[2])  # Set the color of x-axis ticks
    ax_theta.axhline(thermo_fit_fixed["potential_temperature"].get_x_split(), color="black", linestyle="--", label="split level")



    for ax in axs.flatten():
        ax.axhline(ds_cloud.alt, color="red", linestyle="--", label="PSD measurement")
        ax.legend(handler_map=handler_map_alpha())
        ax.tick_params(axis="x", labelrotation=-33)

    fig.suptitle(f"Dropsondes related to cloud {chosen_id}")
    fig.tight_layout()
    fig.savefig(subfig_path / Path("dropsonde_properties_fit_fixed_x_split.png"))

    # %% [markdown]
    # # Export everything as yaml

    # %%

    def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
        """
        Converts a numpy ndarray to a list and represents it for YAML dumping.

        Parameters:
        dumper (yaml.Dumper): The YAML dumper instance.
        array (np.ndarray): The numpy ndarray to represent.

        Returns:
        yaml.Node: The represented list.
        """
        return dumper.represent_list(array.tolist())

    def float64_representer(dumper: yaml.Dumper, array: np.float64) -> yaml.Node:
        """
        Represents a numpy float64 for YAML dumping.

        Parameters:
        dumper (yaml.Dumper): The YAML dumper instance.
        array (np.float64): The numpy float64 to represent.

        Returns:
        yaml.Node: The represented float.
        """
        return dumper.represent_float(array)

    def thermofit_representer(dumper: yaml.Dumper, obj: transfer.ThermodynamicSplitLinear) -> yaml.Node:
        """
        Represents a ThermodynamicSplitLinear object for YAML dumping.

        Parameters:
        dumper (yaml.Dumper): The YAML dumper instance.
        obj (transfer.ThermodynamicSplitLinear): The ThermodynamicSplitLinear object to represent.

        Returns:
        yaml.Node: The represented dictionary of parameters.
        """
        parameters = {"slopes": obj.get_slopes()[0]}
        parameters.update(obj.get_parameters())

        data = {
            "parameters": parameters,
            "type": obj.type,
        }
        return dumper.represent_dict(data)

    def psdfit_representer(dumper: yaml.Dumper, obj: transfer.PSD_LnNormal) -> yaml.Node:
        """
        Represents a PSD_LnNormal object for YAML dumping.

        Parameters:
        dumper (yaml.Dumper): The YAML dumper instance.
        obj (transfer.PSD_LnNormal): The PSD_LnNormal object to represent.

        Returns:
        yaml.Node: The represented dictionary of parameters.
        """
        data = {
            "parameters": obj.get_parameters(),
            "type": obj.type,
        }
        return dumper.represent_dict(data)

    def add_representer() -> None:
        """Adds custom representers for numpy ndarrays, numpy float64s,
        ThermodynamicSplitLinear objects, and PSD_LnNormal objects to the YAML
        dumper."""
        yaml.add_representer(np.ndarray, ndarray_representer)
        yaml.add_representer(np.float64, float64_representer)
        yaml.add_representer(transfer.ThermodynamicSplitLinear, thermofit_representer)
        yaml.add_representer(transfer.PSD_LnNormal, psdfit_representer)

    add_representer()

    version_control = dict(
        git_hash=get_git_revision_hash(),
        date=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S"),
    )

    output_dictonary = dict(
        particle_size_distribution=psd_fit,
        thermodynamics=thermo_fit_fixed,
        cloud=cloud_information,
        version_control=version_control,
    )

    output_filepath = output_dir / f"{identification_type}_{chosen_id}.yaml"

    with open(output_filepath, "w") as file:
        s = yaml.dump(output_dictonary)
        file.write(s)

    with open(subfig_path / "input_CLEO.yaml", "w") as file:
        s = yaml.dump(output_dictonary)
        file.write(s)

    plt.close("all")


for id in identified_clouds.cloud_id.data:
    id = int(id)
    plt.close('all')
    try:
        main(id)
        logger.info(f"Created input for cloud {id}")
    except Exception as e:
        message = traceback.format_exception(None, e, e.__traceback__)
        logger.error(f'Failed to create input for cloud {id}: '+ repr(e) + '\n' + ''.join(message))
        continue
