# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sdm_eurec4a import RepositoryPath, conversions
from sdm_eurec4a.input_processing import transfer
from sdm_eurec4a.reductions import shape_dim_as_dataarray
from sdm_eurec4a.visulization import (
    adjust_lightness_array,
    handler_map_alpha,
    set_custom_rcParams,
)


# %%
def set_xticks_time(ax):
    xticks = [0, 500, 1000]
    ax.set_xticks(xticks)


def set_yticks_height(ax):
    yticks = [0, 500, 1000, 1500, 2000]
    ax.set_yticks(yticks)


def set_yticks_height_km(ax):
    yticks = [0, 0.5, 1, 1.5, 2]
    ax.set_yticks(yticks)


def set_logxticks_meter(ax):
    xticks = [1e-6, 1e-3]
    xticklabels = [r"$10^{-6}$", r"$10^{-3}$"]
    ax.set_xticks(xticks, xticklabels)


def set_logxticks_micrometer(ax):
    xticks = [1e-3, 1e0, 1e3]
    xticklabels = [r"$10^{-3}$", r"$10^{0}$", r"$10^{3}$"]
    ax.set_xticks(xticks, xticklabels)


def set_logtyticks_psd(ax):
    yticks = [1e0, 1e6]
    yticklabels = [r"$10^0$", r"$10^6$"]
    ax.set_yticks(yticks, yticklabels)


def set_yticks_lwc(ax):
    ax.set_yticks([0, 0.1, 0.2])


# %%
plt.style.use("default")
default_colors = set_custom_rcParams()
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
)

dark_colors = adjust_lightness_array(default_colors, amount=0.5)

repo_path = RepositoryPath("levante")()
print(repo_path)

# data_dict = "output_v3.2/stationary_no_physics"
# data_dict = "output_v3.2/stationary_condensation"
subdata_dir = "output_v3.2-v3.4_v2"

data_dict = dict(
    no_physics_512=dict(
        path=Path("/home/m/m301096/CLEO/data/")
        / "output_v3.2"
        / "stationary_no_physics"
        / "combined/eulerian_dataset_combined.nc",
        microphysics="Null microphysics (512 SDs)",
        linestyle="dashdot",
        color="black",
    ),
    no_physics_1024=dict(
        path=Path("/home/m/m301096/CLEO/data/")
        / "output_v3.3"
        / "stationary_no_physics"
        / "combined/eulerian_dataset_combined.nc",
        microphysics="Null microphysics (1024 SDs)",
        linestyle=":",
        color="black",
    ),
    condensation_512=dict(
        path=Path("/home/m/m301096/CLEO/data/")
        / "output_v3.2"
        / "stationary_condensation"
        / "combined/eulerian_dataset_combined.nc",
        microphysics="Condensation/Evaporation (512 SDs)",
        linestyle="dashdot",
        color="black",
    ),
    condensation_1024=dict(
        path=Path("/home/m/m301096/CLEO/data/")
        / "output_v3.3"
        / "stationary_condensation"
        / "combined/eulerian_dataset_combined.nc",
        microphysics="Condensation/Evaporation (1024 SDs)",
        linestyle="-",
        color="black",
    ),
    condensation_2048=dict(
        path=Path("/home/m/m301096/CLEO/data/")
        / "output_v3.4"
        / "stationary_condensation"
        / "combined/eulerian_dataset_combined.nc",
        microphysics="Condensation/Evaporation (2048 SDs)",
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        color="black",
    ),
    collision_condensation_512=dict(
        path=Path("/home/m/m301096/CLEO/data/")
        / "output_v3.2"
        / "stationary_collision_condensation"
        / "combined/eulerian_dataset_combined.nc",
        microphysics="Collision and cond./evap. (512 SDs)",
        linestyle="dashdot",
        color="black",
    ),
    collision_condensation_1024=dict(
        path=Path("/home/m/m301096/CLEO/data/")
        / "output_v3.3"
        / "stationary_collision_condensation"
        / "combined/eulerian_dataset_combined.nc",
        microphysics="Collision and cond./evap. (1024 SDs)",
        linestyle="--",
        color="black",
    ),
    collision_condensation_2048=dict(
        path=Path("/home/m/m301096/CLEO/data/")
        / "output_v3.4"
        / "stationary_collision_condensation"
        / "combined/eulerian_dataset_combined.nc",
        microphysics="Collision and cond./evap. (2048 SDs)",
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        color="black",
    ),
)

# THE PATH TO THE SCRIPT DIRECTORY
script_dir = Path("/home/m/m301096/repositories/sdm-eurec4a/notebooks/presentation/results/")
print(script_dir)


fig_dir = repo_path / "results" / script_dir.relative_to(repo_path) / subdata_dir
print(fig_dir)

fig_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load datasets

# %%
for key in data_dict:
    ds = xr.open_dataset(data_dict[key]["path"])
    # ds = ds.sel(cloud_id = [18, 301])
    ds.attrs.update(microphysics=data_dict[key]["microphysics"])
    data_dict[key]["combined_output"] = ds

# %% [markdown]
# ## Calculations

# %%
from functools import reduce


print("all cloud ids which are present in all simulations")
in_all = reduce(
    np.intersect1d, [data_dict[key]["combined_output"]["cloud_id"].data for key in data_dict]
)
print(len(in_all))
print(in_all)


# %%
clouds_dict = {
    "222": dict(
        cloud_id=222,
        color="r",
    ),
    "142": dict(
        cloud_id=142,
        color="b",
    ),
}

# %%
for key in data_dict:
    ds = data_dict[key]["combined_output"]
    for ID in clouds_dict:
        is_in = ID in ds["cloud_id"]
        cloud_id = clouds_dict[ID]["cloud_id"]
        print(f"{cloud_id}, {key}, {is_in}")

# %% [markdown]
# ### Calculate masks and coord3 and volume


# %%
def add_variables(combined_output: xr.Dataset):
    combined_output["sub_cloud_layer_mask"] = combined_output["gridbox"] < combined_output["max_gridbox"]
    combined_output["sub_cloud_layer_mask"].attrs.update(
        long_name="Sub Cloud Layer Mask",
        description="Boolean mask indicating if the gridbox is part of the sub cloud layer",
        units="1",
    )

    combined_output["sub_cloud_layer_no_bottom"] = (combined_output["sub_cloud_layer_mask"]) & (
        combined_output["gridbox"] > 0
    )
    combined_output["sub_cloud_layer_no_bottom"].attrs.update(
        long_name="Sub Cloud Layer Mask without bottom",
        description="Boolean mask indicating if the gridbox is part of the sub cloud layer and not the bottom gridbox",
        units="1",
    )

    combined_output["pseudo_coord3"] = combined_output["gridbox"] * 20
    combined_output["pseudo_coord3"].attrs.update(
        long_name="pseudo Coordinate 3",
        description="pseudo Coordinate 3. Which is the gridbox number times 20m. It thus is wrong for the top gridbox for each cloud. This can be identified by the variable 'max_gridbox'. So use the mask 'sub_cloud_layer_mask' to identify the sub cloud layer.",
        units="m",
    )

    combined_output["pseudo_coord3_normalized"] = (
        combined_output["pseudo_coord3"]
        / combined_output["pseudo_coord3"][combined_output["max_gridbox"] - 1]
    )

    Z = np.nan_to_num(
        x=(combined_output["pseudo_coord3"].where(combined_output["sub_cloud_layer_mask"])),
        nan=0,
    )
    Z_max = np.nan_to_num(
        x=(
            combined_output["pseudo_coord3"].where(
                combined_output["gridbox"] == combined_output["max_gridbox"]
            )
        )
        * 0
        + combined_output["max_gridbox"] * 20
        + (100 - 20) / 2,
        nan=0,
    )
    Z = Z + Z_max
    combined_output["pseudo_coord3_full"] = (("gridbox", "cloud_id"), Z)
    combined_output["pseudo_coord3_full"] = combined_output["pseudo_coord3_full"].where(
        combined_output["gridbox"] <= combined_output["max_gridbox"]
    )
    combined_output["pseudo_coord3_full"].attrs.update(
        long_name="pseudo Coordinate 3 Full",
        description="pseudo Coordinate 3. Each gridbox below cloud layer is 20m thick and cloud layer is 100m thick.",
        units="m",
    )

    combined_output["pseudo_gribox_volume"] = combined_output["pseudo_coord3"] * 0 + 20**3
    combined_output["pseudo_gribox_volume"].attrs.update(
        long_name="Simple Pseudo Gridbox Volume",
        description="pseudo Gridbox Volume. Which is the gridbox volume for each gridbox in the sub cloud layer. It is 20m x 20m x 20m for each gridbox.",
        units="m^3",
    )

    combined_output["pseudo_gribox_volume_full"] = (
        ("gridbox", "cloud_id"),
        np.nan_to_num(
            x=(
                combined_output["pseudo_gribox_volume"].where(
                    combined_output["gridbox"] != combined_output["max_gridbox"]
                )
            ),
            nan=20 * 20 * 100,
        ),
    )
    combined_output["pseudo_gribox_volume_full"] = combined_output["pseudo_gribox_volume_full"].where(
        combined_output["gridbox"] <= combined_output["max_gridbox"]
    )
    combined_output["pseudo_gribox_volume"].attrs.update(
        long_name="pseudo Gridbox Volume",
        description="pseudo Gridbox Volume. Which is the gridbox volume for each gridbox in the sub cloud layer. It is 20m x 20m x 20m for each gridbox in subcloud layer. 20m x 20m x 100m in cloud layer. nan for gridboxes above cloud layer.",
        units="m^3",
    )
    vertical_profile_mass = (
        combined_output["mass_represented"].sel(time=slice(400, None)).sum("radius_bins").mean("time")
    )
    vertical_profile_mass = vertical_profile_mass.where(combined_output["sub_cloud_layer_mask"])
    vertical_profile_mass = 1e3 * vertical_profile_mass / combined_output["pseudo_gribox_volume"]
    vertical_profile_mass.attrs.update(
        long_name="Mass",
        description="Vertical Profile of Mass. The mass is averaged over time and radius bins and then divided by the gridbox volume.",
        units=r"$g m^{-3}$",
    )
    combined_output["vertical_profile_mass"] = vertical_profile_mass

    vertical_profile_mass_std = (
        combined_output["mass_represented"].sel(time=slice(400, None)).sum("radius_bins").std("time")
    )
    vertical_profile_mass_std = vertical_profile_mass_std.where(combined_output["sub_cloud_layer_mask"])
    vertical_profile_mass_std = 1e3 * vertical_profile_mass_std / combined_output["pseudo_gribox_volume"]
    vertical_profile_mass_std.attrs.update(
        long_name="Mass",
        description="Vertical Profile of Mass. The mass is averaged over time and radius bins and then divided by the gridbox volume.",
        units=r"$g m^{-3}$",
    )
    combined_output["vertical_profile_mass_std"] = vertical_profile_mass_std

    combined_output["top_subcloud_mass"] = (
        combined_output["vertical_profile_mass"]
        .where(combined_output["gridbox"] == combined_output["max_gridbox"] - 1)
        .mean("gridbox")
    )
    combined_output["top_subcloud_mass"].attrs.update(
        long_name="Top Subcloud Mass",
        description="Mass of the top subcloud layer. The mass is averaged over time and radius bins and then divided by the gridbox volume.",
        units=r"$g m^{-3}$",
    )

    combined_output["bottom_subcloud_mass"] = combined_output["vertical_profile_mass"].sel(gridbox=0)
    combined_output["bottom_subcloud_mass"].attrs.update(
        long_name="Bottom Subcloud Mass",
        description="Mass of the bottom subcloud layer. The mass is averaged over time and radius bins and then divided by the gridbox volume.",
        units=r"$g m^{-3}$",
    )

    vertical_profile_mass_fraction = (
        100
        * combined_output["vertical_profile_mass"]
        / combined_output["vertical_profile_mass"].sel(gridbox=combined_output["max_gridbox"] - 1)
    )
    vertical_profile_mass_fraction.attrs.update(
        long_name="Mass Fraction",
        description="Vertical Profile of Mass Fraction. The mass is divided by the mass of the topmost sub-cloud-layer gridbox.",
        units="%",
    )
    combined_output["vertical_profile_mass_fraction"] = vertical_profile_mass_fraction

    vertical_profile_mass_fraction_std = (
        100
        * combined_output["vertical_profile_mass_std"]
        / combined_output["vertical_profile_mass"].sel(gridbox=combined_output["max_gridbox"] - 1)
    )
    vertical_profile_mass_fraction_std.attrs.update(
        long_name="Mass Fraction",
        description="Vertical Profile of Mass Fraction. The mass is divided by the mass of the topmost sub-cloud-layer gridbox.",
        units="%",
    )
    combined_output["vertical_profile_mass_fraction_std"] = vertical_profile_mass_fraction_std

    vertical_profile_mass_diff = (
        combined_output["mass_difference"].sel(time=slice(400, None)).sum("radius_bins").mean("time")
    )
    vertical_profile_mass_diff = vertical_profile_mass_diff.where(
        combined_output["sub_cloud_layer_no_bottom"]
    )
    vertical_profile_mass_diff = (
        1e3 * vertical_profile_mass_diff / combined_output["pseudo_gribox_volume"]
    )
    vertical_profile_mass_diff.attrs.update(
        long_name="Mass Difference",
        description="Vertical Profile of Mass Difference. The mass difference is averaged over time and radius bins and then divided by the gridbox volume. ",
        units=r"$g m^{-3} s^{-1}$",
    )
    combined_output["vertical_profile_mass_diff"] = vertical_profile_mass_diff

    vertical_evaporation_fraction = 100 * vertical_profile_mass_diff / vertical_profile_mass
    vertical_evaporation_fraction.attrs.update(
        long_name="Vertical Evaporation Fraction",
        description="Vertical Evaporation Fraction. The mass difference is divided by the mass to get the fraction of mass which evaporates in the sub cloud layer.",
        units=r"$\%$",
    )
    combined_output["vertical_evaporation_fraction"] = vertical_evaporation_fraction

    # total values
    combined_output["mass_difference_total"] = (
        combined_output["vertical_profile_mass_diff"] * combined_output["pseudo_gribox_volume"]
    ).sum("gridbox", keep_attrs=True) / (
        combined_output["pseudo_gribox_volume"]
        .where(combined_output["sub_cloud_layer_no_bottom"])
        .sum("gridbox")
    )
    combined_output["mass_difference_total"].attrs.update(
        long_name="Total Mass Difference",
        description="Total Mass Difference. The mass difference is summed over all gridboxes and divided by the total volume of the sub cloud layer.",
        units=r"$g m^{-3} s^{-1}$",
    )

    combined_output["mass_total"] = combined_output["vertical_profile_mass"].sum("gridbox")
    combined_output["mass_total"].attrs.update(
        long_name="Total Mass",
        description="Total Mass. The mass is summed over all gridboxes of the sub cloud layer.",
        units=r"$g$",
    )

    combined_output["mass_fraction_total"] = vertical_profile_mass_fraction.sel(gridbox=1)
    combined_output["mass_fraction_total"].attrs.update(
        long_name="Total Mass Fraction",
        description="Total Mass Fraction. The mass fraction of the mass that reaches the bottom.",
        units="%",
    )

    combined_output["liquid_water_content"] = 1e3 * (
        combined_output["mass_represented"] / combined_output["pseudo_gribox_volume_full"]
    )
    combined_output["liquid_water_content"].attrs.update(
        long_name="Liquid Water Content",
        description="Liquid Water Content per gridbox for the sub cloud layer.",
        units=r"$g m^{-3}$",
    )

    combined_output["liquid_water_content_smooth"] = (
        combined_output["liquid_water_content"].rolling(time=50, center=True).mean()
    )
    combined_output["liquid_water_content_smooth"].attrs.update(
        long_name="Liquid Water Content Smooth",
        description="Smoothed Liquid Water Content per gridbox for the sub cloud layer.",
        units=r"$g m^{-3}$",
    )

    vertical_profile_lwc = (
        combined_output["liquid_water_content"]
        .sel(time=slice(400, None))
        .sum("radius_bins", keep_attrs=True)
        .mean("time", keep_attrs=True)
    )
    vertical_profile_lwc = vertical_profile_lwc.where(
        combined_output["gridbox"] <= combined_output["max_gridbox"]
    )
    vertical_profile_lwc.attrs.update(
        long_name="Liquid Water Content",
        description="Vertical Profile of liquid water content.",
    )
    combined_output["vertical_profile_lwc"] = vertical_profile_lwc.where(
        combined_output["sub_cloud_layer_mask"]
    )
    combined_output["vertical_profile_lwc_including_cloud"] = vertical_profile_lwc

    vertical_profile_lwc_fraction = combined_output["vertical_profile_lwc"] - combined_output[
        "vertical_profile_lwc"
    ].sel(gridbox=combined_output["max_gridbox"] - 1)
    vertical_profile_lwc_fraction.attrs.update(
        long_name="Liquid Water Content fraction",
        description="Vertical Profile of liquid water content fraction.",
        units=combined_output["liquid_water_content"].units,
    )
    combined_output["vertical_profile_lwc_fraction"] = vertical_profile_lwc_fraction


# %%
for key in data_dict:
    print(f"Calculating for {key}")
    add_variables(data_dict[key]["combined_output"])

print("done")
# %%
# for key in data_dict:
#     combined_output = data_dict[key]["combined_output"]
#     combined_output = combined_output.sel(cloud_id=combined_output["max_gridbox"] < 40)
#     combined_output = combined_output.sel(gridbox=slice(None, 40))
#     combined_output = combined_output.sel(time=slice(400, None))
#     data = (
#         1e3 * combined_output["mass_difference_timestep"] / combined_output["pseudo_gribox_volume_full"]
#     )
#     # data = data.where(combined_output['sub_cloud_layer_no_bottom'])
#     cumsum_loss = data.sortby(100 - data["gridbox"]).cumsum(dim="gridbox")
#     cumsum_loss = cumsum_loss.sortby("gridbox")
#     cumsum_loss.attrs.update(
#         long_name="Cumulative Mass Loss",
#         description="Cumulative Mass Loss. The mass loss is summed over all gridboxes from the bottom to the top.",
#         units=r"$g m^{-3}$",
#     )
#     combined_output["cumsum_loss"] = cumsum_loss

#     vertical_cumsum_loss = combined_output["cumsum_loss"].sum("radius_bins").mean("time")
#     vertical_cumsum_loss = vertical_cumsum_loss.where(combined_output["sub_cloud_layer_mask"])
#     vertical_cumsum_loss.attrs.update(
#         long_name="Cumulative Mass Loss",
#         description="Cumulative Mass Loss. The mass loss is summed over all gridboxes from the bottom to the top.",
#         units=r"$g m^{-3}$",
#     )
#     combined_output["vertical_cumsum_loss"] = vertical_cumsum_loss
#     data_dict[key]["combined_output"] = combined_output

# %% [markdown]
# ### visuals


# %%
def plot_single_microphysics(ax, data_dict, data_keys, variable, cloud_dict):
    ax_microlabel = ax.twinx()
    ax_microlabel.set_yticks([])

    plot_cloud_lines = True
    for key in data_keys:
        combined_output = data_dict[key]["combined_output"]
        linestyle = data_dict[key]["linestyle"]
        color = data_dict[key]["color"]
        combined_output = combined_output.sel(cloud_id=combined_output["max_gridbox"] < 40)

        for ID in cloud_dict:
            cloud_id = cloud_dict[ID]["cloud_id"]
            ax.plot(
                combined_output.sel(cloud_id=cloud_id)[variable],
                combined_output["pseudo_coord3"],
                color=cloud_dict[ID]["color"],
                alpha=1,
                linestyle=linestyle,
                zorder=3,
            )
        if plot_cloud_lines == True:
            for ID in cloud_dict:
                ax.plot(
                    np.nan,
                    np.nan,
                    color=cloud_dict[ID]["color"],
                    alpha=1,
                    label=f"cloud {ID}",
                    linestyle="-",
                    zorder=3,
                )
            plot_cloud_lines = False

        ax_microlabel.plot(
            np.nan,
            np.nan,
            color="k",
            alpha=0,
            label=f"{data_dict[key]['microphysics']}",
            linestyle=linestyle,
        )

        ax.plot(
            combined_output[variable].quantile(0.5, "cloud_id"),
            combined_output["pseudo_coord3"],
            color="k",
            alpha=1,
            linestyle=linestyle,
            zorder=1,
        )
        ax.fill_betweenx(
            combined_output["pseudo_coord3"],
            combined_output[variable].quantile(0.25, "cloud_id"),
            combined_output[variable].quantile(0.75, "cloud_id"),
            color="k",
            alpha=0.1,
            linestyle=linestyle,
            zorder=1,
        )

    xlabel = combined_output[variable].long_name + f" {combined_output[variable].attrs['units']}"
    ax.set_xlabel(xlabel)

    # # ax.set_title(ind_cloud_output1[vertical_profile_mass_fraction].long_name)

    set_yticks_height(ax)
    ax.set_ylim(0, 800)
    ax.set_ylabel("Height $m$")

    ax_microlabel.legend(handler_map=handler_map_alpha(), loc="upper center")
    ax.legend(handler_map=handler_map_alpha(), loc="upper right")

    return ax


def plot_single_microphysics_all(ax, data_dict, data_keys, variable, cloud_dict):
    ax_microlabel = ax.twinx()
    ax_microlabel.set_yticks([])

    plot_cloud_lines = True
    for key in data_keys:
        combined_output = data_dict[key]["combined_output"]
        linestyle = data_dict[key]["linestyle"]
        color = data_dict[key]["color"]
        combined_output = combined_output.sel(cloud_id=combined_output["max_gridbox"] < 40)

        for ID in cloud_dict:
            cloud_id = cloud_dict[ID]["cloud_id"]
            ax.plot(
                combined_output.sel(cloud_id=cloud_id)[variable],
                combined_output["pseudo_coord3"],
                color=cloud_dict[ID]["color"],
                alpha=1,
                linestyle=linestyle,
                zorder=3,
            )
        if plot_cloud_lines == True:
            for ID in cloud_dict:
                ax.plot(
                    np.nan,
                    np.nan,
                    color=cloud_dict[ID]["color"],
                    alpha=1,
                    label=f"cloud {ID}",
                    linestyle="-",
                    zorder=3,
                )
            plot_cloud_lines = False

        ax_microlabel.plot(
            np.nan,
            np.nan,
            color="k",
            alpha=0,
            label=f"{data_dict[key]['microphysics']}",
            linestyle=linestyle,
        )

        ax.plot(
            combined_output[variable].quantile(0.5, "cloud_id"),
            combined_output["pseudo_coord3"],
            color="k",
            alpha=1,
            linestyle=linestyle,
            zorder=1,
        )
        ax.plot(
            combined_output[variable].T,
            combined_output["pseudo_coord3"],
            color=[0.3, 0.3, 0.3],
            alpha=0.4,
            linestyle="-",
            linewidth=0.3,
            zorder=1,
        )

    xlabel = combined_output[variable].long_name + f" {combined_output[variable].attrs['units']}"
    ax.set_xlabel(xlabel)

    # # ax.set_title(ind_cloud_output1[vertical_profile_mass_fraction].long_name)

    set_yticks_height(ax)
    ax.set_ylim(0, 800)
    ax.set_ylabel("Height $m$")

    ax_microlabel.legend(handler_map=handler_map_alpha(), loc="upper center")
    ax.legend(handler_map=handler_map_alpha(), loc="upper right")

    return ax


# %%
variable = "vertical_profile_mass_fraction"
keys = ["no_physics_1024", "condensation_1024", "collision_condensation_1024"]
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4.5))

plot_single_microphysics(
    ax=ax,
    data_dict=data_dict,
    data_keys=keys,
    variable=variable,
    cloud_dict=clouds_dict,
)

ax.set_xticks([80, 90, 100])
ax.set_xlim(80, 105)
ax.axvline(100, color="k", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")


fig.savefig(fig_dir / "mass_fraction_1024.svg", transparent=True)

# %%
variable = "vertical_profile_mass_fraction"
keys = ["condensation_1024"]
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4.5))

plot_single_microphysics_all(
    ax=ax,
    data_dict=data_dict,
    data_keys=keys,
    variable=variable,
    cloud_dict=clouds_dict,
)

ax.set_xticks([60, 80, 100])
ax.set_xlim(80, 105)
ax.axvline(100, color="k", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")


fig.savefig(fig_dir / "mass_fraction_1024_all.svg", transparent=True)

# %%
variable = "vertical_profile_lwc_fraction"
keys = ["no_physics_1024", "condensation_1024", "collision_condensation_1024"]
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4.5))

plot_single_microphysics(
    ax=ax,
    data_dict=data_dict,
    data_keys=keys,
    variable=variable,
    cloud_dict=clouds_dict,
)

ax.set_xticks(
    [
        0,
        -0.01,
    ]
)
ax.set_xlim(-0.01, 0.002)
ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")
ax.set_xlabel(r"Evaporated mass $g m^{-3}$")

fig.savefig(fig_dir / "evaporated_mass_1024.svg", transparent=True)

# %%
variable = "vertical_profile_lwc_fraction"
keys = ["condensation_512", "condensation_1024", "condensation_2048"]
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4.5))

plot_single_microphysics(
    ax=ax,
    data_dict=data_dict,
    data_keys=keys,
    variable=variable,
    cloud_dict=clouds_dict,
)

ax.set_xticks(
    [
        0,
        -0.01,
    ]
)
ax.set_xlim(-0.01, 0.002)
ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

ax.set_xlabel(r"Evaporated mass $g m^{-3}$")

fig.savefig(fig_dir / "evaporated_mass_condensation.svg", transparent=True)

# %%
variable = "vertical_profile_lwc_fraction"
keys = ["condensation_1024"]
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4.5))

plot_single_microphysics(
    ax=ax,
    data_dict=data_dict,
    data_keys=keys,
    variable=variable,
    cloud_dict=clouds_dict,
)

ax.set_xticks(
    [
        0,
        -0.01,
    ]
)
ax.set_xlim(-0.01, 0.002)
ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

ax.set_xlabel(r"Evaporated mass $g m^{-3}$")

fig.savefig(fig_dir / "evaporated_mass_condensation_1024.svg", transparent=True)

# %%
variable = "vertical_profile_lwc_fraction"
keys = ["condensation_2048"]
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4.5))

plot_single_microphysics(
    ax=ax,
    data_dict=data_dict,
    data_keys=keys,
    variable=variable,
    cloud_dict=clouds_dict,
)

ax.set_xticks(
    [
        0,
        -0.01,
    ]
)
ax.set_xlim(-0.01, 0.002)
ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

ax.set_xlabel(r"Evaporated mass $g m^{-3}$")

fig.savefig(fig_dir / "evaporated_mass_condensation_2048.svg", transparent=True)

# %%
variable = "vertical_profile_lwc_fraction"
keys = ["condensation_1024"]
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4.5))

plot_single_microphysics_all(
    ax=ax,
    data_dict=data_dict,
    data_keys=keys,
    variable=variable,
    cloud_dict=clouds_dict,
)

ax.set_xticks(
    [
        0,
        -0.01,
    ]
)
ax.set_xlim(-0.01, 0.002)
ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

ax.set_xlabel(r"Evaporated mass $g m^{-3}$")

fig.savefig(fig_dir / "evaporated_mass_condensation_1024_all.svg", transparent=True)

# %%
variable = "vertical_profile_lwc_fraction"
keys = ["collision_condensation_512", "collision_condensation_1024", "collision_condensation_2048"]
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4.5))

plot_single_microphysics(
    ax=ax,
    data_dict=data_dict,
    data_keys=keys,
    variable=variable,
    cloud_dict=clouds_dict,
)

ax.set_xticks(
    [
        0,
        -0.01,
    ]
)
ax.set_xlim(-0.01, 0.002)
ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")
ax.set_xlabel(r"Evaporated mass $g m^{-3}$")

fig.savefig(fig_dir / "evaporated_mass_collision.svg", transparent=True)

# %%
variable = "vertical_profile_lwc_fraction"
keys = ["no_physics_512", "no_physics_1024"]
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4.5))

plot_single_microphysics(
    ax=ax,
    data_dict=data_dict,
    data_keys=keys,
    variable=variable,
    cloud_dict=clouds_dict,
)

ax.set_xticks(
    [
        0,
        -0.01,
    ]
)
ax.set_xlim(-0.01, 0.002)
ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

ax.set_xlabel(r"Evaporated mass $g m^{-3}$")

fig.savefig(fig_dir / "evaporated_mass_null_microphysics.svg", transparent=True)

# %%
variable = "number_superdroplets"
keys = ["collision_condensation_512", "collision_condensation_1024", "collision_condensation_2048"]

fig, axs = plt.subplots(ncols=len(keys), nrows=1, figsize=(15, 4.5))

idx = 0
for key in keys:
    combined_output = data_dict[key]["combined_output"]
    linestyle = "-"
    color = data_dict[key]["color"]
    ax = axs[idx]

    combined_output = data_dict[key]["combined_output"]
    combined_output = combined_output.sel(cloud_id=combined_output["max_gridbox"] < 40)
    combined_output = combined_output.sel(gridbox=slice(None, 40))
    combined_output = combined_output.sel(time=slice(400, None))
    data = combined_output[variable]
    data = data.where(data <= 1e6)
    data = data.sum("radius_bins").mean("time")
    data = data.where(data != 0)
    ax.plot(
        data.T,
        combined_output["pseudo_coord3"],
        label=key,
        linestyle=linestyle,
        color=color,
        alpha=0.1,
    )
    ax.plot(
        data.quantile(0.5, "cloud_id"),
        combined_output["pseudo_coord3"],
        label=key,
        linestyle=linestyle,
        alpha=1,
        color=color,
    )

    title = data_dict[key]["microphysics"]
    ax.set_title(title)

    idx += 1

for ax in axs:
    ax.set_xlim(0, 400)
    ax.set_xlabel("Number of Superdroplets")
    ax.set_ylabel("Height $m$")

fig.tight_layout()
fig.savefig(fig_dir / "number_superdroplets_collision.svg", transparent=True)

# %%
variable = "number_superdroplets"
keys = ["collision_condensation_512", "collision_condensation_1024", "collision_condensation_2048"]

fig, axs = plt.subplots(ncols=len(keys), nrows=1, figsize=(15, 4.5))

idx = 0
for key in keys:
    combined_output = data_dict[key]["combined_output"]
    linestyle = "-"
    color = data_dict[key]["color"]
    ax = axs[idx]

    combined_output = data_dict[key]["combined_output"]
    combined_output = combined_output.sel(cloud_id=combined_output["max_gridbox"] < 40)
    combined_output = combined_output.sel(gridbox=slice(None, 40))
    combined_output = combined_output.sel(time=slice(400, None))
    data = combined_output[variable]
    data = data.where(combined_output["sub_cloud_layer_mask"])
    data = data.where(data <= 1e6)
    data = data.sum("radius_bins").mean("time")
    data = data.where(data != 0)

    ax.plot(
        data.T,
        combined_output["pseudo_coord3"],
        label=key,
        linestyle=linestyle,
        color=color,
        alpha=0.1,
    )
    ax.plot(
        data.quantile(0.5, "cloud_id"),
        combined_output["pseudo_coord3"],
        label=key,
        linestyle=linestyle,
        alpha=1,
        color=color,
    )

    title = data_dict[key]["microphysics"]
    ax.set_title(title)

    idx += 1

for ax in axs:
    ax.set_xlim(0, 400)
    ax.set_xlabel("Number of Superdroplets")
    ax.set_ylabel("gridbox")

fig.tight_layout()
fig.savefig(fig_dir / "number_superdroplets_collision_no_cloud_layer.svg", transparent=True)

# %%
