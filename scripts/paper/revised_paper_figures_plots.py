### functions adapted from ``./notebooks/paper/17-NN-figure-review-version.ipynb``
### Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import seaborn as sns
import textwrap
import xarray as xr

from pathlib import Path
from tqdm import tqdm
from ruamel.yaml import YAML
from typing import Tuple, Literal, Union

from sdm_eurec4a.visulization import (
    set_paper_rcParams,
    adjust_lightness_array,
    adjust_lightness,
    label_from_attrs,
    add_additional_axis,
    add_subplotlabel,
    save_figure,
)
from sdm_eurec4a import RepositoryPath
from sdm_eurec4a import data_loading
from sdm_eurec4a.constants import TimeSlices
from sdm_eurec4a import conversions
from sdm_eurec4a.input_processing import models as smodels
from sdm_eurec4a.identifications import match_clouds_and_cloudcomposite, match_clouds_and_dropsondes
from sdm_eurec4a.constants import TimeSlices
from sdm_eurec4a.visulization import (
    set_custom_rcParams,
    set_paper_rcParams,
    label_from_attrs,
    adjust_lightness_array,
    plot_one_one,
    handler_map_alpha,
    save_figure,
    add_subplotlabel,
)

from sdm_eurec4a import data_loading
from sdm_eurec4a.reductions import mean_and_stderror_of_mean
from sdm_eurec4a.conversions import (
    msd_from_psd_dataarray,
    potential_temperature_from_temperature_pressure,
    relative_humidity_from_tps,
    temperature_from_potential_temperature_pressure,
)
from sdm_eurec4a.input_processing import models as smodels
from sdm_eurec4a.identifications import match_clouds_and_cloudcomposite, match_clouds_and_dropsondes
from sdm_eurec4a.constants import TimeSlices

from sdm_eurec4a.reductions import mean_and_stderror_of_mean

from revised_paper_figures_src import propagate_mean_std

### Figure Sizes
small_fig_size = np.array((16 / 3, 9 / 3))
square_fig_size = small_fig_size[0], small_fig_size[0]
large_figure_multiplicator = 12 / 8.3
large_fig_size = small_fig_size * large_figure_multiplicator
large_square_fig_size = large_fig_size[0], large_fig_size[0]
wide_fig_size = large_fig_size[0], small_fig_size[1]

###  Color Settings and Colormap defintion
def fetch_truncated_colormap(minval=0.0, maxval=1.0, n=256):
    def truncate_colormap(cmap, minval, maxval, n):
        """https://stackoverflow.com/a/18926541/16372843"""
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)),
        )
        return new_cmap

    # strength_cmap = sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)
    full_strength_cmap = sns.color_palette("mako_r", as_cmap=True)
    return truncate_colormap(full_strength_cmap, minval=minval, maxval=maxval, n=n)

default_colors = set_paper_rcParams()
# pltrcParams.update({'figure.autolayout': True}) # make sure that figures are not cut off
default_dark_colors = adjust_lightness_array(default_colors, 0.75)
strength_cmap = fetch_truncated_colormap()

### Plotting Function Definitions
def scatter_and_errorbar(
    ax: plt.Axes,
    x_var: str,
    y_var: str,
    ds: xr.Dataset,
    ds_sem: xr.Dataset,
    microphysics: Literal[
        "null_microphysics",
        "condensation",
        "collision_condensation",
        "coalbure_condensation_small",
        "coalbure_condensation_large",
    ] = "condensation",
    x_multiply: float = 1.0,
    y_multiply: float = 1.0,
    plot_patch: bool = True,
    plot_annotations: bool = True,
    patch_width: float = 1,
    scatter_kwargs: Union[dict, None] = None,
    error_kwargs: dict = dict(fmt="", label="mean ± SEM", color="black", capsize=5, linewidth=2),
    annotation_kwargs: dict = dict(
        fontsize=12,
        color="black",
    ),
) -> Tuple[dict, dict]:

    x_attrs = ds[x_var].attrs.copy()
    x = x_multiply * ds[x_var].sel(microphysics=microphysics)
    x_sem = x_multiply * ds_sem[x_var].sel(microphysics=microphysics)
    # x_sem = x * 0
    x_mean = x.mean("cloud_id")
    x_std = propagate_mean_std(x, x_sem, dim="cloud_id")

    y_attrs = ds[y_var].attrs.copy()
    y = y_multiply * ds[y_var].sel(microphysics=microphysics)
    y_sem = y_multiply * ds_sem[y_var].sel(microphysics=microphysics)
    y_mean = y.mean("cloud_id")
    y_std = propagate_mean_std(y, y_sem, dim="cloud_id")

    if scatter_kwargs == None:
        microphysics_styles = data_loading.MicrophysicsStyles()
        scatter_kwargs = microphysics_styles.get_style(key=microphysics)
    else:
        pass

    pathcollection = ax.scatter(x, y, **scatter_kwargs)

    error_container = ax.errorbar(
        x=x_mean,
        y=y_mean,
        xerr=x_std,
        yerr=y_std,
        **error_kwargs,
    )

    # for (x, y), label in zip(
    #     (, (x_mean, y_mean + 4 * y_std)),
    #     (fr"{x_mean.data:.2f}$\pm${x_std.data:.2f}", fr"{y_mean.data:.2f}$\pm${y_std.data:.2f}"),
    # ) :

    increase = max(patch_width, 1)
    offset = 1.1  # offset by 10 % to the right and top

    x_xy = ((x_mean.data + x_std.data), y_mean.data)
    x_xytext = (offset * (x_mean.data + increase * x_std.data), y_mean.data)
    x_xy = (0, 0)
    x_xytext = (0, 0)

    x_label = rf"{x_mean.data:.2f} $\pm$ {x_std.data:.2f} ${x_attrs['units']}$"

    y_xy = (x_mean.data, (y_mean.data + y_std.data))
    y_xytext = (x_mean.data, offset * (y_mean.data + increase * y_std.data))
    y_xy = (0, 0)
    y_xytext = (0, 0)

    y_label = rf"{y_mean.data:.2f} $\pm$ {y_std.data:.2f} ${y_attrs['units']}$"

    print(x_label, y_label)

    if plot_annotations:

        x_annotation = ax.annotate(
            x_label,
            xy=x_xy,
            xytext=x_xytext,
            ha="left",
            va="center",
            **annotation_kwargs,
        )

        y_annotation = ax.annotate(
            y_label,
            xy=y_xy,
            xytext=y_xytext,
            ha="center",
            va="bottom",
            rotation=90,
            **annotation_kwargs,
        )
    else:
        y_annotation = None
        x_annotation = None

    # Create a Rectangle patch

    if plot_patch:
        if patch_width > 1.0:
            wide_error_kwargs = error_kwargs.copy()
            wide_error_kwargs.update(alpha=0.1)
            error_container_wide = ax.errorbar(
                x=x_mean,
                y=y_mean,
                xerr=patch_width * x_std,
                yerr=patch_width * y_std,
                **wide_error_kwargs,
            )

        xy = x_mean.data - patch_width * x_std.data, y_mean.data - patch_width * y_std.data
        dx = patch_width * 2 * x_std.data
        dy = patch_width * 2 * y_std.data

        rect = patches.Rectangle(xy, dx, dy, linewidth=1, edgecolor="None", facecolor="k", alpha=0.1)

        ax.add_patch(rect)
    else:
        rect = None

    return (
        dict(
            x_mean=x_mean,
            y_mean=y_mean,
            x_std=x_std,
            y_std=y_std,
            x_label=x_label,
            y_label=y_label,
        ),
        dict(
            pathcollection=pathcollection,
            error_container=error_container,
            rect=rect,
            y_annotation=y_annotation,
            x_annotation=x_annotation,
        ),
    )

def plot_4_all(identified_clusters,
               cloud_composite,
               dropsonde,
               ds_distances, 
               ds_cleo_null,
               ds_cleo_sem_null, 
               ds_fitted_linear, 
               da_potential_temperature, 
               da_relative_humidity,
               da_pressure, 
               ds_relative_humidity_parameters, 
               cloud_id=396,
               pressure_reference=100000):

    fig, axs = plt.subplot_mosaic(
        [
            [
                "ax_pt",
                "ax_rh",
            ],
            [
                "ax_psd",
                "ax_lwc",
            ],
        ],
        figsize=large_square_fig_size,
        layout="constrained",
    )

    ax_psd: plt.Axes = axs["ax_psd"]
    ax_lwc: plt.Axes = axs["ax_lwc"]
    ax_pt: plt.Axes = axs["ax_pt"]
    ax_rh: plt.Axes = axs["ax_rh"]

    # PARTICLE SIZE DISTRIBUTION

    ds_observed_cloud = match_clouds_and_cloudcomposite(
        ds_clouds=identified_clusters.sel(cloud_id=cloud_id),
        ds_cloudcomposite=cloud_composite,
    )

    ds_dropsonde_cloud = match_clouds_and_dropsondes(
        ds_clouds=identified_clusters.sel(cloud_id=cloud_id),
        ds_sonde=dropsonde,
        ds_distance=ds_distances,
        max_temporal_distance=np.timedelta64("3", "h"),
        max_spatial_distance=100,
    )
    ds_dropsonde_cloud = ds_dropsonde_cloud.sel(altitude=slice(0, 1200))

    x_observed = ds_observed_cloud["radius"] * 1e6
    y_observed = ds_observed_cloud["particle_size_distribution"]

    ds_cleo_cloud = ds_cleo_null.sel(gridbox=ds_cleo_null["max_gridbox"])
    ds_cleo_cloud = ds_cleo_cloud.sel(cloud_id=cloud_id)
    x_cleo = ds_cleo_cloud["radius_bins"]

    w_cleo = (x_cleo.shift(radius_bins=-1) - x_cleo.shift(radius_bins=1)) / 2
    w_cleo = w_cleo.interpolate_na("radius_bins", method="linear", fill_value="extrapolate")
    w_cleo = w_cleo * 1e-6  # convert µm to m

    y_cleo = ds_cleo_cloud["xi_temporal_mean"] / w_cleo / ds_cleo_cloud["gridbox_volume"]
    y_cleo = y_cleo.isel(microphysics=0)
    y_cleo.attrs.update(
        units="m^{-3} m^{-1}",
        long_name="Number concentration",
    )

    ds_fitted_cloud = ds_fitted_linear.sel(cloud_id=cloud_id)
    y_fitted = ds_fitted_cloud
    x_fitted = ds_fitted_cloud["radius"] * 1e6

    # plot observed and fitted PSD
    ax_psd.plot(
        x_observed,
        y_observed,
        linestyle="",
        marker=".",
        markersize=2,
        color=[0.5, 0.5, 0.5],
        alpha=0.3,
    )
    ax_psd.plot(
        x_observed,
        y_observed.mean("time"),
        linestyle="",
        linewidth=1,
        marker=".",
        markersize=4,
        color=[0.3, 0.3, 0.3],
        alpha=0.75,
        label="Mean. Obs.",
    )

    ax_psd.plot(
        x_fitted,
        y_fitted.T,
        linestyle="-",
        color=[0.1, 0.1, 0.1],
        lw=2,
        label="Fit",
    )

    ax_psd.set_xscale("log")
    ax_psd.set_yscale("symlog", linthresh=0.5, linscale=0.3)
    ax_psd.set_ylim(-0.5, 1e13)
    ax_psd.set_xlabel(r"Radius $[\mu m]$")
    ax_psd.set_ylabel(r"Number Concentration $[m^{-3} m^{-1}]$")

    # ----------------------
    # Liquid Water Content
    # ----------------------

    x = identified_clusters["mean_rain_water_content"]
    xerr = identified_clusters["sem_rain_water_content"]
    y = ds_cleo_null["cloud_liquid_water_content"].sel(microphysics="null_microphysics")
    yerr = ds_cleo_sem_null["cloud_liquid_water_content"].sel(microphysics="null_microphysics")

    y.attrs.update(long_name="Model " + y.attrs["long_name"])

    corr = xr.corr(x, y, dim="cloud_id")
    corr_loglog = xr.corr(np.log(x + 1e-28), np.log(y + 1e-28), dim="cloud_id")

    ax_lwc.scatter(
        x=x,
        y=y,
        marker=".",
        color="k",
        alpha=0.75,
        # label = f"Pearson correlation coefficient: {corr.values:.2f}"
    )

    ax_lwc.errorbar(
        x=x,
        y=y,
        xerr=xerr,
        yerr=yerr,
        fmt=".",
        color="k",
        alpha=0.1,
        # label = f"Pearson correlation coefficient: {corr.values:.2f}"
    )

    xlim = ax_lwc.get_xlim()
    ylim = ax_lwc.get_ylim()
    lim = 0, max(xlim[1], ylim[1])

    ax_lwc.set_xlim(lim)
    ax_lwc.set_ylim(lim)

    plot_one_one(ax_lwc, color="grey", linestyle="-")
    ax_lwc.set_xlabel(label_from_attrs(x, name_width=25))
    ax_lwc.set_ylabel(label_from_attrs(y, name_width=20))
    # ax.legend(loc="upper left")

    ax_lwc.annotate(
        text=f"$R = {corr.values:.2f}$\n$R_{{log}} = {corr_loglog.values:.2f}$",
        xy=(0.65, 0.1),
        xycoords="axes fraction",
        # fontsize=10,
    )

    # ----------------------
    # POTENTIAL TEMPERATURE
    # ----------------------

    y_observed = ds_dropsonde_cloud["altitude"]
    x_observed = ds_dropsonde_cloud["potential_temperature"].transpose(..., "time")

    x_cleo = potential_temperature_from_temperature_pressure(
        air_temperature=ds_cleo_null["air_temperature"].sel(cloud_id=cloud_id),
        pressure=ds_cleo_null["pressure"].sel(cloud_id=cloud_id),
        pressure_reference=pressure_reference,
    )
    y_cleo = ds_cleo_null["gridbox_coord3"].sel(cloud_id=cloud_id)

    x_fitted = da_potential_temperature.sel(cloud_id=cloud_id)
    y_fitted = da_pressure["altitude"]

    ax_pt.plot(
        x_observed,
        y_observed,
        linestyle="-",
        color=default_colors[0],
        alpha=0.3,
    )
    ax_pt.plot(
        x_observed.mean("time"),
        y_observed,
        linestyle="-",
        color=default_dark_colors[0],
        alpha=0.75,
        lw=1,
        label="Mean. Obs.",
    )

    ax_pt.plot(
        x_fitted.T,
        y_fitted.T,
        linestyle="-",
        color=default_dark_colors[0],
        lw=2,
        label="Fit",
    )

    ax_pt.set_xlabel(r"Potential Temperature $[K]$")
    ax_pt.set_ylabel(r"Altitude $[m]$")

    # ----------------------
    # RELATIVE HUMIDITY
    # ----------------------

    y_observed = ds_dropsonde_cloud["altitude"]
    x_observed = ds_dropsonde_cloud["relative_humidity"].transpose(..., "time")

    x_cleo = relative_humidity_from_tps(
        temperature=ds_cleo_null["air_temperature"].sel(cloud_id=cloud_id),
        pressure=ds_cleo_null["pressure"].sel(cloud_id=cloud_id),
        specific_humidity=ds_cleo_null["specific_mass_vapour"].sel(cloud_id=cloud_id),
    )
    y_cleo = ds_cleo_null["gridbox_coord3"].sel(cloud_id=cloud_id)

    x_fitted = da_relative_humidity.sel(cloud_id=cloud_id)
    y_fitted = da_relative_humidity["altitude"]

    ax_rh.plot(
        x_observed,
        y_observed,
        linestyle="-",
        color=default_colors[1],
        alpha=0.3,
    )
    ax_rh.plot(
        x_observed.mean("time"),
        y_observed,
        linestyle="-",
        color=default_dark_colors[1],
        alpha=0.75,
        lw=1,
        label="Mean. Obs.",
    )
    ax_rh.plot(
        x_fitted.T,
        y_fitted.T,
        linestyle="-",
        color=default_dark_colors[1],
        lw=2,
        label="Fit",
    )

    yticks = ax_pt.get_yticks()
    ax_pt.set_xticks(np.arange(296, 302, 2))

    ax_rh.set_xlabel(r"Relative Humidity $[\%]$")

    for _ax in [ax_pt, ax_rh]:
        _ax.axhline(
            ds_relative_humidity_parameters["x_split"].sel(cloud_id=cloud_id).data,
            color="red",
            linestyle=":",
        )
        _ax.axhline(ds_observed_cloud["altitude"].mean("time").data, color="k", linestyle=":")
        _ax.set_yticks(np.arange(0, 1250, 400))
        _ax.set_ylim(0, 1200)

    ax_psd.legend(loc="lower left")

    ax_psd.set_title(f"Measurements {len(ds_observed_cloud['time'].data)}")
    ax_pt.set_title(f"Measurements {len(ds_dropsonde_cloud['time'].data)}")
    ax_rh.set_title(f"Measurements {len(ds_dropsonde_cloud['time'].data)}")

    fig.suptitle(
        f"Cloud ID {cloud_id} | Pressure Ref. {pressure_reference/100:.2f} hPa | Time {ds_observed_cloud['time'].data[0].astype('datetime64[m]')}"
    )
    fig.tight_layout()
    return fig, axs

def add_zoom_box(ax, xlims, ylims):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (xlims[0], ylims[0]),
        xlims[1] - xlims[0],
        ylims[1] - ylims[0],
        linewidth=1,
        linestyle="--",
        edgecolor='darkblue',
        facecolor='none'
    )
    ax.add_patch(rect)

    return ax


### Plot specific figure functions
def plot_figure_1(cloud_composite,
                  dropsonde,
                  identified_clusters,
                  ds_distances,
                  ds_parameters_linear,
                  ds_potential_temperature_parameters,
                  ds_pressure_parameters,
                  ds_relative_humidity_parameters,
                  ds,
                  ds_cleo_null,
                  ds_cleo_sem_null,
                  cloud_id=396,
                  pressure_reference=100000
               ):
    ### Get DSD Fits
    radius = np.geomspace(50e-6, 3e-3, 100)
    # create radius test DataArray with radius as dimension and cloud_id as coordinate
    # this dataset will be used to compute the fitted distributions
    t_test = xr.DataArray(
        radius,
        dims="radius",
        coords={"radius": radius},
    )
    t_test = t_test.expand_dims(cloud_id=ds["cloud_id"])

    # weight bin width for log spaced bins
    w_test = (t_test["radius"].shift(radius=-1) - t_test["radius"].shift(radius=1)) / 2
    w_test = w_test.interpolate_na("radius", method="linear", fill_value="extrapolate")

    # fit the double log-normal distribution
    ds_fitted_linear: xr.DataArray = smodels.double_log_normal_distribution_all(
        x=t_test,  # type: ignore
        mu1=ds_parameters_linear["geometric_mean1"],  # type: ignore
        mu2=ds_parameters_linear["geometric_mean2"],  # type: ignore
        sigma1=ds_parameters_linear["geometric_std_dev1"],  # type: ignore
        sigma2=ds_parameters_linear["geometric_std_dev2"],  # type: ignore
        scale1=ds_parameters_linear["scale_factor1"],  # type: ignore
        scale2=ds_parameters_linear["scale_factor2"],  # type: ignore
        parameter_space="geometric",
        x_space="linear",
    )  # type: ignore

    # multiply fitted psd with bin width to get number concentration and then compute the mass size distribution
    fitted_linear_psd = ds_fitted_linear * w_test
    fitted_linear_msd = msd_from_psd_dataarray(ds_fitted_linear * w_test)

    ### Get Thermo Fits
    da_potential_temperature: xr.DataArray = smodels.split_linear_func(
        x=dropsonde["altitude"],  # type: ignore
        f_0=ds_potential_temperature_parameters["f_0"],  # type: ignore
        slope_1=ds_potential_temperature_parameters["slope_1"],  # type: ignore
        slope_2=ds_potential_temperature_parameters["slope_2"],  # type: ignore
        x_split=ds_potential_temperature_parameters["x_split"],  # type: ignore
    )  # type: ignore
    da_potential_temperature = da_potential_temperature.sel(altitude=slice(0, 1200))

    da_relative_humidity: xr.DataArray = smodels.split_linear_func(
        x=dropsonde["altitude"],  # type: ignore
        f_0=ds_relative_humidity_parameters["f_0"],  # type: ignore
        slope_1=ds_relative_humidity_parameters["slope_1"],  # type: ignore
        slope_2=ds_relative_humidity_parameters["slope_2"],  # type: ignore
        x_split=ds_relative_humidity_parameters["x_split"],  # type: ignore
    )
    da_relative_humidity = da_relative_humidity.sel(altitude=slice(0, 1200))

    da_pressure: xr.DataArray = smodels.linear_func(
        x=dropsonde["altitude"],  # type: ignore
        slope=ds_pressure_parameters["slope"],  # type: ignore
        f_0=ds_pressure_parameters["f_0"],  # type: ignore
    )
    da_pressure = da_pressure.sel(altitude=slice(0, 1200))

    ### Plotting
    fig, axs = plot_4_all(identified_clusters,
               cloud_composite,
               dropsonde,
               ds_distances, 
               ds_cleo_null,
               ds_cleo_sem_null, 
               ds_fitted_linear, 
               da_potential_temperature, 
               da_relative_humidity,
               da_pressure, 
               ds_relative_humidity_parameters, 
               cloud_id=cloud_id,
               pressure_reference=pressure_reference)

    add_subplotlabel(
      axs=axs.values(),
      location="upper left",
  )

    axs["ax_psd"].set_yticks([0, 1e0, 1e3, 1e6, 1e9, 1e12])
    lwc_ticks = np.arange(0, 0.8, 0.2)
    axs["ax_lwc"].set_yticks(lwc_ticks)
    axs["ax_lwc"].set_xticks(lwc_ticks)
    axs["ax_lwc"].set_ylabel("Fitted Rain Water\nContent " + r"$[g m^{-3}]$")
    axs["ax_lwc"].set_xlabel("Observed Rain Water\nContent " + r"$[g m^{-3}]$")

    axs["ax_lwc"].scatter(
        x=identified_clusters["mean_rain_water_content"].sel(cloud_id=cloud_id),
        y=ds_cleo_null["cloud_liquid_water_content"]
        .sel(microphysics="null_microphysics")
        .sel(cloud_id=cloud_id),
        marker=".",
        color="r",
        alpha=1,
        zorder=10,
    )

    for key in axs:
        try:
            axs[key].get_legend().remove()
        except AttributeError:
            pass

    for _ax in [axs["ax_pt"], axs["ax_rh"]]:
        xlim = _ax.get_xlim()
        xy = (xlim[0], 200)
        width = xlim[1] - xlim[0]
        height = 500 - 200
        rect = mpatches.Rectangle(
            xy=xy,
            width=width,
            height=height,
            linewidth=0,
            edgecolor="None",
            facecolor=[0.5, 0.5, 0.5, 0.1],
        )

        # Add the patch to the Axes
        _ax.add_patch(rect)

    fig.suptitle("")
    for _ax in axs.values():
        _ax.set_title("")

    fig.tight_layout()

    return fig

def plot_figure_2(ds_conservation):
    rolling_indices = 30
    xlim = (0, 3600)
    ylim = (0, 11.5)

    fig, ax = plt.subplots()
    ax: plt.Axes = ax

    x = ds_conservation["time"]

    y = -ds_conservation["outflow_precipitation"].transpose("time", ...)
    y_rolling = y.rolling(time=rolling_indices, center=True).mean()
    x_rolling = x.rolling(time=rolling_indices, center=True).mean()

    y_mean, y_sem = mean_and_stderror_of_mean(y.sel(time=TimeSlices.quasi_stationary_state), dims=("time",))

    total_mean = y_mean.mean("cloud_id")
    total_std = propagate_mean_std(data=y_mean, data_std=y_sem, dim="cloud_id")

    total_median = y_mean.median("cloud_id")

    x = x.isel(time=slice(0, -2))
    y = y.isel(time=slice(0, -2))

    ax.plot(
        x_rolling,
        y_rolling,
        color="grey",
        alpha=0.2,
        linewidth=0.5,
        zorder=10,
    )

    ax.plot(
        x_rolling,
        y_rolling.mean("cloud_id"),
        color="k",
        alpha=1,
        linestyle="--",
        zorder=10,
        label=rf"Mean: {total_mean.data:.2f}$\pm${total_std.data:.2f} ${y.attrs['units']}$",
    )
    ax.fill_between(
        x_rolling,
        y_rolling.mean("cloud_id") + y_rolling.std("cloud_id"),
        y_rolling.mean("cloud_id") - y_rolling.std("cloud_id"),
        label="Std.Dev.",
        color=adjust_lightness("grey", 1.7),
        alpha=1,
        zorder=3,
    )


    ax.plot(
        x_rolling,
        y_rolling.median("cloud_id"),
        color="k",
        alpha=1,
        linestyle="-",
        zorder=10,
        label=rf"Median: {total_median.data:.2f} ${y.attrs['units']}$",
    )

    ax.fill_between(
        x_rolling,
        y_rolling.quantile(0.25, "cloud_id"),
        y_rolling.quantile(0.75, "cloud_id"),
        label=f"25-75%",
        color=adjust_lightness("grey", 1.2),
        alpha=1,
        zorder=4,
    )


    ax.fill_betweenx(
        [10, 11],
        400,
        TimeSlices.quasi_stationary_state.stop,
        color=default_colors[2],
        alpha=0.1,
        edgecolor="none",
    )
    ax.fill_betweenx(
        [10, 11],
        TimeSlices.quasi_stationary_state.start,
        TimeSlices.quasi_stationary_state.stop,
        color=default_colors[2],
        alpha=0.1,
        edgecolor="none",
    )
    ax.annotate(
        text="Quasi-Stationary State",
        xy=(
            (TimeSlices.quasi_stationary_state.start + TimeSlices.quasi_stationary_state.stop) / 2,
            10.5,
        ),
        ha="center",
        va="center",
        fontsize=8,
        color=default_dark_colors[2],
    )
    ax.fill_betweenx(
        [10, 11],
        0,
        400,
        color=default_colors[1],
        alpha=0.1,
        edgecolor="none",
    )
    ax.annotate(
        text="Spin-Up",
        xy=(
            (0 + 400) / 2,
            10.5,
        ),
        ha="center",
        va="center",
        fontsize=8,
        color=default_dark_colors[1],
    )

    ax.set_ylim(ylim)
    ax.set_ylabel(label_from_attrs(y, name_width=25))
    ax.set_xlabel(r"Simulation time $[s]$")
    ax.set_xlim(xlim)
    # ax.set_yticks(yticks)
    ax.legend(loc="center right")

    ax.axvline(
        TimeSlices.quasi_stationary_state.start,
        color=default_colors[2],
        linestyle="-",
        linewidth=1,
        alpha=1,
        zorder=20,
        label="Stationary State",
    )

    fig.tight_layout()

    return fig

def plot_figure_3(ds, ds_sem, microphysics_styles):
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5.33, 4))

    ylim = (0, 80)
    ax_x_hist: plt.Axes = axs[0]
    ax_y_hist: plt.Axes = axs[1]

    x = ds["cloud_liquid_water_content"]
    x_sem = ds_sem["cloud_liquid_water_content"]
    y = conversions.EvaporationUnits(data=-ds["outflow_energy"], input_type="energy").convert_to(
        "precipitation"
    )
    y_sem = conversions.EvaporationUnits(data=-ds_sem["outflow_energy"], input_type="energy").convert_to(
        "precipitation"
    )

    x_mean = x.mean("cloud_id")
    x_std = propagate_mean_std(x, x_sem, dim="cloud_id")

    y_mean = y.mean("cloud_id")
    y_std = propagate_mean_std(y, y_sem, dim="cloud_id")

    x_median = x.median("cloud_id")
    y_median = y.median("cloud_id")

    x_bins = np.arange(0, 1, 0.05)
    y_bins = np.arange(0, 10, 0.2)

    x_dict = dict(
        data=x,
        mean=x_mean,
        std=x_std,
        median=x_median,
        bins=x_bins,
    )
    y_dict = dict(
        data=y,
        mean=y_mean,
        std=y_std,
        median=y_median,
        bins=y_bins,
    )

    for mp in ["condensation"]:
        style = microphysics_styles.get_style(mp)
        for d, _ax, rounding in zip([x_dict, y_dict], [ax_x_hist, ax_y_hist], [2, 2]):

            units = d["data"].attrs.get("units", "")
            units = rf"${units}$"

            median = d["median"].sel(microphysics=mp)
            m, s = d["mean"].sel(microphysics=mp), d["std"].sel(microphysics=mp)

            # round values for display
            m_rounded = np.round(m, rounding)
            s_rounded = np.round(s, rounding)
            median_rounded = np.round(median, rounding)

            # if rounded value is integer, convert to int for display
            if rounding == 0:
                m_rounded = m_rounded.astype(int)
                s_rounded = s_rounded.astype(int)
                median_rounded = median_rounded.astype(int)

            _ax.hist(
                d["data"].sel(microphysics=mp),
                bins=d["bins"],
                histtype="step",
                color=style["color"],
                lw=2,
            )
            _ax.axvline(
                m,
                color=style["color"],
                linestyle="--",
                lw=2,
                label=rf"Mean: {m_rounded.data} $\pm$ {s_rounded.data} {units}",
            )
            _ax.fill_betweenx(
                ylim,
                m - s,
                m + s,
                color=style["color"],
                alpha=0.1,
            )
            _ax.axvline(
                median,
                color=style["color"],
                linestyle="-",
                lw=2,
                label=f"Median: {median_rounded.data} {units}",
            )

    ax_x_hist.set_xlabel(label_from_attrs(x))
    ax_y_hist.set_xlabel(label_from_attrs(y))

    ax_y_hist.set_xticks(np.arange(0, 12, 2))
    for _ax in axs.flatten():
        _ax.set_ylabel("Counts")
        _ax.set_ylim(ylim)
        _ax.legend(loc="upper right")
        _ax.set_xlim(0, None)

    y_ticks = xr.DataArray(ax_y_hist.get_xticks(), attrs=y.attrs.copy())
    new_y_ticks: xr.DataArray = conversions.EvaporationUnits(data=y_ticks, input_type="precipitation").convert_to(
        "energy"
    )
    new_ticks_func = lambda _: [f"{round(new_x, 2):.0f}" for x, new_x in zip(y_ticks, new_y_ticks.data)]
    add_additional_axis(
        ax=ax_y_hist,
        new_ticks_func=new_ticks_func,
        label=label_from_attrs(da=new_y_ticks),
        position="top",
        offset_position=["axes", 1],
    )

    add_subplotlabel([ax_x_hist, ax_y_hist], location="title", zorder=100)

    fig.tight_layout()

    return fig

def plot_figure_4(ds, ds_sem, microphysics_styles):

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5.33, 4.4))

    ylim = (0, 40)
    ax_x_hist: plt.Axes = axs[0]
    ax_y_hist: plt.Axes = axs[1]

    x = -ds["source_energy"]
    x_sem = -ds_sem["source_energy"]

    y = ds["evaporation_fraction"]
    y_sem = ds_sem["evaporation_fraction"]

    x_mean = x.mean("cloud_id")
    x_std = propagate_mean_std(x, x_sem, dim="cloud_id")

    y_mean = y.mean("cloud_id")
    y_std = propagate_mean_std(y, y_sem, dim="cloud_id")

    x_median = x.median("cloud_id")
    y_median = y.median("cloud_id")

    x_bins = np.arange(0, 1000, 50)
    y_bins = np.arange(0, 101, 5)

    x_dict = dict(
        data=x,
        mean=x_mean,
        std=x_std,
        median=x_median,
        bins=x_bins,
    )
    y_dict = dict(
        data=y,
        mean=y_mean,
        std=y_std,
        median=y_median,
        bins=y_bins,
    )

    for mp in ["condensation"]:
        style = microphysics_styles.get_style(mp)
        for d, _ax, rounding in zip([x_dict, y_dict], [ax_x_hist, ax_y_hist], [1, 1]):

            units = d["data"].attrs.get("units", "")
            units = rf"${units}$"

            median = d["median"].sel(microphysics=mp)
            m, s = d["mean"].sel(microphysics=mp), d["std"].sel(microphysics=mp)

            # round values for display
            m_rounded = np.round(m, rounding)
            s_rounded = np.round(s, rounding)
            median_rounded = np.round(median, rounding)

            # if rounded value is integer, convert to int for display
            if rounding == 0:
                m_rounded = m_rounded.astype(int)
                s_rounded = s_rounded.astype(int)
                median_rounded = median_rounded.astype(int)

            _ax.hist(
                d["data"].sel(microphysics=mp),
                bins=d["bins"],
                histtype="step",
                color=style["color"],
                lw=2,
            )
            _ax.axvline(
                m,
                color=style["color"],
                linestyle="--",
                lw=2,
                label=rf"Mean: {m_rounded.data} $\pm$ {s_rounded.data} {units}",
            )
            _ax.fill_betweenx(
                ylim,
                m - s,
                m + s,
                color=style["color"],
                alpha=0.1,
            )
            _ax.axvline(
                median,
                color=style["color"],
                linestyle="-",
                lw=2,
                label=f"Median: {median_rounded.data} {units}",
            )

    ax_x_hist.set_xlabel(label_from_attrs(x))
    ax_y_hist.set_xlabel(label_from_attrs(y))

    for _ax in axs.flatten():
        _ax.set_ylabel("Counts")
        _ax.set_ylim(ylim)
        _ax.legend(loc="upper right")
        _ax.set_xlim(0, None)

    add_subplotlabel([ax_x_hist, ax_y_hist], location="title", zorder=100)

    # update the title to be on the left
    ax_x_hist.set_title("")

    x_ticks = xr.DataArray(ax_x_hist.get_xticks(), attrs=x.attrs.copy())
    new_x_ticks: xr.DataArray = conversions.EvaporationUnits(data=x_ticks, input_type="energy").convert_to(
        "precipitation"
    )
    new_ticks_func = lambda _: [f"{round(new_x, 2):.2f}" for x, new_x in zip(x_ticks, new_x_ticks.data)]
    add_additional_axis(
        ax=ax_x_hist,
        new_ticks_func=new_ticks_func,
        label=label_from_attrs(da=new_x_ticks),
        position="top",
        offset_position=["axes", 1],
    )
    ax_x_hist.set_xlabel(label_from_attrs(da=x))


    fig.tight_layout()

    return fig

def plot_figure_5(ds, ds_normalized, microphysics_styles):
    y_ticks = np.arange(0, 1.01, 0.25)

    fig = plt.figure(figsize=wide_fig_size)
    gs = fig.add_gridspec(nrows=1, ncols=1)

    ax = fig.add_subplot(gs[:, :])

    plot_microphysics = ["condensation"]

    x = -ds_normalized["evaporation_rate_energy"]
    y = ds_normalized["normalized_gridbox_coord3"]

    c = ds_normalized["liquid_water_content"]

    norm = mcolors.Normalize(vmin=0, vmax=ds["cloud_liquid_water_content"].max().data)


    for mp in plot_microphysics:
        _x = x.sel(microphysics=mp)
        _y = y

        # select all but the top most gridboxes
        _x = _x.sel(normalized_gridbox_coord3=slice(0, 0.99))
        _y = _y.sel(normalized_gridbox_coord3=slice(0, 0.99))
        _c = c.sel(normalized_gridbox_coord3=slice(0, 0.99))

        md_mean = _x.mean("cloud_id")
        style_full = microphysics_styles[mp].copy()

        _xx = _x
        _yy = _y.expand_dims(cloud_id=_x["cloud_id"])
        _cc = _c.sel(microphysics=mp)

        sc = ax.scatter(
            _xx,
            _yy,
            c=_cc,
            s=0,
            alpha=1,
            marker=".",
            cmap=strength_cmap,
            norm=norm,
        )

        for cloud_id in x["cloud_id"]:

            xx = np.flip(_xx.sel(cloud_id=cloud_id).data)
            yy = np.flip(_yy.sel(cloud_id=cloud_id).data)
            cc = np.flip(_cc.sel(cloud_id=cloud_id).data)

            rng = np.arange(0, len(xx) - 1)
            lines = [[(xx[i], yy[i]), (xx[i + 1], yy[i + 1])] for i in rng]
            colors = strength_cmap(norm(cc[rng]))

            lc = mcollections.LineCollection(segments=lines, colors=colors)  # Use a random colormap
            lc.set_linewidth(0.75)  # Set line width
            lc.set_alpha(1)  # Set line width
            ax.add_collection(lc)  # Add the line collection to the axes

        ax.set_yticks(y_ticks)
        ax.set_yticklabels([])

        ### mean and std
        ### median and IQR
        ax.plot(
            _x.median("cloud_id"),
            _y,
            label=style_full["name"] + " Median",
            color=style_full["dark_color"],
            linestyle="-",
            zorder=4,
        )
        ax.fill_betweenx(
            _y,
            _x.quantile(0.25, "cloud_id"),
            _x.quantile(0.75, "cloud_id"),
            alpha=0.3,
            color=adjust_lightness(style_full["light_color"], 1.5),
            zorder=2,
            label=style_full["name"] + " IQR",
        )

        ax.plot(
            md_mean,
            _y,
            label=style_full["name"] + " Mean",
            color=style_full["color"],
            linestyle="--",
            zorder=4,
        )

    ax.set_yticks(y_ticks, y_ticks)
    ax.legend(loc="upper right")

    ax.set_xlim(0, None)
    ax.set_ylim(0, 1)

    ax.set_xlabel(label_from_attrs(x))
    ax.set_ylabel(label_from_attrs(y, return_units=False, name_width=25))

    fig.colorbar(sc, ax=ax, label=label_from_attrs(c, name_width=20, linebreak=True))

    fig.tight_layout()

    return fig

def plot_figure_6(ds, microphysics_styles):
    def plot_subfigure(ax):

        x = ds["inflow_energy"]
        y = -ds["source_energy"]

        for mp in ["condensation"]:
            style = microphysics_styles.get_style(mp)
            ax.scatter(
                x.sel(microphysics=mp),
                y.sel(microphysics=mp),
                **style,
            )

        return ax, x, y

    def plot_isolines(ax, xlim, ylim, high_res=False):
        lims = np.concatenate([xlim, ylim])
        p_x_values = np.geomspace(lims.min(), lims.max(), 100)

        values_label_size = 10

        if high_res:
            # p_list = [1, 0.31622, 0.1, 0.031622, 0.01]
            p_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
            _x_list = [7.5] + [None]*8 + [280, 3e3]
            _y = 25
        else:
            p_list = [1, 0.1, 0.01]
            _x_list = [0.13, 13, 130]
            _y = 0.9

        for i in range(len(p_list)):
            p = p_list[i]
            style = dict(color="black", alpha=p ** (1 / 3.5))
            ax.plot(p_x_values, p * p_x_values, "--", linewidth=1, zorder=0, **style)

            if high_res:
                _x =_x_list[i]
                if _x is not None:
                    ax.annotate(
                        f"{100 * p:.0f} %",
                        xy=(_x, _y),
                        xytext=(0, 0),
                        textcoords="offset points",
                        va="top",
                        ha="left",
                        size=values_label_size,
                        **style,
                    )
            else:
                _x =_x_list[i]
                if _x is not None:
                    ax.annotate(
                        f"{100 * p:.0f} %",
                        xy=(_x, _y),
                        xytext=(0, 0),
                        textcoords="offset points",
                        va="top",
                        ha="left",
                        size=values_label_size,
                        **style,
                    )

        return ax

    def add_second_axis(ax, x, y):
        x_ticks = xr.DataArray(ax.get_xticks(), attrs=x.attrs.copy())
        new_x_ticks: xr.DataArray = conversions.EvaporationUnits(data=x_ticks, input_type="energy").convert_to(
            "precipitation"
        )

        factor = new_x_ticks / x_ticks

        assert (
            np.abs(factor.std() / factor.mean()) < 1e-6
        ), f"Conversion factor is not constant: std={factor.std}, mean={factor.mean}"
        factor = factor.mean().data    

        # add a second xaxis
        ax2 = ax.twiny()
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlim(factor * xlim)  # Sync the x-limits
        ax2.set_ylim(ylim)  # Sync the x-limits

        ax2.set_xlabel(label_from_attrs(da=new_x_ticks))
        ax.grid(color="grey", alpha=0.25, linewidth=0.75)
        ax2.grid(color="grey", alpha=0.25, linewidth=0.75, linestyle=":")

        ax.set_xlabel(label_from_attrs(x))
        ax.set_ylabel(label_from_attrs(y, name_width=20))

        return ax

    fig, axs = plt.subplots(1, 2, figsize=(large_fig_size[0] * 1.25, large_fig_size[1]))

    ax0, x, y = plot_subfigure(axs[0])
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_ylim(5e-3, 1e3)
    ax0.set_xlim(1e-1, 1e4)
    xlim = np.array(ax0.get_xlim())
    ylim = np.array(ax0.get_ylim())
    ax0 = plot_isolines(ax0, xlim, ylim)
    ax0 = add_second_axis(ax0, x, y)

    ax1, x, y = plot_subfigure(axs[1])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(5, 1e3)
    ax1.set_xlim(7, 1e4)
    xlim = np.array(ax1.get_xlim())
    ylim = np.array(ax1.get_ylim())
    ax1 = plot_isolines(ax1, xlim, ylim, high_res=True)
    ax1 = add_second_axis(ax1, x, y)

    add_zoom_box(ax0,
                 [ax1.get_xlim()[0]*1.025, ax1.get_xlim()[1]*0.95],
                 [ax1.get_ylim()[0]*1.025, ax1.get_ylim()[1]*0.95])
    add_zoom_box(ax1,
                 [ax1.get_xlim()[0]*1.025, ax1.get_xlim()[1]*0.975],
                 [ax1.get_ylim()[0]*1.025, ax1.get_ylim()[1]*0.975])

    fig.tight_layout()

    return fig

def plot_figure_7(ds, ds_correlations_EF, ds_correlations_CIE, ds_correlations_MEH, microphysics_styles):
    fig, axs = plt.subplots(ncols=3, figsize=wide_fig_size)

    axs_ef: plt.Axes = axs[1]
    axs_cie: plt.Axes = axs[0]
    axs_meh: plt.Axes = axs[2]

    y = ds["evaporation_fraction"].sel(microphysics="condensation")
    x = ds["cloud_mass_radius_mean"].sel(microphysics="condensation")
    correlation = ds_correlations_EF[x.name].sel(microphysics="condensation")
    axs_ef.set_title(f" R = {correlation.data:.2f}")
    axs_ef.scatter(
        x,
        y,
        **microphysics_styles.get_style("condensation"),
    )
    axs_ef.set_xlabel(label_from_attrs(x, name_width=20))
    axs_ef.set_ylabel(label_from_attrs(y, name_width=20))

    y = -ds["source_energy"].sel(microphysics="condensation")
    x = ds["inflow_energy"].sel(microphysics="condensation")
    correlation = ds_correlations_CIE[x.name].sel(microphysics="condensation")
    axs_cie.set_title(f" R = {correlation.data:.2f}")
    axs_cie.scatter(
        x,
        y,
        **microphysics_styles.get_style("condensation"),
    )
    axs_cie.set_xlabel(label_from_attrs(x, name_width=20))
    axs_cie.set_ylabel(label_from_attrs(y, name_width=20))

    y = ds["mean_evaporation_height"].sel(microphysics="condensation")
    x = ds["cloud_mass_radius_mean"].sel(microphysics="condensation")
    correlation = ds_correlations_MEH[x.name].sel(microphysics="condensation")
    axs_meh.set_title(f" R = {correlation.data:.2f}")
    axs_meh.scatter(
        x,
        y,
        **microphysics_styles.get_style("condensation"),
    )
    axs_meh.set_xlabel(label_from_attrs(x, name_width=20))
    axs_meh.set_ylabel(label_from_attrs(y, name_width=20))

    add_subplotlabel(axs=axs, location="upper left")

    fig.tight_layout()

    return fig

def plot_figure_8(ds_normalized):
    fig, axs = plt.subplots(1, 3, figsize=(small_fig_size[0]*2.0, small_fig_size[1]))
    #fig, axs = plt.subplots(2, 2, figsize=(large_fig_size[0], large_fig_size[1]*1.25))
    axs = axs.flatten()
    cmap = fetch_truncated_colormap()

    def plot_hexbin(ax, x, y, w, xscale="linear", vmin=0.0, vmax=None, tile=True):
        if tile:
            y_flat = np.tile(y.values, x.sizes.get("cloud_id", 1))
        else:
            y_flat = y.values.flatten()
        hb = ax.hexbin(x.values.flatten(), y_flat, w.values.flatten(), reduce_C_function=np.mean,
                       gridsize=(5,3), xscale=xscale,  cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(hb, ax=ax, label='Evaporation rate [$W \\, m^{-3}$]')

    w = -ds_normalized.evaporation_rate_energy.sel(microphysics="condensation")

    # evaporation_rate on 'x' vs. normalised height
    keys = ("relative_humidity", "liquid_water_content", "mass_radius_mean")

    labels = {
        "mass_radius_mean": label_from_attrs(ds_normalized["mass_radius_mean"], return_units=False),
        "liquid_water_content": label_from_attrs(ds_normalized["liquid_water_content"], return_units=True),
        "relative_humidity": label_from_attrs(ds_normalized["relative_humidity"], return_units=True),
    }
    labels["mass_radius_mean"] = "Mean mass radius [µm]"

    y = ds_normalized.normalized_gridbox_coord3
    for k, key in enumerate(keys):
        ax, label = axs[k], labels[key]
        x = ds_normalized[key].sel(microphysics="condensation")
        xscale="linear"
        vmax = 0.5
        if key == "liquid_water_content":
            vmax = 1.5
        
        plot_hexbin(ax, x, y, w, xscale=xscale, vmax=vmax)

        if k == 0:
            ax.set_ylabel("Normalized height []") 
        ax.set_xlabel(label)

        fig.tight_layout()

    # ### evaporation_rate on LWC vs RelH
    # ax = axs[-1]
    # x = ds_normalized["liquid_water_content"].sel(microphysics="condensation")
    # y = ds_normalized["relative_humidity"].sel(microphysics="condensation")
    # plot_hexbin(ax, x, y, w, xscale="linear", tile=False) 
    # ax.set_xlabel("liquid water content")
    # ax.set_ylabel("relative humidity") 

    fig.tight_layout()

    return fig

def plot_figure_9(ds, ds_no_ventilation, evaporation_fraction_ventilation, evaporation_fraction, microphysics_styles):
    fig, ax = plt.subplots(1, 1)

    x = ds["cloud_mass_radius_mean"]
    y = ds["evaporation_fraction"]

    x_no_ventilation = ds_no_ventilation["cloud_mass_radius_mean"]
    y_no_ventilation = ds_no_ventilation["evaporation_fraction"]

    for mp in ["condensation"]:
        style = microphysics_styles.get_style(mp)
        style["label"] += r" $\mathbf{with} \, f_v$"
        ax.scatter(
            x.sel(microphysics=mp),
            y.sel(microphysics=mp),
            **style,
        )
        style = microphysics_styles.get_style(mp, colortype="light")
        style["label"] += r" $\mathbf{without} \, f_v$"
        style["color"] = "grey"
        style["marker"] = "."
        ax.scatter(
            x_no_ventilation.sel(microphysics=mp),
            y_no_ventilation.sel(microphysics=mp),
            **style,
        )

    ax.plot(
        ds["radius_bins"],
        1e2 * evaporation_fraction_ventilation,
        label=r"Theory $\mathbf{with} \, f_v$",
        color="black",
        linestyle="--",
    )

    ax.plot(
        ds["radius_bins"],
        1e2 * evaporation_fraction,
        label=r"Theory $\mathbf{without} \, f_v$",
        color="grey",
        linestyle="--",
    )

    ax.set_xlim(50, None)
    ax.set_ylim(1, None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(label_from_attrs(x))
    ax.set_ylabel(label_from_attrs(y))
    ax.legend(loc="lower left")

    fig.tight_layout()

    return fig

def plot_figure_10(ds_normalized, ds_normalized_sem, microphysics_styles):
    x_all = -ds_normalized["evaporation_rate_energy"]
    x_refernce = x_all.sel(microphysics="condensation")

    x_sem_all = ds_normalized_sem["evaporation_rate_energy"]
    x_sem_refernce = x_sem_all.sel(microphysics="condensation")

    attrs = x_all.attrs.copy()

    A = x_all
    B = x_refernce
    dA = x_sem_all
    dB = x_sem_refernce

    f = (A - B) / B
    df = ((1 / B * dA) ** 2 + (-A / B**2 * dB) ** 2) ** 0.5

    x = f * 100
    x_sem = df * 100

    x.attrs.update(
        long_name=f"Relative difference of {attrs['long_name']} compared to {microphysics_styles['condensation']['name']}",
        units=r"\%",
    )

    y = ds_normalized["normalized_gridbox_coord3"]


    y_ticks = [0, 0.5, 1]

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5, 5), sharey=True)


    plot_microphysics = [
        "collision_condensation",
        "coalbure_condensation_small",
        "coalbure_condensation_large",
    ]

    for _ax, mp in zip(axs, plot_microphysics):

        _x = x.sel(microphysics=mp)
        _x_std = x_sem.sel(microphysics=mp)
        md_mean = _x.mean("cloud_id")
        md_std = propagate_mean_std(_x, _x_std, dim="cloud_id")

        style_full = microphysics_styles[mp].copy()

        _ax.set_title(microphysics_styles.get_setup(mp)["name"])

        _ax.plot(
            md_mean,
            y,
            color=style_full["dark_color"],
            linestyle="--",
            label="Mean",
            zorder=10,
        )
        _ax.plot(
            _x.median("cloud_id"),
            y,
            color=style_full["dark_color"],
            linestyle="-",
            label="Median",
            zorder=10,
        )

        _ax.fill_betweenx(
            y,
            _x.quantile(0.1, "cloud_id"),
            _x.quantile(0.9, "cloud_id"),
            zorder=1,
            color=adjust_lightness(style_full["light_color"], 1.5),
            label="10-90%",
        )
        _ax.fill_betweenx(
            y,
            _x.quantile(0.25, "cloud_id"),
            _x.quantile(0.75, "cloud_id"),
            zorder=2,
            color=adjust_lightness(style_full["light_color"], 1.4),
            label="25-75%",
        )
        _ax.fill_betweenx(
            y,
            _x.quantile(0.33, "cloud_id"),
            _x.quantile(0.66, "cloud_id"),
            zorder=3,
            color=adjust_lightness(style_full["light_color"], 1.3),
            label="33-66%",
        )


    for _ax in axs:
        _ax.axvline(0, color="k", linestyle="--", alpha=0.5, zorder=10)
        _ax.set_ylim(0, 1)
        _ax.set_yticks(y_ticks)

    for _ax in [axs[0], axs[1]]:
        _ax.set_xlim(-15, 15)
    axs[2].set_xlim(-15, 150)

    fig.supxlabel(label_from_attrs(x, name_width=40))
    fig.supylabel(label_from_attrs(y))

    add_subplotlabel(axs=list(axs))

    fig.tight_layout()

    return fig

def plot_figure_11(ds, microphysics_styles):
    def plot_relative_differences(
        ax: plt.Axes,
        ds: xr.Dataset,
        x_var_name: str,
        y_var_name: str,
        microphysics_list: list = [
            "collision_condensation",
            "coalbure_condensation_small",
            "coalbure_condensation_large",
        ],
    ):
        x = ds[x_var_name]
        if x_var_name == "inflow_energy":
            x = conversions.EvaporationUnits(data=x, input_type="energy").convert_to("precipitation")

        y_all = ds[y_var_name]
        y_refernce = y_all.sel(microphysics="condensation")
        attrs = y_all.attrs.copy()
        y = (y_all - y_refernce) / y_refernce * 100

        y.attrs.update(
            # long_name=f"{attrs['long_name']} relative difference to {microphysics_styles['condensation']['name']}",
            long_name=f"{attrs['long_name']} relative difference",
            units=r"\%",
        )

        x = x.sel(microphysics=microphysics_list)
        y = y.sel(microphysics=microphysics_list)

        for mp in microphysics_list:
            style = microphysics_styles.get_style(mp).copy()
            ax.scatter(
                x.sel(microphysics=mp),
                y.sel(microphysics=mp),
                alpha=0.75,
                **style,
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, zorder=0)

        ax.set_xlabel(label_from_attrs(x))
        ax.set_ylabel(label_from_attrs(y, name_width=25))

        return ax

    variable_combinations = [
        ("inflow_energy", "source_energy"),
        ("inflow_energy", "source_energy"),
        # ("cloud_mass_radius_mean", "source_energy"),
        ("inflow_energy", "evaporation_fraction"),
        ("inflow_energy", "evaporation_fraction"),
        # ("cloud_mass_radius_mean", "evaporation_fraction"),
    ]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=large_fig_size * 1.2)

    for _ax, (_x, _y) in zip(
        axs.flatten(),
        variable_combinations,
    ):
        _ax = plot_relative_differences(
            ax=_ax,
            ds=ds,
            x_var_name=_x,
            y_var_name=_y,
            microphysics_list=[
                "coalbure_condensation_large",
                "collision_condensation",
                "coalbure_condensation_small",
            ],
        )

    # remove repeated labels
    for ax in axs[0,:]:
        ax.set_xlabel(None)
    for ax in axs[:,1]:
        ax.set_ylabel(None)

    for _ax in axs.flatten():
        _ax.set_xscale("log")

    axs[0,0].legend(loc="upper left")

    for _ax in [axs[0,0], axs[1,0]]:
        _ax.set_ylim(-25, 800)

    for _ax in [axs[0,1], axs[1,1]]:
        _ax.set_ylim(-25, 100)
        _ax.set_xlim(left=1e-1)

    add_zoom_box(axs[0,0],
                 [axs[0,1].get_xlim()[0]*1.015, axs[0,1].get_xlim()[1]*0.95],
                 [axs[0,1].get_ylim()[0]*0.975, axs[0,1].get_ylim()[1]])

    add_zoom_box(axs[0,1],
                 [axs[0,1].get_xlim()[0]*1.015, axs[0,1].get_xlim()[1]*0.975],
                 [axs[0,1].get_ylim()[0]*0.975, axs[0,1].get_ylim()[1]])

    add_zoom_box(axs[1,0],
                 [axs[1,1].get_xlim()[0]*1.015, axs[1,1].get_xlim()[1]*0.95],
                 [axs[1,1].get_ylim()[0]*0.975, axs[1,1].get_ylim()[1]])

    add_zoom_box(axs[1,1],
                 [axs[1,1].get_xlim()[0]*1.015, axs[1,1].get_xlim()[1]*0.975],
                 [axs[1,1].get_ylim()[0]*0.975, axs[1,1].get_ylim()[1]])

    fig.tight_layout()

    return fig

def plot_figure_appdx_1(ds, ds_sem, microphysics_styles):
    fig, axs = plt.subplots(
        2, 2, figsize=large_square_fig_size, width_ratios=[1, 0.3], height_ratios=[0.3, 1]
    )

    ax_empty = axs[0, 1]
    ax_empty.axis("off")
    ax_x_hist = axs[0, 0]
    ax_y_hist = axs[1, 1]
    ax_scatter = axs[1, 0]

    ax_x_hist.sharex(ax_scatter)
    ax_y_hist.sharey(ax_scatter)

    x = -ds["source_energy"]
    y = ds["evaporation_fraction"]
    c = ds["cloud_liquid_water_content"]

    x_bins = np.arange(0, 1000, 50)
    y_bins = np.arange(0, 101, 5)

    ax_scatter.set_xlim(x_bins[0], x_bins[-1])
    ax_scatter.set_ylim(y_bins[0], y_bins[-1])

    for i, mp in enumerate(microphysics_styles):

        style = microphysics_styles.get_style(mp)

        data_dict, plot_dict = scatter_and_errorbar(
            ax=ax_scatter,
            x_var=x.name,
            y_var=y.name,
            ds=ds,
            ds_sem=ds_sem,
            microphysics=mp,
            x_multiply=-1,
            y_multiply=1,
            plot_patch=False,
            plot_annotations=True,
            patch_width=2.5,
            scatter_kwargs=dict(
                color=adjust_lightness(style["color"], 1.5),
                marker=style["marker"],
                alpha=1,
            ),
            error_kwargs=dict(
                fmt="",
                label="mean ± SEM",
                color=style["color"],
                capsize=5,
                linewidth=2,
            ),
            annotation_kwargs=dict(
                color=style["color"],
            ),
        )

        # add correlation annotation
        x_var = x.name
        y_var = y.name
        correlation = xr.corr(-ds[x_var], ds[y_var], dim="cloud_id")

        ax_scatter.annotate(
            f" R = {correlation.sel(microphysics=mp).data:.2f}",
            xy=(900, 80 + 5 * i),
            color=style["color"],
            xycoords="data",
            ha="right",
            va="center",
        )

        ax_x_hist.hist(
            x.sel(microphysics=mp),
            bins=x_bins,
            histtype="step",
            color=style["color"],
            lw=2,
        )
        ax_y_hist.hist(
            y.sel(microphysics=mp),
            bins=y_bins,
            histtype="step",
            color=style["color"],
            lw=2,
            orientation="horizontal",
        )

        x_annotation = plot_dict["x_annotation"]
        x_annotation.set(
            x=50,
            y=80 + 5 * i,
            ha="left",
            va="center",
        )

        y_annotation = plot_dict["y_annotation"]
        y_annotation.set(
            x=900,
            y=45 + 5 * i,
            rotation=0,
            ha="right",
            va="center",
        )


    ax_scatter.plot(
        x,
        y,
        color=adjust_lightness("grey", 1.75),
        zorder=0,
    )

    ax_scatter.set_xlabel(label_from_attrs(x))
    ax_scatter.set_ylabel(label_from_attrs(y))

    ax_x_hist.set_ylabel("Counts")
    ax_y_hist.set_xlabel("Counts")


    # add additional x axis with converted units

    x_ticks = xr.DataArray(ax_scatter.get_xticks(), attrs=x.attrs.copy())
    new_x_ticks: xr.DataArray = conversions.EvaporationUnits(data=x_ticks, input_type="energy").convert_to(
        "precipitation"
    )
    new_ticks_func = lambda _: [f"{round(new_x, 2):.2f}" for x, new_x in zip(x_ticks, new_x_ticks.data)]
    add_additional_axis(
        ax=ax_scatter,
        new_ticks_func=new_ticks_func,
        label=label_from_attrs(da=new_x_ticks),
        position="bottom",
        offset_position=["axes", -0.2],
    )
    ax_scatter.set_xlabel(label_from_attrs(da=x))

    for _ax in axs.flatten():
        _ax.grid(linestyle="-", alpha=0.2, color="grey")

    add_subplotlabel([ax_scatter, ax_x_hist, ax_y_hist], location="title")

    return fig

def plot_figure_appdx_2(ds,
                        ds_correlations_EF,
                        ds_correlations_CIE,
                        ds_correlations_MEH,
                        ds_correlations_log_EF,
                        ds_correlations_log_CIE,
                        ds_correlations_log_MEH,
                        microphysics_styles):
    correlation_vars = (
        "cloud_mass_radius_mean",
        "cloud_liquid_water_content",
        "inflow_precipitation",
        "relative_humidity_mean",
        "cloud_base_height",
    )
    
    fig, axs = plt.subplots(nrows=3, ncols=len(correlation_vars), figsize=(2 * len(correlation_vars), 7.5))

    axs_ef: Tuple[plt.Axes, plt.Axes, plt.Axes] = axs[0]
    axs_ef[1].sharey(axs_ef[0])
    axs_ef[2].sharey(axs_ef[0])

    axs_cie: Tuple[plt.Axes, plt.Axes, plt.Axes] = axs[1]
    axs_cie[1].sharey(axs_cie[0])
    axs_cie[2].sharey(axs_cie[1])

    axs_meh: Tuple[plt.Axes, plt.Axes, plt.Axes] = axs[2]
    axs_meh[1].sharey(axs_meh[0])
    axs_meh[2].sharey(axs_meh[1])

    # for the evaporation fraction
    for i, var in enumerate(correlation_vars):
        y = ds["evaporation_fraction"].sel(microphysics="condensation")
        x = ds[var].sel(microphysics="condensation")
        correlation = ds_correlations_EF[var].sel(microphysics="condensation")
        correlation_log = ds_correlations_log_EF[var].sel(microphysics="condensation")
        axs_ef[i].set_title("     " + r"$R$" + f"={correlation.data:.2f}")
        axs_ef[i].scatter(
            x,
            y,
            **microphysics_styles.get_style("condensation"),
        )
        axs_ef[i].set_xlabel(label_from_attrs(x, name_width=20, linebreak=True))

    axs_ef[0].set_ylabel(label_from_attrs(y, name_width=15))

    # for the column integrate evaporation
    for i, var in enumerate(correlation_vars):
        y = -ds["source_energy"].sel(microphysics="condensation")
        x = ds[var].sel(microphysics="condensation")
        correlation = ds_correlations_CIE[var].sel(microphysics="condensation")
        correlation_log = ds_correlations_log_CIE[var].sel(microphysics="condensation")
        axs_cie[i].set_title("     " + r"$R$" + f"={correlation.data:.2f}")
        axs_cie[i].scatter(
            x,
            y,
            **microphysics_styles.get_style("condensation"),
        )
        axs_cie[i].set_xlabel(label_from_attrs(x, name_width=20, linebreak=True))

    axs_cie[0].set_ylabel(label_from_attrs(y, name_width=15, linebreak=True))

    # for the mean evaporation height
    for i, var in enumerate(correlation_vars):
        y = ds["mean_evaporation_height"].sel(microphysics="condensation")
        x = ds[var].sel(microphysics="condensation")
        correlation = ds_correlations_MEH[var].sel(microphysics="condensation")
        correlation_log = ds_correlations_log_MEH[var].sel(microphysics="condensation")
        axs_meh[i].set_title("     " + r"$R$" + f"={correlation.data:.2f}")
        axs_meh[i].scatter(
            x,
            y,
            **microphysics_styles.get_style("condensation"),
        )
        axs_meh[i].set_xlabel(label_from_attrs(x, name_width=20, linebreak=True))

    axs_meh[0].set_ylabel(label_from_attrs(y, name_width=15))


    for _axs in axs[:, 1:]:
        for _ax in _axs.flatten():
            _ax.set_ylabel("")
    for _axs in [axs_ef, axs_cie]:
        for _ax in _axs.flatten():
            _ax.set_xlabel("")
    add_subplotlabel(axs=axs.flatten(), location="title")


    fig.tight_layout()

    return fig