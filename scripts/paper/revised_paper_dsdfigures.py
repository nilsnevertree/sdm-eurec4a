# %%
def get_mfdataset(datapath, microphysics, valid_cloud_ids=None, normheight=False):
  import glob
  import xarray as xr

  if normheight:
    ds_name = "droplet_pdfdistribs_normheight.zarr"
  else:
    ds_name = "droplet_pdfdistribs.zarr"
 
  datasets = sorted(glob.glob(str(datapath / microphysics / "cluster_*" / "processed" / ds_name)))
  cluster_names = [Path(d).parent.parent.name for d in datasets]

  if valid_cloud_ids is not None:
    valid_cloud_id_names = ["cluster_"+str(x) for x in valid_cloud_ids]
    ds_name_pairs = [
        (ds, name)
        for ds, name in zip(datasets, cluster_names)
        if name in valid_cloud_id_names
    ]
    datasets = [dataset for dataset, _ in ds_name_pairs]
    cluster_names = [cluster_name for _, cluster_name in ds_name_pairs]

  print(f"{len(cluster_names)} valid clusters found with {ds_name}:\n", cluster_names)

  ds = xr.open_mfdataset(datasets,
                           engine="zarr",
                           combine="nested",
                           concat_dim="cluster"
                           )

  def get_number_after_underscore(s):
    parts = s.split("_")
    return int(parts[-1]) if len(parts) > 1 else None

  cluster_nums = dict(cluster=("cluster", [get_number_after_underscore(c) for c in cluster_names]))
  ds = ds.assign_coords(cluster_nums)
  
  ds.attrs = {
     "microphysics":  microphysics,
     "normalised_by_height": normheight
     }

  return ds

# %%
def sanity_check_plot(datasets, microphysics_styles, extra_ds, clusters2plot=None, normheight=False):
    import matplotlib.pyplot as plt
    import numpy as np

    print("Distributions at Top and Bottom of Domain for all Microphysics Setups")

    default_ds = datasets["condensation"]
    if clusters2plot is None:
       clusters2plot = default_ds.cluster.values


    for cluster in clusters2plot:
        if normheight:
            heights2plot = [1.0, 0.0]
        else:
            coord3 = extra_ds.gridbox_coord3.sel(microphysics="condensation", cloud_id=cluster)
            heights2plot = [coord3.max(), coord3.min()]

        ncols=len(datasets.keys())
        nrows=len(heights2plot)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,5))
        fig.suptitle(f"cluster {cluster}")

        for i, h in enumerate(heights2plot):
            axs = axes[i,:]
            if normheight:
                text = f"level={h:.0f}"
            else:
                text = f"{h:.0f}m"
            axs[0].text(
                0.02,
                0.98,
                text,
                transform=axs[0].transAxes,
                ha="left",
                va="top",
            )
            for a, (microphys, ds) in enumerate(datasets.items()):
                ax = axs[a]
                if microphys == "condensation_novent":
                   title = microphysics_styles["condensation"]["name"]
                else:
                   title = microphysics_styles[microphys]["name"]
                ax.set_title(title)

                if normheight:
                    # data = ds.widths.values*ds.nsupers_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                    # data = ds.numconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                    data = ds.massconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                else:
                    # data = ds.widths.values*ds.nsupers_pdf.sel(cluster=cluster).sel(height=h, method="nearest")
                    # data = ds.numconc_pdf.sel(cluster=cluster).sel(height=h, method="nearest")
                    data = ds.massconc_pdf.sel(cluster=cluster).sel(height=h, method="nearest")
                ax.step(ds.centers, data, where="mid")

        for ax in axes.flatten():
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim([1e-1, 5e4])
        for ax in axes[-1,:]:
            ax.set_xlabel("radius / microns")
        for ax in axes[:,0]:
            # ax.set_ylabel("nsupers")
            # ax.set_ylabel("numconc / cm^-3 m^-1")
            ax.set_ylabel("massconc / g m^-3 m^-1")

        plt.tight_layout()
        plt.show()

# %% Input Settings
import numpy as np

from pathlib import Path
from ruamel.yaml import YAML
from sdm_eurec4a import RepositoryPath, data_loading
from sdm_eurec4a.visulization import (
    save_figure,
)

import revised_paper_figures_src as src
RepoPaths = RepositoryPath("levante_m300950")

data_dir = RepoPaths.CLEO_data_dir / "output_v4.2"
data_dir_novent = RepoPaths.CLEO_data_dir / "output_v4.2_novent"
data_dir_obs = RepoPaths.data_dir / "observation"

fig_dir = RepoPaths.fig_dir / Path("paper-revised") / Path("dsd_plots")
fig_dir.mkdir(exist_ok=True, parents=False)

yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
valid_cloud_ids_file = data_dir / Path("valid_cloud_ids.yaml")
valid_cloud_ids = sorted(yaml.load(valid_cloud_ids_file)["valid_cloud_ids"])

print(f"Using data from {data_dir}")
print(f"Using novent data from: {data_dir_novent}")
print(f"Using obs data from {data_dir_obs}")
print(f"using {len(valid_cloud_ids)} valid_cloud_ids from: {valid_cloud_ids_file}")
# %% Load datasets, takes circa. 2mins
microphysics_styles = data_loading.MicrophysicsStyles()
microphysics_datasets = {
   m: get_mfdataset(data_dir, m, valid_cloud_ids=valid_cloud_ids, normheight=False) for m in tuple(microphysics_styles)
}
microphysics_datasets["condensation_novent"] = get_mfdataset(data_dir_novent, "condensation", normheight=False)
microphysics_datasets
# %% Load datasets, takes circa. 2mins
microphysics_datasets_norm = {
   m: get_mfdataset(data_dir, m, valid_cloud_ids=valid_cloud_ids, normheight=True) for m in tuple(microphysics_styles)
}
microphysics_datasets_norm["condensation_novent"] = get_mfdataset(data_dir_novent, "condensation", normheight=True)
microphysics_datasets_norm
# %% Load Extra Datasets
_, _, extra_ds, \
    extra_ds_sem, \
    extra_ds_normalized, \
    extra_ds_normalized_sem, \
    extra_ds_no_ventilation, \
    extra_ds_sem_no_ventilation, \
    _, _, _ = src.load_required_datasets_and_remove_outliers(data_dir, \
                                                                      data_dir_novent, \
                                                                        data_dir_obs, \
                                                                          microphysics_styles)
# %%
clusters2plot = np.random.choice(microphysics_datasets["condensation"].cluster.values, size=10, replace=False)
# sanity_check_plot(microphysics_datasets_norm, microphysics_styles, extra_ds, clusters2plot=clusters2plot, normheight=True)
sanity_check_plot(microphysics_datasets, microphysics_styles, extra_ds, clusters2plot=clusters2plot)
# %% Select which clouds to plot for example DSDs of effect of collisions

def select_collisions_diff_clouds(ds_normalized, microphysics_styles):
    import matplotlib.pyplot as plt
    plot_limits = {
        "collision_condensation" : 17,
        "coalbure_condensation_small": 32,
        "coalbure_condensation_large": 975,
   }
    selected_clusters = {
        "collision_condensation" : [],
        "coalbure_condensation_small": [],
        "coalbure_condensation_large": [],
   }

    x_all = -ds_normalized["evaporation_rate_energy"]
    x_refernce = x_all.sel(microphysics="condensation")
    attrs = x_all.attrs.copy()
    x = (x_all - x_refernce) / x_refernce * 100
    x.attrs.update(
        long_name=f"Relative difference of {attrs['long_name']} compared to {microphysics_styles['condensation']['name']}",
        units=r"\%",
    )
    y = ds_normalized["normalized_gridbox_coord3"]

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5, 8), sharey=True)
    
    plot_microphysics = [
        "collision_condensation",
        "coalbure_condensation_small",
        "coalbure_condensation_large",
    ]
    for _ax, mp in zip(axs, plot_microphysics):

        style_full = microphysics_styles[mp].copy()
        _ax.set_title(microphysics_styles.get_setup(mp)["name"])

        for id in ds_normalized.cloud_id:
            x_cloud = x.sel(microphysics=mp, cloud_id=id)
            if mp == "collision_condensation":
                compare = -x_cloud
            else:
                compare = x_cloud
            if np.any(compare > plot_limits[mp]):
                color=style_full["dark_color"]
                selected_clusters[mp].append(id.values)
            else:
                color="grey"
            _ax.plot(
                x_cloud,
                y,
                color=color,
                linestyle="--",
                label="Mean",
                zorder=10,
            )

    for _ax in axs:
        _ax.axvline(0, color="k", linestyle="--", alpha=0.5, zorder=10)
        _ax.set_ylim(0, 1)
        _ax.set_yticks([0, 0.5, 1])

    axs[0].set_xlim(-25, 10)
    axs[1].set_xlim(-15, 80)
    axs[2].set_xlim(-15, 1100)

    for ax in axs:
        ax.set_ylabel("Normalized Height []")
        ax.set_xlabel("Rel Diff. to EvapOnly [%]")

    fig.tight_layout()

    save_figure(fig=fig, filepath=fig_dir / "dsd_effect_breakup_selection_criteria")

    for key, value in selected_clusters.items():
        selected_clusters[key] = np.asarray(value)

    return selected_clusters
selected_clusters = select_collisions_diff_clouds(extra_ds_normalized, microphysics_styles)
selected_clusters
# %%
import matplotlib.pyplot as plt

nlevels = 5
levels2plot = np.flip(np.linspace(0.0, 1.0, nlevels))
plot_numconc = False

fig, axes = plt.subplots(nrows=nlevels, ncols=3, figsize=(9, 9), sharex=True, sharey=True)

ds_ref = microphysics_datasets_norm["condensation"]

# plot all clouds' distrib at level 1.0
if plot_numconc:
    data = ds_ref.numconc_pdf.sel(level=1.0, method="nearest")
else:
    data = ds_ref.massconc_pdf.sel(level=1.0, method="nearest")
for ax in axes[0,:]:
    ax.step(ds_ref.centers, data.T, where="mid", color="lightgrey")

for i, (mp, clusters2plot) in enumerate(selected_clusters.items()):
    ds = microphysics_datasets_norm[mp]
    color = microphysics_styles[mp]["dark_color"]
    for cluster in clusters2plot:
        for j, h in enumerate(levels2plot):
            ax = axes[j, i]
            if plot_numconc:
                ylabel = "number concentration distribution  / cm$^{-3}$ m$^{-1}$"
                data_ref = ds_ref.numconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                data = ds.numconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
            else:
                ylabel = "mass concentration distribution / g m$^{-3}$ m$^{-1}$"
                data_ref = ds_ref.massconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                data = ds.massconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
            ax.text(
                0.02,
                0.98,
                f"level={h:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
            ax.step(ds.centers, data_ref, where="mid", color=microphysics_styles["condensation"]["dark_color"], alpha=0.5)
            ax.step(ds.centers, data, where="mid", color=color)

for ax in axes.flatten():
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1e-1, 5e4])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
for ax in axes[-1,:]:
    ax.set_xlabel("radius / $\u03BC$m")
axes[2,0].set_ylabel(ylabel)

axes[0,0].set_title("EvapCoal")
axes[0,1].set_title("EvapCoalBuRe-few")
axes[0,2].set_title("EvapCoalBuRe-many")

plt.tight_layout()
save_figure(fig=fig, filepath=fig_dir / "dsd_effect_breakup")
plt.show()
# %% Select which clouds to plot for example DSDs of condensation setup
def select_condonly_diff_clouds(ds_normalized):
    plot_limits = {
        "little_evap" : [0.11, 0.13],  # clusters with any value1 < height evap < any value2
        "top_heavy": 0.04,  # clusters with any height evap > evap_surface + value
        "extreme_evap" : 1.25,  # clusters with any height evap > value
   }
    selected_clusters = {
        "little_evap" : [],
        "top_heavy": [],
        "extreme_evap" : [],
   }
    selected_clusters_colors = {
        "little_evap" : "mediumseagreen",
        "top_heavy": "magenta",
        "extreme_evap" : "midnightblue",
   }
 
    fig = plt.figure(figsize=(16 / 3 * 12 / 8.3, 9 / 3))
    gs = fig.add_gridspec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[:, :])

    mp = "condensation"
    x = -ds_normalized.sel(microphysics=mp)["evaporation_rate_energy"]
    y = ds_normalized.sel(microphysics=mp)["normalized_gridbox_coord3"]

    # select all but the top most gridboxes
    x = x.sel(normalized_gridbox_coord3=slice(0, 0.99))
    y = y.sel(normalized_gridbox_coord3=slice(0, 0.99))
    y = y.expand_dims(cloud_id=x["cloud_id"])

    for cloud_id in x["cloud_id"]:
        xx = np.flip(x.sel(cloud_id=cloud_id).data)
        yy = np.flip(y.sel(cloud_id=cloud_id).data)
        xsurf = x.sel(cloud_id=cloud_id, normalized_gridbox_coord3=0).values
        if (np.any(xx > plot_limits["little_evap"][0]) and np.all(xx < plot_limits["little_evap"][1])):
            selected_clusters["little_evap"].append(cloud_id)
            color=selected_clusters_colors["little_evap"]
            linestyle = "dashdot"
        elif np.any(xx > plot_limits["extreme_evap"]):
            selected_clusters["extreme_evap"].append(cloud_id)
            color=selected_clusters_colors["extreme_evap"]
            linestyle = "-"
        elif np.any(xx > xsurf + plot_limits["top_heavy"]):
            selected_clusters["top_heavy"].append(cloud_id)
            color=selected_clusters_colors["top_heavy"]
            linestyle = "--"
        else:
            color = "grey"
            linestyle = "dotted"
        ax.plot(xx, yy, color=color, linestyle=linestyle)

    y_ticks = np.arange(0, 1.01, 0.25)
    ax.set_yticks(y_ticks, y_ticks)
    ax.legend(loc="upper right")

    ax.set_xlim(0, None)
    ax.set_ylim(0, 1)

    ax.set_ylabel("Normalized Height []")
    ax.set_xlabel("Evap. Rate W m^-3")

    fig.tight_layout()

    save_figure(fig=fig, filepath=fig_dir / "dsd_different_condevaps_selection_criteria")

    for key, value in selected_clusters.items():
        selected_clusters[key] = np.asarray(value)

    return selected_clusters, selected_clusters_colors
selected_clusters, selected_clusters_colors = select_condonly_diff_clouds(extra_ds_normalized)
selected_clusters
# %%
import matplotlib.pyplot as plt

nlevels = 5
levels2plot = np.flip(np.linspace(0.0, 1.0, nlevels))
plot_numconc = False

fig, axes = plt.subplots(nrows=nlevels, ncols=1, figsize=(5, 9), sharex=True, sharey=True)

ds = microphysics_datasets_norm["condensation"]

# plot all clouds' distrib at level 1.0
if plot_numconc:
    data = ds.numconc_pdf.sel(level=1.0, method="nearest")
else:
    data = ds.massconc_pdf.sel(level=1.0, method="nearest")
axes[0].step(ds.centers, data.T, where="mid", color="lightgrey")

# plot selected clouds' distrib at selected levels
for j, h in enumerate(levels2plot):
    for i, (selection, clusters2plot) in enumerate(selected_clusters.items()):
        for cluster in clusters2plot:
            ax = axes[j]
            if plot_numconc:
                ylabel = "number concentration distribution  / cm$^{-3}$ m$^{-1}$"
                data = ds.numconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
            else:
                ylabel = "mass concentration distribution / g m$^{-3}$ m$^{-1}$"
                data = ds.massconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
            if cluster == clusters2plot[0]:
                label = selection
            else:
                label = None
            ax.step(ds.centers, data, where="mid", color=selected_clusters_colors[selection], label=label)
    ax.text(
        0.02,
        0.98,
        f"level={h:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

for ax in axes.flatten():
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1e-1, 5e4])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].legend()
axes[-1].set_xlabel("radius / $\u03BC$m")
axes[2].set_ylabel(ylabel)

plt.tight_layout()
save_figure(fig=fig, filepath=fig_dir / "dsd_different_condevaps")
plt.show()
# %%
