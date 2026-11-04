# %%
condonly_plot_limits = {
    "random_sample": [],
    ### limits for 3 clouds selected in each category
    "little_evap" : [0.03, 0.06],  # clusters with any value1 < height evap < any value2
    "top_heavy": 0.055,  # clusters with any height evap > evap_surface + value
    "extreme_evap" : 1.345,  # clusters with any height evap > value
    ### limits for 5 clouds selected in each category
    # "little_evap" : [0.03, 0.075],  # clusters with any value1 < height evap < any value2
    # "top_heavy": 0.04,  # clusters with any height evap > evap_surface + value
    # "extreme_evap" : 1.25,  # clusters with any height evap > value
}

collisions_plot_limits = {
    "random_sample": [],
    ### limits for 3 clouds selected in each category
    "collision_condensation" : 20,
    "coalbure_condensation_small": 45,
    "coalbure_condensation_large": 1010,
    ### limits for 5 clouds selected in each category
    # "collision_condensation" : 17,
    # "coalbure_condensation_small": 32,
    # "coalbure_condensation_large": 975,
}

def get_mfdataset(datapath, microphysics, valid_cloud_ids=None, normheight=False):
  import glob
  import xarray as xr
  from pathlib import Path

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


def plot_collisions_diff_clouds(ax, selected_clusters, ds_normalized, microphysics_styles, mp, do_append=True, sample_size=3):
    import numpy as np

    style_full = microphysics_styles[mp].copy()

    x_all = -ds_normalized["evaporation_rate_energy"]
    x_refernce = x_all.sel(microphysics="condensation")
    attrs = x_all.attrs.copy()
    x = (x_all - x_refernce) / x_refernce * 100
    x.attrs.update(
        long_name=f"Relative difference of {attrs['long_name']} compared to {microphysics_styles['condensation']['name']}",
        units=r"\%",
    )
    y = ds_normalized["normalized_gridbox_coord3"]

    all_others = []
    for id in ds_normalized.cloud_id:
        x_cloud = x.sel(microphysics=mp, cloud_id=id)
        if mp == "collision_condensation":
            compare = -x_cloud
        else:
            compare = x_cloud
        if np.any(compare > collisions_plot_limits[mp]):
            color=style_full["dark_color"]
            linestyle = "-"
            if do_append:
                selected_clusters[mp].append(id.values)
        else:
            color="grey"
            linestyle = "--"
            all_others.append(id.values)
        ax.plot(
            x_cloud,
            y,
            color=color,
            linestyle=linestyle,
            label="Mean",
            zorder=10,
        )

    if do_append:
       selected_clusters["condensation"] = np.random.choice(all_others, size=sample_size, replace=False)

    return selected_clusters

def select_collisions_diff_clouds(ds_normalized, microphysics_styles):
    import matplotlib.pyplot as plt
    import numpy as np

    selected_clusters = {
        "condensation": [],
        "collision_condensation" : [],
        "coalbure_condensation_small": [],
        "coalbure_condensation_large": [],
   }

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5, 8), sharey=True)

    plot_microphysics = [
        "collision_condensation",
        "coalbure_condensation_small",
        "coalbure_condensation_large",
    ]
    for _ax, mp in zip(axs, plot_microphysics):
        _ax.set_title(microphysics_styles.get_setup(mp)["name"])
        selected_clusters = plot_collisions_diff_clouds(_ax, selected_clusters, ds_normalized, microphysics_styles, mp)

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

    for key, value in selected_clusters.items():
        selected_clusters[key] = np.asarray(value)

    return fig, selected_clusters

def plot_collision_effect_evaporation_dsds(microphysics_datasets_norm, collisions_selected_clusters, microphysics_styles):
    import matplotlib.pyplot as plt
    import numpy as np

    nlevels = 5
    levels2plot = np.flip(np.linspace(0.0, 1.0, nlevels))
    plot_numconc = False

    fig, axes = plt.subplots(nrows=nlevels, ncols=4, figsize=(9, 9), sharex=True, sharey=True)

    ds_ref = microphysics_datasets_norm["condensation"]

    # plot all clouds' distrib at level 1.0
    if plot_numconc:
        data = ds_ref.numconc_pdf.sel(level=1.0, method="nearest")
    else:
        data = ds_ref.massconc_pdf.sel(level=1.0, method="nearest")
    for ax in axes[0,:]:
        ax.step(ds_ref.centers, data.T, where="mid", color="lightgrey")

    for i, (mp, clusters2plot) in enumerate(collisions_selected_clusters.items()):
        ds = microphysics_datasets_norm[mp]
        color = microphysics_styles[mp]["dark_color"]
        for cluster in clusters2plot:
            for j, h in enumerate(levels2plot):
                ax = axes[j, i]
                if plot_numconc:
                    ylabel = "number concentration distribution  / cm$^{-3}$ m$^{-1}$"
                    # data_ref = ds_ref.numconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
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
                ax.step(ds.centers, data_ref, where="mid", color=microphysics_styles["condensation"]["dark_color"], alpha=0.3, linestyle="dotted")
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

    axes[0,0].set_title("EvapOnly")
    axes[0,1].set_title("EvapCoal")
    axes[0,2].set_title("EvapCoalBuRe-few")
    axes[0,3].set_title("EvapCoalBuRe-many")

    plt.tight_layout()

    return fig

def plot_condonly_diff_clouds(ax, ds_normalized, selected_clusters, selected_clusters_colors, do_append=True, sample_size=3):
    import numpy as np

    mp = "condensation"
    x = -ds_normalized.sel(microphysics=mp)["evaporation_rate_energy"]
    y = ds_normalized.sel(microphysics=mp)["normalized_gridbox_coord3"]

    # select all but the top most gridboxes
    x = x.sel(normalized_gridbox_coord3=slice(0, 0.99))
    y = y.sel(normalized_gridbox_coord3=slice(0, 0.99))
    y = y.expand_dims(cloud_id=x["cloud_id"])

    a, b, c = True, True, True

    all_others = []
    for cloud_id in x["cloud_id"]:
        xx = np.flip(x.sel(cloud_id=cloud_id).data)
        yy = np.flip(y.sel(cloud_id=cloud_id).data)
        xsurf = x.sel(cloud_id=cloud_id, normalized_gridbox_coord3=0).values
        label = None
        if (np.any(xx > condonly_plot_limits["little_evap"][0]) and np.all(xx < condonly_plot_limits["little_evap"][1])):
            if do_append:
                selected_clusters["little_evap"].append(cloud_id)
            color=selected_clusters_colors["little_evap"]
            linestyle = "dashdot"
            if a:
                label, a = "little_evap", False
        elif np.any(xx > condonly_plot_limits["extreme_evap"]):
            if do_append:
                selected_clusters["extreme_evap"].append(cloud_id)
            color=selected_clusters_colors["extreme_evap"]
            linestyle = "-"
            if b:
                label, b = "extreme_evap", False
        elif np.any(xx > xsurf + condonly_plot_limits["top_heavy"]):
            if do_append:
                selected_clusters["top_heavy"].append(cloud_id)
            color=selected_clusters_colors["top_heavy"]
            linestyle = "--"
            if c:
                label, c = "top_heavy", False
        else:
            if do_append:
                if np.any(xx > condonly_plot_limits["little_evap"][0]):
                    all_others.append(cloud_id) 
            color = "grey"
            linestyle = "dotted"
            label = None
        ax.plot(xx, yy, color=color, linestyle=linestyle, label=label)

    if do_append:
       selected_clusters["random_sample"] = np.random.choice(all_others, size=sample_size, replace=False)

    d = True
    for cloud_id in x["cloud_id"]:
        if cloud_id.values in list(selected_clusters["random_sample"]):
            xx = np.flip(x.sel(cloud_id=cloud_id).data)
            yy = np.flip(y.sel(cloud_id=cloud_id).data)
            label = None
            if d:
                label, d = "random_sample", False                
            ax.plot(xx, yy, color="black", linestyle=(0, (3, 1, 1, 1, 1, 1)), label=label, zorder=-1)

    return selected_clusters

def select_condonly_diff_clouds(ds_normalized):
    import matplotlib.pyplot as plt
    import numpy as np

    selected_clusters = {
        "random_sample": [],
        "little_evap" : [],
        "top_heavy": [],
        "extreme_evap" : [],
   }
    selected_clusters_colors = {
        "random_sample": "black",
        "little_evap" : "mediumseagreen",
        "top_heavy": "magenta",
        "extreme_evap" : "midnightblue",
   }
 
    fig = plt.figure(figsize=(16 / 3 * 12 / 8.3, 9 / 3))
    gs = fig.add_gridspec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[:, :])

    selected_clusters = plot_condonly_diff_clouds(ax, ds_normalized, selected_clusters, selected_clusters_colors)

    y_ticks = np.arange(0, 1.01, 0.25)
    ax.set_yticks(y_ticks, y_ticks)
    ax.legend(loc="upper right")

    ax.set_xlim(0, None)
    ax.set_ylim(0, 1)

    ax.set_ylabel("Normalized Height []")
    ax.set_xlabel("Evap. Rate W m^-3")

    fig.tight_layout()

    for key, value in selected_clusters.items():
        selected_clusters[key] = np.asarray(value)

    return fig, selected_clusters, selected_clusters_colors

def plot_bottom_top_heavy_evaporation_dsds(microphysics_datasets_norm, condonly_selected_clusters, condonly_selected_clusters_colors):
    import matplotlib.pyplot as plt
    import numpy as np

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
        for i, (selection, clusters2plot) in enumerate(condonly_selected_clusters.items()):
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
                ax.step(ds.centers, data, where="mid", color=condonly_selected_clusters_colors[selection], label=label)
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

    return fig


def plot_figure_appdx_2_version2(microphysics_datasets_norm, ds_normalized, condonly_selected_clusters, condonly_selected_clusters_colors):
    import matplotlib.pyplot as plt
    import numpy as np
    import textwrap

    from sdm_eurec4a.visulization import add_subplotlabel

    from matplotlib.gridspec import GridSpec

    nlevels = 5
    levels2plot = np.flip(np.linspace(0.0, 1.0, nlevels))
    plot_numconc = False

    fig = plt.figure(figsize=[9, 9], layout="constrained")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.3,1,1])
    ax0 = fig.add_subplot(gs[0, :])
    selection_axes = {'random_sample': fig.add_subplot(gs[1, 0]),
            'little_evap': fig.add_subplot(gs[1, 1]),
            'top_heavy': fig.add_subplot(gs[2, 0]),
            'extreme_evap': fig.add_subplot(gs[2, 1])
            }

    plot_condonly_diff_clouds(ax0, ds_normalized, condonly_selected_clusters, condonly_selected_clusters_colors, do_append=False)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.set_ylabel("Normalized Height [-]")
    ax0.set_xlabel("Evaporation Rate [$W m^-3$]")
    ax0.legend(fontsize=12)

    ds = microphysics_datasets_norm["condensation"]
    radius_mm = ds.centers / 1e3

    # plot all clouds' distrib at level 1.0
    if plot_numconc:
        data = ds.numconc_pdf.sel(level=1.0, method="nearest")
    else:
        data = ds.massconc_pdf.sel(level=1.0, method="nearest")

    for ax in selection_axes.values():
        ax.step(radius_mm, data.T, where="mid", color="lightgrey")

    colors = ["black", "blue", "purple", "fuchsia", "crimson"]

    # plot selected clouds' distrib at selected levels
    for j, h in enumerate(levels2plot):
        for i, (selection, clusters2plot) in enumerate(condonly_selected_clusters.items()):
            for cluster in clusters2plot:
                ax = selection_axes[selection]
                if plot_numconc:
                    ylabel = "number concentration distribution  / cm$^{-3}$ m$^{-1}$"
                    data = ds.numconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                else:
                    ylabel = "mass concentration distribution / g m$^{-3}$ m$^{-1}$"
                    data = ds.massconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                if cluster == clusters2plot[0]:
                    label = f"level={h:.2f}",
                else:
                    label = None
                ax.text(
                    0.02,
                    0.98,
                    selection,
                    color=condonly_selected_clusters_colors[selection],
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                )
                ax.step(radius_mm, data, where="mid", label=label, color=colors[j])

    selection_axes["top_heavy"].legend(loc=(0.8,0.8))

    for ax in selection_axes.values():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([1e-4, 5e1])
        ax.set_ylim([1e-6, 1e3])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in [selection_axes["top_heavy"], selection_axes["extreme_evap"]]:
        ax.set_xlabel("radius / mm")
    for ax in [selection_axes["random_sample"], selection_axes["top_heavy"]]: 
        ax.set_ylabel(textwrap.fill(ylabel, 30))

    add_subplotlabel([ax0] + list(selection_axes.values()), location=[[-0.05, 0.0], [1.1, 0.0]])

    return fig

def plot_figure_appdx_2(microphysics_datasets_norm, ds_normalized, condonly_selected_clusters):
    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib.gridspec import GridSpec

    from sdm_eurec4a.visulization import add_subplotlabel

    nlevels = 5
    levels2plot = np.flip(np.linspace(0.0, 1.0, nlevels))
    plot_numconc = False

    selected_clusters_colors = {
        "random_sample": "black",
        "little_evap" : "dodgerblue",
        "top_heavy": "magenta",
        "extreme_evap" : "mediumseagreen",
    }

    fig = plt.figure(figsize=[9, 12], layout="constrained")
    gs = GridSpec(nlevels+1, 4, figure=fig, height_ratios=[1.3]+[1]*nlevels)
    ax0 = fig.add_subplot(gs[0, :])
    selection_axes = {'random_sample': [fig.add_subplot(gs[n+1, 0]) for n in range(nlevels)],
            'little_evap': [fig.add_subplot(gs[n+1, 1]) for n in range(nlevels)],
            'top_heavy': [fig.add_subplot(gs[n+1, 2]) for n in range(nlevels)],
            'extreme_evap': [fig.add_subplot(gs[n+1, 3]) for n in range(nlevels)],
            }

    plot_condonly_diff_clouds(ax0, ds_normalized, condonly_selected_clusters, selected_clusters_colors, do_append=False)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.set_ylabel("Normalized Height [-]")
    ax0.set_xlabel("Evaporation Rate [$W m^{-3}$]")
    ax0.legend(fontsize=12, loc=(0.75, 0.4))

    ds = microphysics_datasets_norm["condensation"]
    radius_mm = ds.centers / 1e3

    # plot all clouds' distrib at level 1.0
    if plot_numconc:
        data = ds.numconc_pdf.sel(level=1.0, method="nearest")
    else:
        data = ds.massconc_pdf.sel(level=1.0, method="nearest")
    for axs in selection_axes.values():
        axs[0].step(radius_mm, data.T, where="mid", color="lightgrey")

    # plot selected clouds' distrib at selected levels
    for i, (selection, clusters2plot) in enumerate(condonly_selected_clusters.items()):
        for j, h in enumerate(levels2plot):
            for cluster in clusters2plot:
                ax = selection_axes[selection][j]
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
                ax.step(radius_mm, data, where="mid", color=selected_clusters_colors[selection], label=label)

            ref_data = ds.massconc_pdf.sel(level=h, method="nearest")
            ax.step(radius_mm, ref_data.T, where="mid", color="lightgrey", zorder=0, alpha=0.3)

            ax.text(
                0.02,
                0.98,
                f"level={h:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
            )

    for selection, axs in selection_axes.items():
        for ax in axs:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim([1e-4, 5e1])
            ax.set_ylim([1e-9, 5e3])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axs[0].set_title(selection, color=selected_clusters_colors[selection])
        axs[-1].set_xlabel("radius / mm")
    selection_axes["random_sample"][2].set_ylabel(ylabel)

    axes_list = [ax0]
    for j in range(len(levels2plot)):
        for axs in selection_axes.values():
            axes_list.append(axs[j])
    add_subplotlabel(axs=axes_list, location=[[-0.1, -0.05], [1.06, 0.0]])

    plt.tight_layout()

    return fig

def plot_figure_appdx_4(microphysics_datasets_norm, ds_normalized, collisions_selected_clusters, microphysics_styles):
    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib.gridspec import GridSpec

    from sdm_eurec4a.visulization import add_subplotlabel

    nlevels = 5
    levels2plot = np.flip(np.linspace(0.0, 1.0, nlevels))
    plot_numconc = False

    fig = plt.figure(figsize=[9, 12], layout="constrained")
    gs = GridSpec(nlevels+1, 3, figure=fig, height_ratios=[1.3]+[1]*nlevels)
    axes0 = [fig.add_subplot(gs[0, n]) for n in range(3)]
    selection_axes = {
            'collision_condensation': [fig.add_subplot(gs[n+1, 0]) for n in range(nlevels)],
            'coalbure_condensation_small': [fig.add_subplot(gs[n+1, 1]) for n in range(nlevels)],
            'coalbure_condensation_large': [fig.add_subplot(gs[n+1, 2]) for n in range(nlevels)],
            }

    for _ax, mp in zip(axes0, [
        "collision_condensation",
        "coalbure_condensation_small",
        "coalbure_condensation_large",
    ]):
        _ax.set_title(microphysics_styles.get_setup(mp)["name"], color=microphysics_styles[mp]["dark_color"])
        plot_collisions_diff_clouds(_ax, collisions_selected_clusters, ds_normalized, microphysics_styles, mp, do_append=False)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    axes0[1].set_xlabel(f"Relative difference of Evaporation Rate compared to {microphysics_styles['condensation']['name']} [%]")
    axes0[0].set_ylabel("Normalized Height [-]")

    ds_ref = microphysics_datasets_norm["condensation"]
    radius_ref_mm = ds_ref.centers / 1e3

    # plot all clouds' distrib at level 1.0
    if plot_numconc:
        data = ds_ref.numconc_pdf.sel(level=1.0, method="nearest")
    else:
        data = ds_ref.massconc_pdf.sel(level=1.0, method="nearest")
    for axs in selection_axes.values():
        axs[0].step(radius_ref_mm, data.T, where="mid", color="lightgrey")

    for mp in selection_axes.keys():
        clusters2plot = collisions_selected_clusters[mp]
        ds = microphysics_datasets_norm[mp]
        color = microphysics_styles[mp]["dark_color"]
        radius_mm = ds.centers / 1e3
        for cluster in clusters2plot:
            for j, h in enumerate(levels2plot):
                ax = selection_axes[mp][j]
                if plot_numconc:
                    ylabel = "number concentration distribution  / cm$^{-3}$ m$^{-1}$"
                    data_ref = ds_ref.numconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                    data = ds.numconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                else:
                    ylabel = "mass concentration distribution / g m$^{-3}$ m$^{-1}$"
                    data_ref = ds_ref.massconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                    data = ds.massconc_pdf.sel(cluster=cluster).sel(level=h, method="nearest")
                ax.text(
                    0.015,
                    0.98,
                    f"level={h:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                )
                ax.step(radius_mm, data_ref, where="mid",
                        color=microphysics_styles["condensation"]["dark_color"],
                        alpha=0.8, linestyle="dotted")
                ax.step(radius_mm, data, where="mid", color=color)

    for axs in selection_axes.values():
        for ax in axs:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim([1e-4, 5e1])
            ax.set_ylim([1e-9, 5e3])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axs[-1].set_xlabel("radius / mm")
    
    selection_axes["collision_condensation"][2].set_ylabel(ylabel)

    axes_list = axes0
    for j in range(len(levels2plot)):
        for axs in selection_axes.values():
            axes_list.append(axs[j])
    add_subplotlabel(axs=axes_list, location=[[-0.1, -0.05], [1.06, 0.0]])

    plt.tight_layout()

    return fig
# %%
