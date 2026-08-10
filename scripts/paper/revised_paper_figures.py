### script adapted from ``./notebooks/paper/17-NN-figure-review-version.ipynb``
# %% Imports
import numpy as np
import xarray as xr

from importlib import reload
from pathlib import Path

from sdm_eurec4a import RepositoryPath
from sdm_eurec4a import data_loading
from sdm_eurec4a.visulization import (
    set_paper_rcParams,
    adjust_lightness_array,
    adjust_lightness,
    label_from_attrs,
    add_additional_axis,
    add_subplotlabel,
    save_figure,
)

import revised_paper_figures_src
import revised_paper_figures_plots
src = reload(revised_paper_figures_src)
plots = reload(revised_paper_figures_plots)

# %% Input Settings
RepoPaths = RepositoryPath("levante_m300950")

data_dir_obs = RepoPaths.data_dir / "observation"
data_dir_model_input =  RepoPaths.data_dir / "model" / "input_v4.2"

data_dir = RepoPaths.CLEO_data_dir / Path("output_v4.2")
data_dir_novent = Path("/work/mh1126/m300950/rain-evap-nils/sdm-eurec4a-CLEO/data/output_v4.2_novent/")

fig_dir = RepoPaths.fig_dir / Path("paper-revised")
fig_dir.mkdir(exist_ok=True, parents=False)
appendix_fig_dir = fig_dir / "appendix"
appendix_fig_dir.mkdir(exist_ok=True, parents=False)

print(f"Using data from {data_dir}")
print(f"data_dir_novent: {data_dir_novent}")
print(f"data_dir_obs: {data_dir_obs}")
print(f"data_dir_model_input: {data_dir_model_input}")

# %% Load Datasets
microphysics_styles = data_loading.MicrophysicsStyles()
print(tuple(microphysics_styles))

valid_cloud_ids, \
    all_cloud_ids, \
    ds, \
    ds_sem, \
    ds_normalized, \
    ds_normalized_sem, \
    ds_no_ventilation, \
    ds_sem_no_ventilation, \
    identified_clusters, \
    ds_cleo_null, \
    ds_cleo_sem_null = src.load_required_datasets_and_remove_outliers(data_dir, \
                                                                      data_dir_novent, \
                                                                        data_dir_obs, \
                                                                          microphysics_styles)

ds_eulerian, ds_conservation = src.load_eulerian_and_conservation_datasets_and_remove_outliers(data_dir, valid_cloud_ids)

ds = src.add_extra_plotting_variables_to_datasets(ds, ds_normalized)

ds_correlations_CIE, ds_correlations_EF, ds_correlations_MEH = src.get_correlations_dataset(ds)
ds_correlations_log_CIE, ds_correlations_log_EF, ds_correlations_log_MEH = src.get_logarithmic_correlations_dataset(ds)

# %% Get observational and fitted data datasets
### open observations datasets and edit identified_clusters
dropsonde, ds_distances, cloud_composite, identified_clusters = src.get_observational_data_fitting(data_dir_obs, identified_clusters)

### open fitted dsd parameters in linear space
ds_parameters_linear = xr.open_dataset(
    data_dir_model_input / "particle_size_distribution_parameters_linear_space.nc"
)

### open fitted thermodynamics profile
ds_potential_temperature_parameters = xr.open_dataset(
    data_dir_model_input / "potential_temperature_parameters.nc"
)
ds_pressure_parameters = xr.open_dataset(data_dir_model_input / "pressure_parameters.nc")
ds_relative_humidity_parameters = xr.open_dataset(data_dir_model_input / "relative_humidity_parameters.nc")

# %% PLOT FIGURE 1
fig = plots.plot_figure_1(cloud_composite,
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
               )
save_figure(
    fig=fig,
    filepath=fig_dir / "fig1",
)
# %% PLOT FIGURE 2
fig = plots.plot_figure_2(ds_conservation)
save_figure(fig=fig,
            filepath=fig_dir / "fig2"
)
# %% PLOT FIGURE 3
fig = plots.plot_figure_3(ds, ds_sem, microphysics_styles)
save_figure(fig=fig,
            filepath=fig_dir / "fig3")

# %% PLOT FIGURE 4
fig = plots.plot_figure_4(ds, ds_sem, microphysics_styles)
save_figure(fig=fig,
            filepath=fig_dir / "fig4")

# %% PLOT FIGURE 5
fig = plots.plot_figure_5(ds, ds_normalized, microphysics_styles)
save_figure(fig, fig_dir / "fig5")

# %% PLOT FIGURE 6
fig = plots.plot_figure_6(ds, microphysics_styles)
save_figure(fig=fig,
            filepath=fig_dir / "fig6")
# %% PLOT FIGURE 7
fig = plots.plot_figure_7(ds,
                          ds_correlations_EF,
                          ds_correlations_CIE,
                          ds_correlations_MEH,
                          microphysics_styles)
save_figure(fig=fig,
            filepath=fig_dir / "fig7")

# %% PLOT FIGURE 8
fig = plots.plot_figure_8(ds)
save_figure(fig=fig,
            filepath=fig_dir / "fig8")

# %% PLOT FIGURE 9
evaporation_fraction, \
    evaporation_fraction_ventilation = src.get_theory_evaporation_fraction(ds)

fig = plots.plot_figure_9(ds,
                          ds_no_ventilation,
                          evaporation_fraction_ventilation,
                          evaporation_fraction,
                          microphysics_styles)
save_figure(fig=fig,
            filepath=fig_dir / "fig9")

# %% PLOT FIGURE 10
fig = plots.plot_figure_10(ds_normalized, ds_normalized_sem, microphysics_styles)
save_figure(fig=fig,
            filepath=fig_dir / "fig10")

# %% PLOT FIGURE 11
fig = plots.plot_figure_11(ds, microphysics_styles)
save_figure(fig=fig,
            filepath=fig_dir / "fig11")

# %% PLOT FIGURE A1
fig = plots.plot_figure_appdx_1(ds, ds_sem, microphysics_styles)
save_figure(fig=fig, filepath=fig_dir / "fig_appdx_1")

# %% PLOT FIGURE A2
fig = plots.plot_figure_appdx_2(ds, ds_sem, microphysics_styles)
save_figure(fig=fig, filepath=fig_dir / "fig_appdx_2")

# %% PLOT FIGURE A3
fig = plots.plot_figure_appdx_3(ds,
                                ds_correlations_EF,
                                ds_correlations_CIE,
                                ds_correlations_MEH,
                                ds_correlations_log_EF,
                                ds_correlations_log_CIE,
                                ds_correlations_log_MEH,
                                microphysics_styles)
save_figure(fig=fig, filepath=fig_dir / "fig_appdx_3")
