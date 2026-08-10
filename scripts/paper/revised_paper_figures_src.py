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

### Ensemble Mean and Spread Definitions
def propagate_mean_sem(data, data_std, dim: str):

    N = len(data[dim])

    # Inter-model spread (std of model means)
    inter_model_spread = data.std(dim=dim, ddof=1) / N**0.5

    # Individual model uncertainty propagation
    individual_model_error = (data_std**2).sum(dim) ** 0.5 / N

    # Total propagated SEM
    total_sem = (inter_model_spread**2 + individual_model_error**2) ** 0.5

    return total_sem


def propagate_mean_std(data, data_std, dim: str):

    N = len(data[dim])

    # Inter-model spread (std of model means)
    inter_model_spread = data.std(dim=dim, ddof=1)

    # Individual model uncertainty propagation
    individual_model_error = (data_std**2).sum(dim) ** 0.5 / N

    # Total propagated SEM
    total_sem = (inter_model_spread**2 + individual_model_error**2) ** 0.5

    return total_sem

### Outlier Function Definitions
def remove_invalid_combined(data_dir, microphysics_styles, ds):
    '''
    Remove Datasets for which combined dataset values differ to individual
    (See notebook 17-NN-figure-review-version.ipynb for details)
    '''
    atol = 1e-10
    invalid_combined_dataset_ids = set()
    error_combined_dataset_ids = set()

    for mp in microphysics_styles:
        print(mp)
        for cloud_id in tqdm(ds["cloud_id"]):
            cloud_id = int(cloud_id.data)
            p = data_dir / Path(f"{mp}/cluster_{cloud_id}/processed/conservation_dataset.nc")

            if p.is_file():
                ds_single = xr.open_dataset(p).sel(time=TimeSlices.quasi_stationary_state)
                inflow_diff = np.abs(
                    ds_single["inflow"].mean("time").data
                    - ds["inflow"].sel(microphysics=mp).sel(cloud_id=cloud_id).data
                )
                outflow_diff = np.abs(
                    ds_single["outflow"].mean("time").data
                    - ds["outflow"].sel(microphysics=mp).sel(cloud_id=cloud_id).data
                )
                source_diff = np.abs(
                    ds_single["source"].mean("time").data
                    - ds["source"].sel(microphysics=mp).sel(cloud_id=cloud_id).data
                )

                if inflow_diff > atol or outflow_diff > atol or source_diff > atol:
                    invalid_combined_dataset_ids.add(cloud_id)

            else:
                error_combined_dataset_ids.add(cloud_id)

    invalid_union = invalid_combined_dataset_ids.union(error_combined_dataset_ids)
    print(
        f"The following {len(invalid_union)} clouds have invalid combined datasets {invalid_union}"
    )

    return invalid_combined_dataset_ids, error_combined_dataset_ids

def remove_invalid_conservation(data_dir, ds_error, error_microphysics, relative_to_variables):
    '''
    Remove datasets which have >10% error in conservation
    (See notebook 17-NN-figure-review-version.ipynb for details)
    '''
    conservation_data_dir = data_dir

    conservation_list = []
    for mp in error_microphysics:
        _ds = xr.open_dataset(
            data_loading.__conservation_data_path__(data_dir=conservation_data_dir, microphysic=mp)
        )
        conservation_list.append(_ds.expand_dims(microphysics=[mp]))

    select_ds_conservation = xr.concat(
        conservation_list,
        dim="microphysics",
    )

    total = (
        select_ds_conservation["inflow"]
        + select_ds_conservation["outflow"]
        + select_ds_conservation["source"]
        - select_ds_conservation["reservoir_change"]
    )
    total = total.sel(time=TimeSlices.quasi_stationary_state).mean("time")

    error = {}
    for key in ["inflow", "outflow", "source"]:
        e = total / ds_error[key] * 100
        e = e.where(np.isfinite(e), np.nan)
        error[key] = e
        error[key].attrs.update(ds_error[key].attrs)
        error[key].attrs.update(units=r"\%", description=f"Relative error of {key} per gridbox per cloud")
        error[key] = error[key].expand_dims(which=[key])

    da_error = xr.concat(
        error.values(),
        dim="which",
    )
    da_maximum_error = (
        np.abs(da_error)
        .sel(which=relative_to_variables)
        .max(dim="which", skipna=True)
        .expand_dims(which=["maximum"])
    )
    error["maximum"] = da_maximum_error

    da_error = xr.concat(
        error.values(),
        dim="which",
    )

    # where is the error of the conversation larger than 10% relative to any of the inflow, outflow, source
    invalid_derivate_mass_conservation_ids = set(
        da_error["cloud_id"]
        .where(da_error.sel(which="maximum").max("microphysics") >= 10, drop=True)
        .data.astype(int)
        .tolist()
    )

    print(f"The following {len(invalid_derivate_mass_conservation_ids)} clouds have invalid conservation of mass {invalid_derivate_mass_conservation_ids}")

    return invalid_derivate_mass_conservation_ids

def remove_invalid_precipitation(ds, ds_error, ds_error_sem):
    '''
    Remove clouds with precipitation which exceed the inter-cloud
    mean by more than 4 standard deviations. 
    (see notebook 17-NN-figure-review-version.ipynb for details)
    '''
    data = ds_error["inflow_precipitation"].sel(microphysics="condensation")
    data_sem = ds_error_sem["inflow_precipitation"].sel(microphysics="condensation")
    m = data.mean("cloud_id").data
    s = propagate_mean_std(data, data_sem, dim="cloud_id")

    print(f"mean: {m:.2f}, std: {s:.2f} mm/h")
    print(f"mean + 4 std: {m + 4 * s:.2f} mm/h")
    invalid_cloud_base_precipitation_ids = set(
        [int(_d) for _d in ds["cloud_id"].where(data > m + 4 * s, drop=True).data]
    )

    print(f"The following {len(invalid_cloud_base_precipitation_ids)} clouds have invalid base precipitation {invalid_cloud_base_precipitation_ids}")

    return invalid_cloud_base_precipitation_ids

def remove_invalid_evaporation(ds, ds_error, ds_error_sem):
    '''
    Exclude clouds with column integrated evaporation which exceed the inter-cloud
    mean by more than 4 standard deviations.
    (see notebook 17-NN-figure-review-version.ipynb for details)
    '''
    data = -ds_error["source_precipitation"].sel(microphysics="condensation")
    data_sem = ds_error_sem["source_precipitation"].sel(microphysics="condensation")
    m = data.mean("cloud_id").data
    s = propagate_mean_std(data, data_sem, dim="cloud_id")

    print(f"mean: {m:.2f}, std: {s:.2f} mm/h")
    print(f"mean + 4 std: {m + 4 * s:.2f} mm/h")
    invalid_column_integrated_evaporation_ids = set(
        [int(_d) for _d in ds["cloud_id"].where(data > m + 4 * s, drop=True).data]
    )

    print(f"The following {len(invalid_column_integrated_evaporation_ids)} clouds have invalid column integrated evaporation {invalid_column_integrated_evaporation_ids}")

    return invalid_column_integrated_evaporation_ids

def remove_invalid_rwc(ds_cleo_null, identified_clusters):
    '''
    remove clouds with rain water content too far from observations
    (see notebook 17-NN-figure-review-version.ipynb for details)
    '''
    # we want to omit outliers, where the rain water content in the model is different from the observations.T
    # There are two parameters to define the valid range:
    # 1. A multiplicative factor
    # 2. An additive factor to account for systematic offsets
    # Here we choose a factor of 1.5 and an additive factor of 0.05 g/m²
    factor = 1.5  # units: dimensionless
    addition = 0.02  # units: g m-2
    upper = lambda x: x * factor + addition
    lower = lambda x: x / factor - addition / factor

    fig, ax = plt.subplots()

    cloud_ids = ds_cleo_null["cloud_id"].data
    x = np.geomspace(1e-9, 0.6, 100)
    ax.fill_between(
        x,
        lower(x),
        upper(x),
        color="grey",
        alpha=0.1,
        zorder=1,
    )
    ax.plot(
        x,
        x,
        color="grey",
        alpha=0.3,
        zorder=2,
    )
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)

    x = identified_clusters["mean_rain_water_content"].sel(cloud_id=cloud_ids)
    x.attrs["long_name"] = "Observed Cloud Rain Water Content"
    xerr = identified_clusters["sem_rain_water_content"].sel(cloud_id=cloud_ids)
    y = (
        ds_cleo_null["cloud_liquid_water_content"]
        .sel(microphysics="null_microphysics")
        .sel(cloud_id=cloud_ids)
    )
    yerr = y * 0

    ax.errorbar(
        x=x,
        y=y,
        xerr=xerr,
        yerr=yerr,
        label="All clouds",
        color="gray",
        marker="None",
        linestyle="None",
        alpha=0.5,
        zorder=3,
    )

    ax.set_xlabel(label_from_attrs(x, linebreak=True))
    ax.set_ylabel(label_from_attrs(y, linebreak=True))

    mask = (x > lower(y)) & (x < upper(y))
    rwc_valid_ids = x["cloud_id"].where(mask, drop=True).data
    rwc_invalid_ids = x["cloud_id"].where(~(mask), drop=True).data
    print(rwc_valid_ids.size, "out of", len(x))

    ax.scatter(
        x=x.sel(cloud_id=rwc_valid_ids),
        y=y.sel(cloud_id=rwc_valid_ids),
        color="k",
        marker=".",
        zorder=3,
    )
    ax.plot(
        x.sel(cloud_id=rwc_invalid_ids),
        y.sel(cloud_id=rwc_invalid_ids),
        linestyle="None",
        color="orange",
        marker="x",
        markersize=3,
        zorder=3,
    )

    set_rwc_invalid_ids = set(rwc_invalid_ids.astype(int))

    print(f"The following {len(rwc_invalid_ids)} clouds have invalid rain water content {set_rwc_invalid_ids}")

    return set_rwc_invalid_ids

def combine_and_write_outliers(data_dir, microphysics_styles, ds, ds_cleo_null, identified_clusters):
    '''
    Remove Outliers and identify valid cloud ids and write to file
    (see notebook 17-NN-figure-review-version.ipynb for details)
    '''
    relative_to_variables = ["inflow", "outflow", "source"]
    error_microphysics = (
        "null_microphysics",
        "condensation",
        "collision_condensation",
        "coalbure_condensation_small",
        "coalbure_condensation_large",
    )
    cleo_dataset_error = data_loading.CleoDataset(
        data_dir=data_dir,
        microphysics=error_microphysics,
    )
    # get physical height cleo output data
    ds_error, ds_error_sem = cleo_dataset_error()

    invalid_combined_dataset_ids, error_combined_dataset_ids = remove_invalid_combined(data_dir, microphysics_styles, ds)
    invalid_derivate_mass_conservation_ids = remove_invalid_conservation(data_dir, ds_error, error_microphysics, relative_to_variables)
    invalid_cloud_base_precipitation_ids = remove_invalid_precipitation(ds, ds_error, ds_error_sem)
    invalid_column_integrated_evaporation_ids = remove_invalid_evaporation(ds, ds_error, ds_error_sem)
    set_rwc_invalid_ids = remove_invalid_rwc(ds_cleo_null, identified_clusters)

    do_write = True
    exclude_rwc_invalid = True

    all_cloud_ids = set(ds["cloud_id"].data.astype(int).tolist())

    if do_write:
        invalid_data_cloud_ids = invalid_combined_dataset_ids.union(error_combined_dataset_ids).union(
            invalid_derivate_mass_conservation_ids
        )
        invalid_value_cloud_ids = invalid_cloud_base_precipitation_ids.union(
            invalid_column_integrated_evaporation_ids
        )
        if exclude_rwc_invalid:
            set_invalid_cloud_ids = invalid_data_cloud_ids.union(invalid_value_cloud_ids).union(set_rwc_invalid_ids)
        else:
            set_invalid_cloud_ids = invalid_data_cloud_ids.union(invalid_value_cloud_ids)

        # remove invalid clouds
        set_valid_cloud_ids = all_cloud_ids - set_invalid_cloud_ids
        valid_cloud_ids = sorted(set_valid_cloud_ids)

        with open(
            data_dir / Path("valid_cloud_ids.yaml"),
            "w",
        ) as f:
            f.write(
                textwrap.dedent(
                    f"""\
            # The following cloud ids are valid for the CLEO data
            # and can be used for the analysis
            valid_cloud_ids:
            """
                )
            )
            for _id in valid_cloud_ids:
                f.write(f" - {_id}\n")

        print("writing to:", data_dir / Path("valid_cloud_ids.yaml"))
        print(f"Number of clouds with valid CLEO data is {len(valid_cloud_ids)} of {len(all_cloud_ids)}")

    # load valid cloud ids
    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    d = yaml.load(data_dir / Path("valid_cloud_ids.yaml"))
    valid_cloud_ids = d["valid_cloud_ids"]

    print("loading from:", data_dir / Path("valid_cloud_ids.yaml"))
    print(f"Number of clouds with valid CLEO data is {len(valid_cloud_ids)} of {len(all_cloud_ids)}")

    return valid_cloud_ids, all_cloud_ids

### Load Datasets
def load_required_datasets_and_remove_outliers(data_dir, data_dir_novent, data_dir_obs, microphysics_styles):
    ''' 1) load original datasets, 2) remove outliers, 3) return valid cloud ids and datasets'''
    ### 1) load original datasets
    cleo_dataset = data_loading.CleoDataset(
        data_dir=data_dir,
        microphysics=tuple(microphysics_styles),
    )

    # get physical height cleo output data
    ds, ds_sem = cleo_dataset()
    cleo_dataset.normalize_gridboxes()
    # get normalized height cleo output data
    ds_normalized, ds_normalized_sem = cleo_dataset()

    # get non ventilated cleo output data
    cleo_dataset_no_ventilation = data_loading.CleoDataset(
        data_dir=data_dir_novent,
        microphysics=('condensation',),
    )
    ds_no_ventilation, ds_sem_no_ventilation = cleo_dataset_no_ventilation()

    # load identified clusters dataset
    identified_clusters = xr.open_dataset(
        data_dir_obs
        / "cloud_composite" / "processed" / "identified_clusters" / "identified_clusters_rain_mask_5.nc"
        )
    identified_clusters = identified_clusters.swap_dims({"time": "cloud_id"})

    # get null microphysics data
    cleo_dataset_null = data_loading.CleoDataset(
        data_dir=data_dir,
        microphysics=("null_microphysics",),
    )
    ds_cleo_null, ds_cleo_sem_null = cleo_dataset_null()

    ### 2) Now remove outliers
    valid_cloud_ids, all_cloud_ids = combine_and_write_outliers(data_dir, microphysics_styles, ds, ds_cleo_null, identified_clusters)
    print(f"Number of clouds with valid CLEO data is {len(valid_cloud_ids)} of {len(all_cloud_ids)}")
    print(f"{len(valid_cloud_ids) / len(all_cloud_ids) * 100:.2f}% of the clouds have valid CLEO data")

    ### 3) Return valid cloud ids and datasets
    ds = ds.sel(cloud_id=valid_cloud_ids)
    ds_sem = ds_sem.sel(cloud_id=valid_cloud_ids)

    ds_normalized = ds_normalized.sel(cloud_id=valid_cloud_ids)
    ds_normalized_sem = ds_normalized_sem.sel(cloud_id=valid_cloud_ids)

    ds_cleo_null = ds_cleo_null.sel(cloud_id=valid_cloud_ids)
    ds_cleo_sem_null = ds_cleo_sem_null.sel(cloud_id=valid_cloud_ids)

    identified_clusters = identified_clusters.sel(cloud_id=valid_cloud_ids)

    valid_cloud_ids_no_ventilation = [
        cid for cid in valid_cloud_ids if cid in ds_no_ventilation.cloud_id.values
    ]
    ds_no_ventilation = ds_no_ventilation.sel(cloud_id=valid_cloud_ids_no_ventilation)
    ds_sem_no_ventilation = ds_sem_no_ventilation.sel(cloud_id=valid_cloud_ids_no_ventilation)

    print(f"Number of clouds after removing invalid data: {len(ds['cloud_id'])}")

    return valid_cloud_ids, all_cloud_ids, ds, ds_sem, ds_normalized, ds_normalized_sem, ds_no_ventilation, ds_sem_no_ventilation, identified_clusters, ds_cleo_null, ds_cleo_sem_null

### Further post-processsing for plotting variables
def add_extra_plotting_variables_to_datasets(ds, ds_normalized):
    '''
    Add extra variables to the datasets for plotting purposes
    (see notebook 17-NN-figure-review-version.ipynb for details)
    '''
    ds["cloud_base_height"] = ds["gridbox_coord3"].sel(gridbox=ds["max_gridbox"])
    ds["cloud_base_height"].attrs = {
        "long_name": "Cloud base height",
        "units": ds["gridbox_coord3"].attrs["units"],
    }
    ds["relative_humidity_mean"] = (
        ds["relative_humidity"] * ds["gridbox_volume"] / ds["gridbox_volume"].sum("gridbox")
    ).sum("gridbox")
    ds["relative_humidity_mean"].attrs = {
        "long_name": "Mean relative humidity",
        "units": ds["relative_humidity"].attrs["units"],
    }

    # Calculate mean evaporation height
    # for each gridbox the evaporation energy is given by E * V
    ev = ds["evaporation_rate_energy"] * ds["gridbox_volume"]
    # MEH is then the height average weighted by E * V.
    # We need to weight, because the gridbox volume is not constant along the vertical
    meh = ((ds["gridbox_coord3"] * ev) / ev.sum("gridbox")).sum("gridbox")
    meh = meh / ds["gridbox_coord3"].sel(gridbox=ds["max_gridbox"])

    ds["mean_evaporation_height"] = meh
    ds["mean_evaporation_height"].attrs = dict(
        long_name="Mean evaporation height",
        units=ds_normalized["normalized_gridbox_coord3"].attrs["units"],
    )

    ds["radius_bins"].attrs.update(
        long_name="Radius",
        units="µm",
    )

    # update the name and units for the xi temporal mean
    radius_bin_width = (ds["radius_bins"].shift(radius_bins=-1) - ds["radius_bins"].shift(radius_bins=1)) / 2
    radius_bin_width = radius_bin_width.interpolate_na(dim="radius_bins", method="linear")
    ds["radius_bin_width"] = radius_bin_width
    ds["radius_bin_width"].attrs = dict(
        long_name="Radius bin width",
        units="µm",
        description="Width of the radius bin given by a linear interpolation of the radius bins",
    )

    ds["number_concentration"] = ds["xi_temporal_mean"] / ds["gridbox_volume"] / ds["radius_bin_width"]
    ds["number_concentration"].attrs = dict(
        long_name="Number concentration",
        units="m^{-3} µm^{-1}",
    )

    return ds

def load_eulerian_and_conservation_datasets_and_remove_outliers(data_dir, valid_cloud_ids):
    ds_eulerian = xr.open_dataset(
        data_loading.__eulerian_data_path__(data_dir=data_dir, microphysic="coalbure_condensation_small")
    )
    ds_eulerian = ds_eulerian.sel(time=TimeSlices.full_state)
    ds_eulerian = data_loading.__post_process_eulerian_dataset__(ds=ds_eulerian)

    ds_conservation = xr.open_dataset(
        data_loading.__conservation_data_path__(data_dir=data_dir, microphysic="coalbure_condensation_small")
    )
    ds_conservation = ds_conservation.sel(time=TimeSlices.full_state)
    ds_conservation = data_loading.__post_process_conservation_dataset__(
        ds=ds_conservation,
        da_surface_area=ds_eulerian["surface_area"].mean("gridbox"),
        timestep=ds_conservation["time"].diff("time").mean().values,
    )

    ds_eulerian = ds_eulerian.sel(cloud_id=valid_cloud_ids)
    ds_conservation = ds_conservation.sel(cloud_id=valid_cloud_ids)

    print(f"Number of eulerian clouds after removing invalid data: {len(ds_eulerian['cloud_id'])}")
    print(f"Number of conservation clouds after removing invalid data: {len(ds_conservation['cloud_id'])}")

    return ds_eulerian, ds_conservation

def get_observational_data_fitting(data_dir_obs, identified_clusters):
    # load dropsonde dataset
    dropsonde = xr.open_dataset(
        data_dir_obs / "dropsonde" / "processed" / "drop_sondes.nc"
    )

    # load dropsonde distances dataset
    ds_distances = xr.open_dataset(
        data_dir_obs
        / "combined" / "distance" / "distance_dropsondes_identified_clusters_rain_mask_5.nc",
    )

    # load cloud composite dataset
    cloud_composite = xr.open_dataset(
        data_dir_obs / "cloud_composite" / "processed" / "cloud_composite_SI_units_20241025.nc",
    )
    cloud_composite["radius2D"] = cloud_composite["radius"].expand_dims(time=cloud_composite["time"])
    cloud_composite = cloud_composite.transpose("radius", ...)
    cloud_composite = cloud_composite.sel(radius=slice(10e-6, None))

    identified_clusters = identified_clusters.where(
        (
            (identified_clusters.duration.dt.seconds >= 3)
            & (identified_clusters.altitude < 1200)
            & (identified_clusters.altitude > 500)
        ),
        drop=True,
    )

    return dropsonde, ds_distances, cloud_composite, identified_clusters

def get_correlations_dataset(ds):
    correlation_vars = (
        "cloud_mass_radius_mean",
        "cloud_liquid_water_content",
        "inflow_precipitation",
        "inflow_energy",
        "relative_humidity_mean",
        "cloud_base_height",
    )

    correlated_var = -ds["source_precipitation"]
    correlations = dict()
    for var in correlation_vars:
        x = ds[var]
        correlation = xr.corr(correlated_var, x, dim="cloud_id")
        correlations[var] = correlation

    # store correlations in dataset
    ds_correlations_CIE = xr.Dataset(correlations)

    correlated_var = ds["evaporation_fraction"]
    correlations = dict()
    for var in correlation_vars:
        x = ds[var]
        correlation = xr.corr(correlated_var, x, dim="cloud_id")
        correlations[var] = correlation

    # store correlations in dataset
    ds_correlations_EF = xr.Dataset(correlations)

    correlated_var = ds["mean_evaporation_height"]
    correlations = dict()
    for var in correlation_vars:
        x = ds[var]
        correlation = xr.corr(correlated_var, x, dim="cloud_id")
        correlations[var] = correlation

    # store correlations in dataset
    ds_correlations_MEH = xr.Dataset(correlations)

    return ds_correlations_CIE, ds_correlations_EF, ds_correlations_MEH

def get_logarithmic_correlations_dataset(ds):
    correlation_vars = (
        "cloud_mass_radius_mean",
        "cloud_liquid_water_content",
        "inflow_precipitation",
        "inflow_energy",
        "relative_humidity_mean",
        "cloud_base_height",
    )

    correlated_var = -ds["source_precipitation"]
    correlations = dict()
    for var in correlation_vars:
        x = ds[var]
        correlation = xr.corr(np.log(correlated_var), np.log(x), dim="cloud_id")
        correlations[var] = correlation

    # store correlations in dataset
    ds_correlations_log_CIE = xr.Dataset(correlations)

    correlated_var = ds["evaporation_fraction"]
    correlations = dict()
    for var in correlation_vars:
        x = ds[var]
        correlation = xr.corr(np.log(correlated_var), np.log(x), dim="cloud_id")
        correlations[var] = correlation

    # store correlations in dataset
    ds_correlations_log_EF = xr.Dataset(correlations)

    correlated_var = ds["mean_evaporation_height"]
    correlations = dict()
    for var in correlation_vars:
        x = ds[var]
        correlation = xr.corr(np.log(correlated_var), np.log(x), dim="cloud_id")
        correlations[var] = correlation

    # store correlations in dataset
    ds_correlations_log_MEH = xr.Dataset(correlations)

    return ds_correlations_log_CIE, ds_correlations_log_EF, ds_correlations_log_MEH

def get_theory_evaporation_fraction(ds):
    RH = ds["relative_humidity"].mean().data / 100
    H = 1000

    rhow = 0.998e3
    rhoa = 1.2
    eta = 1.85e-5
    g = 9.81
    nu = eta / rhoa
    T = 294.41807507
    p = 1e5
    Dv0 = 0.211 * (T / 273.15) ** (1.94) * (1013.25e2 / p) * 1e-4  # PK97 (13-3)
    Sc = 0.71  # nu/Dv0
    gamma = 73e-3
    Coo = 0.26
    Cgamma = 18.4
    lgamma = np.sqrt(gamma / (rhow * g))
    kb = 1.380649e-23
    Rconst = 8.314
    Rv = 461.5
    lv = 2.5e6
    ka = 26.19e-3

    def psat_water(T):
        theta = T - 273.15
        psat = 6.1121e2 * np.exp((18.678 - theta / 234.5) * (theta / (257.14 + theta)))
        return psat


    def rhosat_water(T):
        rho = psat_water(T) * 18.01528e-3 / (Rconst * T)
        return rho


    Dv = Dv0 / (1 + lv * Dv0 * rhosat_water(T) / (ka * T) * (lv / (Rv * T) - 1))


    def theoretical_evaporation_fraction(r0s: xr.DataArray) -> xr.DataArray:
        bU = np.sqrt(8 / 3 * rhow / rhoa * g / 0.5)
        dr52 = 5 / 2 * Dv * H / bU * (1 - RH) * rhosat_water(T) / rhow
        efftheo = 1 - (1 - dr52 / r0s ** (5 / 2)) ** (6 / 5)
        efftheo = np.minimum(efftheo.fillna(1), 1)
        return efftheo


    def fv(a, v):
        """Arguments are mass and velocity"""
        Re = 2 * a * np.abs(v) / nu
        x = Sc ** (1 / 3) * Re ** (1 / 2)
        if a < 60e-6:
            return 1 + 0.108 * x**2
        else:
            return 0.78 + 0.308 * x


    def fv_xr(a: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
        """Arguments are mass and velocity"""
        Re = 2 * a * np.abs(v) / nu
        x = Sc ** (1 / 3) * Re ** (1 / 2)
        low = 1 + 0.108 * x**2
        high = 0.78 + 0.308 * x

        return xr.where(a < 60e-6, low, high)


    def vtlim(a):
        """Terminal velocity in m/s"""
        c1 = Coo ** (1 / 2)
        c2 = (12 * nu / a) ** (1 / 2)
        c3 = (8 * rhow * g * a / (3 * rhoa)) ** (1 / 2)
        return ((np.sqrt(c2**2 + 4 * c1 * c3) - c2) / (2 * c1)) ** 2


    def vt(a):
        """Terminal velocity in m/s"""
        c1 = Coo ** (1 / 2) * (1 + Cgamma * (a / lgamma) ** 3) ** (1 / 6)
        c2 = (12 * nu / a) ** (1 / 2)
        c3 = (8 * rhow * g * a / (3 * rhoa)) ** (1 / 2)
        return ((np.sqrt(c2**2 + 4 * c1 * c3) - c2) / (2 * c1)) ** 2


    ventilation_coefficient = fv_xr(ds["radius_bins"] * 1e-6, vt(ds["radius_bins"] * 1e-6))

    evaporation_fraction = theoretical_evaporation_fraction(ds["radius_bins"] * 1e-6)
    evaporation_fraction_ventilation = evaporation_fraction * ventilation_coefficient
    evaporation_fraction_ventilation: xr.DataArray = np.minimum(
        evaporation_fraction_ventilation.fillna(1), 1
    )

    return evaporation_fraction, evaporation_fraction_ventilation