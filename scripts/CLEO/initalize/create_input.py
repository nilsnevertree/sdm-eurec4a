# %%
import datetime
import logging
import sys

from pathlib import Path

import lmfit
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


# %%
# THE PATH TO THE SCRIPT DIRECTORY
script_dir = Path("/home/m/m301096/repositories/sdm-eurec4a/scripts/CLEO/initalize")
print(script_dir)

# REPOSITORY_ROOT = Path(script_dir).parents[2]
REPOSITORY_ROOT = RepositoryPath("levante")()
print(REPOSITORY_ROOT)

# ======================================
# %% Define output and figure directories

final_dir = "output_v3.0"

output_dir = REPOSITORY_ROOT / "data/model/input/" / final_dir
output_dir.mkdir(parents=True, exist_ok=True)


fig_path = (
    REPOSITORY_ROOT / "results" / script_dir.relative_to(REPOSITORY_ROOT) / "create_input" / final_dir
)
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
handler = logging.FileHandler(log_dir / "create_input.log")
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
# input('all correct?')


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
    (identified_clouds["alt"] <= 1200)
    & (identified_clouds["alt"] >= 500)
    & (identified_clouds["vertical_extent"] <= 150)
    & (identified_clouds["duration"] >= np.timedelta64(3, "s"))
    # & (identified_clouds["liquid_water_content"] / identified_clouds["duration"].dt.seconds >=0.1)
    ,
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
    temperature=drop_sondes["air_temperature"],
    pressure=drop_sondes["pressure"],
    specific_humidity=drop_sondes["specific_humidity"],
)

# %% [markdown]
# print all the ids of clouds which are used selected by this criteria


# %%
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

    cloud_information["altitude"] = ds_cloud["alt"].data
    cloud_information["duration"] = ds_cloud["duration"].dt.seconds.data
    cloud_information["time"] = ds_cloud["time"].dt.strftime("%Y-%m-%d %H:%M:%S").data

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

    logging.info(f"Number of dropsondes: {ds_dropsondes['time'].shape}")
    logging.info(f"Number of cloudcomposite: {ds_cloudcomposite['time'].shape}")

    if ds_cloudcomposite["time"].shape[0] == 0:
        raise ValueError("No cloudcomposite data found for cloud")
    if ds_dropsondes["time"].shape[0] == 0:
        raise ValueError("No dropsonde data found for cloud")

    # # Split data into cloud and rain

    ds_lower = ds_cloudcomposite.sel(radius=slice(None, split_radius))
    ds_rain = ds_cloudcomposite.sel(radius=slice(split_radius, None))

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

    # Fit the ThermodynamicLinear models
    thermodynamic_split_linear_dict = dict(
        air_temperature=transfer.ThermodynamicSplitLinear(),
        specific_humidity=transfer.ThermodynamicSplitLinear(),
        potential_temperature=transfer.ThermodynamicSplitLinear(),
        relative_humidity=transfer.ThermodynamicSplitLinear(),
    )
    for var in thermodynamic_split_linear_dict:
        logging.info(f"Fitting {var}")
        thermodynamic_split_linear_dict[var] = transfer.fit_thermodynamics(
            da_thermo=ds_dropsondes[var],
            thermo_fit=thermodynamic_split_linear_dict[var],
            dim="alt",
            x_split=None,
        )

    mean_x_split = np.mean(
        [thermodynamic_split_linear_dict[key].get_x_split() for key in thermodynamic_split_linear_dict]
    )
    logging.info(f"Split level: {mean_x_split} m")

    # Fit the ThermodynamicLinear models
    thermodynamic_split_linear_dict_fixed = dict(
        air_temperature=transfer.ThermodynamicSplitLinear(),
        specific_humidity=transfer.ThermodynamicSplitLinear(),
        potential_temperature=transfer.ThermodynamicSplitLinear(),
        relative_humidity=transfer.ThermodynamicSplitLinear(),
        pressure=transfer.ThermodynamicLinear(),
    )
    for var in thermodynamic_split_linear_dict_fixed:
        logging.info(f"Fitting {var}")
        thermodynamic_split_linear_dict_fixed[var] = transfer.fit_thermodynamics(
            da_thermo=ds_dropsondes[var],
            thermo_fit=thermodynamic_split_linear_dict_fixed[var],
            dim="alt",
            x_split=mean_x_split,
        )

    # # Export everything as yaml
    add_representer()

    version_control = dict(
        git_hash=get_git_revision_hash(),
        date=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S"),
    )

    thermodynamic_parameters = thermodynamic_split_linear_dict_fixed

    output_dictonary = dict(
        particle_size_distribution=psd_fit,
        thermodynamics=thermodynamic_parameters,
        cloud=cloud_information,
        version_control=version_control,
    )

    # return output_dictonary

    output_filepath = output_dir / f"{identification_type}_{chosen_id}.yaml"

    with open(output_filepath, "w") as file:
        s = yaml.dump(output_dictonary)
        file.write(s)

    with open(subfig_path / "input_CLEO.yaml", "w") as file:
        s = yaml.dump(output_dictonary)
        file.write(s)


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


def thermofit_split_linear_representer(
    dumper: yaml.Dumper, obj: transfer.ThermodynamicSplitLinear
) -> yaml.Node:
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


def thermofit_linear_representer(dumper: yaml.Dumper, obj: transfer.ThermodynamicLinear) -> yaml.Node:
    """
    Represents a ThermodynamicSplitLinear object for YAML dumping.

    Parameters:
    dumper (yaml.Dumper): The YAML dumper instance.
    obj (transfer.ThermodynamicSplitLinear): The ThermodynamicSplitLinear object to represent.

    Returns:
    yaml.Node: The represented dictionary of parameters.
    """
    parameters = obj.get_parameters()

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
    yaml.add_representer(transfer.ThermodynamicSplitLinear, thermofit_split_linear_representer)
    yaml.add_representer(transfer.ThermodynamicLinear, thermofit_linear_representer)
    yaml.add_representer(transfer.PSD_LnNormal, psdfit_representer)


# %%

# for id in identified_clouds.cloud_id.data:
#     id = int(id)
#     plt.close('all')
#     try:
#         main(id)
#         logger.info(f"Created input for cloud {id}")
#     except Exception as e:
#         message = traceback.format_exception(None, e, e.__traceback__)
#         logger.error(f'Failed to create input for cloud {id}: '+ repr(e) + '\n' + ''.join(message))
#         continue
