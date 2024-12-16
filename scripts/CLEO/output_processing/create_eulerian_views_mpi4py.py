import argparse

from pathlib import Path
import os
import sys

from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

import awkward as ak
import numpy as np
import xarray as xr
import mpi4py
import logging
from mpi4py import MPI
import datetime

from sdm_eurec4a import RepositoryPath

from pySD.sdmout_src import pygbxsdat, pysetuptxt, supersdata
from sdm_eurec4a.conversions import relative_humidity_from_tps
import sdm_eurec4a.input_processing.models as smodels

from pySD.initsuperdropsbinary_src.probdists import DoubleLogNormal
from typing import Tuple

from sdm_eurec4a.visulization import set_custom_rcParams

from sdm_eurec4a.conversions import msd_from_psd_dataarray

from sdm_eurec4a import RepositoryPath
import secrets

set_custom_rcParams()

RP = RepositoryPath("levante")

repo_dir = RP.repo_dir
sdm_data_dir = RP.data_dir


# === mpi4py ===
try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # [0,1,2,3,4,5,6,7,8,9]
    npro = comm.Get_size()  # 10
except:
    print("::: Warning: Proceeding without mpi4py! :::")
    rank = 0
    npro = 1

# create shared logging directory
if rank == 0:
    # Generate a shared directory name based on UTC time and random hex
    time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    random_hex = secrets.token_hex(4)
    log_dir = repo_dir / "logs" / f"create_eulerian_views/{time_str}-{random_hex}"
    log_dir.mkdir(exist_ok=True, parents=True)
else:
    log_dir = None

# Broadcast the shared directory name to all processes
log_dir = comm.bcast(log_dir, root=0)
# create individual log file
log_file_path = log_dir / f"{rank}.log"


# === logging ===
# create log file


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(log_file_path)
handler.setLevel(logging.INFO)

# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

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


logging.info(f"====================")
logging.info(f"Start with rank {rank} of {npro}")


parser = argparse.ArgumentParser(
    description="Create eulerian view for data_dir which contains zarr dir in ./eurec4a1d_sol.zarr and config files in ./config/eurec4a1d_setup.txt"
)

# Add arguments
parser.add_argument("-d", "--data_dir", type=str, help="Path to data directory", required=True)
# Parse arguments
args = parser.parse_args()

master_data_dir = Path(args.data_dir)
subfolder_pattern = "cluster*"


logging.info(f"Enviroment: {sys.prefix}")
logging.info(f"Create eulerian view in: {master_data_dir}")
logging.info(f"Subfolder pattern: {subfolder_pattern}")
data_dir_list = sorted(list(master_data_dir.glob(subfolder_pattern)))


def ak_differentiate(sa: supersdata.SupersAttribute) -> supersdata.SupersAttribute:
    """
    This function calculates the difference of the data in the
    supersdata.SupersAttribute along the last axis. The difference is calculated as the
    difference of the next value minus the current value. The last value is set to nan,
    to make sure, that the mass change is at the same timestep, as the original value.

    Notes
    -----
    - The function is designed to work with awkward arrays
    - It is intended to be used on relatively regular arrays, where the last axis has at least 1 value or best 2 values.
    - Arrays which are empty along the last axis will be filled with a nan after execution. So use this function with caution due to high increase in memory usage.

    Parameters
    ----------
    sa : supersdata.SupersAttribute
        The attribute, which should be differentiated.
        Assuming it has the shape (N, M, var), the differentiation is done along the last axis.

    Returns
    -------
    supersdata.SupersAttribute
        The differentiated attribute.
        The output has the same shape as the input, but the last value along the last axis is nan.
        The new name of the attribute is the old name with "_difference" appended.
        All metadata is copied and the long_name is appended with "difference".
    """

    data = sa.data
    # logging.info(data)
    # It is very important, to concate the nan values at the END of the array, so that the last value is nan.
    # This makes sure, that the mass change is at the same timestep, as the original value.
    # With this, the evapoartion fraction can not exceed 1.
    data: ak.Array = ak.concatenate([data, np.nan], axis=-1)

    # if the data has entries, which have only one value, append another nan value
    if ak.min(ak.num(data, axis=-1)) < 2:
        data = ak.concatenate([data, np.nan], axis=-1)

    # calculate the difference
    diff = data[..., 1:] - data[..., :-1]

    # create a new attribute
    result = supersdata.SupersAttribute(
        name=sa.name + "_difference",
        data=diff,
        units=sa.units,
        metadata=sa.metadata.copy(),
    )

    # update metadata
    updated_metadata = sa.metadata.copy()
    try:
        updated_metadata["long_name"] = updated_metadata["long_name"] + " difference"
    except KeyError:
        pass
    result.set_metadata(metadata=updated_metadata)

    return result


def ak_last(sa: supersdata.SupersAttribute) -> supersdata.SupersAttribute:
    """
    This function only keeps the last value along axis 1. The rest will be replaced by
    nans.

    Notes
    -----
    - The function is designed to work with awkward arrays
    - It is intended to be used on relatively regular arrays, where the last axis has at least 1 value or best 2 values.

    Parameters
    ----------
    sa : supersdata.SupersAttribute
        The attribute, which should be lasted.
        Assuming it has the shape (N, M, var), the last values is kept along the last axis.

    Returns
    -------
    supersdata.SupersAttribute
        The lasted attribute.
        The output has the same shape as the input, but the last value along the last axis is nan.
        The new name of the attribute is the old name with "_last" appended.
        All metadata is copied and the long_name is appended with "last".
    """

    data = sa.get_data()

    # in order to remove all values except the last one, we need to create a new array with the same shape
    # data = [
    #     [1, 2, 3, 4],
    #     [5, 6, 7, 8],
    #    ]
    # after concatenate
    # data = [
    #     [n, n, n, n, 1],
    #     [n, n, n, n, 1],
    #    ]
    # after multiplication
    # data = [
    #     [n, n, n, n, 4],
    #     [n, n, n, n, 8],
    #    ]
    # after the slice, the result is
    # data = [
    #     [n, n, n, 4],
    #     [n, n, n, 8],
    #    ]

    last = (ak.concatenate([data * np.nan, 1], axis=-1) * data[..., -1])[..., 1:]
    # create a new attribute
    result = supersdata.SupersAttribute(
        name=sa.name + "_last",
        data=last,
        units=sa.units,
        metadata=sa.metadata.copy(),
    )

    # update metadata
    updated_metadata = sa.metadata.copy()
    try:
        updated_metadata["long_name"] = updated_metadata["long_name"] + " last"
    except KeyError:
        pass
    result.set_metadata(metadata=updated_metadata)

    return result


def create_lagrangian_dataset(dataset: supersdata.SupersDataNew) -> supersdata.SupersDataNew:
    """
    This function creates a lagrangian view of the SupersDataset. Within this setup, the
    following variables are calculated and added to the dataset:

    - mass_difference : The mass difference per second
    - mass_difference_timestep : The mass difference per timestep
    - xi_difference : The multiplicity difference per second
    - evaporated_fraction : The evaporated fraction per second

    Then a eulerian view is created by binning the data by sdId.
    The following variables are given in the dataset:
    - mass : mass which is represented by a superdroplet
    - mass_difference : mass difference which is represented by a superdroplet
    - radius : radius of a superdroplet
    - evaporated_fraction : evaporated fraction per second of a superdroplet
    - xi : multiplicity of a superdroplet
    - number_superdroplets : 1 for each superdroplet. This can be summed over to get the number of superdroplets in a bin.

    Parameters
    ----------
    dataset : supersdata.SupersDataNew
        The dataset which should be transformed to a lagrangian view.

    Returns
    -------
    supersdata.SupersDataNew
        The dataset in the lagrangian view.
        It contains the following attributes:
        - mass : mass which is represented by a superdroplet
        - mass_difference : mass difference which is represented by a superdroplet
        - radius : radius of a superdroplet
        - evaporated_fraction : evaporated fraction per second of a superdroplet
        - xi : multiplicity of a superdroplet
        - number_superdroplets : 1 for each superdroplet. This can be summed over to get the number of superdroplets in a bin.

        It has the following coordinates:
        - sdId : the superdroplet id
    """

    dataset.flatten()

    # ============
    # 1. Create the necessary indexes and pass if they already exist
    # ============
    # make time an indexer which correspondataset to the unique values of the time attribute
    try:
        dataset.set_attribute(dataset["time"].attribute_to_indexer_unique())
    except KeyError:
        pass
    try:
        dataset.set_attribute(dataset["sdId"].attribute_to_indexer_unique())
    except KeyError:
        pass

    # ============
    # 2. Create the Lagrangian view to calculate the mass change
    # ============

    # bin by the superdroplet id and calcuate the difference of the mass
    dataset.index_by_indexer(index=dataset["sdId"])

    time_diff = ak_differentiate(dataset["time"])
    time_diff.set_metadata(
        metadata={
            "long_name": "Time difference per timestep",
        }
    )
    time_diff.set_name("time_difference")
    time_diff.set_units("s")

    # calculate the difference of the mass as the total mass change per timestep
    mass_rep_diff = ak_differentiate(dataset["mass_represented"])
    mass_rep_diff.set_metadata(
        metadata={
            "long_name": "Mass difference per timestep",
            "notes": r"Mass here is mass represented by a superdroplet: $m = \xi \cdot m_{sd}$",
        }
    )
    mass_rep_diff.set_name("mass_difference_timestep")
    dataset.set_attribute(mass_rep_diff)

    # calculate the difference of the mass as the total mass change per second
    mass_rep_diff_per_second = mass_rep_diff / time_diff
    mass_rep_diff_per_second.set_metadata(
        metadata={
            "long_name": "Mass difference per second",
            "notes": r"Mass here is mass represented by a superdroplet: $m = \xi \cdot m_{sd}$",
        }
    )
    mass_rep_diff_per_second.set_name("mass_difference")
    dataset.set_attribute(mass_rep_diff_per_second)

    # calculate the difference of the multiplicity per second
    xi_diff = ak_differentiate(dataset["xi"]) / time_diff
    xi_diff.set_name("xi_difference")
    xi_diff.set_metadata(
        metadata={
            "long_name": "Multiplicity difference per second",
        }
    )
    dataset.set_attribute(xi_diff)

    # calculate the evaporated fraction per second
    evaporated_fraction = dataset["mass_difference"] / dataset["mass_represented"] * -100
    evaporated_fraction.set_name("evaporated_fraction")
    evaporated_fraction.set_metadata(
        metadata={
            "long_name": "evaporated fraction per second",
            "notes": r"Evaporated fraction is calculated as $\frac{\Delta m}{m} \cdot 100$",
        }
    )
    evaporated_fraction.set_units("%")
    dataset.set_attribute(evaporated_fraction)

    # number of superdroplets
    counts = dataset.get_data("xi")
    counts = counts * 0 + 1
    number_superdroplets = supersdata.SupersAttribute(
        name="number_superdroplets",
        data=counts,
        units="#",
        metadata={
            "long_name": "number of superdroplets",
        },
    )
    dataset.set_attribute(number_superdroplets)

    # calculate total mass which left domain
    mass_left = ak_last(dataset["mass_represented"])
    mass_left.set_name("mass_left")
    mass_left.set_metadata(
        metadata={
            "long_name": "mass which left domain",
            "note": r"this is the last represented mass which a super droplet has during the simulation.\nMass represented by a superdroplet: $m = \xi \cdot m_{sd}$",
        }
    )
    mass_left.set_units("kg")
    dataset.set_attribute(mass_left)

    # calculate total mass which left domain
    xi_left = ak_last(dataset["xi"])
    xi_left.set_name("xi_left")
    xi_left.set_metadata(
        metadata={
            "long_name": "multiplicity of droplets, which left domain",
            "note": r"this is the last represented multiplicity which a super droplet has during the simulation.",
        }
    )
    xi_left.set_units("kg")
    dataset.set_attribute(xi_left)

    # calculate total number of superdroplets which left domain
    counts_left = ak_last(dataset["number_superdroplets"])
    counts_left.set_name("number_superdroplets_left")
    counts_left.set_metadata(
        metadata={
            "long_name": "umber of superdroplets only 1 if left domain",
            "note": r"This is the number of superdroplets leave domain has during the simulation.",
        }
    )
    counts_left.set_units("#")
    dataset.set_attribute(counts_left)

    return dataset


def create_eulerian_dataset(
    dataset: supersdata.SupersDataNew, radius_bins: np.ndarray = np.logspace(-7, 7, 150)
) -> supersdata.SupersDataNew:
    """
    This function creates a eulerian view of the SupersDataset. First, a lagrangian view
    is created by binning the data by the superdroplet id. Within this setup, the
    following variables are calculated:

    - mass_difference : The mass difference per second
    - mass_difference_timestep : The mass difference per timestep
    - xi_difference : The multiplicity difference per second
    - evaporated_fraction : The evaporated fraction per second

    Then a eulerian view is created by binning the data by time, gridbox and radius_bins.
    The following variables are given in the dataset:
    - mass : mass which is represented by a superdroplet
    - mass_difference : mass difference which is represented by a superdroplet
    - radius : radius of a superdroplet
    - evaporated_fraction : evaporated fraction per second of a superdroplet
    - xi : multiplicity of a superdroplet
    - number_superdroplets : 1 for each superdroplet. This can be summed over to get the number of superdroplets in a bin.

    Parameters
    ----------
    dataset : supersdata.SupersDataNew
        The dataset which should be transformed to a eulerian view.
    radius_bins : np.ndarray, optional
        The bins for the radius, by default np.logspace(-7, 7, 150)

    Returns
    -------
    supersdata.SupersDataNew
        The dataset in the eulerian view.
        It contains the following attributes:
        - mass : mass which is represented by a superdroplet
        - mass_difference : mass difference which is represented by a superdroplet
        - radius : radius of a superdroplet
        - evaporated_fraction : evaporated fraction per second of a superdroplet
        - xi : multiplicity of a superdroplet
        - number_superdroplets : 1 for each superdroplet. This can be summed over to get the number of superdroplets in a bin.

        It has the following coordinates:
        - gridbox : the gridbox index
        - time : the time index
        - radius_bins : the radius bin index
    """

    # ============
    # 1. Create the Lagrangian view to calculate the mass change
    # ============

    dataset = create_lagrangian_dataset(dataset)
    dataset.flatten()

    # ============
    # 2. Create the necessary indexes asnd pass if they already exist
    # ============
    try:
        dataset.set_attribute(dataset["time"].attribute_to_indexer_unique())
    except KeyError:
        pass
    try:
        dataset.set_attribute(dataset["sdId"].attribute_to_indexer_unique())
    except KeyError:
        pass

    try:
        # make time an indexer which correspondataset to the unique values of the time attribute
        dataset.set_attribute(dataset["sdgbxindex"].attribute_to_indexer_unique(new_name="gridbox"))
    except KeyError:
        pass

    try:
        dataset.set_attribute(
            dataset["radius"].attribute_to_indexer_binned(bins=radius_bins, new_name="radius_bins")
        )
    except KeyError:
        pass

    # ============
    # 3. Create eulerian view
    # ============

    dataset.index_by_indexer(index=dataset["time"])
    dataset.index_by_indexer(index=dataset["gridbox"])
    dataset.index_by_indexer(index=dataset["radius_bins"])

    # ============
    # 4. Rename mass_represented to mass and mass to mass_individual
    # this helps to be consiten with the naming of the mass in a eulerian view
    # ============

    # create an attribute which counts the number of superdroplets
    counts = dataset.get_data("xi")
    counts = counts * 0 + 1
    number_superdroplets = supersdata.SupersAttribute(
        name="number_superdroplets",
        data=counts,
        units="#",
        metadata={
            "long_name": "number of superdroplets",
        },
    )
    dataset.set_attribute(number_superdroplets)

    return dataset


def create_eulerian_xr_dataset(
    dataset: supersdata.SupersDataNew,
    radius_bins: np.ndarray = np.logspace(-7, 7, 150),
) -> xr.Dataset:
    """
    This function creates a eulerian view of the SupersDataset and transforms it to a
    xarray dataset. For this, the function create_eulerian_dataset is used to create the
    eulerian view.

    Note
    ----
    The ``dataset`` is mutated.
    It will be transformed to a eulerian view by binning the data by time, gridbox and radius_bins.
    New variable names will be added to the dataset.
    Load the dataset new, if you want to use the original dataset again.

    The following variables are calculated:
    - mass : The total mass in the bin
    - mass_difference : The mass difference per second in the bin
    - radius : The mean radius in the bin
    - evaporated_fraction : The mean evaporated fraction per second in the bin
    - xi : The total multiplicity in the bin
    - number_superdroplets : The total number of superdroplets in the bin

    Parameters
    ----------
    dataset : supersdata.SupersDataNew
        The dataset which should be transformed to a eulerian view.
    radius_bins : np.ndarray, optional
        The bins for the radius, by default np.logspace(-7, 7, 150)

    Returns
    -------
    xr.Dataset
        The dataset in the eulerian view.
        It contains the following variables:
        - mass : The total mass in the bin
        - mass_difference : The mass difference per second in the bin
        - radius : The mean radius in the bin
        - evaporated_fraction : The evaporated fraction per second in the bin
        - xi : The total multiplicity in the bin
        - number_superdroplets : The number of superdroplets in the bin
        - mass_left : The mass which left the domain in the bin
        - number_superdroplets_left : The number of superdroplets which left the domain in the bin

        It has the following coordinates:
        - gridbox : the gridbox index
        - time : the time index
        - radius_bins : the radius bin index
    """

    # create a eulerian view of the data

    eulerian = create_eulerian_dataset(dataset, radius_bins=radius_bins)

    eulerian.__update_attributes__()

    # create a dataset with the necessary reductions
    sum_reduction = dict(
        reduction_func=ak.sum,
        add_metadata={"reduction_func": "ak.sum Summation over all SDs in the bin"},
    )
    mean_reduction = dict(
        reduction_func=ak.mean, add_metadata={"reduction_func": "ak.mean Mean over all SDs in the bin"}
    )

    reduction_map = {
        # basic attributes and variables
        "radius": mean_reduction,
        "xi": sum_reduction,
        "number_superdroplets": sum_reduction,
        "mass_represented": sum_reduction,
        # differentiated attributes
        "mass_difference": sum_reduction,
        "mass_difference_timestep": sum_reduction,
        "evaporated_fraction": mean_reduction,
        # left attributes
        "mass_left": sum_reduction,
        "xi_left": sum_reduction,
        "number_superdroplets_left": sum_reduction,
    }

    # create individual DataArrays
    da_list = []
    for varname in reduction_map:
        reduction_func = reduction_map[varname]["reduction_func"]
        add_metadata = reduction_map[varname]["add_metadata"]
        da = eulerian.attribute_to_DataArray_reduction(
            attribute_name=varname,
            reduction_func=reduction_func,
        )
        da.attrs.update(add_metadata)
        da_list.append(da)

    # create the dataset by merging the DataArrays
    ds = xr.merge(da_list)

    ds["time"].attrs["long_name"] = "time"
    ds["time"].attrs["units"] = "s"

    return ds


# Functions to add thermodynamic variables and gridbox properties to the dataset


def add_thermodynamics(ds: xr.Dataset, ds_zarr: xr.Dataset) -> None:
    """
    This function adds the thermodynamic variables to the dataset. The
    following variables are added:

    - pressure : The pressure in Pascals
    - air_temperature : The air temperature in Kelvin
    - specific_mass_vapour : The specific mass of vapour in Kg/Kg
    - relative_humidity : The relative humidity in percent

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to which the variables should be added.
    ds_zarr : xr.Dataset
        The dataset which contains the thermodynamic variables.
    """

    ds["pressure"] = ds_zarr["press"].mean("time", keep_attrs=True)
    ds["pressure"] = ds["pressure"] * 100  # convert from hectoPascals to Pascals
    ds["pressure"].attrs["units"] = "Pa"

    ds["air_temperature"] = ds_zarr["temp"].mean("time", keep_attrs=True)

    ds["specific_mass_vapour"] = ds_zarr["qvap"].mean("time", keep_attrs=True)
    ds["specific_mass_vapour"] = ds["specific_mass_vapour"] / 1000  # convert from g/Kg to Kg/Kg
    ds["specific_mass_vapour"].attrs["units"] = "kg/kg"

    ds["relative_humidity"] = relative_humidity_from_tps(
        temperature=ds["air_temperature"],
        pressure=ds["pressure"],
        specific_humidity=ds["specific_mass_vapour"],
    )


def add_gridbox_properties(ds: xr.Dataset, gridbox_dict: dict, gridbox_key: str = "gridbox") -> None:
    ds["gridbox_top"] = (("gridbox",), gridbox_dict["zhalf"][1:])
    ds["gridbox_top"].attrs.update(
        long_name="Gridbox top",
        description=f"Gridbox top. Which is the upper bound of the gridbox for each gridbox.",
        units="$m$",
    )
    ds["gridbox_bottom"] = (("gridbox",), gridbox_dict["zhalf"][:-1])
    ds["gridbox_bottom"].attrs.update(
        long_name="Gridbox bottom",
        description=f"Gridbox bottom. Which is the lower bound of the gridbox for each gridbox.",
        units="$m$",
    )

    ds["gridbox_coord3"] = (("gridbox",), gridbox_dict["zfull"])
    ds["gridbox_coord3"].attrs.update(
        long_name="Gridbox center coordinate 3",
        units="$m$",
    )

    ds["surface_area"] = (
        ("gridbox",),
        np.full_like(ds["gridbox"], np.diff(gridbox_dict["xhalf"]) * np.diff(gridbox_dict["yhalf"])),
    )
    ds["surface_area"].attrs.update(
        long_name="Gridbox center coordinate 3",
        units="$m$",
    )

    ds["gridbx_coord3_norm"] = ds["gridbox_coord3"] / ds["gridbox_coord3"].max()
    ds["gridbx_coord3_norm"].attrs.update(
        long_name="Normalized gridbox center coordinate 3",
        description="Normalized gridbox center coordinate 3.",
        units="",
    )

    ds["gridbox_volume"] = (("gridbox",), gridbox_dict["gbxvols"][0, 0, :])
    ds["gridbox_volume"].attrs.update(
        long_name="Gridbox Volume",
        description=f"Gridbox Volume",
        units="$m^3$",
    )


def add_liquid_water_content(ds: xr.Dataset) -> None:
    ds["liquid_water_content"] = ds["mass_represented"] / ds["gridbox_volume"]
    ds["liquid_water_content"].attrs.update(
        long_name="Liquid Water Content",
        description="Liquid Water Content per gridbox",
        units=r"$kg m^{-3}$",
    )


def add_vertical_profiles(ds: xr.Dataset, time_slice=slice(1500, None)):
    # ds["vertical_liquid_water_content"] = (
    #     ds["liquid_water_content"].sel(time=time_slice).sum("radius_bins").mean("time")
    # )

    # ds["vertical_liquid_water_content"].attrs.update(
    #     long_name="Vertical Liquid Water Content",
    #     description=f"Vertical Profile of LWC. Sum over all radius bins and mean over time of the stationary state ({time_slice.start}-{time_slice.stop}).",
    #     units=ds["liquid_water_content"].attrs["units"],
    # )

    ds["mass_difference_per_volume"] = ds["mass_difference"] / ds["gridbox_volume"]
    ds["mass_difference_per_volume"].attrs.update(
        long_name="Mass Difference per volume",
        description="Mass Difference per Volume. The mass difference is divided by the gridbox volume.",
        units=r"$kg m^{-3} s^{-1}$",
    )

    # ds["vertical_mass_difference_per_volume"] = (
    #     ds["mass_difference_per_volume"].sel(time=time_slice).sum("radius_bins").mean("time")
    # )
    # ds["vertical_mass_difference_per_volume"].attrs.update(
    #     long_name="Vertical mass difference per voluem",
    #     description=f"Vertical Profile of mass difference per voluem. Sum over all radius bins and mean over time of the stationary state ({time_slice.start}-{time_slice.stop}).",
    #     units=r"$kg m^{-3} s^{-1}$",
    # )


def add_precipitation(ds):
    """
    Uses the mass_left variable in the dataset to calculate the precipitation rate.

    The precipitation rate is the ``mass_left`` in gridbox 0 divided by the time step.
    To convert from kg to mm/h, the value is divided by the surface area of gridbox 0 and multiplied by 3600.
    The result is added to the dataset as ``precipitation``.
    """

    precipitation_rate = ds["mass_left"].sel(gridbox=0) / ds["time"].diff("time").shift(time=-1)
    # unit conversion from kg to mm/h
    precipitation_rate = precipitation_rate / ds["surface_area"].sel(gridbox=0) * 3600
    ds["precipitation"] = precipitation_rate
    ds["precipitation"].attrs.update(
        long_name="Precipitation",
        description="Precipitation rate in mm/h",
        units="mm/h",
    )


def parameters_dataset_to_dict(ds: xr.Dataset, mapping: Union[dict[str, str], Tuple[str]]) -> dict:
    """
    Convert selected parameters from an xarray Dataset to a dictionary.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the parameters.
    mapping : Union[dict[str, str], Tuple[str]]
        A mapping of parameter names to extract from the Dataset. If a dictionary is provided,
        the keys are the new names for the parameters and the values are the names in the Dataset.
        If a tuple or list is provided, the parameter names are used as-is.

    Returns
    -------
    dict
        A dictionary where the keys are the parameter names (or new names if a dictionary was provided)
        and the values are the corresponding values from the Dataset.

    Raises
    ------
    TypeError
        If the mapping is not a dictionary or a tuple/list.
    """

    if isinstance(mapping, (list, tuple)):
        parameters = {key: float(ds[key].values) for key in mapping}
    elif isinstance(mapping, dict):
        parameters = {mapping[key]: float(ds[key].values) for key in mapping}
    else:
        raise TypeError("mapping must be a dict or a tuple")

    return parameters


ds_psd_parameters = xr.open_dataset(
    sdm_data_dir / "model/input_v4.0/particle_size_distribution_parameters_linear_space.nc"
)

# define the bin edges in micrometers
radii_edges = np.geomspace(10, 4e3, 151)

mapping = dict(
    geometric_mean1="geometric_mean1",
    geometric_mean2="geometric_mean2",
    geometric_std_dev1="geometric_std_dev1",
    geometric_std_dev2="geometric_std_dev2",
    scale_factor1="scale_factor1",
    scale_factor2="scale_factor2",
)

#
sublist_data_dirs = np.array_split(np.array(data_dir_list), npro)[rank]
total_npro = len(sublist_data_dirs)

sucessful = []

for step, data_dir in enumerate(sublist_data_dirs):

    logging.info(f"--------------------")
    logging.info(f"Rank {rank+1} {step+1}/{total_npro}")
    logging.info(f"processing {data_dir}")
    try:
        cloud_id = int(data_dir.name.split("_")[1])

        output_dir = data_dir / "processed"
        output_dir.mkdir(exist_ok=True, parents=False)

        output_path = output_dir / "eulerian_dataset.nc"
        output_path.parent.mkdir(exist_ok=True)

        setupfile_path = data_dir / "config" / "eurec4a1d_setup.txt"
        statsfile_path = data_dir / "config" / "eurec4a1d_stats.txt"
        zarr_path = data_dir / "eurec4a1d_sol.zarr"
        gridfile_path = data_dir / "share/eurec4a1d_ddimlessGBxboundaries.dat"

        # read in constants and intial setup from setup .txt file
        config = pysetuptxt.get_config(str(setupfile_path), nattrs=3, isprint=False)
        consts = pysetuptxt.get_consts(str(setupfile_path), isprint=False)
        gridbox_dict = pygbxsdat.get_gridboxes(str(gridfile_path), consts["COORD0"], isprint=False)

        ds_zarr = xr.open_zarr(zarr_path, consolidated=False)
        ds_zarr = ds_zarr.rename({"gbxindex": "gridbox"})

        # Use the SupersDataNew class to read the dataset
        dataset = supersdata.SupersDataNew(dataset=str(zarr_path), consts=consts)

        logging.info("create the eulerian xarray Dataset from the raw dataset")
        ds = create_eulerian_xr_dataset(
            dataset=dataset,
            radius_bins=(radii_edges),
        )
        mini = ds["gridbox"].min().values
        if mini > 0:
            logging.info(f"Adding missing bottom gridboxes ({mini}) missing")
            ds2 = ds.isel(gridbox=slice(0, mini)) * np.nan
            ds2["gridbox"] = np.arange(mini)
            ds = xr.concat([ds2, ds], dim="gridbox")
        else:
            logging.info(f"All bottom gridboxes are already present")

        logging.info("Add thermodynamic variables and gridbox properties to the dataset")

        add_thermodynamics(ds, ds_zarr)
        add_gridbox_properties(ds, gridbox_dict)
        add_liquid_water_content(ds)
        add_vertical_profiles(ds)
        add_precipitation(ds)

        # add the monitor massdelta condensation
        ds["massdelta_condensation"] = (
            1e18 * 1e-3 * ds_zarr["massdelta_cond"] / ds["time"].diff("time") / ds["gridbox_volume"]
        )
        ds["massdelta_condensation"].attrs.update(
            long_name="Condensation mass",
            description="Condensation mass as caputured by CLEOs condensation monitor. The massdelta condensation is converted from g to kg m-3 s-1.",
            units=r"$kg m^{-3} s^{-1}$",
        )

        logging.info(f"Make sure to have float precission for all variables to be able to include NaNs")

        for var in ds:
            if np.issubdtype(ds[var].dtype, np.floating):
                pass
            else:
                logging.info(f"Convert {var} to float32")
                ds[var] = ds[var].astype(np.float32)

        logging.info(f"Attempt to store dataset to: {output_path}")
        ds.to_netcdf(output_path)
        logging.info("Successfully stored dataset")

        logging.info(f"Plot the PSD and MSD for cloud_id: {cloud_id}")

        fig_dir = data_dir / "figures"

        psd_params = ds_psd_parameters.sel(cloud_id=cloud_id)
        psd_params_dict = parameters_dataset_to_dict(psd_params, mapping)

        radii = 1e-6 * ds["radius_bins"]
        bin_width = 0.5 * ((radii.shift(radius_bins=-1) - radii.shift(radius_bins=1)))
        bin_width = bin_width.fillna(0)

        ds_psd = DoubleLogNormal(**psd_params_dict)(radii=radii) * bin_width

        # r = ~np.isnan(ds_psd)
        observation_psd = ds_psd
        cleo_psd = (ds["xi"] / ds["gridbox_volume"]).isel(gridbox=-1).mean("time")  # .where(r)

        observation_msd = msd_from_psd_dataarray(
            da=observation_psd, radius_name="radius_bins", radius_scale_factor=1e-6
        )
        cleo_msd = msd_from_psd_dataarray(
            da=cleo_psd, radius_name="radius_bins", radius_scale_factor=1e-6
        )

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        for i, (tt, cc) in enumerate(zip([observation_psd, observation_msd], [cleo_psd, cleo_msd])):
            axs[i].plot(
                1e-3 * ds["radius_bins"],
                tt,
                color="blue",
                label="Observational fit",
            )
            axs[i].plot(
                1e-3 * ds["radius_bins"],
                cc,
                color="red",
                label="CLEO data",
            )
            axs[i].plot(
                1e-3 * ds["radius_bins"],
                cc - tt,
                color="k",
                label="Difference CLEO - Observation",
            )

            axs[i].set_title(
                f"Below are the sums over all radii for\nObservation fit,  CLEO data,  Differences\n{np.nansum(tt):.2e},  {np.nansum(cc):.2e},  {np.nansum(cc - tt):.2e}"
            )
            axs[i].set_xscale("log")
            axs[i].set_yscale("log")
            axs[i].legend()

        axs[0].set_xlabel(r"Radius [$mm$]")
        axs[1].set_xlabel(r"Radius [$mm$]")
        axs[0].set_ylabel(r"Number concentration [$m^{-3}$]")
        axs[1].set_ylabel(r"Mass concentration [$kg m^{-3}$]")

        fig.suptitle(f"Cloud ID: {cloud_id}")
        fig.tight_layout()

        fig.savefig(fig_dir / f"comparison_psd_msd_cluster_{cloud_id}.png")

        sucessful.append(cloud_id)

    except Exception as e:
        logging.exception(e)
        continue


logging.info("Collecting sucessful cloud_ids from all processes")

# Gather the lists from all ranks
all_sucessful = comm.gather(sucessful, root=0)

if rank == 0:
    # Combine the lists from all ranks
    combined_sucessful = np.concatenate(all_sucessful)
    number_sucessful = len(combined_sucessful)
    number_total = len(data_dir_list)
    logging.info(f"All processes finished with {number_sucessful}/{number_total} sucessful")
    logging.info(f"Sucessful clouds are: {combined_sucessful}")
