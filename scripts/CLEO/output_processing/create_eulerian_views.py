# %%
import argparse

from pathlib import Path


# Create argument parser
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


import os
import sys

from pathlib import Path
from typing import Union

import awkward as ak
import numpy as np
import xarray as xr

from pySD.sdmout_src import pygbxsdat, pysetuptxt, supersdata


# parser = argparse.ArgumentParser(
#     description="Create eulerian view for data_dir which contains zarr dir in ./eurec4a1d_sol.zarr and config files in ./config/eurec4a1d_setup.txt"
# )

# # Add arguments
# parser.add_argument("-d", "--data_dir", type=str, help="Path to data directory", required=True)
# # Parse arguments
# args = parser.parse_args()

# data_dir = Path(args.data_dir)
data_dir = Path("/home/m/m301096/CLEO/data/output_v3.5/condensation/clusters_222/")

print(f"Enviroment: {sys.prefix}")
print("Create eulerian view in:")
print(data_dir)


# %%
def ak_differentiate(sa: supersdata.SupersAttribute) -> supersdata.SupersAttribute:
    """
    This function calculates the difference of the data in the
    supersdata.SupersAttribute along the last axis. The difference is
    calculated as the difference of the next value minus the current value. The
    last value is set to nan, to make sure, that the mass change is at the same
    timestep, as the original value.

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

    data = sa.get_data()

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
    This function only keeps the last value along axis 1. The rest will be
    replaced by nans.

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
    This function creates a lagrangian view of the SupersDataset. Within this
    setup, the following variables are calculated and added to the dataset:

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

    time_diff = ak_differentiate(dataset["time"])
    time_diff.set_metadata(
        metadata={
            "long_name": "Time difference per timestep",
        }
    )
    time_diff.set_name("time_difference")
    time_diff.set_units("s")

    # calculate the difference of the mass as the total mass change per second
    mass_rep_diff_time = mass_rep_diff / time_diff
    mass_rep_diff_time.set_metadata(
        metadata={
            "long_name": "Mass difference",
            "notes": r"Mass here is mass represented by a superdroplet: $m = \xi \cdot m_{sd}$",
        }
    )
    mass_rep_diff_time.set_name("mass_difference")
    dataset.set_attribute(mass_rep_diff_time)

    # calculate the difference of the multiplicity per second
    xi_diff = ak_differentiate(dataset["xi"]) / time_diff
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
    This function creates a eulerian view of the SupersDataset. First, a
    lagrangian view is created by binning the data by the superdroplet id.
    Within this setup, the following variables are calculated:

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
    output_path: Union[None, Path] = None,
    hand_out: bool = True,
) -> xr.Dataset:
    """
    This function creates a eulerian view of the SupersDataset and transforms
    it to a xarray dataset. For this, the function create_eulerian_dataset is
    used to create the eulerian view.

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
    output_path : Union[None, Path], optional
        The path where the dataset should be saved, by default None
        If None, the dataset is not saved.
    hand_out : bool, optional
        If True, the dataset is returned, by default True

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
        "mass_difference": sum_reduction,
        "mass_difference_timestep": sum_reduction,
        "mass_represented": sum_reduction,
        "radius": mean_reduction,
        "evaporated_fraction": mean_reduction,
        "xi": sum_reduction,
        "number_superdroplets": sum_reduction,
        "mass_left": sum_reduction,
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

    if output_path is not None:
        # Save the dataset
        ds.to_netcdf(output_path)
    if hand_out is True:
        return ds


# %%
eulerian_dataset_path = data_dir / "processed/"
eulerian_dataset_path.mkdir(exist_ok=True)

setupfile = data_dir / "config" / "eurec4a1d_setup.txt"
statsfile = data_dir / "config" / "eurec4a1d_stats.txt"
zarr_dataset = data_dir / "eurec4a1d_sol.zarr"
gridfile = data_dir / "share/eurec4a1d_ddimlessGBxboundaries.dat"

ds_zarr = xr.open_zarr(zarr_dataset, consolidated=False)

# read in constants and intial setup from setup .txt file
config = pysetuptxt.get_config(str(setupfile), nattrs=3, isprint=False)
consts = pysetuptxt.get_consts(str(setupfile), isprint=False)
gridbox_dict = pygbxsdat.get_gridboxes(str(gridfile), consts["COORD0"], isprint=False)

# %%
dataset = supersdata.SupersDataNew(dataset=str(zarr_dataset), consts=consts)
# create the eulerian dataset
ds = create_eulerian_xr_dataset(
    dataset=dataset,
    radius_bins=np.logspace(-7, 7, 150),
    output_path=None,
    hand_out=True,
)

ds_zarr = ds_zarr.rename({"gbxindex": "gridbox"})

print("Eulerian dataset created")

# %%
from sdm_eurec4a.conversions import relative_humidity_from_tps


# Add thermodynamic data to the dataset


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
    ds["vertical_liquid_water_content"] = (
        ds["liquid_water_content"].sel(time=time_slice).sum("radius_bins").mean("time")
    )

    ds["vertical_liquid_water_content"].attrs.update(
        long_name="Vertical Liquid Water Content",
        description=f"Vertical Profile of LWC. Sum over all radius bins and mean over time of the stationary state ({time_slice.start}-{time_slice.stop}).",
        units=ds["liquid_water_content"].attrs["units"],
    )

    ds["mass_difference_per_volume"] = ds["mass_difference_timestep"] / ds["gridbox_volume"]
    ds["mass_difference_per_volume"].attrs.update(
        long_name="Mass Difference per volume",
        description="Mass Difference per Volume. The mass difference is divided by the gridbox volume.",
        units=r"$kg m^{-3} s^{-1}$",
    )

    ds["vertical_mass_difference_per_volume"] = (
        ds["mass_difference_per_volume"].sel(time=time_slice).sum("radius_bins").mean("time")
    )
    ds["vertical_mass_difference_per_volume"].attrs.update(
        long_name="Vertical mass difference per voluem",
        description=f"Vertical Profile of mass difference per voluem. Sum over all radius bins and mean over time of the stationary state ({time_slice.start}-{time_slice.stop}).",
        units=r"$kg m^{-3} s^{-1}$",
    )


def add_precipitation(ds):
    """
    Uses the mass_left variable in the dataset to calculate the precipitation
    rate.

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


add_thermodynamics(ds, ds_zarr)
add_gridbox_properties(ds, gridbox_dict)
add_liquid_water_content(ds)
add_vertical_profiles(ds)
add_precipitation(ds)

# %%
