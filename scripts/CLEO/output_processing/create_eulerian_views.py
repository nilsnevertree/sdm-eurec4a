# %%
import argparse
from pathlib import Path
# Create argument parser
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook() == False:
    parser = argparse.ArgumentParser(description='Concatenate eulerian views')

    # Add arguments
    parser.add_argument(
        '-d', '--data_dir', 
        type=str, 
        help='Path to data directory', 
        required=True
    )
    # Parse arguments
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

import os
import sys


import numpy as np
import awkward as ak 
import xarray as xr

from pySD.sdmout_src import pysetuptxt
from pySD.sdmout_src import supersdata
from pathlib import Path
from typing import Union

print(f"Enviroment: {sys.prefix}")
print("Create eulerian view for all subdirectories in:")
print(data_dir)

# %%
def ak_differentiate(sa : supersdata.SupersAttribute) -> supersdata.SupersAttribute:
    """
    This function calculates the difference of the data in the supersdata.SupersAttribute along the last axis.
    The difference is calculated as the difference of the next value minus the current value.
    The last value is set to nan, to make sure, that the mass change is at the same timestep, as the original value.

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

    # It is very important, to concate the nan values at the END of the array, so that the last value is nan.
    # This makes sure, that the mass change is at the same timestep, as the original value.
    # With this, the evapoartion fraction can not exceed 1.
    data = ak.concatenate([data, np.nan], axis = -1)

    # if the data has entries, which have only one value, append another nan value
    if ak.min(ak.num(data, axis = -1)) < 2:
        data = ak.concatenate([data, np.nan], axis = -1)
        
    # calculate the difference
    diff = data[..., 1:] - data[..., :-1]
    
    # create a new attribute
    result = supersdata.SupersAttribute(
        name = sa.name + "_difference", 
        data = diff,
        units = sa.units,
        metadata = sa.metadata.copy(),
    )

    # update metadata
    updated_metadata = sa.metadata.copy()
    try : 
        updated_metadata["long_name"] = updated_metadata["long_name"] + " difference"
    except KeyError:
        pass        
    result.set_metadata(
        metadata= updated_metadata
    )

    return result

def create_lagrangian_dataset(
    dataset : supersdata.SupersDataNew
) -> supersdata.SupersDataNew :
    """
    This function creates a lagrangian view of the SupersDataset.
    Within this setup, the following variables are calculated and added to the dataset:
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
    try :
        dataset.set_attribute(
            dataset["time"].attribute_to_indexer_unique()
        )
    except KeyError:
        pass
    try:
        dataset.set_attribute(
            dataset["sdId"].attribute_to_indexer_unique()
        )
    except KeyError:
        pass

    # ============
    # 2. Create the Lagrangian view to calculate the mass change
    # ============

    # bin by the superdroplet id and calcuate the difference of the mass
    dataset.index_by_indexer(index = dataset["sdId"])

    

    # calculate the difference of the mass as the total mass change per timestep
    mass_rep_diff = ak_differentiate(dataset["mass_represented"])
    mass_rep_diff.set_metadata(
        metadata = {
            "long_name" : "Mass difference per timestep",
            "notes" : r"Mass here is mass represented by a superdroplet: $m = \xi \cdot m_{sd}$"
        }
    )
    mass_rep_diff.set_name("mass_difference_timestep")
    dataset.set_attribute(mass_rep_diff)

    time_diff = ak_differentiate(dataset["time"])
    time_diff.set_metadata(
        metadata = {
            "long_name" : "Time difference per timestep",
        }
    )
    time_diff.set_name("time_difference")
    time_diff.set_units("s")

    # calculate the difference of the mass as the total mass change per second
    mass_diff = mass_rep_diff / time_diff
    mass_diff.set_metadata(
        metadata = {
            "long_name" : "Mass difference",
            "notes" : r"Mass here is mass represented by a superdroplet: $m = \xi \cdot m_{sd}$"
        }
    )
    mass_diff.set_name("mass_difference")
    dataset.set_attribute(mass_diff)

    # calculate the difference of the multiplicity per second
    xi_diff = ak_differentiate(dataset["xi"]) / time_diff
    xi_diff.set_metadata(
        metadata = {
            "long_name" : "Multiplicity difference per second",
        }
    )
    dataset.set_attribute(xi_diff)

    # calculate the evaporated fraction per second
    evaporated_fraction = dataset["mass_difference"] / dataset["mass_represented"] * -100
    evaporated_fraction.set_name("evaporated_fraction")
    evaporated_fraction.set_metadata(
        metadata = {
            "long_name" : "evaporated fraction per second",
            "notes" : r"Evaporated fraction is calculated as $\frac{\Delta m}{m} \cdot 100$"
        }
    )
    evaporated_fraction.set_units("%")
    dataset.set_attribute(evaporated_fraction)


    return dataset



def create_eulerian_dataset(
    dataset : supersdata.SupersDataNew,
    radius_bins : np.ndarray = np.logspace(-7, 7, 150)
) -> supersdata.SupersDataNew :
    """
    This function creates a eulerian view of the SupersDataset.
    First, a lagrangian view is created by binning the data by the superdroplet id.
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
    try :
        dataset.set_attribute(
            dataset["time"].attribute_to_indexer_unique()
        )
    except KeyError:
        pass
    try:
        dataset.set_attribute(
            dataset["sdId"].attribute_to_indexer_unique()
        )
    except KeyError:
        pass

    try:
        # make time an indexer which correspondataset to the unique values of the time attribute
        dataset.set_attribute(
            dataset["sdgbxindex"].attribute_to_indexer_unique(new_name = "gridbox")
        )
    except KeyError:
        pass
    try : 
        dataset.set_attribute(
            dataset["radius"].attribute_to_indexer_binned(
                bins= radius_bins,
                new_name= "radius_bins")
        )
    except KeyError:
            pass

    # ============
    # 3. Create eulerian view
    # ============

    dataset.index_by_indexer(index = dataset["time"])
    dataset.index_by_indexer(index = dataset["gridbox"])
    dataset.index_by_indexer(index = dataset["radius_bins"])

    # ============
    # 4. Rename mass_represented to mass and mass to mass_individual
    # this helps to be consiten with the naming of the mass in a eulerian view
    # ============

    # create an attribute which counts the number of superdroplets
    counts = dataset.get_data("xi")
    counts = counts * 0 + 1
    number_superdroplets = supersdata.SupersAttribute(
        name = "number_superdroplets",
        data = counts,
        units = "#",
        metadata = {
            "long_name" : "number of superdroplets",
        }
    )
    dataset.set_attribute(number_superdroplets)

    return dataset

def create_eulerian_xr_dataset(
        dataset : supersdata.SupersDataNew,
        radius_bins : np.ndarray = np.logspace(-7, 7, 150),
        output_path : Union[None, Path] = None,
        hand_out : bool = True,
        ) -> xr.Dataset:
    """
    This function creates a eulerian view of the SupersDataset and transforms it to a xarray dataset.
    For this, the function create_eulerian_dataset is used to create the eulerian view.

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
            reduction_func = ak.sum,
            add_metadata = {
                "reduction_func" : "ak.sum Summation over all SDs in the bin"
            }
        )
    mean_reduction = dict(
            reduction_func = ak.mean,
            add_metadata = {
                "reduction_func" : "ak.mean Mean over all SDs in the bin"
            }
        )

    reduction_map = {
        "mass_difference" : sum_reduction,
        "mass_difference_timestep":  sum_reduction,
        "mass_represented" : sum_reduction,
        "radius" : mean_reduction,
        "evaporated_fraction": mean_reduction,
        "xi": sum_reduction,
        "number_superdroplets" : sum_reduction,
    }

    # create individual DataArrays
    da_list = []
    for varname in reduction_map:
        reduction_func = reduction_map[varname]["reduction_func"]
        add_metadata = reduction_map[varname]["add_metadata"]
        da = eulerian.attribute_to_DataArray_reduction(
                attribute_name = varname,
                reduction_func = reduction_func,
            )
        da.attrs.update(add_metadata)
        da_list.append(da)

    # create the dataset by merging the DataArrays
    ds = xr.merge(da_list)



    ds["time"].attrs["long_name"] = "time"
    ds["time"].attrs["units"] = "s"

    if output_path is not None:    
        # Save the dataset
        ds.to_netcdf(
            output_path
        )
    if hand_out is True:
        return ds


subdirectories = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) and name.startswith("clusters_")]


for sub_dir_name in subdirectories: 
    print(f"Processing {sub_dir_name}")
    sub_dir = data_dir / Path(sub_dir_name)

    eulerian_dataset_path  = sub_dir / "processed/"
    eulerian_dataset_path.mkdir(exist_ok=True)

    # basepath = Path("/home/m/m301096/CLEO/data/newoutput/stationary_no_physics/clusters_18/")

    setupfile = sub_dir / "config" / "eurec4a1d_setup.txt"
    zarr_dataset = sub_dir / "eurec4a1d_sol.zarr"

    try : 
        # read in constants and intial setup from setup .txt file
        config = pysetuptxt.get_config(setupfile, nattrs=3, isprint=False)
        consts = pysetuptxt.get_consts(setupfile, isprint=False)
    
        try : 
            dataset = supersdata.SupersDataNew(dataset = str(zarr_dataset), consts=consts)

            try : 
                # create the eulerian dataset
                create_eulerian_xr_dataset(
                    dataset = dataset,
                    radius_bins = np.logspace(-7, 7, 150),
                    output_path = eulerian_dataset_path / "eulerian_dataset.nc",
                    hand_out = False
                )

            except Exception:
                print(f"Error in creating eulerian dataset for {zarr_dataset}")
                continue
        except Exception:
            print(f"Error in reading {zarr_dataset}")
            continue
    
    # initialize the superdroplets data class
    except Exception:
        print(f"Error in reading {setupfile}")
    

