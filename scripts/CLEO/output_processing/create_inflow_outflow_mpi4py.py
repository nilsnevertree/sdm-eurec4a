# %%
import argparse

from pathlib import Path
import sys

from pathlib import Path
from typing import Union

import awkward as ak
import numpy as np
import xarray as xr
import logging
from mpi4py import MPI
import datetime

from sdm_eurec4a import RepositoryPath

from pySD.sdmout_src import pygbxsdat, pysetuptxt, supersdata

from typing import Tuple

from sdm_eurec4a.visulization import set_custom_rcParams


from sdm_eurec4a import RepositoryPath, get_git_revision_hash
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
    log_dir = repo_dir / "logs" / f"create_inflow_outflow/{time_str}-{random_hex}"
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
    description="Create inflow outflow dataset for data_dir which contains zarr dir in ./eurec4a1d_sol.zarr and config files in ./config/eurec4a1d_setup.txt"
)
# %%
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


def create_inflow_outflow_reservoir_dataset(
    dataset: supersdata.SupersDataNew,
    dim0_name: str = "time",
    dim1_name: str = "sdgbxindex",
    attribute_names: Union[Tuple[str], None, str] = None,
) -> Tuple[
    supersdata.SupersDataSimple,
    supersdata.SupersDataSimple,
    supersdata.SupersDataSimple,
]:

    # use only the Superdroplets, which are in more than one timestep!
    # For this, the ak.num > 1
    # An example would be this array:
    # [
    #      [0,1,2,3],   -> usable
    #      [0,1],       -> usable
    #      [0],         -> UNUSABLE
    #      [3, 4, 5],   -> usable
    #  ]
    data = dataset[dim0_name].data
    mask = ak.num(data, axis=-1) > 1

    logging.info(f"Number of usable Superdroplets: {ak.sum(mask)}")

    # create the empty dataset for the inflow, outflow and reservoir
    dataset_inflow = supersdata.SupersDataSimple([])
    dataset_outflow = supersdata.SupersDataSimple([])
    dataset_reservoir = supersdata.SupersDataSimple([])

    # if no attribute names are given, compute all attributes
    if isinstance(attribute_names, (str,)):
        attribute_names = (attribute_names,)

    if attribute_names is None:
        attribute_names = tuple(dataset.attributes.keys())

    # iterate over all attributes and create the inflow, outflow and reservoir
    # also iterate over the dimensions to have them avaiable for the indexing
    for key in set(attribute_names + (dim0_name, dim1_name)):
        logging.info(f"Processing variable {key}")
        attribute = dataset[key]
        data = attribute.data
        data = data[mask]

        # The inflow is the second value of the array, because the first is the initialisation!
        # The first value of the array would be in the cloud gridbox
        inflow_array = data[:, 1]
        # The outflow of the data is the last value along the SD-Id dimension
        outflow_array = data[:, -1]
        # The reservoir is the data without the first and last value of the dataset
        reservoir_data = data[:, 1:-1]
        reservoir_data = ak.flatten(reservoir_data, axis=-1)

        dataset_inflow.set_attribute(
            supersdata.SupersAttribute(
                name=key, data=inflow_array, units=attribute.units, metadata=attribute.metadata
            )
        )
        dataset_outflow.set_attribute(
            supersdata.SupersAttribute(
                name=key, data=outflow_array, units=attribute.units, metadata=attribute.metadata
            )
        )
        dataset_reservoir.set_attribute(
            supersdata.SupersAttribute(
                name=key, data=reservoir_data, units=attribute.units, metadata=attribute.metadata
            )
        )

    logging.info(f"Indexing the datasets")
    dataset_inflow.set_attribute(dataset_inflow[dim0_name].attribute_to_indexer_unique())
    dataset_inflow.index_by_indexer(dataset_inflow[dim0_name])

    dataset_outflow.set_attribute(dataset_outflow[dim0_name].attribute_to_indexer_unique())
    dataset_outflow.index_by_indexer(dataset_outflow[dim0_name])

    dataset_reservoir.set_attribute(dataset_reservoir[dim0_name].attribute_to_indexer_unique())
    dataset_reservoir.set_attribute(dataset_reservoir[dim1_name].attribute_to_indexer_unique())
    dataset_reservoir.index_by_indexer(dataset_reservoir[dim0_name])
    dataset_reservoir.index_by_indexer(dataset_reservoir[dim1_name])

    return dataset_inflow, dataset_outflow, dataset_reservoir


def create_inflow_outflow_reservoir_xr_dataset(
    dataset: supersdata.SupersDataNew,
    dim0_name: str = "time",
    dim1_name: str = "sdgbxindex",
    attribute_name: Union[Tuple[str], str, None] = "mass_represented",
) -> xr.Dataset:

    dataset_inflow, dataset_outflow, dataset_reservoir = create_inflow_outflow_reservoir_dataset(
        dataset=dataset,
        dim0_name=dim0_name,
        dim1_name=dim1_name,
        attribute_names=attribute_name,
    )

    # create the xarray dataset
    dataset_inflow = dataset_inflow.attribute_to_DataArray_reduction(
        "mass_represented", reduction_func=ak.nansum
    )
    dataset_outflow = dataset_outflow.attribute_to_DataArray_reduction(
        "mass_represented", reduction_func=ak.nansum
    )
    dataset_reservoir = dataset_reservoir.attribute_to_DataArray_reduction(
        "mass_represented", reduction_func=ak.nansum
    )

    # outflow should be negative
    dataset_outflow = -dataset_outflow
    dataset_reservoir = dataset_reservoir

    ds_box_model = xr.Dataset(
        {
            "inflow": dataset_inflow,
            "outflow": dataset_outflow,
            "reservoir": dataset_reservoir,
        }
    )

    # !!!!!!!!!!!!
    # The data is now given in kg per timestep, which we keep it!

    ds_box_model = ds_box_model.rename({"sdgbxindex": "gridbox"})
    ds_box_model = ds_box_model.fillna(0)
    attrs = {key: ds_box_model[key].attrs.copy() for key in ds_box_model.data_vars}

    ds_box_model = ds_box_model

    ds_box_model["reservoir"] = ds_box_model["reservoir"].sum("gridbox")

    for key in ds_box_model.data_vars:
        ds_box_model[key].attrs = attrs[key]

    ds_box_model["inflow"].attrs = dict(
        long_name="Inflow",
        description="Inflow of mass into the domain. Given in total mass per timestep.",
        units="kg dT^{-1}",
    )
    ds_box_model["outflow"].attrs = dict(
        long_name="Outflow",
        description="Outflow of mass out of the domain. Given in total mass per timestep.",
        units="kg dT^{-1}",
    )
    ds_box_model["reservoir"].attrs = dict(
        long_name="Reservoir",
        description="Reservoir of mass in the domain. Given in total mass inside the domain.",
        units="kg",
    )

    # for the first timestep, the reservoir is equal to the inflow
    # ds_box_model['reservoir'][0] = ds_box_model['inflow'][0]

    # ds_box_model['inflow_integrate'] = ds_box_model['inflow'].cumsum('time', keep_attrs=True)
    # ds_box_model['inflow_integrate'] = ds_box_model['inflow_integrate'].shift(time = 0)
    # ds_box_model['inflow_integrate'].attrs['units'] = 'kg'

    # ds_box_model['outflow_integrate'] = ds_box_model['outflow'].cumsum('time', keep_attrs=True)
    # ds_box_model['outflow_integrate'] = ds_box_model['outflow_integrate'].shift(time = 0)
    # ds_box_model['outflow_integrate'].attrs['units'] = 'kg'

    ds_box_model["reservoir_change"] = ds_box_model["reservoir"].diff("time")
    ds_box_model["reservoir_change"] = ds_box_model["reservoir_change"].shift(time=0)
    ds_box_model["reservoir_change"].attrs = dict(
        long_name="Reservoir change",
        description="Change of the reservoir mass in the domain per timestep dT.",
        units="kg dT^{-1}",
    )

    # the first change in the reservoir is the first timestep - 0
    ds_box_model["reservoir_change"][0] = ds_box_model["reservoir"][0] - 0
    return ds_box_model


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

        output_path = output_dir / "conservation_dataset.nc"
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
        dataset = supersdata.SupersDataNew(
            dataset=ds_zarr,
            consts=consts,
        )
        dataset.set_attribute(dataset["sdId"].attribute_to_indexer_unique())
        dataset.index_by_indexer(dataset["sdId"])

        ds = create_inflow_outflow_reservoir_xr_dataset(
            dataset=dataset,
            dim0_name="time",
            dim1_name="sdgbxindex",
            attribute_name="mass_represented",
        )

        # add the monitor massdelta condensation
        # massdelta_cond is given in g, so we need to convert it to kg with 1e3
        # NOTE: if you use an old version of CLEO, a factor of 1e18 might be missing.
        # This is resolved at least in v0.30.1

        ds["source"] = 1e-3 * ds_zarr["massdelta_cond"].sel(
            gridbox=slice(0, ds_zarr["gridbox"].max() - 1)
        ).sum("gridbox").shift(time=0)

        # TODO: due to the issue of every half timestep, we apply a rolling mean over the source terms
        ds["source"] = ds["source"].rolling(time=2).mean()

        ds["source"].attrs = dict(
            long_name="Source term",
            description="Source term of mass in the domain. Given in total mass per timestep. It is the condensation of water vapor.",
            units="kg dT^{-1}",
        )

        logging.info(f"Make sure to have float precission for all variables to be able to include NaNs")

        for var in ds:
            if np.issubdtype(ds[var].dtype, np.floating):
                pass
            else:
                logging.info(f"Convert {var} to float32")
                ds[var] = ds[var].astype(np.float32)

        logging.info("Remove gridbox coordinate")
        ds = ds.drop_vars(names=("gridbox",))

        logging.info(f"Attempt to store dataset to: {output_path}")
        ds.attrs.update(
            author="Nils Niebaum",
            date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S"),
            description="Eulerian dataset created from raw data",
            git_hash=get_git_revision_hash(),
        )

        ds.to_netcdf(output_path)
        logging.info("Successfully stored dataset")
        sucessful.append(cloud_id)

    except Exception as e:
        logging.exception(e)
        continue

# %%

logging.info("Collecting sucessful cloud_ids from all processes")

# Gather the lists from all ranks
all_sucessful = comm.gather(sucessful, root=0)
import pandas as pd

if rank == 0:
    # Combine the lists from all ranks
    combined_sucessful = list(pd.core.common.flatten(all_sucessful))
    number_sucessful = len(combined_sucessful)
    number_total = len(data_dir_list)
    logging.info(f"All processes finished with {number_sucessful}/{number_total} sucessful")
    logging.info(f"Sucessful clouds are: {combined_sucessful}")
