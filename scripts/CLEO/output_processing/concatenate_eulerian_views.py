# %%

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sdm_eurec4a import RepositoryPath
import datetime

RP = RepositoryPath("levante")
repo_dir = RP.repo_dir


# add domain masks
def add_domain_masks(ds: xr.Dataset) -> None:
    # create domain mask and sub cloud layer mask
    ds["domain_mask"] = ds["gridbox"] <= ds["max_gridbox"]
    ds["domain_mask"].attrs.update(
        long_name="Domain Mask",
        description="Boolean mask indicating valid gridbox in the domain for each cloud",
        units="1",
    )

    ds["cloud_layer_mask"] = ds["gridbox"] == ds["max_gridbox"]
    ds["cloud_layer_mask"].attrs.update(
        long_name="Cloud Layer Mask",
        description="Boolean mask indicating if the gridbox is part of the cloud layer",
        units="1",
    )

    ds["sub_cloud_layer_mask"] = ds["gridbox"] < ds["max_gridbox"]
    ds["sub_cloud_layer_mask"].attrs.update(
        long_name="Sub Cloud Layer Mask",
        description="Boolean mask indicating if the gridbox is part of the sub cloud layer",
        units="1",
    )


# === mpi4py ===
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # [0,1,2,3,4,5,6,7,8,9]
    number_ranks = comm.Get_size()  # 10
except:
    print("::: Warning: Proceeding without mpi4py! :::")
    rank = 0
    number_ranks = 1

# === logging ===
# create log file

time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

log_file_dir = repo_dir / "logs" / f"concatenate_eulerian_views/{time_str}"
log_file_dir.mkdir(exist_ok=True, parents=True)
log_file_path = log_file_dir / f"{rank}.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(log_file_path)
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


# %%

logging.info(f"====================")
logging.info(f"Start with rank {rank} of {number_ranks}")

# path2CLEO = Path("/home/m/m301096/CLEO")
# path2CLEO.is_dir()

# microphysics = "condensation"
# master_data_dir = path2CLEO / f"data/output_v4.1/{microphysics}"

parser = argparse.ArgumentParser(
    description="Create eulerian view for data_dir which contains all subfolders of cloud data it should contain the subfolder cluster_*/processed/eulerian_dataset.nc"
)

# Add arguments
parser.add_argument("-d", "--data_dir", type=str, help="Path to data directory", required=True)
# Parse arguments
args = parser.parse_args()

master_data_dir = Path(args.data_dir)


output_dir = master_data_dir / f"combined"
output_dir.mkdir(exist_ok=True, parents=False)
output_file_path = output_dir / f"eulerian_dataset_combined.nc"

relative_path_to_eulerian_dataset = Path("processed/eulerian_dataset.nc")
pattern = "cluster_*/"

logging.info(f"Enviroment: {sys.prefix}")
logging.info("Concatenate eulerian views in:")
logging.info(master_data_dir)
logging.info(f"Subfolder pattern: {pattern}")
logging.info(f"Relative path to eulerian dataset: {relative_path_to_eulerian_dataset}")
logging.info(f"Save the combined eulerian view to: {output_file_path}")

data_dir_list = np.array(sorted(list(master_data_dir.glob(pattern))))
eulerian_dataset_path_list = data_dir_list / relative_path_to_eulerian_dataset

sublist_data_dirs = np.array_split(np.array(data_dir_list), number_ranks)[rank]
sublist_eulerian_dataset_paths = np.array_split(np.array(eulerian_dataset_path_list), number_ranks)[rank]
total_npro = len(sublist_data_dirs)

sucessful = []

max_gridbox_list = []
file_path_list = []
cloud_id_list = []

for step, (data_dir, eulerian_dataset_path) in enumerate(
    zip(
        sublist_data_dirs,
        sublist_eulerian_dataset_paths,
    )
):
    logging.info(f"Rank {rank+1} {step+1}/{total_npro}")
    cloud_id = int(data_dir.name.split("_")[1])
    logging.info(f"Start {cloud_id}")
    try:
        logging.info(f"Open {eulerian_dataset_path}")
        euler_dataset = xr.open_dataset(eulerian_dataset_path)

        logging.info(f"Find max gridbox")
        max_gridbox_list.append(euler_dataset["gridbox"].max())

        logging.info(f"Add cloud to combination list")
        file_path_list.append(eulerian_dataset_path)
        cloud_id_list.append(cloud_id)

        logging.info(f"Processed {cloud_id}")

    except FileNotFoundError as e:
        logger.error(e)

temporary_dir = output_dir / f"temporary"
temporary_dir.mkdir(exist_ok=True, parents=False)
temporary_file_path = temporary_dir / f"temporary{rank}.nc"

number_sucessful = len(cloud_id_list)
number_total = len(sublist_data_dirs)

# %%
if number_ranks == 1:

    logging.info(f"All processes finished with {number_sucessful}/{number_total} sucessful")
    logging.info(f"Sucessful clouds are: {list(cloud_id_list)}")

    logging.info("Attempt to open all sucessful clouds and combine them with xr.open_mfdataset")

    cloud_id_index = pd.Index(cloud_id_list, name="cloud_id")
    ds = xr.open_mfdataset(file_path_list, combine="nested", concat_dim=[cloud_id_index], parallel=True)

    logging.info("Add cloud_id and max_gridbox to the dataset")
    ds["cloud_id"].attrs.update(dict(long_name="Cloud identification number", units=""))
    max_gridbox = xr.concat(max_gridbox_list, dim=cloud_id_index)
    ds["max_gridbox"] = max_gridbox
    ds["max_gridbox"].attrs.update(
        dict(
            long_name="Maximum gridbox value for each cloud. Above this value, the cloud has no data.",
            units="",
        )
    )

    ds = ds.sortby("cloud_id")

    ds.attrs.update(
        dict(
            description="Combined eulerian view for all clouds.",
            author="Nils Niebaum",
            date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )
    )

    logging.info("Add domain masks")
    add_domain_masks(ds)

    logging.info(f"Attempt to save combined dataset to: {output_file_path}")
    ds.to_netcdf(output_file_path)
    logging.info(f"Closing dataset")
    logging.info(f"Done")

else:

    logging.info("Attempt to open all sucessful clouds and combine them with xr.open_mfdataset")

    cloud_id_index = pd.Index(cloud_id_list, name="cloud_id")
    ds = xr.open_mfdataset(file_path_list, combine="nested", concat_dim=[cloud_id_index], parallel=True)

    logging.info("Add cloud_id and max_gridbox to the dataset")
    ds["cloud_id"].attrs.update(dict(long_name="Cloud identification number", units=""))
    max_gridbox = xr.concat(max_gridbox_list, dim=cloud_id_index)
    ds["max_gridbox"] = max_gridbox
    ds["max_gridbox"].attrs.update(
        dict(
            long_name="Maximum gridbox value for each cloud. Above this value, the cloud has no data.",
            units="",
        )
    )

    ds = ds.sortby("cloud_id")

    ds.attrs.update(
        dict(
            description="Combined eulerian view for all clouds.",
            author="Nils Niebaum",
            date=str(pd.Timestamp.now()),
        )
    )

    logging.info("Add domain masks")
    add_domain_masks(ds)

    logging.info(f"Attempt to save combined dataset to: {temporary_file_path}")
    ds.to_netcdf(temporary_file_path)
    logging.info(f"Closing dataset")
    logging.info(f"Done")

    # combine the results from all ranks
    logging.info("Combine all netcdf files from all ranks")

    all_temporaray_file_path_list = comm.gather(temporary_file_path, root=0)
    all_cloud_id_list = comm.gather(cloud_id_list, root=0)
    all_max_gridbox_list = comm.gather(max_gridbox_list, root=0)
    logging.info(f"{all_cloud_id_list}")

    if rank != 0:
        logging.info(f"Finished with rank {rank}")

    else:

        logging.info("====================================")
        logging.info("Rank 0 collects the results and creates the combined dataset")
        # Combine the lists from all ranks
        all_cloud_id = np.concatenate(all_cloud_id_list)
        all_max_gridbox = np.concatenate(all_max_gridbox_list)
        all_temporary_file_path = all_temporaray_file_path_list

        number_sucessful = len(all_cloud_id)
        number_total = len(data_dir_list)

        logging.info(f"All processes finished with {number_sucessful}/{number_total} sucessful")
        logging.info(f"Sucessful clouds are: {list(all_cloud_id)}")

        logging.info("Attempt to open all temporary datasets")
        ds = xr.open_mfdataset(
            all_temporary_file_path, combine="nested", chunks={"cloud_id": 1}, parallel=True
        )
        ds.sortby("cloud_id")

        logging.info(f"Attempt to save combined dataset to: {output_file_path}")
        ds.to_netcdf(output_file_path)

        logging.info(f"Saved combined dataset to: {output_file_path}")

        # remove temporary files
        logging.info("Close dataset")

        logging.info("Remove temporary files")
        for temporary_file_path in all_temporary_file_path:
            temporary_file_path.unlink()
        logging.info("Finished complete process")
