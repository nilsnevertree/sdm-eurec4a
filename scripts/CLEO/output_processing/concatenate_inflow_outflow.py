# %%

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sdm_eurec4a import RepositoryPath, get_git_revision_hash
import datetime

RP = RepositoryPath("levante")
repo_dir = RP.repo_dir

# === logging ===
# create log file

time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

log_file_dir = repo_dir / "logs" / f"concatenate_inflow_outflow/{time_str}"
log_file_dir.mkdir(exist_ok=True, parents=True)
log_file_path = log_file_dir / f"main.log"

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

# path2CLEO = Path("/home/m/m301096/CLEO")
# path2CLEO.is_dir()

# microphysics = "condensation"
# master_data_dir = path2CLEO / f"data/output_v4.1/{microphysics}"

parser = argparse.ArgumentParser(
    description="Create eulerian view for data_dir which contains all subfolders of cloud data it should contain the subfolder cluster_*/processed/conservation_dataset.nc"
)

# Add arguments
parser.add_argument("-d", "--data_dir", type=str, help="Path to data directory", required=True)
# Parse arguments
args = parser.parse_args()

master_data_dir = Path(args.data_dir)


output_dir = master_data_dir / f"combined"
output_dir.mkdir(exist_ok=True, parents=False)
output_file_path = output_dir / f"conservation_dataset_combined.nc"

relative_path_to_conservation_dataset = Path("processed/conservation_dataset.nc")
pattern = "cluster_*/"

logging.info(f"Enviroment: {sys.prefix}")
logging.info("Concatenate eulerian views in:")
logging.info(master_data_dir)
logging.info(f"Subfolder pattern: {pattern}")
logging.info(f"Relative path to eulerian dataset: {relative_path_to_conservation_dataset}")
logging.info(f"Save the combined eulerian view to: {output_file_path}")

data_dir_list = np.array(sorted(list(master_data_dir.glob(pattern))))
conservation_dataset_path_list = data_dir_list / relative_path_to_conservation_dataset

sucessful = []

max_gridbox_list = []
file_path_list = []
cloud_id_list = []

for step, (data_dir, conservation_dataset_path) in enumerate(
    zip(
        data_dir_list,
        conservation_dataset_path_list,
    )
):
    cloud_id = int(data_dir.name.split("_")[1])
    logging.info(f"Start {cloud_id}")
    try:
        logging.info(f"Open {conservation_dataset_path}")
        xr.open_dataset(conservation_dataset_path)

        logging.info(f"Add cloud to combination list")
        file_path_list.append(conservation_dataset_path)
        cloud_id_list.append(cloud_id)

        logging.info(f"Processed {cloud_id}")

    # allow either ValueError or FileNotFoundError
    except (FileNotFoundError, ValueError) as e:
        logger.error(e)

number_sucessful = len(cloud_id_list)

# %%
logging.info(f"Sucessful clouds are: {list(cloud_id_list)}")

logging.info("Attempt to open all sucessful clouds and combine them with xr.open_mfdataset")

cloud_id_index = pd.Index(cloud_id_list, name="cloud_id")
ds = xr.open_mfdataset(file_path_list, combine="nested", concat_dim=[cloud_id_index], parallel=True)

logging.info("Add cloud_id and max_gridbox to the dataset")
ds["cloud_id"].attrs.update(dict(long_name="Cloud identification number", units=""))

ds = ds.sortby("cloud_id")

ds.attrs.update(
    dict(
        description="Combined eulerian view for all clouds.",
        author="Nils Niebaum",
        date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
)

logging.info(f"Attempt to save combined dataset to: {output_file_path}")
ds.to_netcdf(output_file_path)
logging.info(f"Closing dataset")
logging.info(f"Done")
