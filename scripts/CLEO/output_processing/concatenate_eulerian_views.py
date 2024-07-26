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


if is_notebook() == False:
    parser = argparse.ArgumentParser(description="Concatenate eulerian views")

    # Add arguments
    parser.add_argument("-d", "--data_dir", type=str, help="Path to data directory", required=True)
    parser.add_argument("-o", "--output_dir", type=str, help="Path to output directory", required=True)
    parser.add_argument(
        "-r",
        "--result_file_name",
        type=str,
        default="eulerian_dataset_combined.nc",
        help="Name of output file",
    )
    # Parse arguments
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_file_path = output_dir / args.result_file_name

import logging
import os
import sys

import pandas as pd
import xarray as xr


log_dir = output_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(name="logger")
logger.setLevel(logging.INFO)
logging.captureWarnings(True)

# create a file handler
file_handler = logging.FileHandler(log_dir / f"concatenate_eulerian_views.log")
file_handler.setLevel(logging.INFO)
# create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
logger_handlers = [type(handler) for handler in logger.handlers]
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# add the handlers to the logger
if logging.FileHandler not in logger_handlers:
    logger.addHandler(file_handler)
if logging.StreamHandler not in logger_handlers:
    logger.addHandler(console_handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical(
        "Execution terminated due to an Exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


sys.excepthook = handle_exception


logger.info(f"Enviroment: {sys.prefix}")
logger.info("Combine all eulerian views from:")
logger.info(data_dir)
logger.info("Save the combined eulerian view to:")
logger.info(output_file_path)

subdirectories = [
    name
    for name in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, name)) and name.startswith("clusters_")
]

dataset_list = []
file_path_list = []
cloud_id_list = []

for sub_dir_name in subdirectories:
    sub_dir = data_dir / Path(sub_dir_name)
    cloud_id = int(sub_dir_name.split("_")[1])

    eulerian_dataset_path = sub_dir / "processed/eulerian_dataset.nc"

    try:
        euler_dataset = xr.open_dataset(eulerian_dataset_path)
        max_gridbox_temp = euler_dataset["gridbox"].max()

        file_path_list.append(eulerian_dataset_path)

        dataset_list.append(max_gridbox_temp)
        cloud_id_list.append(cloud_id)

        logger.info(f"Process {sub_dir_name}")

    except FileNotFoundError as e:
        logger.error(f"Skip {sub_dir_name} : {type(e)} {e}")


cloud_id_index = pd.Index(cloud_id_list, name="cloud_id")
ds = xr.open_mfdataset(file_path_list, combine="nested", concat_dim=[cloud_id_index])

ds["cloud_id"].attrs.update(dict(long_name="Cloud identification number", units=""))

max_gridbox = xr.concat(dataset_list, dim=cloud_id_index)
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

logger.info(f"Attempt to save combined dataset to: {output_file_path}")
ds.to_netcdf(output_file_path)
logger.info(f"Saved combined dataset to: {output_file_path}")
