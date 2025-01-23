from pathlib import Path
import xarray as xr
import numpy as np

from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask.config

from tempfile import NamedTemporaryFile, TemporaryDirectory  # Creating temporary Files/Dirs
from getpass import getuser  # Libaray to copy things


def init_dask_slurm_cluster(
    scale: int = 1,
    processes: int = 16,
    walltime: str = "00:30:00",
    memory: str = "64GiB",
    name: str = "m301096-dask-cluster",
    account: str = "mh1126",
    data_dir: str = "/scratch/m/m301096/dask_temp",
    log_dir: str = "/scratch/m/m301096/dask_logs",
    scheduler_options: dict = {"dashboard_address": ":8989"},
):
    """
    Initialize a dask slurm cluster with the given parameters to be used with dask for xarray.
    This script is a slightly modified copy of the script provided by the `python on levante` course on GWDG from MPI-Met in Hamburg.

    Parameters:
    -----------
    scale : int
        Number of nodes to be used.
        Default is 1.
    processes : int
        Number of processes per worker.
        Default is 16.
    walltime : str
        Walltime for the job, the format is "HH:MM:SS" as used for all slurm jobs.
        Default is "00:30:00".
    memory : str
        Memory per node, the format is "XXGiB" where XX is the amount of memory.
        Default is "64GiB".
    name : str
        Name of the job.
        Default is "m301096-dask-cluster".
    account : str
        Account to be used for the job.
        Default is "mh1126".
    data_dir : str
        Directory to store the temporary data.
        This directory should be accessible from all nodes.
        The best option is to use the scratch directory.
        Default is "/scratch/m/m301096/dask_temp".
    log_dir : str
        Directory to store the logs.
        Default is "/scratch/m/m301096/dask_logs".
    scheduler_options : dict
        Dictionary of scheduler options.
        For instance to set the dashboard port address.
        Default is {"dashboard_address": ":8989"}.

    Returns:
    --------
    client : dask.distributed.Client
        Dask client to be used for the computations.
    scluster : dask_jobqueue.SLURMCluster
        Dask SLURM cluster to be used for the computations

    """

    dask.config.set(
        {
            "distributed.worker.data-directory": data_dir,
            "distributed.worker.memory.target": 0.75,
            "distributed.worker.memory.spill": 0.85,
            "distributed.worker.memory.pause": 0.95,
            "distributed.worker.memory.terminate": 0.98,
        }
    )

    scluster = SLURMCluster(
        queue="compute",
        walltime=walltime,
        memory=memory,
        cores=processes,
        processes=processes,
        account=account,
        name=name,
        interface="ib0",
        asynchronous=False,
        log_directory=log_dir,
        local_directory=data_dir,
        scheduler_options=scheduler_options,
    )

    client = Client(scluster)
    scluster.scale(jobs=scale)
    print(scluster.job_script())
    nworkers = scale * processes
    client.wait_for_workers(nworkers)  # waits for all workers to be ready, can be submitted now

    return client, scluster
