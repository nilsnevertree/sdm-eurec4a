# can be named as slurm_cluster.py
# the relative path should be "/src/slurm_cluster.py"
# function that you can call in your script by importing.

from pathlib import Path
import xarray as xr
import numpy as np

from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask.config

from tempfile import NamedTemporaryFile, TemporaryDirectory  # Creating temporary Files/Dirs
from getpass import getuser  # Libaray to copy things


def init_dask_slurm_cluster(scale=1, processes=16, walltime="00:30:00", memory="64GiB"):
    dask.config.set(
        {
            "distributed.worker.data-directory": "/scratch/m/m301096/dask_temp",
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
        account="mh1126",
        name="m301096-dask-cluster",
        interface="ib0",
        asynchronous=False,
        log_directory="/scratch/m/m301096/dask_logs/",
        local_directory="/scratch/m/m301096/dask_temp/",
        scheduler_options={"dashboard_address": ":8989"},
    )

    client = Client(scluster)
    scluster.scale(jobs=scale)
    print(scluster.job_script())
    nworkers = scale * processes
    client.wait_for_workers(nworkers)  # waits for all workers to be ready, can be submitted now

    return client, scluster
