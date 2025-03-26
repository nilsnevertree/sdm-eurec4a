#!/bin/bash
#SBATCH --job-name=e1d_conservation
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=um1487
#SBATCH --output=/home/m/m301096/repositories/sdm-eurec4a/logs/conservation_master/%j_out.log
#SBATCH --error=/home/m/m301096/repositories/sdm-eurec4a/logs/conservation_master/%j_err.log

### ---------------------------------------------------- ###
### ------------------ Input Parameters ---------------- ###
### ------ You MUST edit these lines to set your ------- ###
### ----- environment, build type, directories, the ---- ###
### --------- executable(s) to compile and your -------- ###
### --------------  python script to run. -------------- ###
### ---------------------------------------------------- ###

### ------------------ Load Modules -------------------- ###
source ${HOME}/.bashrc
env=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
env=/work/um1487/m301096/conda/envs/sdm_pysd_python312
conda activate ${env}

# ------------------ Set Variables --------------------- #
echo "--------------------------------------------"
echo "START RUN"
date
echo "git hash: $(git rev-parse HEAD)"
echo "git branch: $(git symbolic-ref --short HEAD)"
echo "============================================"

# Set microphysics setup
# microphysics="null_microphysics"
microphysics="condensation"
# microphysics="collision_condensation"
# microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"

path2CLEO=${HOME}/CLEO/
path2sdm_eurec4a=${HOME}/repositories/sdm-eurec4a

create_inflow_outflow=true
concatenate_inflow_outflow=true

inflow_outflow_pyhtonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/create_inflow_outflow_mpi4py.py
concatenate_io_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/concatenate_inflow_outflow.py

path2data=${path2CLEO}/data/output_v4.2/${microphysics}/

echo "============================================"
echo "path2data: ${path2data}"
echo "microphysics: ${microphysics}"

if [ ! -d "$path2data" ]; then
    echo "Invalid path to data"
    exit 1
elif [ ! -f "$inflow_outflow_pyhtonscript" ]; then
    echo "Python script not found: ${eulerian_view_pythonscript}"
    exit 1
elif [ ! -f "$concatenate_io_pythonscript" ]; then
    echo "Python script not found: ${concatenate_ev_pythonscript}"
    exit 1
else
    echo "All paths are valid"
fi
echo "============================================"

if [ "$create_inflow_outflow" = true ]; then
    echo "Create Inflow Outflow"
    # python ${inflow_outflow_pyhtonscript} --data_dir ${path2data}
    mpirun -np 30 python ${inflow_outflow_pyhtonscript} --data_dir ${path2data}
    wait
    echo "============================================"
fi

if [ "$concatenate_inflow_outflow" = true ]; then
    echo "Concatenate Inflow Outflow datasets"
    python ${concatenate_io_pythonscript} --data_dir ${path2data}
    echo "============================================"
fi
echo "============================================"
