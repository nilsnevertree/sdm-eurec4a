#!/bin/bash
#SBATCH --job-name=e1d_eulerian_master
#SBATCH --partition=compute
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=/home/m/m301096/repositories/sdm-eurec4a/logs/eulerian_master/%j_out.log
#SBATCH --error=/home/m/m301096/repositories/sdm-eurec4a/logs/eulerian_master/%j_err.log

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

create_eulerian_view=true
# concatenate_eulerian_view=true

# inflow_outflow=true
# concatenate_inflow_outflow=true


create_script=${path2sdm_eurec4a}/scripts/CLEO/output_processing/create_eulerian_views_mpi4py.py
concatenate_script=${path2sdm_eurec4a}/scripts/CLEO/output_processing/concatenate_eulerian_views.py

# inflow_outflow_pyhtonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/create_inflow_outflow_mpi4py.py
# concatenate_io_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/concatenate_inflow_outflow.py

path2data=${path2CLEO}/data/output_v4.0/${microphysics}/

echo "============================================"
echo "path2data: ${path2data}"
echo "microphysics: ${microphysics}"

if [ ! -d "$path2data" ]; then
    echo "Invalid path to data"
    exit 1
elif [ ! -f "$create_script" ]; then
    echo "Python script not found: ${create_script}"
    exit 1
elif [ ! -f "$concatenate_script" ]; then
    echo "Python script not found: ${concatenate_script}"
    exit 1
else
    echo "All paths are valid"
fi
echo "============================================"

if [ "$create_eulerian_view" = true ]; then
    echo "Create eulerian views"
    mpirun -np 15 python ${create_script} --data_dir ${path2data}
    wait
    echo "============================================"
fi

if [ "$concatenate_eulerian_view" = true ]; then
    echo "Concatenate Eulerian Views with dependency of create eulerian views"
    python ${concatenate_script} --data_dir ${path2data}
    echo "============================================"
fi

# if [ "$inflow_outflow" = true ]; then
#     echo "Create Inflow Outflow"
#     # python ${inflow_outflow_pyhtonscript} --data_dir ${path2data}
#     mpirun -np 20 python ${inflow_outflow_pyhtonscript} --data_dir ${path2data}
#     wait
#     echo "============================================"
# fi

# if [ "$concatenate_inflow_outflow" = true ]; then
#     echo "Concatenate Inflow Outflow datasets"
#     python ${concatenate_io_pythonscript} --data_dir ${path2data}
#     echo "============================================"
# fi


echo "============================================"
