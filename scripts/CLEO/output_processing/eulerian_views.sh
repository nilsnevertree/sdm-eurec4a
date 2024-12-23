#!/bin/bash
#SBATCH --job-name=e1d_eulerian_master
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=00:10:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eulerian_view/combined/%j_out.out
#SBATCH --error=./logfiles/eulerian_view/combined/%j_err.out


### ---------------------------------------------------- ###
### ------------------ Input Parameters ---------------- ###
### ------ You MUST edit these lines to set your ------- ###
### ----- environment, build type, directories, the ---- ###
### --------- executable(s) to compile and your -------- ###
### --------------  python script to run. -------------- ###
### ---------------------------------------------------- ###

### ------------------ Load Modules -------------------- ###
env=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
mamba activate ${env}
python=${env}/bin/python

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

create=true
concatenate=false

create_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/create_eulerian_views_mpi4py.py
concatenate_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/concatenate_eulerian_views.py


path2data=${path2CLEO}/data/output_v4.1/${microphysics}/

echo "============================================"
echo "path2data: ${path2data}"
echo "microphysics: ${microphysics}"

if [ ! -d "$path2data" ]; then
    echo "Invalid path to data"
    exit 1
elif [ ! -f "$create_pythonscript" ]; then
    echo "Python script not found: ${create_pythonscript}"
    exit 1
else
    echo "All paths are valid"
fi
echo "============================================"

if [ "$create" = true ]; then
    echo "Create eulerian views"
    mpirun -np 40 python ${create_pythonscript} --data_dir ${path2data}
    echo "============================================"
fi

if [ "$concatenate" = true ]; then
    echo "Concatenate Eulerian Views with dependency of create eulerian views"
    python ${concatenate_pythonscript} --data_dir ${path2data}
    echo "============================================"
fi
echo "============================================"
