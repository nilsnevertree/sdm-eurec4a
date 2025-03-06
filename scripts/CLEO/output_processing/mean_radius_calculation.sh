#!/bin/bash
#SBATCH --job-name=e1d_conservation
#SBATCH --partition=compute
#SBATCH --mem=1G
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
env=/work/um1487/m301096/conda/envs/sdm_pysd_python312
conda activate ${env}

# ------------------ Set Variables --------------------- #
echo "--------------------------------------------"
echo "START RUN"
date
echo "git hash: $(git rev-parse HEAD)"
echo "git branch: $(git symbolic-ref --short HEAD)"
echo "============================================"

path2CLEO=${HOME}/CLEO/
path2sdm_eurec4a=${HOME}/repositories/sdm-eurec4a

mean_radius_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/mean_radius_calculation.py

path2data=${path2CLEO}/data/output_v4.1/

echo "============================================"
echo "path2data: ${path2data}"

if [ ! -d "$path2data" ]; then
    echo "Invalid path to data"
    exit 1
elif [ ! -f "$mean_radius_pythonscript" ]; then
    echo "Python script not found: ${mean_radius_pythonscript}"
    exit 1
else
    echo "All paths are valid"
fi
echo "============================================"

echo "Create Mean radius script"
python ${mean_radius_pythonscript} --data_dir ${path2data}
echo "============================================"
