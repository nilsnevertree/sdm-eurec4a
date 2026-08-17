#!/bin/bash
#SBATCH --job-name=dsd_datasets
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --mail-user=clara.bayley@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=/home/m/m300950/rain-evap-nils/sdm-eurec4a/logs/dsd_datasets/%j_out.log
#SBATCH --error=/home/m/m300950/rain-evap-nils/sdm-eurec4a/logs/dsd_datasets/%j_err.log

### ---------------------------------------------------- ###
### ------------------ Input Parameters ---------------- ###
### ------ You MUST edit these lines to set your ------- ###
### ----- environment, build type, directories, the ---- ###
### --------- executable(s) to compile and your -------- ###
### --------------  python script to run. -------------- ###
### ---------------------------------------------------- ###

### ------------------ Load Modules -------------------- ###
source ${HOME}/.bashrc
env=/home/m/m300950/mamba/envs/sdm_eurec4a_env312
micromamba activate ${env}

# ------------------ Set Variables --------------------- #
echo "--------------------------------------------"
echo "START RUN"
date
echo "git hash: $(git rev-parse HEAD)"
echo "git branch: $(git symbolic-ref --short HEAD)"
echo "python : $(which python)"
echo "============================================"

# Set microphysics setup
# microphysics="null_microphysics"
microphysics="condensation"
# microphysics="collision_condensation"
# microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"

path2sdm_eurec4a=/home/m/m300950/rain-evap-nils/sdm-eurec4a
path2sdm_eurec4a_cleo=/home/m/m300950/rain-evap-nils/sdm-eurec4a-CLEO
path2data=/work/mh1126/m300950/rain-evap-nils/sdm-eurec4a-CLEO/data/output_v4.2/${microphysics}/

pythonscript=${path2sdm_eurec4a}/scripts/paper/dsd_datasets_for_all_clusters.py


echo "============================================"
echo "path2data: ${path2data}"
echo "microphysics: ${microphysics}"

if [ ! -d "$path2data" ]; then
    echo "Invalid path to data: ${path2data}"
    exit 1
elif [ ! -d "$path2sdm_eurec4a_cleo" ]; then
    echo "sdm-eurec4a-cleo directory not found: ${path2sdm_eurec4a_cleo}"
    exit 1
elif [ ! -f "$pythonscript" ]; then
    echo "Python script not found: ${pythonscript}"
    exit 1
else
    echo "All paths are valid"
fi
echo "============================================"

echo "===== CREATING DSD DATASETS FOR ALL CLUSTERS ====="
srun python ${pythonscript} ${path2sdm_eurec4a_cleo} ${path2data}
wait
echo "============================================"
if [ $? -ne 0 ]; then
    echo "Error: One or more process failed!"
    exit 1
fi
echo "=============== JOB FINISHED ==============="
