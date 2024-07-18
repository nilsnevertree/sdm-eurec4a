#!/bin/bash
#SBATCH --job-name=e1d_eulerian_view
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eulerian_view/%A/%A_%a_out.out
#SBATCH --error=./logfiles/eulerian_view/%A/%A_%a_err.out
#SBATCH --array=0-110



### ---------------------------------------------------- ###
### ------------------ Input Parameters ---------------- ###
### ------ You MUST edit these lines to set your ------- ###
### ----- environment, build type, directories, the ---- ###
### --------- executable(s) to compile and your -------- ###
### --------------  python script to run. -------------- ###
### ---------------------------------------------------- ###

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

rawdirectory=${HOME}/CLEO/data/output_v3.5/${microphysics}/

path2sdm_eurec4a=${HOME}/repositories/sdm-eurec4a
subdir_pattern=clusters_

create_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/create_eulerian_views.py

### ------------------ Load Modules -------------------- ###
env=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
python=${env}/bin/python
source activate ${env}


directories=($(find ${rawdirectory} -maxdepth 1 -type d -name 'clusters*' -printf '%P\n' | sort))

#echo "Directories: ${directories[@]}"
echo "Number of directories: ${#directories[@]}"
echo "Current array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Current directory: ${directories[${SLURM_ARRAY_TASK_ID}]}"

path2inddir=${rawdirectory}/${directories[${SLURM_ARRAY_TASK_ID}]}


### ------------------ Create Eulerian View --------------- ###
${python}  ${create_pythonscript} --data_dir ${path2inddir}
### ---------------------------------------------------- ###

echo "============================================"
