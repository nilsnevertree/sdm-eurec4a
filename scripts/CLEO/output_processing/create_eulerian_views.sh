#!/bin/bash
#SBATCH --job-name=e1d_eulerian_create
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=00:10:00
#SBATCH --mail-user=clara.bayley@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eulerian_view/create/%A/%A_%a_out.out
#SBATCH --error=./logfiles/eulerian_view/create/%A/%A_%a_err.out
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
# microphysics="condensation"
# microphysics="collision_condensation"
# microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"

# path2data=/work/mh1126/m300950/rain-evap-nils/sdm-eurec4a-CLEO/data/output_v3.5/${microphysics}/
echo "Init path2data: ${path2data}"
echo "Init python script: ${create_pythonscript}"
echo "============================================"

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

### ------------------ Load Modules -------------------- ###
env=/home/m/m300950/mamba/envs/sdm_eurec4a_env312
python=${env}/bin/python
source activate ${env}

# select the directory to process
directories=($(find ${path2data} -maxdepth 1 -type d -name 'clusters*' -printf '%P\n' | sort))
path2inddir=${path2data}/${directories[${SLURM_ARRAY_TASK_ID}]}
echo "Number of directories: ${#directories[@]}"
echo "Current array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Current directory: ${path2inddir}"


### ------------------ Create Eulerian View --------------- ###
${python}  ${create_pythonscript} --data_dir ${path2inddir}
### ---------------------------------------------------- ###

echo "============================================"
