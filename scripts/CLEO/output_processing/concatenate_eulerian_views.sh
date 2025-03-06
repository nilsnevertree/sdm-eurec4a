#!/bin/bash
#SBATCH --job-name=e1d_concatenate_eulerian
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=30GB
#SBATCH --time=00:15:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=um1487
#SBATCH --output=/home/m/m301096/repositories/sdm-eurec4a/logs/concatenate_master/%j_out.out
#SBATCH --error=/home/m/m301096/repositories/sdm-eurec4a/logs/concatenate_master/%j_err.out



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

### ------------------ Load Modules -------------------- ###
source ${HOME}/.bashrc
env=/work/um1487/m301096/conda/envs/sdm_pysd_python312
env=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
conda activate ${env}
python=${env}/bin/python

# path2data=${HOME}/CLEO/data/output_v4.1/coalbure_condensation_large/
# concatenate_pythonscript=${HOME}/repositories/sdm-eurec4a/scripts/CLEO/output_processing/concatenate_eulerian_views.py

echo "Init path2data: ${path2data}"
echo "Init python script: ${concatenate_pythonscript}"
echo "============================================"



if [ ! -d "$path2data" ]; then
    echo "Invalid path to data"
    exit 1
elif [ ! -f "$concatenate_pythonscript" ]; then
    echo "Python script not found: ${concatenate_pythonscript}"
    exit 1
else
    echo "All paths are valid"
fi
echo "============================================"


# ### ------------------ Concatenate Eulerian View --------------- ###
python  ${concatenate_pythonscript} --data_dir ${path2data}
echo "============================================"
