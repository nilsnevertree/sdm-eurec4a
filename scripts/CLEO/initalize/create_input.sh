#!/bin/bash
#SBATCH --job-name=sdm_eurec4a1d_create_input
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/sdm_eurec4a1d_create_input.%j_out.out
#SBATCH --error=./logfiles/sdm_eurec4a1d_create_input.%j_err.out


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


path2sdmeurec4a=${HOME}/repositories/sdm-eurec4a/

pythonscript=${path2sdmeurec4a}/scripts/CLEO/initalize/create_input.py

### ------------------ Load Modules -------------------- ###
env=/work/mh1126/m301096/conda/envs/sdm_eurec4a_env312
python=${env}/bin/python
source activate ${env}

### ------------------ Run Python Script --------------- ###
${python}  ${pythonscript}
