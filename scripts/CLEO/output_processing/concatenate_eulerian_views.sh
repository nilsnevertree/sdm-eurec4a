#!/bin/bash
#SBATCH --job-name=e1d_eulerian_view
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eulerian_view/%j/%j_out.out
#SBATCH --error=./logfiles/eulerian_view/%j/%j_err.out



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

output_dir=${rawdirectory}/combined
result_file_name=eulerian_dataset_combined.nc

concatenate_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/concatenate_eulerian_views.py

### ------------------ Load Modules -------------------- ###
env=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
python=${env}/bin/python
source activate ${env}

# ### ------------------ Concatenate Eulerian View --------------- ###
${python}  ${concatenate_pythonscript} --data_dir ${data_dir} --output_dir ${output_dir} --result_file_name ${result_file_name}
echo "============================================"
