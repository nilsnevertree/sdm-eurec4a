#!/bin/bash
#SBATCH --job-name=sdm_eurec4a1d_eulerian_view
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/sdm_eurec4a1d_eulerian_view.%j_out.out
#SBATCH --error=./logfiles/sdm_eurec4a1d_eulerian_view.%j_err.out


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

path2sdm_eurec4a=${HOME}/repositories/sdm-eurec4a
subdir_pattern=clusters_

# # NO PHYSICS
# data_dir=${HOME}/CLEO/data/output_v3.3/stationary_no_physics/

# CONDENSATION
data_dir=${HOME}/CLEO/data/output_v3.3/stationary_condensation

# # COLLISION AND CONDENSATION
# data_dir=${HOME}/CLEO/data/output_v3.4/stationary_collision_condensation


output_dir=${data_dir}/combined
result_file_name=eulerian_dataset_combined.nc

create_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/create_eulerian_views.py
concatenate_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/concatenate_eulerian_views.py


### ------------------ Load Modules -------------------- ###
env=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
python=${env}/bin/python
source activate ${env}

### ------------------ Create Eulerian View --------------- ###
echo "CREATE EULERIAN VIEWS"
for exp_folder in ${data_dir}/${subdir_pattern}*; do
    echo "::::::::::::::::::::::::::::::::::::::::::::"
    echo "Create eulerian views for experiment:"
    echo "in ${exp_folder}"
    {
        ${python}  ${create_pythonscript} --data_dir ${exp_folder}
    } || {
        echo "--------------------------------------------"
        echo "EXCECUTION ERROR: in ${exp_folder}"
        echo "--------------------------------------------"
    }
    echo "::::::::::::::::::::::::::::::::::::::::::::"
done
### ---------------------------------------------------- ###

### ------------------ Concatenate Eulerian View --------------- ###
${python}  ${concatenate_pythonscript} --data_dir ${data_dir} --output_dir ${output_dir} --result_file_name ${result_file_name}
echo "============================================"
