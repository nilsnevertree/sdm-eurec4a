#!/bin/bash
#SBATCH --job-name=e1d_eulerian_master
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100M
#SBATCH --time=00:10:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eulerian_view/master/%j/%j_out.out
#SBATCH --error=./logfiles/eulerian_view/master/%j/%j_err.out


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
microphysics="null_microphysics"
# microphysics="condensation"
# microphysics="collision_condensation"
# microphysics="coalbure_condensation_small"
# microphysics="coalbure_condensation_large"


create=false
concatenate=true

path2sdm_eurec4a=${HOME}/repositories/sdm-eurec4a
path2data=${HOME}/CLEO/data/output_v3.5/${microphysics}/


create_script_path=${path2sdm_eurec4a}/scripts/CLEO/output_processing/create_eulerian_views.sh
create_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/create_eulerian_views.py

concatenate_script_path=${path2sdm_eurec4a}/scripts/CLEO/output_processing/concatenate_eulerian_views.sh
concatenate_pythonscript=${path2sdm_eurec4a}/scripts/CLEO/output_processing/concatenate_eulerian_views.py

directories=($(find ${path2data} -maxdepth 1 -type d -name 'clusters*' -printf '%P\n' | sort))
number_of_dirs=${#directories[@]}
max_number=$(($number_of_dirs - 1))
# max_number=1
#echo "Directories: ${directories[@]}"
echo "Number of directories: ${#directories[@]}"

echo "Update create eulerian views script"
# Update --array=0-max_number
sed -i "s/#SBATCH --array=.*/#SBATCH --array=0-${max_number}/" "$create_script_path"
# Update --ntasks-per-node=1
sed -i "s/#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=1/" "$create_script_path"


echo "============================================"
echo "path2data: ${path2data}"
echo "microphysics: ${microphysics}"

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

if [ "$create" = true ]; then
    echo "Create eulerian views"
    JOBID_create=$(\
        sbatch --export=create_pythonscript=${create_pythonscript},path2data=${path2data} \
        ${create_script_path}\
        )
    echo "JOBID: ${JOBID_create}"
    echo "${JOBID_create}"
    echo "============================================"
fi

if [ "$concatenate" = true ] && [ "$create" = true ]; then
    echo "Concatenate Eulerian Views with dependency of create eulerian views"
    echo "JOBID_create: ${JOBID_create}"
    JOBID_concatenate=$(\
        sbatch --dependency=afterany:${JOBID_create##* } --export=concatenate_pythonscript=${concatenate_pythonscript},path2data=${path2data} \
        ${concatenate_script_path}\
        )
    echo "JOBID: ${JOBID_concatenate}"
    echo "============================================"
elif [ "$concatenate" = true ] && [ "$create" = false ]; then
    echo "Concatenate Eulerian Views"
    JOBID_concatenate=$(\
        sbatch --export=concatenate_pythonscript=${concatenate_pythonscript},path2data=${path2data} \
        ${concatenate_script_path}\
        )
    echo "JOBID: ${JOBID_concatenate}"
    echo "============================================"
fi
echo "============================================"
