#!/bin/bash
#SBATCH --job-name=e1d_eulerian_concatenate
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:25:00
#SBATCH --mail-user=nils-ole.niebaumy@mpimet.mpg.de
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1126
#SBATCH --output=./logfiles/eulerian_view/concatenate/%j/%j_out.out
#SBATCH --error=./logfiles/eulerian_view/concatenate/%j/%j_err.out



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

output_dir=${path2data}/combined
result_file_name=eulerian_dataset_combined_v2.nc

### ------------------ Load Modules -------------------- ###
env=/work/mh1126/m301096/conda/envs/sdm_pysd_env312
python=${env}/bin/python
source activate ${env}

# ### ------------------ Concatenate Eulerian View --------------- ###
${python}  ${concatenate_pythonscript} --data_dir ${path2data} --output_dir ${output_dir} --result_file_name ${result_file_name}
echo "============================================"
