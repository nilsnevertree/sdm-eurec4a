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

output_dir=${path2sdmeurec4a}/data/model/input/output_test_new

identified_cloud_path=${path2sdmeurec4a}/data/observation/cloud_composite/processed/identified_clouds/identified_clusters_rain_mask_5.nc
distance_relation_path=${path2sdmeurec4a}/data/observation/combined/distance_relations/distance_dropsondes_identified_clusters_rain_mask_5.nc
cloud_composite_path=${path2sdmeurec4a}/data/observation/cloud_composite/processed/cloud_composite_si_units.nc
drop_sonde_path=${path2sdmeurec4a}/data/observation/dropsonde/processed/drop_sondes.nc
identification_type=clusters
environment=nils_levante

pythonscript=${path2sdmeurec4a}/scripts/CLEO/initalize/create_input.py

### ------------------ Load Modules -------------------- ###
env=/work/mh1126/m301096/conda/envs/sdm_eurec4a_env312
python=${env}/bin/python
source activate ${env}

### ------------------ Run Python Script --------------- ###
${python} \
    ${pythonscript} \
    --output_dir ${output_dir} \
    --identified_cloud_path ${identified_cloud_path} \
    --distance_relation_path ${distance_relation_path} \
    --cloud_composite_path ${cloud_composite_path} \
    --drop_sonde_path ${drop_sonde_path} \
    --identification_type ${identification_type} \
    --environment ${environment}
