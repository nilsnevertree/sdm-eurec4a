# Data handling pipeline

In order to use the package with all advantages, it makes sense, to pre-process the kind of raw files.


## Download and preprocess files


### ATR cloud composite
To download the already pre-processed ATR data one can use wget and download the [Cloud Composite](https://observations.ipsl.fr/aeris/eurec4a-data/AIRCRAFT/ATR/PMA/PROCESSED/CloudComposite/) dataset.
Excecute the following command in the ``data/observation/cloud_composite/raw`` folder!
```` bash
cd data/observation/cloud_composite/raw
wget -r --no-parent -nH -nd --cut-dirs=1 --reject "index.html*" --accept "*.nc" -P SPECIFYTHIS https://observations.ipsl.fr/aeris/eurec4a-data/AIRCRAFT/ATR/PMA/PROCESSED/CloudComposite/
````

To have a single file with confident variable names, please run the script ``python scripts\preprocessing\cloud_composite.py``

This basically combines the single netCDF files and makes a Variable mapping.
This will also create a log file.

### Drop sonde dataset JOANNE

Again you can use wget for this.

```` bash
cd data/observation/dropsonde
wget -r --no-parent -nH --cut-dirs=1 --reject "index.html*" --accept "*.nc" -P SPECIFYTHIS https://observations.ipsl.fr/aeris/eurec4a-data/PRODUCTS/MERGED-MEASUREMENTS/JOANNE/v2.0.0/SPECIFYTHIS/
````


## Identify individual clouds.

For this one can use the ``python scripts\preprocessing\cloud_identification_general.py`` script. 

Further informations can be found here: 
```{eval-rst}
   ./source/autosummary_index.rst
```