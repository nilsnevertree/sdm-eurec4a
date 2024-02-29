.. _data_preprocessing:
Data preprocessing
==================
In order to use the package with all advantages, it makes sense, to pre-process the kind of raw files.

ATR cloud composite
-------------------
To download the already pre-processed ATR data one can use wget and download the `Cloud Composite <https://doi.org/10.25326/237>`_.
Excecute the following command in the ``data/observation/cloud_composite/raw`` folder!

.. code-block:: console
    cd data/observation/cloud_composite/raw
    wget -r --no-parent -nH -nd --cut-dirs=1 --reject "index.html*" --accept "*.nc" -P SPECIFYTHIS https://observations.ipsl.fr/aeris/eurec4a-data/AIRCRAFT/ATR/PMA/PROCESSED/CloudComposite/


.. note::
    - For further use in this package, the data was chosen to be preprocessed slightly. THe preprocessing includes the following steps:
        - Combination of the individual netCDF files for each flight into one file.
        - Variable name changes to be more consistent with the rest of the package. (This can be seen as emphasized lines).
    - Two scripts exist which basically do the same thing.
    - The first one does not change units and is located under ``scripts/preprocessing/cloud_composite.py``. It stores the result in the file ``cloud_composite.nc`` in the folder mentioned above.
    - The second one *changes units into SI units* and is located under ``scripts/preprocessing/cloud_composite_si_units.py``. It stores the result in the file ``cloud_composite_si_units.nc`` in the folder mentioned above.

.. literalinclude:: ../../scripts/preprocessing/cloud_composite_si_units.py
    :language: python
    :linenos:
    :emphasize-lines: 120-140

Drop sonde dataset JOANNE
-------------------

The drop sonde dataset `JOANNE <https://doi.org/10.25326/246>`_ from EUREC4A is also available for download.
Again you can use ``wget`` download it into the correct directory in the repository directory tree:

.. code-block:: console
    cd data/observation/dropsonde
    wget -r --no-parent -nH --cut-dirs=1 --reject "index.html*" --accept "*.nc" -P SPECIFYTHIS https://observations.ipsl.fr/aeris/eurec4a-data/PRODUCTS/MERGED-MEASUREMENTS/JOANNE/v2.0.0/SPECIFYTHIS/

.. note::
    Also here, a preprocessing of the dataset is performed with a script.
    You can see the script also below.
    The script is located under ``scripts/preprocessing/drop_sondes.py``.
    Again variable name changes to be more consistent with the rest of the package. (This can be seen as emphasized lines).

.. literalinclude:: ../../scripts/preprocessing/drop_sondes.py
    :language: python
    :linenos:
    :emphasize-lines: 65-89