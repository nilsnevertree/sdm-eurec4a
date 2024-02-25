Identifying individual clouds in the ATR dataset
================================================

In the whole cloud composite dataset, many differnt cloud types can be found.
Thus it might make sense to take a look on individual cloud.
Therefore a script is implemented to identify individual clouds in the cloud composite dataset (data\observation\cloud_composite\processed\cloud_composite.nc)

Below you can find a copy of this script.
The important lines are highlighted.

.. note::
    - In lines 131-134 the script uses the ``cloud_mask`` to identify individual clouds.
    - In lines 162-163 and 172 this mid time of a cloud event is chosen as the leading dimension of the new netCDF file.
    - The leading dimension becomes ``time``.

.. literalinclude:: ../../scripts/preprocessing/cloud_identification_general.py
    :language: python
    :linenos:
    :emphasize-lines: 131-134, 162-163, 172
