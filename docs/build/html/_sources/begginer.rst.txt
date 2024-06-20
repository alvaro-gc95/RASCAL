Begginer Tutorials
===================

Prepare your data: Folder Structure
-------------------------------------

**1) Observational data**

The observational data must follow this structure:


| /station_observations_directory/
| ├── /variable/
| │ ├── variable.csv
| │ ├── meta.csv

Where ``variable`` is the name of the variable to reconstruct (ex: TMEAN, PCP, WSPD ...)
and ``meta.csv`` is a .csv file that contains the columns [code, name, latitude, longitude, latitude]
The data must be in daily or sub-daily resolution.

.. note::
    **/station_observations_directory/** should be the same word as in the **code** variable in ``meta.csv``
    The ``variable.csv`` file should contain only the dates, and the data in a column named the same as the file.
   
   
An example of a variable file of mean temperature would be ``TMEAN.csv`` with the following format:

+-------------+---------------+
|             |     TMEAN     |
+=============+===============+
| 2005-01-01  |     -0.1      |
+-------------+---------------+
| 2005-01-02  |      1.2      |
+-------------+---------------+
|     ...     |      ...      |
+-------------+---------------+

An example of a ``meta.csv`` file would be:

+-------------+---------------+-------------+---------------+-------------+
|    code     |     name      |   latitude  |   longitude   |  altitude   |
+=============+===============+=============+===============+=============+
|    St03     |  Station 03   |   40.793056 |   -4.010556   |    1893     |
+-------------+---------------+-------------+---------------+-------------+

Therefore, in this case the folder structure would be as follows:

| /St03/
| ├── /TMEAN/
| │ ├── TMEAN.csv
| │ ├── meta.csv


**2) Reanalysis data**

The reanalysis data must follow this structure:

| /reanalysis_directory/
| ├── /y_YYYY/
| │ ├── YYYY_level_variable.nc

Where ``YYYY`` is the year of the data,
``level`` the level of the variable and 
``variable`` the name of the predictor variable.
The reanalysis data can be in netCDF or GRIB format
The data must be in daily or sub-daily resolution


Make a reconstruction
------------------------

RASCAL is based in four main clases: Station, Predictor, Analogs and Rskill. It can be imported as:

.. code-block:: python

   import rascal


**1) Get observational data**

   To load the observational data (in daily or sub-daily resolution) and the station metadata, the data is loaded from a CSV file with the same name as the desired variable, and a meta.csv file containing the name, code, altitude, longitude and latitude of the station

   .. code-block:: python
   
      station = rascal.analogs.Station(path='./data/observations/station/')
      station_data = station.get_data(variable='PCP')


**2) Load and process predictor fields from large-scale models**
   To load the reanalysis or large-scale model data we use the Predictor class. This example shows how to use the Total Column of Water Vapor Flux from the ERA20C reanalysis. In this reanalysis the components U and V of the TCWVF are named '71.162' and '72.162'. The predictor is set or the years 1900-1910, for each day only the 12:00 is selected through the ``grouping`` argument, the domain is 80ºN-20ºN, 60ºW-20ºE. The ``mosaic`` argument set to *True* concatenates both components U and V in the longitude axis to obtain a single compound variable of size *(time x 2*longitude x latitude)*:

   .. code-block:: python
   
      # Get file paths
      predictor_files = rascal.utils.get_files(
          nwp_path='./data/reanalysis/era20c/',
          variables=['71.162', '72.162'],
          dates=[1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910],
          file_format=".grib"
      )

      # Generate Predictor
      predictors = rascal.analogs.Predictor(
         paths=predictor_files,
         grouping='12h_1D_mean',
         lat_min=20,
         lat_max=80,
         lon_min=-60,
         lon_max=20,
         mosaic=True
     )

**3) Perform Principal Component Analysis on the predictor fields**
   The Principal Component Analysis (PCA) of the compund variable standardized anomalies, with 4 principal components and for the conventionan seasons DJF, MAM, JJA, and SON,  is conducted as follows:
   
   .. code-block:: python
   
      predictor_pcs = predictors.pcs(
         npcs=n_components,
         seasons=[[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
         standardize=True,
         path="./tmp/"
      )


**4) Look at the PC space to find analog days in the historical data**
   After performing the PCA, the obtained values of the principal componets act as the predictor used to perform the reconstructions. First the analog days, in order of euclidean distance, are found.

   .. code-block:: python
   
      analogs = rascal.analogs.Analogs(pcs=predictor_pcs, observations=station_data, dates=test_dates)


**5) Reconstruct or extend missing observational data**
   Later, the reconstuctions are made using one of the following similarity methods: ``closest``, ``average``, or ``quantilemap``.

   .. code-block:: python

      reconstruction = analogs.reconstruct(
          pool_size=30,
          method='closest'
          )

