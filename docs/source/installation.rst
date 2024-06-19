Getting Started
=================

RASCAL is available to download in PyPi and GitHub. To install RASCAL, it is recommended to create a new environment to avoid possible conflicts wioth its required dependencies. 

.. code-block:: console

   (base) $ conda create --name rascal_env
   (base) $ conda activate rascal_env
   
Required dependencies
------------------------

RASCAL runs with *Python 3.10*.

These are the dependencies of RASCAL:

   - **numpy** 1.26.4
   - **pandas** 2.2.1
   - **dask** 2024.4.1
   - **xarray** 2024.3.0
   - **scipy** 1.13.0
   - **tqdm** 4.65.0
   - **scikit-learn** 1.4.1.post1
   - **seaborn** 0.13.2
   - **eofs** 1.4.1
   - **cfgrib** 0.9.12.0
   - **netCDF4** 1.7.0

Installation via PyPi
------------

RASCAL can be installed via PyPi:

.. code-block:: console

   (rascal_env) $ pip install rascal-ties
   
Installation via GitHub
------------

RASCAL can be used via GitHub:

.. code-block:: console

   (rascal_env) $ git clone https://github.com/alvaro-gc95/RASCAL
   
The GitHub repository also contains the following scripts:

   - ``multiple_runs_example.py`` to automatize running several configurations of similarity methods and pool sizes for various stations and variables. 
     This can be configured through the ``config.yaml`` file 
   - ``projection_example.py`` Mostly the same as ``multiple_runs_example.py``, but including a split in training and testing periods for the PCA, and an added year as a projection onto the training period PCs
   - ``RASCAL_evaluation.ipynb`` a Jupyter Notebook to plot and validate the reconstructions
   
   



