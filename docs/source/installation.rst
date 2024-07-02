Getting Started
=================

RASCAL is available to download in PyPi and GitHub. To install RASCAL, it is recommended to create a new environment to avoid possible conflicts with its required dependencies. 

.. code-block:: console

   (base) $ conda create --name rascal_env python==3.10
   (base) $ conda activate rascal_env
   
Required dependencies
------------------------

RASCAL runs with *Python 3.10*.

These are the dependencies of RASCAL:

   - `numpy <https://numpy.org/devdocs/index.html>`_ == 1.26.4
   - `pandas <https://pandas.pydata.org/docs/index.html>`_ == 2.2.1
   - `dask <https://docs.dask.org/en/stable/>`_ == 2024.4.1
   - `xarray <https://docs.xarray.dev/en/stable/>`_ == 2024.3.0
   - `scipy <https://docs.scipy.org/doc/scipy/>`_ == 1.13.0
   - `tqdm <https://tqdm.github.io/>`_ == 4.65.0
   - `scikit-learn <https://scikit-learn.org/stable/>`_ == 1.4.1.post1
   - `seaborn <https://seaborn.pydata.org/>`_ == 0.13.2
   - `eofs <https://ajdawson.github.io/eofs/latest/>`_ == 1.4.1
   - `cfgrib <https://github.com/ecmwf/cfgrib/>`_ == 0.9.12.0
   - `netCDF4 <https://unidata.github.io/netcdf4-python/>`_ == 1.7.0
   - `matplotlib <https://matplotlib.org/stable/index.html>`_ >= 3.5.5

Installation via PyPi
-----------------------

RASCAL can be installed via PyPi:

.. code-block:: console

   (rascal_env) $ pip install rascal-ties
   
Installation via GitHub
-------------------------

RASCAL can be used via GitHub:

.. code-block:: console

   (rascal_env) $ git clone https://github.com/alvaro-gc95/RASCAL
   
The GitHub repository also contains the following scripts:

   - ``multiple_runs_example.py`` to automatize running several configurations of similarity methods and pool sizes for various stations and variables. 
     This can be configured through the ``config.yaml`` file 
   - ``projection_example.py`` Mostly the same as ``multiple_runs_example.py``, but including a split in training and testing periods for the PCA, and an added year as a projection onto the training period PCs
   - ``RASCAL_evaluation.ipynb`` a Jupyter Notebook to plot and validate the reconstructions
   
   



