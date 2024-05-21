Installation
============

Via PyPi
------------

RASCAL can be installed via PyPi:

.. code-block:: console

   (.venv) $ pip install rascal-ties
   
Via GitHub
------------

RASCAL can be used via GitHub:

.. code-block:: console

   (.venv) $ git clone https://github.com/alvaro-gc95/RASCAL
   
The GitHub repository also contains the following scripts:

   - ``multiple_runs_example.py`` to automatize running several configurations of similarity methods and pool sizes for various stations and variables. 
     This can be configured through the ``config.yaml`` file 
   - ``RASCAL_evaluation.ipynb`` a Jupyter Notebook to plot and validate the reconstructions
   
   
Dependencies
-------------

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


