# Analog Time Series Reconstructor (ANATISER)
ANATISER is a python library desinged to reconstruct meteorological data, based on the Analog Method (AM). The AM has been widely used for statistical downscalling, it establish relationships between large-scale predictor variables taken from the reanalysis and a local-scale predictand variable taken from the observational period.
The analogs method is based on the assumption that large-scale atmospheric conditions tend to produce similar local weather patterns, and therefore is possible to predict local conditions finding analog days, with similar large-scale patterns, in the historical record. 
# Installation
In order to use the full functionalities of the model is recommenden to use the conda environment of xESMF: Universal Regridder for Geospatial Data. This is a powerful python package to regrid geospatial data, and its used in the downscaling process. The tutorial for the installation of the environment can be found [here](https://xesmf.readthedocs.io/en/latest/installation.html).

For full functionality, install these other required python libraries:
```
conda install -c conda-forge numpy matplotlib dask PyYAML pygrib xarray tqdm psycopg2 cdsapi

# Conda comands do not owrk in the installation of sklearn
pip install sklearn
```
