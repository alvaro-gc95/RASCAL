# ReconstructorAnalog Time Series Reconstructor (RASCAL)
ANATISER is a python library desinged to reconstruct time series of meteorological data, based on the Analog Method (AM), to use them for climate studies. The AM is a statistical downscalling method, based on the assumption that large-scale atmospheric conditions tend to produce similar local weather patterns, and therefore is possible to predict local conditions finding analog days, with similar large-scale patterns, in the historical record. 
The objective of ANATISER is to generate complete time series, based on limited observational data, that can reproduce better than the reanalysis products, the climatic characteristics of the region to study.

### Requirements
To run this library renalaysis and observational data is required. the reanalysis data should cover the whole period to be reconstructed, and should have at least one predictor variable.The observational data temporal cover must overlap with the reanalysis data.

The choice of the predictor variable is flexible. However, if you want to reconstruct a long time series, it's important to consider that the connection between the predictor and the predicted variable should be very robust. This is because certain relationships may change in a changing climate scenario.

To run ANATISER, these other python libraries are required:
- numpy
- pandas
- dask
- xarray

### References
Pending of publication. González-Cervera, A., Durán, L. (2024), A python tool to reconstruct climatic time series, Journal, DOI
