# Reconstruction by AnalogS of ClimatologicAL time series (RASCAL)
RASCAL is a python library desinged to reconstruct time series of climatological data, based on the Analog Method (AM), to use them for climate studies. The AM is a statistical downscalling method, based on the assumption that large-scale atmospheric conditions tend to produce similar local weather patterns, and therefore is possible to predict local conditions finding analog days, with similar large-scale patterns, in the historical record. 
The objective of RASCAL is to generate complete time series, based on limited observational data, that can reproduce the climatic characteristics of the region to study better than the reanalysis products.

## Requirements
To run this library renalaysis and observational data is required. the reanalysis data should cover the whole period to be reconstructed, and should have at least one predictor variable.The observational data temporal cover must overlap with the reanalysis data.

The choice of the predictor variable is flexible. However, if you want to reconstruct a long time series, it's important to consider that the connection between the predictor and the predicted variable should be very robust. This is because certain relationships may change in a changing climate scenario.

RASCAL is based in python 3.10. To run RASCAL, these other python libraries are required:

- numpy 1.26.4
- pandas 2.2.1
- dask 2024.4.1
- xarray 2024.3.0
- scipy 1.13.0
- tqdm 4.65.0
- scikit-learn 1.4.1.post1
- seaborn 0.13.2
- eofs 1.4.1

## Getting Started

RASCAL can be installed through PyPi

```
pip install rascal-ties
```
or using the files in _rascal/_ directory inside this repository


## How to use

RASCAL is based in four main clases: Station, Predictor, Analogs and Rskill

```python
import rascal
```

### 1) Get observational data
To load the observational data (in daily or sub-daily resolution) and the station metadata, the data is loaded from a CSV file with the same name as the desired variable, and a meta.csv file containing the name, code, altitude, longitude and latitude of the station

```python
station = rascal.analogs.Station(path='./data/observations/station/')
station_data = station.get_data(variable='PCP')
```

### 2) Load and process predictor fields from large-scale models
To load the reanalysis or large-scale model data we use the Predictor class. This example shows how to use the Total Column of Water Vapor Flux from the ERA20C reanalysis. In this reanalysis the components U and V of the TCWVF are named '71.162' and '72.162'. The predictor is set or the years 1900-1910, for each day only the 12:00 is selected through the _grouping_ argument, the domain is 80ºN-20ºN, 60ºW-20ºE. The _mosaic_ argument set to ___True___ concatenates both components U and V in the longitude axis to obtain a single compound variable of size _(time x 2*longitude x latitude)_:

```python
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

```
### 3) Perform Principal Component Analysis on the predictor fields
The Principal Component Analysis (PCA) of the compund variable standardized anomalies, with 4 principal components and for the conventionan seasons DJF, MAM, JJA, and SON,  is conducted as follows:
```python
predictor_pcs = predictors.pcs(
    npcs=n_components,
    seasons=[[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
    standardize=True,
    path="./tmp/"
)
```

### 4) Look at the PC space to find analog days in the historical data
After performing the PCA, the obtained values of the principal componets act as the predictor used to perform the reconstructions. First the analog days, in order of euclidean distance, are found.

```python
analogs = rascal.analogs.Analogs(pcs=predictor_pcs, observations=station_data, dates=test_dates)
```

### 5) Reconstruct or extend missing observational data 
Later, the reconstuctions are made using one of the following similarity methods: _closest_, _average_, or _quantilemap_.

```python
reconstruction = analogs.reconstruct(
    pool_size=30,
    method='closest',
)
```
### 6) Evaluate the reconstructions based on your scientific goals
The evaluation of the reconstructions is made with the RSkill class. The Jupyter Notebook _'RASCAL_evaluation'_ contains examples of applications

## References
- Pending of publication. González-Cervera, A., Durán, L. (2024), RASCAL v1.0.0: An Open Source Tool for Climatological Time Series Reconstruction, Extension and Validation. https://egusphere.copernicus.org/preprints/2024/egusphere-2024-958/

- Zenodo: Gonzalez-Cervera. (2024). alvaro-gc95/RASCAL: RASCALv1.0.0 (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.10592595


