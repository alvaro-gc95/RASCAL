# Reconstruction by AnalogS of ClimatologicAL time series (RASCAL)

[![docs](https://readthedocs.org/projects/rascalv100/badge/)](https://rascalv100.readthedocs.io)
[![GMD](https://img.shields.io/badge/PyPi-install-blue)](https://pypi.org/project/rascal-ties/)
[![GMD](https://img.shields.io/badge/GMD-preprint-orange)](https://egusphere.copernicus.org/preprints/2024/egusphere-2024-958/)


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

## Documentation

For a more detalied documentation and tutorials check [RASCAL ReadTheDocs](https://rascalv100.readthedocs.io). 

## Getting Started

RASCAL can be installed through PyPi. It is recommemded to create a virtual environment first.

```
conda create --name rascal_env python==3.10
conda activate rascal_env
python3 -m pip install rascal-ties
```

## How to use

RASCAL is a library based in four main clases: Station, Predictor, Analogs and Rskill, and an additional class CIndex, that allows to calculate relevant climatic indices

To run RASCAl as a python library, you can refer to the tutorial in the documentation: [Make your first reconstruction](https://rascalv100.readthedocs.io/en/latest/begginer.html).

This repository contains a the script **multiple_runs_example.py**, where all the neccesary steps to make reconstructions are already programmed, allowing to make lots of different reconstructions for different stations, variables, analog pool sizes, and similarity methods, only modifying the configuration file **config.yaml** and running:

```python
python3 multiple_runs_example.py
```

To validate and plot the results, and compare its skill to the observations and reference reanalysis, you can use the Jupyter Notebook *RASCAL_evaluation.ipynb*

## References
- Pending of publication. González-Cervera, A., Durán, L. (2024), RASCAL v1.0: An Open Source Tool for Climatological Time Series Reconstruction and Extension. https://egusphere.copernicus.org/preprints/2024/egusphere-2024-958/

- Zenodo: Gonzalez-Cervera. (2024). alvaro-gc95/RASCAL: RASCALv1.0.9 (v1.0.9). Zenodo. https://doi.org/10.5281/zenodo.10592595


