"""
########################################################################################################################
# ------------------------ RASCAL (Reconstruction by AnalogS of ClimatologicAL time series) -------------------------- #
########################################################################################################################
Version 1.0
Contact: alvaro@intermet.es

This is an example of how to use RASCAL step by step.
This example contains the reconstruction of the temperature of a station situated in Sierra de Guadarrama, Spain.
"""

import os
import tqdm
import datetime
import itertools
import rascal.utils
import rascal.statistics
import rascal.skill_evaluation

import pandas as pd
import matplotlib.pyplot as plt

from rascal.utils import Station
from rascal.analogs import Predictor, Analogs

config = rascal.utils.open_yaml('config.yaml')

"""
System Parameters
"""

# Paths
data_path = '/home/alvaro/data/'
era20c_path = data_path + 'NWP/era20c/'
era5_path = data_path + 'NWP/era5/60.0W20.0E20.0N80.0N/'
observations_path = data_path + 'stations/rmpnsg/1h/'
aemet_path = data_path + 'stations/rmpnsg/1d/'
output_path = './output/'


"""
Reconstruction Parameters
"""

initial_year = config.get('initial_year')
final_year = config.get('final_year')
years = [str(y) for y in range(initial_year, final_year + 1)]

# Variable to reconstruct
variables = config.get("variables")

# Training period
training_start = config.get('training_start')
training_end = config.get('training_end')
training_dates = pd.date_range(
    start=datetime.datetime(training_start[0], training_start[1], training_start[2]),
    end=datetime.datetime(training_end[0], training_end[1], training_end[2]),
    freq='1D'
)

# Reconstruct period
test_start = config.get('test_start')
test_end = config.get('test_end')
test_dates = pd.date_range(
    start=datetime.datetime(test_start[0], test_start[1], test_start[2]),
    end=datetime.datetime(test_end[0], test_end[1], test_end[2]),
    freq='1D'
)

# Station information
stations = config.get("stations")


"""
Predictor Parameters
"""

# Predictor domain
predictor_lat_min = config.get("predictor_lat_min")
predictor_lat_max = config.get("predictor_lat_max")
predictor_lon_min = config.get("predictor_lon_min")
predictor_lon_max = config.get("predictor_lon_max")

predictors_for_variable = config.get("predictor_for_variable")
secondary_predictor_for_variable = config.get("secondary_predictor_for_variable")

# 'PCNR': '950_157', relative humidity at 950hPa
# 'PCNR': 'SURF_168.128', Dewpoint temperature

grouping_per_variable = {
    "TMEAN": "1D_mean",
    "TMAX": "1D_max",
    "TMIN": "1D_min",
    "PCNR": "1D_sum"
}

"""
Principal Component Analysis Parameters
"""

seasons = config.get("seasons")
pca_scaling = config.get("pca_scaling")
n_components = config.get("n_components")
standardize_anomalies = config.get("standardize_anomalies")

"""
Analog Method Parameters
"""

similarity_methods = config.get("similarity_methods")
pool_sizes = config.get("analog_pool_size")
sample_sizes = config.get("weighted_mean_sample_size")

if __name__ == '__main__':

    for station_code, variable in itertools.product(stations, variables):

        # 1) Get historical record

        station = Station(path='./data/observations/' + station_code + '/')
        station_data = station.get_data(variable=variable)

        # 2) Get reanalysis data

        predictor_variables = predictors_for_variable[variable]
        secondary_predictor_variables = secondary_predictor_for_variable[variable]

        predictor_files = rascal.utils.get_files(
            nwp_path=era20c_path,
            variables=predictor_variables,
            dates=years,
            file_format=".grib")
        predictors = Predictor(
            paths=predictor_files,
            grouping="12hour_1D_mean",
            lat_min=predictor_lat_min,
            lat_max=predictor_lat_max,
            lon_min=predictor_lon_min,
            lon_max=predictor_lon_max,
            mosaic=True,
            number=None
        )

        if "quantilemap" in similarity_methods:

            if variable == 'PCNR':
                ensemble_member = 0
            else:
                ensemble_member = None

            secondary_predictor_files = rascal.utils.get_files(
                nwp_path=era20c_path,
                variables=secondary_predictor_variables,
                dates=years,
                file_format=".grib")
            secondary_predictors = Predictor(
                paths=secondary_predictor_files,
                grouping=grouping_per_variable[variable],
                lat_min=station.latitude,
                lat_max=station.latitude,
                lon_min=station.longitude,
                lon_max=station.longitude,
                mosaic=False,
                number=ensemble_member
            )

        # 3) Get Principal Components of the reanalysis data

        predictor_pcs = predictors.pcs(
            npcs=n_components,
            seasons=seasons,
            standardize=standardize_anomalies,
            pcscaling=pca_scaling,
            overwrite=False
        )

        # 4) Take an analog pool for each day to reconstruct, and select an analog for each similarity method

        for pool_size in pool_sizes:
            for method in similarity_methods:
                if method == "average":
                    for sample_size in sample_sizes:
                        analogs = Analogs(pcs=predictor_pcs, observations=station_data, dates=test_dates)
                        reconstruction = analogs.reconstruct(
                            pool_size=pool_size,
                            method=method,
                            sample_size=sample_size,
                        )

                        reconstruction_filename = '_'.join([
                            station_code,
                            variable,
                            str(pool_size).zfill(3),
                            method + str(sample_size).zfill(2),
                        ])
                        reconstruction.to_csv("./output/" + reconstruction_filename + ".csv")
                        print(" Saved: " + reconstruction_filename)

                elif method == "quantilemap":
                    analogs = Analogs(pcs=predictor_pcs, observations=station_data, dates=test_dates)
                    reconstruction = analogs.reconstruct(
                        pool_size=pool_size,
                        method=method,
                        reference_variable=secondary_predictors
                    )

                    reconstruction_filename = '_'.join([
                        station_code,
                        variable,
                        str(pool_size).zfill(3),
                        method,
                    ])
                    reconstruction.to_csv("./output/" + reconstruction_filename + ".csv")
                    print(" Saved: " + reconstruction_filename)

                else:
                    analogs = Analogs(pcs=predictor_pcs, observations=station_data, dates=test_dates)
                    reconstruction = analogs.reconstruct(
                        pool_size=pool_size,
                        method=method,
                    )

                    reconstruction_filename = '_'.join([
                        station_code,
                        variable,
                        str(pool_size).zfill(3),
                        method,
                    ])
                    reconstruction.to_csv("./output/" + reconstruction_filename + ".csv")
                    print(" Saved: " + reconstruction_filename)



