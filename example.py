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
import rascal.utils
import rascal.statistics
import rascal.skill_evaluation

import pandas as pd
import matplotlib.pyplot as plt

from rascal.utils import Station
from rascal.analogs import Predictor, Analogs

config = rascal.utils.open_yaml('config.yaml')

initial_year = config.get('initial_year')
final_year = config.get('final_year')
years = [str(y) for y in range(initial_year, final_year + 1)]

# Variable to reconstruct
variable = 'TMPA'

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

# Paths
data_path = '/home/alvaro/data/'
era20c_path = data_path + 'NWP/era20c/'
era5_path = data_path + 'NWP/era5/60.0W20.0E20.0N80.0N/'
observations_path = data_path + 'stations/rmpnsg/1h/'
aemet_path = data_path + 'stations/rmpnsg/1d/'
output_path = './output/'

# Station information
station_code = 'PN001003'

# Navacerrada stations coordinates
station_latitude = 40.793056
station_longitude = -4.010556

# Predictor domain
predictor_lat_min = config.get("predictor_lat_min")
predictor_lat_max = config.get("predictor_lat_max")
predictor_lon_min = config.get("predictor_lon_min")
predictor_lon_max = config.get("predictor_lon_max")

reanalysis_variables = {
    'TMPA': ['SURF_167'],
    'PCNR': ['SURF_228'],
    'WSPD': ['SURF_165', 'SURF_166']
}
# 'PCNR': '950_157', relative humidity at 950hPa
# 'PCNR': 'SURF_168.128', Dewpoint temperature

predictors_for_variable = {
    'TMPA': ['925_129'],
    'WSPD': ['925_129'],
    'PCNR': ['SURF_71.162', 'SURF_72.162']
}
secondary_predictor_for_variable = {
    'TMPA': ['SURF_167'],
    'WSPD': ['SURF_165', 'SURF_166'],
    'PCNR': ['SURF_228']
}
predictor_variables = predictors_for_variable[variable]
secondary_predictor_variables = secondary_predictor_for_variable[variable]

seasons = [
    [12, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11]
]
similarity_methods = ["quantilemap"]
pool_sizes = [10, 20, 30, 40, 50, 70, 100]
sample_sizes = [2, 3, 5, 10]

if __name__ == '__main__':

    # 1) Get historical record
    print("1) Get historical record")
    station = Station(path='./data/observations/' + station_code + '/')
    station_data = station.get_data(variable=variable)

    # 2) Get reanalysis data
    print("2) Get reanalysis data")
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
        lon_max=predictor_lon_max
    )

    if "quantilemap" in similarity_methods:
        secondary_predictor_files = rascal.utils.get_files(
            nwp_path=era20c_path,
            variables=secondary_predictor_variables,
            dates=years,
            file_format=".grib")
        secondary_predictors = Predictor(
            paths=secondary_predictor_files,
            grouping="12hour_1D_mean",
            lat_min=station.latitude,
            lat_max=station.latitude,
            lon_min=station.longitude,
            lon_max=station.longitude,
            mosaic=False
        )

    # 3) Get Principal Components of the reanalysis data
    print("3) Get Principal Components of the reanalysis data")
    predictor_pcs = predictors.pcs(
        npcs=config.get("n_components"),
        seasons=config.get("seasons"),
        standardize=config.get("standardize_anomalies"),
        pcscaling=config.get("pca_scaling"),
        overwrite=False
    )

    # 4) Take an analog pool for each day to reconstruct, and select an analog for each similarity method
    print("4) Take an analog pool for each day to reconstruct, and select an analog for each similarity method")
    for pool_size in tqdm.tqdm(pool_sizes):
        for method in similarity_methods:
            if method == "average":
                for sample_size in sample_sizes:
                    analogs = Analogs(pcs=predictor_pcs, observations=station_data, dates=test_dates)
                    reconstruction = analogs.reconstruct(
                        pool_size=pool_size,
                        method=method,
                        sample_size=sample_size
                    )

                    reconstruction_filename = '_'.join([
                        station_code,
                        variable,
                        str(pool_size).zfill(3),
                        method + str(sample_size).zfill(2),
                    ])
                    reconstruction.to_csv("./output/" + reconstruction_filename + ".csv")
                    print(" Saved: " + reconstruction_filename)

            if method == "quantilemap":
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

    # 5) Analyze the skill pof the reconstruction

    # Evaluate rascal
    # rascal.skill_evaluation.compare_method_skill(
    #        output_path + station_code.code + '_',
    #         observed_variable,
    #         config.get('similarity_methods'),
    #         initial_year, final_year
    #     )
