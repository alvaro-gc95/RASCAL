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
import rascal.analogs
import rascal.statistics
import rascal.skill_evaluation

import pandas as pd

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
    'TMPA': 'SURF_167',
    'PCNR': 'SURF_228',
    'WSPD': ['SURF_165', 'SURF_166']
}
# 'PCNR': '950_157', relative humidity at 950hPa
# 'PCNR': 'SURF_168.128', Dewpoint temperature

observed_variables = ['TMPA']

predictors_acronyms = {
    'TMPA': ['925_129'],
    'WSPD': ['925_129'],
    'PCNR': ['SURF_71.162', 'SURF_72.162']
}
predictor_variables = predictors_acronyms[variable]

seasons = [
    [12, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11]
]


if __name__ == '__main__':

    # 1) Get historical record
    station = Station(path='./data/observations/' + station_code + '/')
    station_data = station.get_data(variable=variable)

    # 2) Get reanalysis data
    predictors = rascal.utils.get_predictor(
        path=era20c_path,
        variable_names=predictors_acronyms[variable],
        years=years,
        latitude_limits=[predictor_lat_min, predictor_lat_max],
        longitude_limits=[predictor_lon_min, predictor_lon_max],
        grouping=None
    )
    """
    secondary_predictors = rascal.utils.ReanalysisSeries(
        path=era20c_path,
        variable_names=variable,
        years=years,
        lat=station_latitude,
        lon=station_longitude
    )
    """

    # 3) Get Principal Components of the reanalysis data
    synoptic_predictor = Predictor(predictors)
    predictor_pcs = synoptic_predictor.pcs(
        npcs=config.get("n_components"),
        seasons=config.get("seasons"),
        standardize=config.get("standardize_anomalies"),
        pcscaling=config.get("pca_scaling")
    )

    # 4) Take an analog pool for each day to reconstruct, and select an analog for each similarity method
    pool_sizes = [10, 20, 30, 40, 50, 70, 100]
    methods = ["closest", "average"]
    sample_sizes = [2, 3, 5, 10]

    for pool_size in tqdm.tqdm(pool_sizes):
        for method in methods:
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




    exit()
    for observed_variable in observed_variables:

        # Name of the predictors to use
        predictor_variables = predictors_acronyms[observed_variable]

        # Open all data from a station
        daily_climatological_variables = rascal.utils.get_daily_stations(station_code.code, observed_variable)

        # Dates to use for the PCA
        training_dates = pd.date_range(
            start=datetime.datetime(training_start[0], training_start[1], training_start[2]),
            end=datetime.datetime(training_end[0], training_end[1], training_end[2]),
            freq='1D')

        # Dates with available data
        observed_dates = sorted(list(
            set(training_dates) &
            set(rascal.utils.clean_dataset(daily_climatological_variables).index)
        ))

        # Testing dates
        test_dates = pd.date_range(
            start=datetime.datetime(test_start[0], test_start[1], test_start[2]),
            end=datetime.datetime(test_end[0], test_end[1], test_end[2]),
            freq='1D')

        # Get the reanalysis predictor data
        predictors = rascal.utils.get_predictor(
            era20c_path,
            predictor_variables,
            years=years,
            latitude_limits=[predictor_lat_max, predictor_lat_min],
            longitude_limits=[predictor_lon_min, predictor_lon_max],
            grouping=None
        )

        # Get reanalysis series in the grid point
        secondary_predictors = rascal.utils.ReanalysisSeries(
            era20c_path,
            observed_variable,
            years,
            lat=station_latitude,
            lon=station_longitude
        )

        for similarity_method in config.get('similarity_methods'):

            # Data rascal
            min_band_columns = [c + ' min band' for c in daily_climatological_variables.columns]
            max_band_columns = [c + ' max band' for c in daily_climatological_variables.columns]
            reconstruction_columns = min_band_columns + max_band_columns + list(daily_climatological_variables.columns)
            reconstructed_series = pd.DataFrame(index=test_dates, columns=reconstruction_columns)

            # Divide the data by seasons
            for season, seasonal_predictors in predictors.groupby('time.season'):

                # Split in seasonal test and training dates
                seasonal_training_dates = pd.to_datetime(seasonal_predictors['time'].values)
                seasonal_test_dates = sorted(list(set(seasonal_training_dates) & set(test_dates)))
                seasonal_observed_dates = sorted(list(set(seasonal_training_dates) & set(observed_dates)))

                # Get seasonal variables
                predictor_anomalies = rascal.analogs.calculate_anomalies(
                    seasonal_predictors,
                    standardize=config.get('standardize_anomalies')
                )

                if secondary_predictors.precipitation is not None:
                    seasonal_precipitation = secondary_predictors.precipitation.loc[seasonal_test_dates]
                    seasonal_precipitation = seasonal_precipitation.iloc[:, 0]
                    seasonal_precipitation = seasonal_precipitation.rename('PCNR reanalysis')
                    seasonal_precipitation = seasonal_precipitation.to_frame()

                if secondary_predictors.temperature is not None:
                    seasonal_temperature = secondary_predictors.temperature.loc[seasonal_test_dates]

                if secondary_predictors.relative_humidity is not None:
                    seasonal_humidity = secondary_predictors.relative_humidity.loc[seasonal_test_dates]

                # Get PCs
                if not os.path.exists('./pca/'):
                    os.makedirs('./pca/')

                # Create a new dir
                solver_name = (
                        './pca/' +
                        season + '_'
                        + ' '.join(predictor_variables) + '_'
                        + str(initial_year) + str(final_year) + '.pkl'
                )
                solver = rascal.analogs.get_pca_solver(predictor_anomalies['z'], solver_name, overwrite=False)
                pcs = solver.pcs(npcs=config.get('n_components'), pcscaling=config.get('pca_scaling'))

                # Plot EOF maps
                # rascal.analogs.plot_pca(solver_name, n_components, vectorial=True)

                # Get the closest days in the PCs space to get an analog pool
                analog_distances, analog_dates = rascal.analogs.get_analog_pool(
                    training_set=pcs.sel(time=seasonal_observed_dates),
                    test_pcs=pcs.sel(time=seasonal_test_dates),
                    pool_size=config.get('analog_pool_size')
                )

                # Reconstruct the season
                if observed_variable == 'TMPA':
                    reference_variable = seasonal_temperature
                elif observed_variable == 'PCNR' and similarity_method == 'threshold':
                    reference_variable = seasonal_humidity
                elif observed_variable == 'PCNR' and similarity_method != 'threshold':
                    reference_variable = seasonal_precipitation

                reconstructed_season = rascal.analogs.reconstruct_by_analogs(
                    observed_data=daily_climatological_variables,
                    analog_dates=analog_dates,
                    similarity_method=similarity_method,
                    sample_size=config.get('weighted_mean_sample_size'),
                    analog_distances=analog_distances,
                    reference_variable=reference_variable
                )

                reconstructed_series.loc[seasonal_test_dates] = reconstructed_season

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            reconstructed_series.to_csv(output_path +
                                        station_code.code + '_' +
                                        observed_variable +
                                        '_reconstruction_' + similarity_method + '_' +
                                        str(initial_year) + str(final_year) + '.csv')

            daily_climatological_variables.to_csv(output_path +
                                                  station_code.code + '_' +
                                                  observed_variable +
                                                  '_observations_' +
                                                  str(initial_year) + str(final_year) + '.csv')
            print(output_path +
                  station_code.code + '_' +
                  observed_variable +
                  '_observations_' +
                  str(initial_year) + str(final_year) + '.csv')
        # Evaluate rascal
        rascal.skill_evaluation.compare_method_skill(
            output_path + station_code.code + '_',
            observed_variable,
            config.get('similarity_methods'),
            initial_year, final_year
        )
