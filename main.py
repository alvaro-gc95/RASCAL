import multiprocessing
import itertools
import reconstruction.downloader as dw
import seaborn as sn
import os
import reconstruction.utils
import reconstruction.analogs
import reconstruction.skill_evaluation
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import autoval.climate
import autoval.utils
import autoval.statistics

initial_year = 1900
final_year = 2010
years = [str(y) for y in range(initial_year, final_year + 1)]

# Paths
data_path = '/home/alvaro/data/'

era20c_path = data_path + 'NWP/era20c/'
era5_path = data_path + 'NWP/era5/60.0W20.0E20.0N80.0N/'
observations_path = data_path + 'stations/rmpnsg/1h/'
aemet_path = data_path + 'stations/rmpnsg/1d/'

output_path = './output/'

# Station information
station = 'PN001003'

# Navacerrada stations coordinates
station_latitude = 40.793056
station_longitude = -4.010556

# Predictor domain
gph_min_lat = 33
gph_max_lat = 48
gph_min_lon = -15
gph_max_lon = 1.5

reanalysis_variables = {
    'TMPA': 'SURF_167',
    'PCNR': 'SURF_228',
    'WSPD': ['SURF_165', 'SURF_166']
}
# 'PCNR': '950_157', relative humidity at 950hPa
# 'PCNR': 'SURF_168.128', Dewpoint temperature

observed_variables = ['PCNR']

predictors_acronyms = {
    'TMPA': ['925_129'],
    'WSPD': ['925_129'],
    'PCNR': ['SURF_71.162', 'SURF_72.162']
}

analog_pool_size = 30
pondered_mean_sample_size = 3
similarity_method = 'percentiles'  # closest, pondered, percentiles, threshold

n_components = 4

if __name__ == '__main__':

    # Request reanalysis data
    inputs = dw.get_downloader_inputs()
    pool = multiprocessing.Pool()
    pool.map(dw.get_era20c, inputs)
    pool.close()
    pool.join()

    for observed_variable in observed_variables:

        # Name of the predictors to use
        predictor_variables = predictors_acronyms[observed_variable]

        # Open all data from a station
        daily_climatological_variables = reconstruction.utils.get_daily_stations(station, observed_variable)

        if observed_variable == 'PCNR' and similarity_method == 'threshold':
            daily_climatological_variables['RHMA'] = reconstruction.utils.get_daily_stations(station, 'RHMA')
            rh_thr = reconstruction.utils.get_humidity_to_precipitation(
                humidity=daily_climatological_variables['RHMA'],
                precipitation=daily_climatological_variables['PCNR']
            )

        # Dates to use for the PCA
        training_dates = pd.date_range(
            start=datetime.datetime(initial_year, 1, 1),
            end=datetime.datetime(final_year, 12, 31),
            freq='1D')

        # Dates with available data
        observed_dates = list(
            set(training_dates) &
            set(autoval.utils.clean_dataset(daily_climatological_variables).index)
        )

        # Testing dates
        test_dates = pd.date_range(
            start=datetime.datetime(initial_year, 1, 1),
            end=datetime.datetime(final_year, 12, 31),
            freq='1D')

        # Get the reanalysis predictor data
        predictors = reconstruction.utils.concatenate_reanalysis_data(
            era20c_path,
            predictor_variables,
            dates=years,
            latitude_limits=[gph_max_lat, gph_min_lat],
            longitude_limits=[gph_min_lon, gph_max_lon],
            grouping=None
        )

        # Get reanalysis series in the grid point
        secondary_predictors = reconstruction.utils.ReanalysisSeries(
            era20c_path,
            observed_variable,
            years,
            lat=station_latitude,
            lon=station_longitude
        )

        for similarity_method in ['percentiles']:

            # Data reconstruction
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
                predictor_anomalies = reconstruction.analogs.calculate_anomalies(seasonal_predictors, standardize=True)

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
                solver_name = (
                        './eofs/' +
                        season + '_'
                        + ' '.join(predictor_variables) + '_'
                        + str(initial_year) + str(final_year) + '.pkl'
                )
                solver = reconstruction.analogs.get_pca(predictor_anomalies['z'], solver_name, overwrite=False)
                pcs = solver.pcs(npcs=n_components, pcscaling=1)

                # Plot EOF maps
                # reconstruction.analogs.plot_pca(solver_name, n_components, vectorial=True)

                # Get the closest days in the PCs space to get an analog pool
                analog_distances, analog_dates = reconstruction.analogs.get_analog_pool(
                    training_set=pcs.sel(time=seasonal_observed_dates),
                    test_pcs=pcs.sel(time=seasonal_test_dates),
                    pool_size=analog_pool_size
                )

                # Reconstruct the season
                if observed_variable == 'TMPA':
                    reference_variable = seasonal_temperature
                elif observed_variable == 'PCNR' and similarity_method == 'threshold':
                    reference_variable = seasonal_humidity
                elif observed_variable == 'PCNR' and similarity_method != 'threshold':
                    reference_variable = seasonal_precipitation

                reconstructed_season = reconstruction.analogs.reconstruct(observed_data=daily_climatological_variables,
                                                                          analog_dates=analog_dates,
                                                                          similarity_method=similarity_method,
                                                                          sample_size=pondered_mean_sample_size,
                                                                          analog_distances=analog_distances,
                                                                          reference_variable=reference_variable,
                                                                          threshold=85)

                reconstructed_series.loc[seasonal_test_dates] = reconstructed_season

            reconstructed_series.to_csv(output_path +
                                        station + '_' +
                                        observed_variable +
                                        '_reconstruction_' + similarity_method + '_' +
                                        str(initial_year) + str(final_year) + '.csv')

            daily_climatological_variables.to_csv(output_path +
                                                  station + '_' +
                                                  observed_variable +
                                                  '_observations_' +
                                                  str(initial_year) + str(final_year) + '.csv')

            forecast = pd.read_csv(output_path +
                                   station + '_' +
                                   observed_variable +
                                   '_reconstruction_' + similarity_method + '_' +
                                   str(initial_year) + str(final_year) + '.csv', index_col=0)
            forecast.index = pd.to_datetime(forecast.index)

            observation = pd.read_csv(output_path +
                                      station + '_' +
                                      observed_variable +
                                      '_observations_' +
                                      str(initial_year) + str(final_year) + '.csv', index_col=0)
            observation.index = pd.to_datetime(observation.index)

            reconstruction_path = output_path + station + '_' + observed_variable

        # Evaluate reconstruction
        reconstruction.skill_evaluation.compare_method_skill(
            output_path + station + '_',
            observed_variable,
            ['closest', 'pondered', 'threshold'],
            initial_year, final_year
        )