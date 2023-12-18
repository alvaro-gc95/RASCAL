"""
########################################################################################################################
# ------------------------ RASCAL (Reconstruction by AnalogS of ClimatologicAL time series) -------------------------- #
########################################################################################################################
Version 1.0
Contact: alvaro@intermet.es
"""

import os
import rascal.utils
import rascal.analogs
import rascal.skill_evaluation
import datetime
import pandas as pd
import rascal.statistics


config = rascal.utils.open_yaml('config.yaml')

initial_year = config.get('initial_year')
final_year = config.get('final_year')
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

observed_variables = ['TMPA']

predictors_acronyms = {
    'TMPA': ['925_129'],
    'WSPD': ['925_129'],
    'PCNR': ['SURF_71.162', 'SURF_72.162']
}

training_start = config.get('training_start')
training_end = config.get('training_end')

test_start = config.get('test_start')
test_end = config.get('test_end')

if __name__ == '__main__':

    # dw.request_reanalysis(dataset='era20c', parallelize=True)

    station = rascal.utils.get_station_meta(station)

    for observed_variable in observed_variables:

        # Name of the predictors to use
        predictor_variables = predictors_acronyms[observed_variable]

        # Open all data from a station
        daily_climatological_variables = rascal.utils.get_daily_stations(station.code, observed_variable)

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
            latitude_limits=[gph_max_lat, gph_min_lat],
            longitude_limits=[gph_min_lon, gph_max_lon],
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
                                        station.code + '_' +
                                        observed_variable +
                                        '_reconstruction_' + similarity_method + '_' +
                                        str(initial_year) + str(final_year) + '.csv')

            daily_climatological_variables.to_csv(output_path +
                                                  station.code + '_' +
                                                  observed_variable +
                                                  '_observations_' +
                                                  str(initial_year) + str(final_year) + '.csv')
            print(output_path +
                                                  station.code + '_' +
                                                  observed_variable +
                                                  '_observations_' +
                                                  str(initial_year) + str(final_year) + '.csv')
        # Evaluate rascal
        rascal.skill_evaluation.compare_method_skill(
            output_path + station.code + '_',
            observed_variable,
            config.get('similarity_methods'),
            initial_year, final_year
        )
