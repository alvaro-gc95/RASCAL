"""
Reconstruct time series

contact: alvaro@intermet.es
"""

import antiser.analogs
import antiser.utils
import autoval.utils
import pandas as pd
import datetime
import os

config = antiser.utils.open_yaml('config.yaml')

training_start = config.get('training_start')
training_end = config.get('training_end')

test_start = config.get('test_start')
test_end = config.get('test_end')


class Reconstructor:
    def __init__(self, pandas_obj, station, variable_name, reanalysis_dataset, method):
        self._original = pandas_obj
        # self._reconstruction =
        if method == 'analogs':
            reconstructed_data = reconstruct_by_analogs(station, variable_name, reanalysis_dataset)

        self._reconstruction = reconstructed_data


def split_dates(training_start, training_end, test_start, test_end, observed_dates, freq='1D'):
    """
    Make DatetimeIndex of the training dates, testing dates, and available observed dates during the training period
    :param training_start: list. (Year, month, day)
    :param training_end:  list. (Year, month, day)
    :param test_start:  list. (Year, month, day)
    :param test_end:  list. (Year, month, day)
    :param observed_dates: DatetimeIndex.
    :param freq: str. Default='1D'.
    """
    # Get dates
    training_dates = pd.date_range(
        start=datetime.datetime(training_start[0], training_start[1], training_start[2]),
        end=datetime.datetime(training_end[0], training_end[1], training_end[2]),
        freq=freq
    )
    test_dates = pd.date_range(
        start=datetime.datetime(test_start[0], test_start[1], test_start[2]),
        end=datetime.datetime(test_end[0], test_end[1], test_end[2]),
        freq=freq
    )
    observed_dates = sorted(list(
        set(training_dates) &
        set(observed_dates)
    ))

    return training_dates, test_dates, observed_dates


def reconstruct_by_analogs(station, variable_name, reanalysis_dataset, similarity_method):

    predictor_variables = config.get('predictor_of_variable')[variable_name]

    observed_daily_data = antiser.utils.get_daily_stations(station, variable_name)

    antiser.utils.get_station(station)

    training_dates, test_dates, observed_dates = split_dates(
        training_start,
        training_end,
        test_start,
        test_end,
        autoval.utils.clean_dataset(observed_daily_data).index
    )

    # Get the reanalysis predictor data
    predictors = antiser.utils.concatenate_reanalysis_data(
        era20c_path,
        predictor_variables,
        dates=years,
        latitude_limits=[gph_max_lat, gph_min_lat],
        longitude_limits=[gph_min_lon, gph_max_lon],
        grouping=None
    )

    # Get reanalysis series in the grid point
    secondary_predictors = antiser.utils.ReanalysisSeries(
        era20c_path,
        observed_variable,
        years,
        lat=station_latitude,
        lon=station_longitude
    )

    # Data antiser
    min_band_columns = [c + ' min band' for c in observed_daily_data.columns]
    max_band_columns = [c + ' max band' for c in observed_daily_data.columns]
    reconstruction_columns = min_band_columns + max_band_columns + list(observed_daily_data.columns)
    reconstructed_series = pd.DataFrame(index=test_dates, columns=reconstruction_columns)

    # Divide the data by seasons
    for season, seasonal_predictors in predictors.groupby('time.season'):

        # Split in seasonal test and training dates
        seasonal_training_dates = pd.to_datetime(seasonal_predictors['time'].values)
        seasonal_test_dates = sorted(list(set(seasonal_training_dates) & set(test_dates)))
        seasonal_observed_dates = sorted(list(set(seasonal_training_dates) & set(observed_dates)))

        # Get seasonal variables
        predictor_anomalies = antiser.analogs.calculate_anomalies(
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
        solver = antiser.analogs.get_pca(predictor_anomalies['z'], solver_name, overwrite=False)
        pcs = solver.pcs(npcs=config.get('n_components'), pcscaling=config.get('pca_scaling'))

        # Plot EOF maps
        # antiser.analogs.plot_pca(solver_name, n_components, vectorial=True)

        # Get the closest days in the PCs space to get an analog pool
        analog_distances, analog_dates = antiser.analogs.get_analog_pool(
            training_set=pcs.sel(time=seasonal_observed_dates),
            test_pcs=pcs.sel(time=seasonal_test_dates),
            pool_size=config.get('analog_pool_size')
        )

        # Reconstruct the season
        if variable_name == 'TMPA':
            reference_variable = seasonal_temperature
        elif variable_name == 'PCNR' and similarity_method == 'threshold':
            reference_variable = seasonal_humidity
        elif variable_name == 'PCNR' and similarity_method != 'threshold':
            reference_variable = seasonal_precipitation

        reconstructed_season = antiser.analogs.reconstruct_by_analogs(
            observed_data=observed_daily_data,
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
                                    station + '_' +
                                    observed_variable +
                                    '_reconstruction_' + similarity_method + '_' +
                                    str(initial_year) + str(final_year) + '.csv')

        observed_daily_data.to_csv(output_path +
                                   station + '_' +
                                   observed_variable +
                                   '_observations_' +
                                   str(initial_year) + str(final_year) + '.csv')
