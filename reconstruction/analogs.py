"""
Reconstruct meteorological data through the "analog method"

contact: alvaro@intermet.es
"""

import os
import pickle

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from eofs.xarray import Eof
from scipy.stats import percentileofscore

import reconstruction.utils

config = reconstruction.utils.open_yaml('config.yaml')


def calculate_anomalies(data_array: xr.DataArray, standardize=False):
    """
    :param data_array: DataArray.
    :param standardize: bool. Default=False. If True divide the anomalies by its standard deviation.
    :return anomalies: DataArray.
    """

    mean = data_array.mean(dim='time')
    anomalies = data_array - mean

    if standardize:
        anomalies = anomalies / anomalies.std(dim='time')

    return anomalies


def get_pca(data_array, file_name, overwrite=True):
    """
    Do Principal Components Analysis and save the solver as object
    :param data_array: DataArray. Field to analyze.
    :param file_name: str. Name of the pickle file.
    :param overwrite: bool. Default=True. Overwrite pickle file if exists.
    :return solver: obj. EoF solver.
    """

    if not os.path.isfile(file_name) or overwrite:
        # Principal Components Analysis
        solver = Eof(data_array, center=False)
        # Save the solver object
        reconstruction.utils.save_object(solver, file_name)
    else:
        # Open solver object
        with open(file_name, 'rb') as inp:
            solver = pickle.load(inp)

    return solver


def plot_pca(file_name, n_components, vectorial=False):
    """
    Plot maps of the EOFs
    :param file_name: str. Name of the solver file.
    :param n_components: int. Number of components to represent.
    :param vectorial: bool. If True represent the EOFs as a contour of the module and quiver plot fot direction.
    """

    # Open solver object
    with open(file_name, 'rb') as inp:
        solver = pickle.load(inp)

    # This is the map projection we want to plot *onto*
    map_proj = ccrs.PlateCarree()

    # EOF maps
    eofs = solver.eofs(neofs=n_components)

    if vectorial:
        # Separate the concatenated vectorial variable in u, v and module
        eofs = reconstruction.utils.separate_concatenated_components(eofs)

        for mode in eofs['mode'].values:
            # Defining the figure
            fig = plt.figure(figsize=(4, 4), facecolor='w',
                             edgecolor='k')

            # Axes with Cartopy projection
            ax = plt.axes(projection=ccrs.PlateCarree())
            p = eofs['module'].sel(mode=mode).plot.contourf(transform=ccrs.PlateCarree(), levels=20)

            # Defining the quiver plot
            quiver = eofs.sel(mode=mode).plot.quiver(x='longitude', y='latitude', u='u', v='v',
                                                     transform=ccrs.PlateCarree(), scale=1)

            # # Vector options declaration
            veclenght = 0.05
            maxstr = '%3.1f kg m-1 s-1' % veclenght
            ax.quiverkey(quiver, -0.1, -0.1, veclenght, maxstr, labelpos='S', coordinates='axes')

            ax.coastlines()
            explained_variance_ratio = solver.varianceFraction(n_components).sel(mode=mode).values
            ax.set_title(
                'Mode ' + str(mode + 1) + ' (exp. var. = ' + str(round(explained_variance_ratio * 100, 2)) + '%)')

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.1, color='k', alpha=1,
                              linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

            plt.savefig(file_name[:-3] + '_mode' + str(mode + 1) + '.png')

    else:
        p = eofs.plot.contourf(transform=ccrs.PlateCarree(),  # the data's projection
                               col='mode', col_wrap=1,  # multiplot settings
                               subplot_kws={'projection': map_proj},
                               levels=20)  # the plot's projection

        # We have to set the map's options on all four axes
        for i, ax in enumerate(p.axes.flat):
            ax.coastlines()
            explained_variance_ratio = solver.varianceFraction(n_components).sel(mode=i).values
            ax.set_title('Mode ' + str(i + 1) + ' (exp. var. = ' + str(round(explained_variance_ratio * 100, 2)) + '%)')

        plt.savefig(file_name[:-3] + '.png')


def calculate_distances(origin, points, distance='euclidean'):
    """
    Calculate distance between one origin point and a set of points.
    :param origin: DataArray.
    :param points: DataArray.
    :param distance: str. Type of distance to calculate.
    :return distances: DataArray.
    """

    if distance == 'euclidean':
        distances = np.sqrt(((origin - points) ** 2).sum(dim='mode'))
    else:
        raise AttributeError('Error: ' + distance + ' distance does not exist')

    return distances


def get_analog_pool(training_set, test_pcs, pool_size=100):
    """
    Get a pool of analogues calculating the N closest neighbours in the PCs space.
    :param training_set: DataArray. PCs of possible analogues.
    :param test_pcs: DataArray. PCs of the day to reconstruct.
    :param pool_size: int. N number of analogues.
    :return analog_distances: DataFrame. Distances to the day to reconstruct of the N closes analogues.
    :return analog_dates: DataFrame. Dates of the N closest analogues.
    """

    analog_dates = pd.DataFrame(index=pd.to_datetime(test_pcs['time'].values), columns=range(pool_size))
    analog_distances = pd.DataFrame(index=pd.to_datetime(test_pcs['time'].values), columns=range(pool_size))

    training_dates = pd.to_datetime(training_set['time'].values)

    for date in test_pcs['time'].values:
        # Delete values close the date to reconstruct
        validation_window = reconstruction.utils.get_validation_window(
            test_date=pd.to_datetime(date),
            dates=pd.to_datetime(test_pcs['time'].values),
            window_size=config.get('validation_window_size'),
            window_type=config.get('validation_window_type')
        )
        validation_window = pd.to_datetime(validation_window)
        validation_dates = sorted(list(set(training_dates) - set(validation_window)))

        # Find distances in the PC space to the point to predict
        distances = calculate_distances(
            origin=test_pcs.sel(time=date),
            points=training_set.sel(time=validation_dates),
            distance=config.get('distance')
        )

        # Sort the distances to find the closest days in the PC space
        distances = distances.sortby(distances)

        # Get the pool of closest days
        distances = distances.isel(time=slice(0, pool_size))

        # Pool of dates and distances
        analog_dates.loc[date] = distances['time'].values
        analog_distances.loc[date] = distances.values

    return analog_distances, analog_dates


def reconstruct_by_analogs(observed_data, analog_dates, similarity_method='closest', **kwargs):
    """
    Reconstruct time series
    :param observed_data: pd.DataFrame. All observations.
    :param analog_dates: pd.DataFrame. Dates in the analog pool for each date to reconstruct.
    :param similarity_method: str. Reconstruction method. Options = ('closet', 'pondered', 'percentile')
    :param kwargs:
    :return:
    """

    # Columns for maximum and minimum values in the analog pool
    min_band_columns = [c + ' min band' for c in observed_data.columns]
    max_band_columns = [c + ' max band' for c in observed_data.columns]

    # Create the reconstruction empty dataframe
    reconstruction_columns = min_band_columns + max_band_columns + list(observed_data.columns)
    reconstructed_data = pd.DataFrame(index=analog_dates.index, columns=reconstruction_columns)

    for variable in observed_data.columns:

        # Pool of reconstructed values from the analog pool ordered by closeness in the PCs space
        reconstructed_pool = analog_dates.copy()
        reconstructed_pool = reconstructed_pool.apply(lambda x: observed_data[variable].loc[x].values)

        if similarity_method == 'closest':
            reconstruction_series, reconstruction_min_band, reconstruction_max_band = get_closest_neighbor(
                analog_pool=reconstructed_pool
            )

        elif similarity_method == 'average':
            if 'sample_size' not in kwargs.keys():
                raise AttributeError('Missing argument: sample_size')
            elif 'analog_distances' not in kwargs.keys():
                raise AttributeError('Missing argument: analog_distances')
            else:
                reconstruction_series, reconstruction_min_band, reconstruction_max_band = get_weighted_average(
                    analog_pool=reconstructed_pool,
                    analog_distances=kwargs['analog_distances'],
                    sample_size=kwargs['sample_size']
                )

        elif similarity_method == 'percentiles':
            if 'reference_variable' not in kwargs.keys():
                raise AttributeError('Missing argument: reference_variable')
            else:

                # Reanalysis data of the analog pool
                reanalysis_pool = analog_dates.copy()
                reanalysis_pool = reanalysis_pool.apply(
                    lambda x: kwargs['reference_variable'][variable + ' reanalysis'].loc[x].values
                )
                kwargs['reference_variable'].index = pd.to_datetime(kwargs['reference_variable'].index)
                reanalysis_pool['original'] = kwargs['reference_variable'][variable + ' reanalysis'].loc[
                    pd.to_datetime(reanalysis_pool.index)]

                reconstruction_series, reconstruction_min_band, reconstruction_max_band = get_closest_percentile(
                    secondary_predictor_pool=reanalysis_pool,
                    analog_dates=list(reconstructed_data.index),
                    analog_pool=reconstructed_pool
                )

        else:
            raise AttributeError('Method ' + similarity_method + ' doesnot exist')

        reconstructed_data[variable + ' min band'] = reconstruction_min_band
        reconstructed_data[variable + ' max band'] = reconstruction_max_band
        reconstructed_data[variable] = reconstruction_series

    return reconstructed_data


def get_closest_neighbor(analog_pool: pd.DataFrame):
    """
    Get the analog day as the closest to the day to reconstruct.
    :param analog_pool: Historical data in the pool of analogues. The columns of the dataframe must be
    sorted by closeness to the original day.
    """
    # Maximum and minimum bands
    reconstruction_min_band = analog_pool.filter(items=analog_pool.columns).min(axis=1)
    reconstruction_max_band = analog_pool.filter(items=analog_pool.columns).max(axis=1)

    # Closest neighbor
    reconstruction_series = analog_pool[0]

    return reconstruction_series, reconstruction_min_band, reconstruction_max_band


def get_weighted_average(analog_pool: pd.DataFrame, analog_distances: pd.DataFrame, sample_size: int):
    """
    Get a synthetic analog made of the weighted average of the "sample_size" closest neighbors. The weight is
    proportional to the inverse squared distance.
    :param analog_pool: Historical data in the pool of analogues. The columns of the dataframe must be
    sorted by closeness to the original day.
    :param analog_distances: Dataframe of the distances of the members of the analog pool sorted by closeness
    :param sample_size: int. Number of member of the pool to use for the average
    """

    # Weight for averaging (Inverse squared distance)
    coefs = analog_distances[range(sample_size)].apply(lambda x: 1/x ** 2)

    # Reconstruction values
    reconstruction_values = analog_pool[range(sample_size)]

    # Maximum and minimum bands
    reconstruction_min_band = reconstruction_values.filter(items=reconstruction_values.columns).min(axis=1)
    reconstruction_max_band = reconstruction_values.filter(items=reconstruction_values.columns).max(axis=1)

    # Weighted average
    reconstruction_series = (coefs * reconstruction_values).sum(axis=1) / coefs.sum(axis=1)

    return reconstruction_series, reconstruction_min_band, reconstruction_max_band


def get_closest_percentile(secondary_predictor_pool: pd.DataFrame,
                           analog_dates: list,
                           analog_pool: pd.DataFrame):
    """
    selects as an analogue the day whose data in the historical series is of the same percentile as the day to be
    reconstructed in the secondary predictor series.
    :param secondary_predictor_pool: Values of the secondary predictor in the analog pool
    :param analog_dates: Dates of the analogs in the pool
    :param analog_pool: Values oof the analog in the pool
    """

    reconstruction_series = pd.DataFrame(index=analog_dates)

    # Calculate reanalysis and observed distributions of the analog pool
    for date in analog_dates:
        # Percentile value of the day to reconstruct using reanalysis data
        secondary_predictor_percentile = percentileofscore(
            secondary_predictor_pool.loc[date].values,
            score=secondary_predictor_pool['original'].loc[date]
        )

        # Percentiles of each observation in the analog pool
        reconstructed_percentile = np.array(
            [percentileofscore(analog_pool.loc[date].values, score=analog)
             for analog in analog_pool.loc[date].values]
        )

        # Minimum distance between the reanalysis percentile and observed percentiles
        closest_percentile = min(abs(reconstructed_percentile - secondary_predictor_percentile))
        closest_percentile_idx = list(abs(reconstructed_percentile - secondary_predictor_percentile)).index(
            closest_percentile)

        # Get the analog day as the closest percentile
        reconstruction_series.loc[date] = analog_pool[closest_percentile_idx].loc[date]

    # Maximum and minimum bands
    reconstruction_min_band = analog_pool.filter(items=analog_pool.columns).min(axis=1)
    reconstruction_max_band = analog_pool.filter(items=analog_pool.columns).max(axis=1)

    return reconstruction_series, reconstruction_min_band, reconstruction_max_band
