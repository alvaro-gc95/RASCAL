"""
Reconstruct meteorological data through the "analog method"
contact: alvaro@intermet.es
"""

import os
import tqdm
import pickle

import numpy as np
import pandas as pd
import xarray as xr
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt

from eofs.xarray import Eof
from dask.diagnostics import ProgressBar
from scipy.stats import percentileofscore
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import rascal.utils

coordinate_names = ["time", "latitude", "longitude"]
prompt_timer = True


class Station:
    """
    Store station metadata (code, name, altitude, longitude and latitude) and calculate daily time series.
    """
    def __init__(self, path: str):
        meta = pd.read_csv(path + 'meta.csv')
        self.path = path

        self.code = meta['code'].values[0]
        self.name = meta['name'].values[0]
        self.longitude = meta['longitude'].values[0]
        self.latitude = meta['latitude'].values[0]
        self.altitude = meta['altitude'].values[0]

    def get_data(self, variable: str, skipna: bool = True) -> pd.DataFrame:
        data = rascal.utils.get_daily_data(self.path, variable, skipna)
        return data

    def get_gridpoint(self, grid_latitudes: list, grid_longitudes: list):
        ilat, ilon = rascal.utils.get_nearest_gridpoint(
            grid_latitudes=grid_latitudes,
            grid_longitudes=grid_longitudes,
            point_longitude=self.longitude,
            point_latitude=self.latitude
        )
        return grid_latitudes[ilat], grid_longitudes[ilon]


class Predictor:
    """
    Predictor class. This contains data about the predictor variable to use for the reconstruction.
    """

    def __init__(
            self,
            paths: dict,
            grouping: str,
            lat_min: float,
            lat_max: float,
            lon_min: float,
            lon_max: float,
            mosaic: bool = True,
            number: int = None
    ):

        self.data = rascal.utils.open_data(
            files_paths=paths,
            grouping=grouping,
            domain=[lat_min, lat_max, lon_min, lon_max],
            number=number
        )

        if mosaic:
            self.data = self.to_mosaic()

    def crop(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> xr.Dataset:
        """
        Crop the domain of the dataframe
        :param lat_min: float.
        :param lat_max: float.
        :param lon_min: float.
        :param lon_max: float.
        """

        self.data = rascal.utils.crop_domain(
            self.data,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )

        return self.data

    def to_mosaic(self) -> xr.Dataset:
        """
        To use various simultaneous predictors or a vectorial variable, concatenate the variables along the longitude
        axis to obtain a single compound variable, easier to use when performing PCA.
        """

        compound_predictor = []
        compound_predictor_name = '_'.join(self.data.data_vars)
        final_lon = 0

        for j, variable_name in enumerate(self.data.data_vars):

            variable_j = self.data.rename({variable_name: compound_predictor_name})[compound_predictor_name]

            if j != 0:
                # Get the differences between the first longitude and the rest of the list.
                # Add 1 so the first element is not zero
                longitude_diffs = [
                    lon - variable_j['longitude'].values[0] + 1 for lon in variable_j['longitude'].values
                ]
                # Get the new longitudes to concatenate
                new_longitude = longitude_diffs + final_lon
                variable_j = variable_j.assign_coords(longitude=new_longitude)

            final_lon = variable_j['longitude'].values[-1]
            compound_predictor.append(variable_j)

        compound_predictor = xr.combine_by_coords(compound_predictor, combine_attrs='drop_conflicts')
        compound_predictor["time"] = pd.to_datetime(compound_predictor["time"].values)
        compound_predictor = compound_predictor.to_array().squeeze()

        self.data = compound_predictor
        return compound_predictor

    def module(self) -> xr.Dataset:
        """
        Get the module of the predictor variables as if they were components of a vector.
        :return:
        """
        vector_module = np.sqrt((self.data ** 2).to_array().sum("variable"))
        self.data = vector_module
        return self

    def anomalies(self, seasons: list = None, standardize: bool = None, mean_period: list = None) -> xr.Dataset:
        """
        Calculate seasonal anomalies of the field. The definition of season is flexible, being only a list of months
        contained within it.
        :param seasons: list. Months of the season. Default = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        :param standardize: bool. Standardize anomalies. Default = True
        :param mean_period: list. Dates to use as mean climatology to calculate the anomalies.
        :return: anomalies. xr.DataSet. dims = [time, latitude, longitude, season]
        """

        if seasons is None:
            seasons = [[int(m) for m in range(1, 13)]]
        if standardize is None:
            standardize = True
        if mean_period is None:
            mean_period = self.data["time"].values

        # Get the seasonal anomalies of the predictor field
        anomalies = []
        for i, season in enumerate(seasons):
            season_dates = [date for date in pd.to_datetime(mean_period) if date.month in season]
            seasonal_predictors = self.data.sel(time=season_dates)
            seasonal_anomalies = rascal.analogs.calculate_anomalies(seasonal_predictors, standardize=standardize)
            seasonal_anomalies = seasonal_anomalies.expand_dims({"season": [i]})
            anomalies.append(seasonal_anomalies)
        anomalies = xr.merge(anomalies)

        return anomalies

    @rascal.utils.timer_func(prompt=prompt_timer)
    def pcs(
            self,
            path: str,
            npcs: int,
            seasons: list = None,
            standardize: bool = None,
            pcscaling: int = None,
            overwrite: bool = None,
            training: list = None,
            project: xr.Dataset = None
    ) -> xr.Dataset:
        """
        Perform Principal Component Analysis. To save computation time, the PCA object can be saved as a pickle, so
        the analysis does not have to be performed every time.
        :param path: str. Path to save the PCA results
        :param npcs: int. Number of components.
        :param seasons: list. List of list of months of every season.
        :param standardize: bool. If True, the anomalies used in the PCA are standardized.
        :param pcscaling: int. Set the scaling of the PCs used to compute covariance. The following values are accepted:
            0 : Un-scaled PCs.
            1 : PCs are scaled to unit variance (divided by the square-root of their eigenvalue) (default).
            2 : PCs are multiplied by the square-root of their eigenvalue.
        :param overwrite: bool. Default = False. If True recalculate the PCA and overwrite the pickle with the PCA
        :param training: pd.DatetimeIndex. Dates to use for calculating the PCA
        :param project: xr.DataSet. Data to project onto the calculated PCA
        results.
        """

        if pcscaling is None:
            pcscaling = 1
        if overwrite is None:
            overwrite = False

        if training is not None:
            training_dates = sorted(list(set(training) & set(pd.to_datetime(self.data["time"].values))))
            if len(training_dates) == len(self.data["time"].values):
                testing_dates = False
                print("! Training period is all the available period." +
                      " All the training period will be used for the PCA ",
                      "(From " + str(training_dates[0]) + " to " + str(training_dates[-1]) + ")")
            else:
                testing_dates = sorted(list(set(pd.to_datetime(self.data["time"].values)) - set(training_dates)))
                print("! Training PCA period: From " + str(training_dates[0]) + " to " + str(training_dates[-1]),
                      ", Testing PCA period (Projection): From "
                      + str(testing_dates[0]) + " to " + str(testing_dates[-1]))
        else:
            training_dates = self.data["time"].values
            testing_dates = False
            print("! No training period especified: "
                  "Training period will be all the reanalysis period. All the training period will be used for the PCA")

        training_anomalies, training_mean, training_std = get_seasonal_anomalies(
            self.data.sel(time=training_dates),
            seasons=seasons,
            standardize=standardize,
            mean_period=training_dates
        )

        if testing_dates:
            testing_anomalies = []
            testing_dataset = self.data.sel(time=testing_dates)
            for i, season in enumerate(seasons):
                season_dates = [
                    date for date in pd.to_datetime(testing_dataset["time"].values) if date.month in season
                ]
                seasonal_test = testing_dataset.sel(time=season_dates)
                anomalies_test = seasonal_test - training_mean.sel(season=[i])
                if standardize:
                    anomalies_test = anomalies_test / anomalies_test.std(dim='time')
                testing_anomalies.append(anomalies_test)
            testing_anomalies = xr.merge(testing_anomalies)

        if project is not None:
            anomalies_to_project = []
            for i, season in enumerate(seasons):
                season_dates = [date for date in pd.to_datetime(project["time"].values) if date.month in season]
                seasonal_projection = project.sel(time=season_dates)
                projection_anomalies = seasonal_projection - training_mean.sel(season=[i])
                if standardize:
                    projection_anomalies = projection_anomalies / projection_anomalies.std(dim='time')
                anomalies_to_project.append(projection_anomalies)
            anomalies_to_project = xr.merge(anomalies_to_project)
            print("! Project new dataset onto the calculated PCA: From "
                  + str(anomalies_to_project["time"].values[0]) + " to "
                  + str(anomalies_to_project["time"].values[-1]))

        print("Load data anomalies in memory ...")
        with ProgressBar():
            training_anomalies = training_anomalies.compute()
        if testing_dates:
            with ProgressBar():
                testing_anomalies = testing_anomalies.compute()
        if project is not None:
            with ProgressBar():
                anomalies_to_project = anomalies_to_project.compute()

        initial_year = str(int(pd.to_datetime(self.data["time"].values[0]).year))
        final_year = str(int(pd.to_datetime(self.data["time"].values[-1]).year))
        pcs = []
        for i, season in enumerate(seasons):

            # Filename to the PCA solver object that contains tha analysis information
            pca_solver_filename = (
                    path +
                    ''.join([str(s).zfill(2) for s in season]) + '_'
                    + str(initial_year) + str(final_year) + '.pkl'
            )

            if overwrite:

                seasonal_anomalies = training_anomalies.sel(season=i).dropna(dim="time")
                seasonal_anomalies = seasonal_anomalies.to_array().squeeze(dim="variable")
                if testing_dates:
                    seasonal_test_anomalies = testing_anomalies.sel(season=i).dropna(dim="time")
                    seasonal_test_anomalies = seasonal_test_anomalies.to_array().squeeze(dim="variable")

                pcs_solver = rascal.analogs.get_pca_solver(
                    seasonal_anomalies,
                    pca_solver_filename,
                    overwrite=overwrite
                )

            else:

                if os.path.exists(pca_solver_filename):
                    # Avoid calculating anomalies if the pca solver already exists and overwrite == False
                    pcs_solver = rascal.analogs.get_pca_solver(None, pca_solver_filename, overwrite=overwrite)

                else:
                    seasonal_anomalies = training_anomalies.sel(season=i).dropna(dim="time")
                    seasonal_anomalies = seasonal_anomalies.to_array().squeeze(dim="variable")
                    if testing_dates:
                        seasonal_test_anomalies = testing_anomalies.sel(season=i).dropna(dim="time")
                        seasonal_test_anomalies = seasonal_test_anomalies.to_array().squeeze(dim="variable")

                    pcs_solver = rascal.analogs.get_pca_solver(
                        seasonal_anomalies,
                        pca_solver_filename,
                        overwrite=overwrite
                    )

            seasonal_pcs = pcs_solver.pcs(npcs=npcs, pcscaling=pcscaling)
            seasonal_pcs = seasonal_pcs.expand_dims({"season": [i]})
            pcs.append(seasonal_pcs)

            if testing_dates:
                if seasonal_test_anomalies.shape[0] > 0:
                    testing_pcs = pcs_solver.projectField(
                        seasonal_test_anomalies,
                        neofs=npcs,
                        eofscaling=pcscaling,
                        weighted=True
                    )
                    testing_pcs = testing_pcs.expand_dims({"season": [i]})
                    testing_pcs = testing_pcs.rename("pcs")
                    pcs.append(testing_pcs)

            if project is not None:
                seasonal_proj_anomalies = anomalies_to_project.sel(season=i).dropna(dim="time")
                seasonal_proj_anomalies = seasonal_proj_anomalies.to_array().squeeze(dim="variable")
                if seasonal_proj_anomalies.shape[0] > 0:
                    projected_pcs = pcs_solver.projectField(
                        seasonal_proj_anomalies,
                        neofs=npcs,
                        eofscaling=pcscaling,
                        weighted=True
                    )
                    projected_pcs = projected_pcs.expand_dims({"season": [i]})
                    projected_pcs = projected_pcs.rename("pcs")
                    pcs.append(projected_pcs)

        pcs = xr.merge(pcs)
        pcs = pcs.sortby('time')

        return pcs


class Analogs:
    """
    Store analog days information
    """

    def __init__(self, pcs: xr.Dataset, observations: pd.DataFrame, dates: list):
        self.pcs = pcs
        self.dates = dates
        self.observations = observations

    def get_pool(
            self,
            size: int = None,
            vw_size: int = None,
            vw_type: str = None,
            distance: str = None
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Get the pool of 'size' closest neighbors to each day
        :param size: int. Number of neighbors in the pool.
        :param vw_size: int. Validation window size. How many data points around each point is ignored to validate the
            reconstruction.
        :param vw_type: str. Type of validation window. Options:
            - forward: The original date is the last date of the window.
            - backward: The original date is the firs date of the window.
            - centered: The original date is in the center of the window.
        :param distance: str. Metric to determine the distance between points in the PCs space. Options:
            - euclidean
            - mahalanobis
        :return analog_dates, analog_distances: pd.DataFrame, pd.DataFrame. 'analog_dates' contains the dates of the
            analogs in the pool for each day. 'analog_distances' contains the  distances in the PCs space of each
            analog to the original day.
        """

        if size is None:
            size = 100

        analog_distances = []
        analog_dates = []
        for season in self.pcs["season"].values:

            seasonal_pcs = self.pcs.sel(season=season).dropna(dim="time").to_array().squeeze(dim="variable")

            # Split in seasonal test and training dates
            seasonal_training_dates = pd.to_datetime(seasonal_pcs["time"].values)
            seasonal_test_dates = sorted(list(set(seasonal_training_dates) & set(pd.to_datetime(self.dates))))
            seasonal_observed_dates = sorted(list(
                set(seasonal_training_dates) &
                set(rascal.utils.clean_dataset(self.observations).index)
            ))

            if len(seasonal_observed_dates) == 0:
                print("Warning: There is not any observational record during the training period")

            seasonal_analog_distances, seasonal_analog_dates = get_analog_pool(
                training_set=seasonal_pcs.sel(time=seasonal_observed_dates),
                test_pcs=seasonal_pcs.sel(time=seasonal_test_dates),
                pool_size=size,
                vw_size=vw_size,
                vw_type=vw_type,
                distance=distance
            )

            analog_distances.append(seasonal_analog_distances)
            analog_dates.append(seasonal_analog_dates)

        analog_distances = pd.concat(analog_distances, axis=0)
        analog_dates = pd.concat(analog_dates, axis=0)

        analog_distances = analog_distances.sort_index()
        analog_dates = analog_dates.sort_index()

        return analog_distances, analog_dates

    @rascal.utils.timer_func(prompt=prompt_timer)
    def reconstruct(
            self,
            pool_size: int = None,
            method: str = None,
            sample_size: int = None,
            mapping_variable: Predictor = None,
            vw_size: int = None,
            vw_type: str = None,
            distance: str = None
    ) -> pd.DataFrame:
        """
        Reconstruct a time series using the analog pool for each day.
        :param pool_size: int. Size of the analog pool for each day.
        :param method: str. Similarity method to select the best analog of the pool. Options are:
            - 'closest': (Selected by default) Select the closest analog in the PCs space
            - 'average': Calculate the weighted average of the 'sample_size' closest analogs in the PCs space.
            - 'quantilemap': Select the analog that represent the same quantile in the observations pool that another
                   mapping variable.
        :param sample_size: int. Number of analogs to average in the 'average' method
        :param mapping_variable: Predictor object. Time series of a variable to use as mapping in 'quantilemap'
                :param vw_size: int. Validation window size. How many data points around each point is ignored to
                validate the reconstruction.
        :param vw_type: str. Type of validation window. Options:
            - forward: The original date is the last date of the window.
            - backward: The original date is the firs date of the window.
            - centered: The original date is in the center of the window.
        :param distance: str. Metric to determine the distance between points in the PCs space. Options:
            - euclidean
            - mahalanobis (Wishlist)
        :return reconstruction: pd.DataFrame.
        """

        if pool_size is None:
            pool_size = 100
        if method is None:
            method = "closest"
        if vw_size is not None and vw_size < 10:
            print("Warning: Validation window size = " + str(vw_size) + " is < 10 data points. To cross validate " +
                  "the series, a size>=10 is recommended")

        analog_distances, analog_dates = self.get_pool(
            size=pool_size,
            vw_size=vw_size,
            vw_type=vw_type,
            distance=distance
        )
        reconstruction = reconstruct_by_analogs(
            observed_data=self.observations,
            analog_dates=analog_dates,
            similarity_method=method,
            analog_distances=analog_distances,
            sample_size=sample_size,
            mapping_variable=mapping_variable
        )

        return reconstruction


def get_seasonal_anomalies(data: xr.Dataset, seasons: list, standardize: bool, mean_period: list) -> xr.Dataset:
    """
    Calculate seasonal anomalies of the field. The definition of season is flexible, being only a list of months
    contained within it.
    :param data: xr.Dataset
    :param seasons: list. Months of the season. Default = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    :param standardize: bool. Standardize anomalies. Default = True
    :param mean_period: list. dates to use as the mean climatology period
    :return: anomalies. xr.DataSet. dims = [time, latitude, longitude, season]
    :return: mean. xr.DataSet. dims = [latitude, longitude, season]
    :return: std. xr.DataSet. dims = [latitude, longitude, season]
    """

    if seasons is None:
        seasons = [[int(m) for m in range(1, 13)]]
    if standardize is None:
        standardize = True
    if mean_period is None:
        mean_period = data["time"].values

    # Get the seasonal anomalies of the predictor field
    anomalies = []
    mean = []
    std = []
    for i, season in enumerate(seasons):
        season_dates = [date for date in pd.to_datetime(mean_period) if date.month in season]
        seasonal_predictors = data.sel(time=season_dates)
        seasonal_anomalies, seasonal_mean, seasonal_std = rascal.analogs.calculate_anomalies(
            seasonal_predictors,
            standardize=standardize,
            mean_period=season_dates
        )
        seasonal_anomalies = seasonal_anomalies.expand_dims({"season": [i]})
        seasonal_mean = seasonal_mean.expand_dims({"season": [i]})
        seasonal_std = seasonal_std.expand_dims({"season": [i]})

        anomalies.append(seasonal_anomalies.to_dataset(name="anomalies"))
        mean.append(seasonal_mean.to_dataset(name="mean"))
        std.append(seasonal_std.to_dataset(name="standard_deviation"))

    anomalies = xr.merge(anomalies)
    mean = xr.merge(mean)
    std = xr.merge(std)

    return anomalies, mean, std


def calculate_anomalies(data_array: xr.DataArray, standardize: bool = False, mean_period: list = None) -> xr.Dataset:
    """
    :param data_array: DataArray.
    :param standardize: bool. Default=False. If True divide the anomalies by its standard deviation.
    :param mean_period: pd.DatetimeIndex. Dates to use to calculate the mean
    :return anomalies, mean, std: DataArray.
    """
    mean = data_array.sel(time=mean_period).mean(dim='time')
    anomalies = data_array - mean
    std = anomalies.std(dim='time')
    if standardize:
        anomalies = anomalies / std

    return anomalies, mean, std


def get_pca_solver(data_array, file_name: str, overwrite: bool = True):
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
        rascal.utils.save_object(solver, file_name)
    else:
        # Open solver object
        with open(file_name, 'rb') as inp:
            solver = pickle.load(inp)

    return solver


# def plot_pca(file_name, n_components, vectorial=False):
#     """
#     Plot maps of the EOFs
#     :param file_name: str. Name of the solver file.
#     :param n_components: int. Number of components to represent.
#     :param vectorial: bool. If True represent the EOFs as a contour of the module and quiver plot fot direction.
#     """
#
#     # Open solver object
#     with open(file_name, 'rb') as inp:
#         solver = pickle.load(inp)
#
#     # This is the map projection we want to plot *onto*
#     map_proj = ccrs.PlateCarree()
#
#     # EOF maps
#     eofs = solver.eofs(neofs=n_components)
#
#     if vectorial:
#         # Separate the concatenated vectorial variable in u, v and module
#         eofs = rascal.utils.separate_concatenated_components(eofs)
#
#         for mode in eofs['mode'].values:
#             # Defining the figure
#             fig = plt.figure(figsize=(4, 4), facecolor='w',
#                              edgecolor='k')
#
#             # Axes with Cartopy projection
#             ax = plt.axes(projection=ccrs.PlateCarree())
#             p = eofs['module'].sel(mode=mode).plot.contourf(transform=ccrs.PlateCarree(), levels=20)
#
#             # Defining the quiver plot
#             quiver = eofs.sel(mode=mode).plot.quiver(x='longitude', y='latitude', u='u', v='v',
#                                                      transform=ccrs.PlateCarree(), scale=1)
#
#             # # Vector options declaration
#             veclenght = 0.05
#             maxstr = '%3.1f kg m-1 s-1' % veclenght
#             ax.quiverkey(quiver, -0.1, -0.1, veclenght, maxstr, labelpos='S', coordinates='axes')
#
#             ax.coastlines()
#             explained_variance_ratio = solver.varianceFraction(n_components).sel(mode=mode).values
#             ax.set_title(
#                 'Mode ' + str(mode + 1) + ' (exp. var. = ' + str(round(explained_variance_ratio * 100, 2)) + '%)')
#
#             gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                               linewidth=0.1, color='k', alpha=1,
#                               linestyle='--')
#             gl.top_labels = False
#             gl.right_labels = False
#             gl.xformatter = LONGITUDE_FORMATTER
#             gl.yformatter = LATITUDE_FORMATTER
#             gl.xlabel_style = {'size': 8}
#             gl.ylabel_style = {'size': 8}
#
#             plt.savefig(file_name[:-3] + '_mode' + str(mode + 1) + '.png')
#
#     else:
#         p = eofs.plot.contourf(transform=ccrs.PlateCarree(),  # the data's projection
#                                col='mode', col_wrap=1,  # multiplot settings
#                                subplot_kws={'projection': map_proj},
#                                levels=20)  # the plot's projection
#
#         # We have to set the map's options on all four axes
#         for i, ax in enumerate(p.axes.flat):
#             ax.coastlines()
#             explained_variance_ratio = solver.varianceFraction(n_components).sel(mode=i).values
#             ax.set_title('Mode ' + str(i + 1) + ' (exp. var. = ' + str(round(explained_variance_ratio * 100, 2)) + '%)')
#
#         plt.savefig(file_name[:-3] + '.png')


def calculate_distances(origin: xr.Dataset, points: xr.Dataset, distance: str = 'euclidean') -> xr.Dataset:
    """
    Calculate distance between one origin point and a set of points.
    :param origin: DataArray.
    :param points: DataArray.
    :param distance: str. Type of distance to calculate.
    :return distances: DataArray.
    """

    if distance == 'euclidean':
        distances = np.sqrt(((origin - points) ** 2).sum(dim='mode'))
    elif distance == "mahalanobis":
        y_mu = origin - np.mean(points)
        cov = np.cov(points.values.T)
        inv_covmat = np.linalg.inv(cov)
        left = np.dot(y_mu, inv_covmat)
        mahal = np.dot(left, y_mu.T)
        return mahal.diagonal()
    else:
        raise AttributeError('Error: ' + distance + ' distance does not exist')

    return distances


def get_analog_pool(
        training_set: xr.Dataset,
        test_pcs: xr.Dataset,
        pool_size: int = 100,
        vw_size: int = 10,
        vw_type: str = "centered",
        distance: str = "euclidean"
) -> (pd.DataFrame, pd.DataFrame):
    """
    Get a pool of analogues calculating the N closest neighbours in the PCs space.
    :param training_set: DataArray. PCs of possible analogues.
    :param test_pcs: DataArray. PCs of the day to reconstruct.
    :param pool_size: int. N number of analogues.
    :param vw_size: int. Validation window size. How many data points around each point is ignored to validate the
        reconstruction.
    :param vw_type: str. Type of validation window. Options:
        forward: The original date is the last date of the window.
        backward: The original date is the firs date of the window.
        centered: The original date is in the center of the window.
    :param distance: str. Metric to determine the distance between points in the PCs space. Options:
        - euclidean
        - mahalanobis
    :return analog_distances: DataFrame. Distances to the day to reconstruct of the N closes analogues.
    :return analog_dates: DataFrame. Dates of the N closest analogues.
    """

    if pool_size is None:
        pool_size = 100
    if vw_size is None:
        vw_size = 10
    if vw_type is None:
        vw_type = "centered"
    if distance is None:
        distance = "euclidean"

    analog_dates = pd.DataFrame(index=pd.to_datetime(test_pcs['time'].values), columns=range(pool_size))
    analog_distances = pd.DataFrame(index=pd.to_datetime(test_pcs['time'].values), columns=range(pool_size))

    training_dates = pd.to_datetime(training_set['time'].values)

    for date in tqdm.tqdm(test_pcs['time'].values, desc='Generating reconstruction'):
        # Delete values close the date to reconstruct
        validation_window = rascal.utils.get_validation_window(
            test_date=pd.to_datetime(date),
            dates=pd.to_datetime(test_pcs['time'].values),
            window_size=vw_size,
            window_type=vw_type
        )
        validation_window = pd.to_datetime(validation_window)
        validation_dates = sorted(list(set(training_dates) - set(validation_window)))

        # Find distances in the PC space to the point to predict
        distances = calculate_distances(
            origin=test_pcs.sel(time=date),
            points=training_set.sel(time=validation_dates),
            distance=distance
        )

        # Sort the distances to find the closest days in the PC space
        distances = distances.sortby(distances)

        # Get the pool of closest days
        distances = distances.isel(time=slice(0, pool_size))

        # Pool of dates and distances
        analog_dates.loc[date] = distances['time'].values
        analog_distances.loc[date] = distances.values

    return analog_distances, analog_dates


def reconstruct_by_analogs(
        observed_data: pd.DataFrame,
        analog_dates: pd.DataFrame,
        similarity_method: str = 'closest',
        **kwargs
) -> pd.DataFrame:
    """
    Reconstruct time series
    :param observed_data: pd.DataFrame. All observations.
    :param analog_dates: pd.DataFrame. Dates in the analog pool for each date to reconstruct.
    :param similarity_method: str. Reconstruction method. Options = ('closet', 'pondered', 'quantilemap')
    :param kwargs:
    :return:
    """

    # Columns for maximum and minimum values in the analog pool
    min_band_columns = [c + ' min band' for c in observed_data.columns]
    max_band_columns = [c + ' max band' for c in observed_data.columns]

    # Create the empty dataframe for the reconstructed values
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
            if "sample_size" not in kwargs.keys() or kwargs["sample_size"] is None:
                raise AttributeError('Missing argument: sample_size')
            elif "analog_distances" not in kwargs.keys() or kwargs["sample_size"] is None:
                raise AttributeError('Missing argument: analog_distances')
            else:
                reconstruction_series, reconstruction_min_band, reconstruction_max_band = get_weighted_average(
                    analog_pool=reconstructed_pool,
                    analog_distances=kwargs['analog_distances'],
                    sample_size=kwargs['sample_size']
                )

        elif similarity_method == 'quantilemap':
            if 'mapping_variable' not in kwargs.keys():
                raise AttributeError('Missing argument: mapping_variable')
            else:
                # Convert the input predictor object to dataframe
                secondary_predictor = kwargs["mapping_variable"]
                secondary_predictor = secondary_predictor.data.to_dataframe().drop(["latitude", "longitude"], axis=1)

                # Reanalysis data of the analog pool
                reanalysis_pool = analog_dates.copy()
                reanalysis_pool = reanalysis_pool.map(lambda x: np.squeeze(secondary_predictor.loc[x].values))

                secondary_predictor.index = pd.to_datetime(secondary_predictor.index)

                # This line gives problems because when creating the secondary predictor, the dates of the year are not
                # selected, the full year is taken, and this nos happens with the analog dates.index. I leave this
                # here just in case this gives problems
                # secondary_predictor.index = pd.to_datetime(analog_dates.index)

                reanalysis_pool['original'] = secondary_predictor.loc[pd.to_datetime(reanalysis_pool.index)]
                reconstruction_series, reconstruction_min_band, reconstruction_max_band = get_closest_percentile(
                    secondary_predictor_pool=reanalysis_pool,
                    analog_dates=list(reconstructed_data.index),
                    analog_pool=reconstructed_pool
                )
                reconstruction_series = reconstruction_series['variable']

        else:
            raise AttributeError('Method ' + similarity_method + ' doesnot exist')

        reconstructed_data[variable + ' min band'] = reconstruction_min_band
        reconstructed_data[variable + ' max band'] = reconstruction_max_band
        reconstructed_data[variable] = reconstruction_series

    return reconstructed_data


def get_closest_neighbor(analog_pool: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
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


def get_weighted_average(
        analog_pool: pd.DataFrame,
        analog_distances: pd.DataFrame,
        sample_size: int
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Get a synthetic analog made of the weighted average of the "sample_size" closest neighbors. The weight is
    proportional to the inverse squared distance.
    :param analog_pool: Historical data in the pool of analogues. The columns of the dataframe must be
    sorted by closeness to the original day.
    :param analog_distances: Dataframe of the distances of the members of the analog pool sorted by closeness
    :param sample_size: int. Number of member of the pool to use for the average
    """

    # Weight for averaging (Inverse squared distance)
    coefs = analog_distances[range(sample_size)].apply(lambda x: 1 / x ** 2)

    # Reconstruction values
    reconstruction_values = analog_pool[range(sample_size)]

    # Maximum and minimum bands
    reconstruction_min_band = reconstruction_values.filter(items=reconstruction_values.columns).min(axis=1)
    reconstruction_max_band = reconstruction_values.filter(items=reconstruction_values.columns).max(axis=1)

    # Weighted average
    reconstruction_series = (coefs * reconstruction_values).sum(axis=1) / coefs.sum(axis=1)

    return reconstruction_series, reconstruction_min_band, reconstruction_max_band


def get_closest_percentile(
        secondary_predictor_pool: pd.DataFrame,
        analog_dates: list,
        analog_pool: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    selects as an analogue the day whose data in the historical series is of the same percentile as the day to be
    reconstructed in the secondary predictor series.
    :param secondary_predictor_pool: Values of the secondary predictor in the analog pool
    :param analog_dates: Dates of the analogs in the pool
    :param analog_pool: Values oof the analog in the pool
    """

    reconstruction_series = pd.DataFrame(index=analog_dates, columns=['variable'])

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
