"""
RASCAL utility functions
contact: alvaro@intermet.es
"""
import itertools
import os
import yaml
import pickle
import typing
import datetime
import functools
import helpers.open_data
import rascal.statistics

import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

from time import time


coordinate_names = ["time", "latitude", "longitude"]
prompt_timer = True


class Station:
    def __init__(self, path):
        meta = pd.read_csv(path + 'meta.csv')
        self.path = path

        self.code = meta['code'].values[0]
        self.name = meta['name'].values[0]
        self.longitude = meta['longitude'].values[0]
        self.latitude = meta['latitude'].values[0]
        self.altitude = meta['altitude'].values[0]

    def get_data(self, variable, skipna=True):
        data = get_daily_data(self.path, variable, skipna)
        return data

    def get_gridpoint(self, grid_latitudes, grid_longitudes):
        ilat, ilon = get_nearest_gridpoint(
            grid_latitudes=grid_latitudes,
            grid_longitudes=grid_longitudes,
            point_longitude=self.longitude,
            point_latitude=self.latitude
        )
        return grid_latitudes[ilat], grid_longitudes[ilon]


class Preprocess:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @staticmethod
    def _check_variable_in_obj(obj, variables_to_check):
        # Verify there is a column with the selected meteorological variable
        if not (set(obj.columns) & set(variables_to_check)):
            raise AttributeError("Must have " + ', '.join(variables_to_check))

    def wind_components(self, substitute=False):
        """
        Transform wind speed amd direction to components U and V
        :param substitute: bool (default=False). If True substitute the WDIR and WSPD values for U and V.
        """
        self._check_variable_in_obj(self._obj, ['WSPD', 'WDIR'])

        self._obj['U'] = self._obj['WSPD'] * np.deg2rad(270 - self._obj['WDIR']).apply(np.cos)
        self._obj['V'] = self._obj['WSPD'] * np.deg2rad(270 - self._obj['WDIR']).apply(np.sin)

        if substitute:
            self._obj.drop(['WSPD', 'WDIR'], axis=1, inplace=True)

    def clear_low_radiance(self, rad_thr=200):
        """
        Delete Shortwave incoming radiance below the "night" threshold
        """
        self._check_variable_in_obj(self._obj, ['RADS01'])

        self._obj['RADS01'] = self._obj['RADS01'].where(self._obj['RADS01'] >= rad_thr, np.nan)

    def calculate_relative_humidity(self):
        """
        Calculate relative humidity from dew point temperature and air temperature
        :return:
        """
        e0 = 0.611  # [kPa]
        l_rv = 5423  # L/Rv [K]
        t0 = 273  # [K]

        # Water vapor pressure
        self._obj['E'] = self._obj['TDEW'].apply(lambda x: e0 * np.exp(l_rv * ((1 / t0) - (1 / x))))

        # Saturation water vapor pressure
        self._obj['ES'] = self._obj['TMPA'].apply(lambda x: e0 * np.exp(l_rv * ((1 / t0) - (1 / x))))

        # Relative humidity
        self._obj['RHMA'] = (self._obj['E'] / self._obj['ES']) * 100

        del self._obj['E'], self._obj['ES']

    def get_daily_variables(self, skipna=True):
        """
        Get relevant daily climatological variables as DataFrame: Maximum, minimum and mean temperature, maximum and
        mean wind velocity, total solar radiation, total precipitation.
        """

        daily_variables = pd.DataFrame()

        if 'TMAX' in self._obj.columns:
            # tmax = self._obj['TMAX'].resample('D').max().rename('TMAX')
            tmax = nan_resampler(self._obj['TMAX'], freq='1D', grouping="max", skipna=skipna)
            tmax = tmax.squeeze().rename("TMAX")
            daily_variables = pd.concat([daily_variables, tmax], axis=1)

        if 'TMIN' in self._obj.columns:
            # tmin = self._obj['TMIN'].resample('D').min().rename('TMIN')
            tmin = nan_resampler(self._obj['TMIN'], freq='1D', grouping="min", skipna=skipna)
            tmin = tmin.squeeze().rename("TMIN")
            daily_variables = pd.concat([daily_variables, tmin], axis=1)

        if 'TMEAN' in self._obj.columns:
            # tmean = self._obj['TMEAN'].resample('D').mean().rename('TMEAN')
            tmean = nan_resampler(self._obj['TMEAN'], freq='1D', grouping="mean", skipna=skipna)
            tmean = tmean.squeeze().rename("TMEAN")
            daily_variables = pd.concat([daily_variables, tmean], axis=1)

        if 'TMPA' in self._obj.columns:
            # tmax = self._obj['TMPA'].resample('D').max().rename('TMAX')
            tmax = nan_resampler(self._obj['TMAX'], freq='1D', grouping="max", skipna=skipna)
            tmax = tmax.squeeze().rename("TMAX")
            # tmin = self._obj['TMPA'].resample('D').min().rename('TMIN')
            tmin = nan_resampler(self._obj['TMIN'], freq='1D', grouping="min", skipna=skipna)
            tmin = tmin.squeeze().rename("TMIN")
            # tmean = self._obj['TMPA'].resample('D').mean().rename('TMEAN')
            tmean = nan_resampler(self._obj['TMEAN'], freq='1D', grouping="mean", skipna=skipna)
            tmean = tmean.squeeze().rename("TMEAN")
            tamp = abs(tmax - tmin)
            tamp = tamp.rename('TAMP')
            daily_variables = pd.concat([daily_variables, tmax, tmin, tmean], axis=1)

        if 'WSPD' in self._obj.columns:
            # vmax = self._obj['WSPD'].resample('D').max().rename('VMAX')
            vmean = nan_resampler(self._obj['WSPD'], freq='1D', grouping="mean", skipna=skipna)
            vmean = vmean.squeeze().rename("VMEAN")
            # vmean = self._obj['WSPD'].resample('D').mean().rename('VMEAN')
            # daily_variables = pd.concat([daily_variables, vmax, vmean], axis=1)
            daily_variables = pd.concat([daily_variables, vmean], axis=1)

        if 'RADS01' in self._obj.columns:
            rascal.utils.Preprocess(self._obj).clear_low_radiance()
            # rads_total = self._obj['RADS01'].resample('D').sum().rename('RADST')
            rads_total = nan_resampler(self._obj['RADS01'], freq='1D', grouping="sum", skipna=skipna)
            rads_total = rads_total.squeeze().rename("RADST")
            rads_total = rads_total.where(rads_total > 0, np.nan)
            daily_variables = pd.concat([daily_variables, rads_total], axis=1)

        if 'PCP' in self._obj.columns:
            # ptot = self._obj['PCP'].resample('D').sum().rename('PCP')
            ptot = nan_resampler(self._obj['PCP'], freq='1D', grouping="sum", skipna=skipna)
            ptot = ptot.squeeze().rename("PCP")
            daily_variables = pd.concat([daily_variables, ptot], axis=1)

        if 'RHMA' in self._obj.columns:
            # rhmax = self._obj['RHMA'].resample('D').max().rename('RHMA')
            rhmax = nan_resampler(self._obj['RHMA'], freq='1D', grouping="max", skipna=skipna)
            rhmax = rhmax.squeeze().rename("RHMAX")
            rhmean = nan_resampler(self._obj['RHMA'], freq='1D', grouping="mean", skipna=skipna)
            rhmean = rhmean.squeeze().rename("RHMEAN")
            daily_variables = pd.concat([daily_variables, rhmax, rhmean], axis=1)

        daily_variables.index = pd.to_datetime(daily_variables.index)

        return daily_variables


class Climatology:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def daily_cycle(self, percentiles=None, to_series=False):
        """
        Calculate the percentiles of the monthly daily cycles
        """
        if percentiles is None:
            percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        columns = [v + '_' + str(p) for v, p in itertools.product(self._obj.columns, percentiles)]
        # Calculate the index of the monthly daily cycles (format = hour_month)
        idx = [str(h) + '_' + str(m) for h, m in itertools.product(range(0, 24), range(1, 13))]

        # Dataframe of climatological percentiles
        climatology_percentiles = pd.DataFrame(index=idx, columns=columns)

        # Calculate the climatological daily cycle for each month for each percentile
        for variable, percentile in itertools.product(self._obj.columns, percentiles):
            for month, month_dataset in self._obj.groupby(self._obj.index.month):
                monthly_climatology = month_dataset[variable].groupby(month_dataset.index.hour).quantile(percentile)
                for hour in monthly_climatology.index:
                    climatology_percentiles.loc[str(hour) + '_' + str(month), variable + '_' + str(percentile)] = \
                        monthly_climatology.loc[hour]

        # transform the monthly daily cycles to time series
        if to_series:
            return table_to_series(climatology_percentiles, self._obj.index)
        else:
            return climatology_percentiles

    def spatial_regression(self, related_site):
        """
        Get the correlation and linear regression with a reference station
        """

        # Calculate the index of tha monthly daily cycles (format = hour_month)
        idx = [str(h) + '_' + str(m) for h, m in itertools.product(range(0, 24), range(1, 13))]
        columns = [v + '_' + lr for v, lr in itertools.product(self._obj.columns, ['coef', 'intercept', 'correlation'])]
        # Dataframe of climatological percentiles
        regression = pd.DataFrame(index=idx, columns=columns)
        residuals = pd.DataFrame(index=self._obj.index, columns=[c + '_residuals' for c in self._obj.columns])

        # Group the data by month
        for month, month_dataset in self._obj.groupby(self._obj.index.month):
            # Fill the dataframe with the climatological daily cycle of each month
            for hour, hour_dataset in month_dataset.groupby(month_dataset.index.hour):

                # Select the data of the reference station by month and hour
                related_site_hm = related_site.loc[
                    (related_site.index.month == month) &
                    (related_site.index.hour == hour)
                ]
                hour_dataset = hour_dataset

                # Correlate the datasets
                correlation = related_site_hm.corrwith(hour_dataset)

                for variable in self._obj.columns:

                    linear_regressor, regr_res = rascal.statistics.linear_regression(
                        x=related_site_hm[variable].to_frame(),
                        y=hour_dataset[variable].to_frame()
                    )

                    # Save the coefficient, intercept, and correlation for the hour and month
                    regression.loc[str(hour) + '_' + str(month),
                                   variable + '_coef'] = np.squeeze(linear_regressor.coef_)
                    regression.loc[str(hour) + '_' + str(month),
                                   variable + '_intercept'] = np.squeeze(linear_regressor.intercept_)
                    regression.loc[str(hour) + '_' + str(month), variable + '_correlation'] = correlation[variable]

                    residuals.loc[regr_res.index, variable + '_residuals'] = regr_res[variable].values

        return regression, residuals

    def mcp(self):
        """
        Wishlist
        :return:
        """
        pass


def timer_func(func: typing.Callable = None, prompt: bool = True) -> typing.Callable:
    """
    This function shows the execution time of  the function object passed
    """

    if func is None:
        return functools.partial(timer_func, prompt=prompt)

    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        if prompt:
            print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def open_yaml(yaml_path):
    """
    Read the configuration yaml file.
    :param yaml_path: str. Path of the yaml file
    :return configuration file: Object. Object containing the information of the configuration file.
    """

    # Check if the yaml exists
    if not os.path.exists(yaml_path):
        raise AttributeError('WARNING: The configuration file ' + yaml_path + ' does not exist')
    else:
        # Read data in ini
        with open(yaml_path, 'r') as stream:
            try:
                configuration_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        return configuration_file


def clean_dataset(df):
    """
    Delete conflictive values from dataset (NaN or inf)
    :param df: DataFrame or Series.
    :return df: DataFrame or Series. Cleaned vesion of original df.
    """
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def get_common_index(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Reduce two dataframes to their common valid data
    """
    # Clean DataFrames of possible conflictive values
    df1 = clean_dataset(df1)
    df2 = clean_dataset(df2)

    # Get only the common data
    common_idx = sorted(list(set(df1.index).intersection(df2.index)))
    df1 = df1.loc[common_idx]
    df2 = df2.loc[common_idx]

    return df1, df2


def get_validation_window(test_date, dates, window_size, window_type='centered'):
    """
    Get a window of dates around an original one.
    :param test_date: Datetime. central date of the window.
    :param dates: list. All available dates to make the window.
    :param window_size: int. Number of total days of the window, without including the original date.
    :param window_type: str. Type of window. Options:
        forward: The original date is the last date of the window.
        backward: The original date is the firs date of the window.
        centered: The original date is in the center of the window.
    :return validation_window: list. Dates in the window.
    """

    if window_type not in ['forward', 'back', 'centered']:
        raise AttributeError('Error: ' + window_type + ' window does not exist')

    else:
        if window_type == 'forward':
            initial_date = test_date - datetime.timedelta(days=window_size)
            final_date = test_date

        if window_type == 'back':
            initial_date = test_date
            final_date = test_date + datetime.timedelta(days=window_size)

        if window_type == 'centered':
            initial_date = test_date - datetime.timedelta(days=int(np.ceil(window_size / 2)))
            final_date = test_date + datetime.timedelta(days=int(np.floor(window_size / 2)))

        validation_window = pd.date_range(start=initial_date, end=final_date, freq='1D')
        validation_window = list(set(validation_window) & set(dates))

        return validation_window


def get_daily_data(path: str, variable: str, skipna=False):
    observations = helpers.open_data.open_observations(path, [variable])
    daily_observations = Preprocess(observations).get_daily_variables(skipna=skipna)
    return daily_observations


@timer_func(prompt=prompt_timer)
def get_files(nwp_path, variables, dates, file_format):
    """
    Get all files
    :param nwp_path: str. Path to the grib files.
    :param variables: list. Variables to open.
    :param dates: list. Dates to open.
    :param file_format: str. File format
    :return all_file_paths: dict. Lists of all data file paths for each variable
    """

    # List of al the grib files
    all_file_paths = {}
    for variable in variables:
        # Get all file paths
        file_paths = [np.nan] * len(dates)
        for i, year in enumerate(dates):
            file_name = (nwp_path + 'y_' + str(year) + '/' + str(year) + '_' + str(variable) + file_format)
            # If the file exists, put the path in the list and save the correspondent date and hours
            if os.path.isfile(file_name):
                # Put the daily grib file name in the total file names array
                file_paths[i] = file_name
            else:
                print(file_name + ' does not exist')
        # Delete empty slots
        file_paths = [item for item in file_paths if not (pd.isnull(item)) is True]
        # Add files to the dictionary
        all_file_paths[variable] = file_paths

    return all_file_paths


def group_data(ds, grouping=None):
    """
    Group data of a dataframe. It can group data in
    By default grouping is None, then the central hour of the day is taken as the representative time of the day.
    It possible to take individual hours, or different frequencies of timesteps based on the usual xarray syntaxis.
    The grouping then is made on the selected frequency. groupings = ['sum', 'mean', 'min', 'max']
    :param ds: xr.DataArray or xr.DataSet
    :param grouping: str. 'hour(optional)_frequency_grouping'
    :return: ds: grouped xr.DataArray or xr.DataSet
    """

    # The default configuration is to take the 12:00 of each day
    if grouping is None:
        grouping = "12hour_1D_mean"

    if len(grouping.split('_')) == 3:
        hour, frequency, group_type = grouping.split('_')
    elif len(grouping.split('_')) == 2:
        frequency, group_type = grouping.split('_')
        hour = False
    else:
        raise AttributeError("Grouping str must have between 2 and 3 elements separated by _")

    if isinstance(hour, str):
        hour = int(hour.replace("hour", ""))
        ds_time = [date for date in pd.to_datetime(ds['time'].values) if date.hour == hour]
        ds = ds.sel(time=ds_time)

    if group_type == 'sum':
        ds = ds.resample(time=frequency).sum()
    elif group_type == 'mean':
        ds = ds.resample(time=frequency).mean()
    elif group_type == 'min':
        ds = ds.resample(time=frequency).min()
    elif group_type == 'max':
        ds = ds.resample(time=frequency).max()
    else:
        raise AttributeError('Grouping method (' + group_type + ') does not exists')

    return ds


def clean_coordinates(ds):
    """
    Delete unidimensional coordinates that might produce merging problems
    """
    variables_not_coords = [variable for variable in ds.variables if variable not in coordinate_names]
    variables_to_drop = [variable for variable in variables_not_coords if ds[variable].size == 1]
    ds = ds.drop_vars(variables_to_drop).squeeze()
    return ds


@timer_func(prompt=prompt_timer)
def open_data(files_paths, grouping=None, number=None, domain=None):
    """
    Combine a list of files (.grib or .nc usually) in one DataArray.
    :param files_paths: list. Paths of the grib file to open
    :param grouping: str. Default=None. Format = frequency_method. frequency=('hourly', 'daily', 'monthly', yearly').
    method=('sum', 'mean', 'min', 'max')
    :param number: int. Default=None. Ensemble member number (Only for ERA20CM products)
    :param domain: list [minimum latitude, maximum latitude, minimum longitude, maximum longitude]
    :return combined: DataArray. All files concatenated in time
    """
    combined_ds = []
    for variable, files in files_paths.items():
        # Check if the file exists
        for file_path in files:
            if not os.path.isfile(file_path):
                print('     The file ' + file_path + ' does not exist')
                files.remove(file_path)
        # Load to xarray
        variable_ds = xr.open_mfdataset(files)
        # Reduce memory usage
        variable_ds = variable_ds.astype(np.float32)
        # Clean unidimensional coordinates to avoid merging problems
        variable_ds = clean_coordinates(variable_ds)
        # Group the data
        variable_ds = group_data(variable_ds, grouping)
        # Select domain
        if domain is not None:
            variable_ds = crop_domain(
                variable_ds,
                lat_min=domain[0],
                lat_max=domain[1],
                lon_min=domain[2],
                lon_max=domain[3]
            )
        # Select ensemble member if possible
        if number is not None:
            variable_ds = variable_ds.sel(number=number).squeeze()
            variable_ds = variable_ds.drop_vars("number")

        combined_ds.append(variable_ds)

    combined_ds = xr.merge(combined_ds)

    return combined_ds


def get_nearest_gridpoint(grid_longitudes, grid_latitudes, point_longitude, point_latitude):
    """
    Find the nearest grid point in a dataset
    :param grid_longitudes: array.
    :param grid_latitudes: array.
    :param point_longitude: float.
    :param point_latitude: float.
    :return data: DataSet. Original dataset in the nearest gridpoint to the selected latitude and longitude.
    """
    nearest_latitude_distance = min(abs(grid_latitudes - point_latitude))
    nearest_longitude_distance = min(abs(grid_longitudes - point_longitude))

    ilat = list(abs(grid_latitudes - point_latitude)).index(nearest_latitude_distance)
    ilon = list(abs(grid_longitudes - point_longitude)).index(nearest_longitude_distance)

    return ilat, ilon


def crop_domain(data, lat_min, lat_max, lon_min, lon_max, grid_buffer=None):
    """
    Crop dataset domain. Works with regular grids. Irregular grids are on wishlist.
    :param data: DataSet.
    :param lat_min: float.
    :param lat_max: float.
    :param lon_min: float.
    :param lon_max: float.
    :param grid_buffer: list. [x buffer (int), y buffer (int)]. Number of gridpoint to expand outside the real closest
    gridpoint in the margins.
    :return data: DataSet. Original dataset cropped.
    """

    if grid_buffer is None:
        grid_buffer = [0, 0]

    grid_latitudes = data['latitude'].values
    grid_longitudes = data['longitude'].values

    # Get index of the closest grid points
    i_min_lat, i_min_lon = get_nearest_gridpoint(
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
        point_longitude=lon_min,
        point_latitude=lat_min
    )
    i_max_lat, i_max_lon = get_nearest_gridpoint(
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
        point_longitude=lon_max,
        point_latitude=lat_max
    )

    # Abb buffer grid points
    if i_min_lat != len(grid_latitudes):
        i_min_lat = i_min_lat + grid_buffer[1]
    if i_max_lat != 0:
        i_max_lat = i_max_lat - grid_buffer[1]

    if i_min_lon != 0:
        i_min_lon = i_min_lon - grid_buffer[0]
    if i_max_lon != len(grid_longitudes):
        i_max_lon = i_max_lon + grid_buffer[0]

    # Crop domain to a point, line or 2D grid
    if i_max_lat == i_min_lat and i_max_lon != i_min_lon:
        data = data.isel(latitude=i_max_lat, longitude=slice(i_min_lon, i_max_lon))
    elif i_max_lat != i_min_lat and i_max_lon == i_min_lon:
        data = data.isel(latitude=slice(i_max_lat, i_min_lat), longitude=i_max_lon)
    elif i_max_lat == i_min_lat and i_max_lon == i_min_lon:
        data = data.isel(latitude=i_max_lat, longitude=i_max_lon)
    else:
        data = data.isel(latitude=slice(i_max_lat, i_min_lat), longitude=slice(i_min_lon, i_max_lon))

    return data


def separate_concatenated_components(data):
    """
    Separete a concatenated array of vectorial components as different variables and the module of the vector.
    :param data: DataArray. Concatenated vectorial data.
    :return data: DataSet. Separated in 'u', 'v' and 'module'.
    """

    # Get the middle longitude of the variable
    middle_index = int(len(data['longitude'].values) / 2)

    # Divide the concatenated longitudes
    u_index = range(middle_index)
    v_index = range(middle_index, len(data['longitude'].values + 1))

    # Split the eof in the u and v components
    data_u = data.isel(longitude=u_index)
    data_v = data.isel(longitude=v_index)

    # Change the longitude values of v to the original latitudes
    data_v = data_v.assign_coords(longitude=data_u['longitude'].values)

    # Change the name of the components
    data_u.name = 'u'
    data_v.name = 'v'

    # Combine in one dataset
    data = xr.combine_by_coords([data_u, data_v])

    # Calculate the module of the vector
    data['module'] = np.sqrt(data['u'] ** 2 + data['v'] ** 2)

    return data


def get_humidity_to_precipitation(humidity: pd.Series, precipitation: pd.Series, precipitation_threshold=0.25):
    """
    Calculate the relative humidity threshold for precipitation.
    :param humidity: pd.Series.
    :param precipitation: pd.Series.
    :param precipitation_threshold: float. Minimum precipitation threshold.
    :return:
    """
    # Get only precipitation above the minimum threshold
    precipitation = precipitation.mask(precipitation < precipitation_threshold)

    # Get common data
    humidity, precipitation = get_common_index(humidity.to_frame(), precipitation.to_frame())

    # Get inter quartile range
    humidity_q1 = humidity.quantile(0.25).values[0]
    humidity_q3 = humidity.quantile(0.75).values[0]
    inter_quartile_range = humidity_q3 - humidity_q1

    # Get lower adjacent value as minimum threshold
    lower_adjacent_value = humidity_q1 - 1.5 * inter_quartile_range

    # Get box plot
    fig, axs = plt.subplots(1)
    sns.violinplot(x=humidity['RHMA'], ax=axs)
    axs.grid()
    axs.set_title('Lower Adjacent Value: ' + str(lower_adjacent_value) + '%')

    return lower_adjacent_value


def get_station_meta(code):
    """
    Get Station latitude, longitude, altitude and full name
    :param code: str. Code of the station.
    :return station_data: obj.
    """
    network_data = pd.read_csv('./docs/stations.csv')
    station_data = network_data.loc[network_data['code'] == code]
    station_data = Station(station_data)

    return station_data


def table_to_series(df: pd.DataFrame, new_index):
    """
    Transform a table of hourly values per month to a time series.
    """

    climatology_series = pd.DataFrame(index=new_index, columns=df.columns)

    for variable in df.columns:
        for month, month_dataset in climatology_series.groupby(climatology_series.index.month):
            for hour, hourly_dataset in month_dataset.groupby(month_dataset.index.hour):
                climatology_series.loc[hourly_dataset.index, variable] = df.loc[str(hour) + '_' + str(month), variable]

    return climatology_series


def nan_resampler(df, grouping, freq, skipna=True):

    resampled_df = df.resample(freq)
    idx = resampled_df.indices
    if grouping == 'mean':
        resampled_df = [[x[0], x[1].mean(skipna=skipna)] for x in resampled_df]
    elif grouping == "median":
        resampled_df = [[x[0], x[1].median(skipna=skipna)] for x in resampled_df]
    elif grouping == 'sum':
        resampled_df = [[x[0], x[1].sum(skipna=skipna)] for x in resampled_df]
    elif grouping == 'min':
        resampled_df = [[x[0], x[1].min(skipna=skipna)] for x in resampled_df]
    elif grouping == 'max':
        resampled_df = [[x[0], x[1].max(skipna=skipna)] for x in resampled_df]
    else:
        print("ERROR: grouping '" + grouping + "' does not exist")
    resampled_df = pd.DataFrame(resampled_df).set_index(0)
    resampled_df = resampled_df.reindex(idx)

    return resampled_df

