import os
import re
import tqdm
import yaml
import pickle
import datetime
import itertools
import rascal.climate

import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

variables_longnames = {
    'temperature': 'TMPA',
    'dewpoint_temperature': 'TDEW',
    'precipitation': 'PCNR',
    'relative_humidity': 'RHMA',
    'wind_speed': 'WSPD',
    'wind_direction': 'WDIR'
}

era_variable_names = {
    'TMPA': 't2m',
    'TDEW': 'd2m',
    'PCNR': 'tp',
    'RHMA': 'r'
}

reanalysis_variables = {
    'TMPA': 'SURF_167',
    'PCNR': 'SURF_228',
    'TDEW': 'SURF_168',
    'WSPD': ['SURF_165', 'SURF_166'],
    'RHMA': '950_157'
}


class ReanalysisSeries:
    def __init__(self, path, variables, years, lat, lon):
        self.variables = variables
        self.dates = years
        self.lat = lat
        self.lon = lon

        self.precipitation, self.temperature, self.u, self.v, self.relative_humidity = \
            get_reanalysis(variables, path, years, lat, lon)


class Station:
    def __init__(self, path):
        meta = pd.read_csv(path + 'meta.csv')
        self.path = path

        self.code = meta['code'].values[0]
        self.name = meta['name'].values[0]
        self.longitude = meta['longitude'].values[0]
        self.latitude = meta['latitude'].values[0]
        self.altitude = meta['altitude'].values[0]

    def get_data(self, variable):
        data = get_daily_data(self.path, variable)
        return data

    def get_gridpoint(self, grid_latitudes, grid_longitudes):
        ilat, ilon = get_nearest_gridpoint(
            grid_latitudes=grid_latitudes,
            grid_longitudes=grid_longitudes,
            point_longitude=self.longitude,
            point_latitude=self.latitude
        )
        return grid_latitudes[ilat], grid_longitudes[ilon]


class Predictor:
    def __init__(self, path):
        pass

    def crop_domain(self, lat_min, lat_max, lon_min, lon_max):
        return crop_domain(self.data, lat_min, lat_max, lon_min, lon_max)


# class Station:
#     def __init__(self, pandas_obj):
#         self.code = pandas_obj['code'].values[0]
#         self.name = pandas_obj['name'].values[0]
#         self.longitude = pandas_obj['longitude'].values[0]
#         self.latitude = pandas_obj['latitude'].values[0]
#         self.altitude = pandas_obj['altitude'].values[0]
#
#     def get_data(self, variable):
#         data = get_daily_stations(self.code, variable)
#         return data
#
#     def get_gridpoint(self, grid_latitudes, grid_longitudes):
#         ilat, ilon = get_nearest_gridpoint(
#             grid_latitudes=grid_latitudes,
#             grid_longitudes=grid_longitudes,
#             point_longitude=self.longitude,
#             point_latitude=self.latitude
#         )
#         return grid_latitudes[ilat], grid_longitudes[ilon]


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
            initial_date = test_date - datetime.timedelta(days=np.ceil(window_size / 2))
            final_date = test_date + datetime.timedelta(days=np.floor(window_size / 2))

        validation_window = pd.date_range(start=initial_date, end=final_date, freq='1D')
        validation_window = list(set(validation_window) & set(dates))

        return validation_window


def open_aemet(path, variable_name):
    """
    Open AEMET observations data format.
    :param path: str. Path of the file.
    :param variable_name: str.
    :return:
    """
    variable_acronyms = {
        'PCNR': 'Precipitacion',
        'TMPA': 'Temperaturas',
        'WSPD': 'viento',
        'RHMA': 'Humedad'
    }
    # List of all the files in the directory of observations of the station
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # Search the desired observed variable file through all the files in the directory
    for file, variable in itertools.product(files, [variable_acronyms[variable_name]]):
        # Open if the file corresponds to the selected variable
        if file.find(variable) != -1:
            # Open the file
            variable_df = pd.read_csv(path + file, encoding='latin3', delimiter=';')

    new_variable_columns = []
    original_variable_columns = []
    variables = []

    if variable_name != 'RHMA':

        for col in variable_df.columns:
            match = re.search(r"[A-Z]{1,4}\d{1,2}", col)
            if match:
                variable = re.search(r"[A-Z]{1,4}", col)
                day = re.search(r"\d{1,2}", col)
                original_variable_columns.append(match.group())
                new_variable_columns.append(variable.group() + '_' + day.group())

                variables.append(variable.group())

        variable_df = variable_df.rename(columns=dict(zip(original_variable_columns, new_variable_columns)))
        initial_date = datetime.datetime(variable_df['AÑO'].iloc[0], variable_df['MES'].iloc[0], 1)
        final_date = datetime.datetime(variable_df['AÑO'].iloc[-1], variable_df['MES'].iloc[-1], 31)
        dates = pd.date_range(start=initial_date, end=final_date, freq='1D')

        variables = list(set(variables))
        if 'MET' in variables:
            variables.remove('MET')
        new_variable_df = pd.DataFrame(index=dates, columns=variables)

        for variable, date in itertools.product(variables, dates):

            value = variable_df.loc[
                (variable_df['AÑO'] == date.year) &
                (variable_df['MES'] == date.month), variable + '_' + str(date.day)].values

            if len(value) == 0:
                value = np.nan
            else:
                value = value[0]

            new_variable_df.loc[date, variable] = value

    else:

        hours = []

        for col in variable_df.columns:
            match = re.search(r"[A-Z]{1,4}\d{1,2}", col)
            if match:
                variable = re.search(r"[A-Z]{1,4}", col)
                hour = re.search(r"\d{1,2}", col)
                original_variable_columns.append(match.group())
                new_variable_columns.append(variable.group() + '_' + hour.group())

                hours.append(hour.group())
                variables.append(variable.group())

        hours = list(set(hours))

        variable_df = variable_df.rename(columns=dict(zip(original_variable_columns, new_variable_columns)))

        initial_date = datetime.datetime(
            variable_df['AÑO'].iloc[0],
            variable_df['MES'].iloc[0],
            variable_df['DIA'].iloc[0]
        )
        final_date = datetime.datetime(
            variable_df['AÑO'].iloc[-1],
            variable_df['MES'].iloc[-1],
            variable_df['DIA'].iloc[-1]
        )
        dates = pd.date_range(start=initial_date, end=final_date, freq='1H')
        dates = [date for date in dates if str(date.hour).zfill(2) in hours]

        variables = list(set(variables))
        if 'MET' in variables:
            variables.remove('MET')
        new_variable_df = pd.DataFrame(index=dates, columns=variables)

        for variable, date in itertools.product(variables, dates):

            value = variable_df.loc[
                (variable_df['AÑO'] == date.year) &
                (variable_df['MES'] == date.month) &
                (variable_df['DIA'] == date.day), variable + '_' + str(date.hour).zfill(2)].values

            if len(value) == 0:
                value = np.nan
            else:
                value = value[0]

            new_variable_df.loc[date, variable] = value

    if 'TMIN' in variables and 'TMAX' in variables:
        new_variable_df['TMEAN'] = (new_variable_df['TMAX'] - new_variable_df['TMIN']) / 2
        new_variable_df = new_variable_df / 10
    if 'P' in variables:
        new_variable_df = new_variable_df.rename(columns={'P': 'PCNR'})
        new_variable_df = new_variable_df / 10
    new_variable_df = new_variable_df.astype(np.float64)
    if 'HU' in variables:
        new_variable_df = new_variable_df.rename(columns={'HU': 'RHMA'})

    return new_variable_df


def open_observations(path: str, variables: list):
    """
    Get and rename all observational data from one directory as pandas DataFrame.
    :param path: str. Path of the files to open.
    :param variables: list. Acronyms as str of the variables to open.
    """
    # Declare an empty dataframe for the complete observations
    data = pd.DataFrame()
    # List of all the files in the directory of observations of the station
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # Search the desired observed variable file through all the files in the directory
    for file, variable in itertools.product(files, variables):
        # Open if the file corresponds to the selected variable
        if file.find(variable) != -1:
            # Open the file
            variable_data = pd.read_csv(path + file, index_col=0)
            # Rename the values column
            variable_data.columns.values[0] = variable
            # Change the format of the index to datetime
            variable_data.index = pd.to_datetime(variable_data.index)
            # Add to the complete DataFrame
            data = pd.concat([data, variable_data], axis=1)
    # Check if the data exists
    if data.empty:
        print('Warning: Empty data. Files may not exist in ' + path)
        exit()
    else:
        return data


def get_daily_data(path: str, variable: str):
    observations = open_observations(path, [variable])
    daily_observations = rascal.climate.Climatology(observations).climatological_variables()
    return daily_observations


# def get_daily_stations(station_code: str, variable: str):
#     """
#     Get observed data from RMPNG format or AEMET format. Transform the data to daily resolution and get the relevant
#     climatological variables.
#     :param station_code: str. Name of the station.
#     :param variable: str. Variable acronym.
#     :return daily_climatological_variables: DataFrame.
#     """
#
#     data_path = '/home/alvaro/data/'
#     observations_path = data_path + 'stations/rmpnsg/1h/'
#     aemet_path = data_path + 'stations/rmpnsg/1d/'
#
#     # Open all data from a station
#     if 'PN' in station_code:
#         observations = open_observations(observations_path + station_code + '/', [variable])
#         daily_climatological_variables = rascal.climate.Climatology(observations).climatological_variables()
#     else:
#         print(aemet_path + station_code + '/', variable)
#         observations = open_aemet(aemet_path + station_code + '/', variable)
#         # Get variables in daily resolution
#         if variable == 'TMPA':
#             daily_climatological_variables = observations.resample('D').mean()
#         else:
#             daily_climatological_variables = rascal.climate.Climatology(observations).climatological_variables()
#
#     return daily_climatological_variables


def open_data(files_paths, grouping=None, number=None):
    """
    Combine a list of files (.grib or .nc usually) in one DataArray.
    :param files_paths: list. Paths of the grib file to open
    :param grouping: str. Default=None. Format = frequency_method. frequency=('hourly', 'daily', 'monthly', yearly').
    method=('sum', 'mean', 'min', 'max')
    :param number: int. Default=None. Ensemble member number (Only for ERA20CM products)
    :return combined: DataArray. All files concatenated in time
    """

    frequencies = {'hourly': "1H", 'daily': "1D", 'monthly': "1M", 'yearly': "1Y"}

    # Declare a list where each element of the list is one selected grib file in array format
    data_series = [0] * len(files_paths)

    # Open each file and fill the list
    for i, file_path in enumerate(files_paths):

        # First check if the file exists
        if not os.path.isfile(file_path):
            print('     The file ' + file_path + ' does not exist')

        else:
            # Load to xarray
            if file_path.split('.')[-1] == 'grib':
                ds = xr.load_dataset(file_path, engine="cfgrib")
            else:
                ds = xr.load_dataset(file_path)
            ds = ds.astype(np.float32)

            # Group the data
            if grouping is None:
                ds_time = [date for date in pd.to_datetime(ds['time'].values) if date.hour == 12]
                ds = ds.sel(time=ds_time)
                ds = ds.resample(time='1D').mean()
            else:

                frequency, group_type = grouping.split('_')

                if group_type == 'sum':
                    ds = ds.resample(time=frequencies[frequency]).sum()
                elif group_type == 'mean':
                    ds = ds.resample(time=frequencies[frequency]).mean()
                elif group_type == 'min':
                    ds = ds.resample(time=frequencies[frequency]).min()
                elif group_type == 'max':
                    ds = ds.resample(time=frequencies[frequency]).max()
                else:
                    raise AttributeError('Grouping method (' + group_type + ') does not exists')

            # Select ensemble member if possible
            if number is not None:
                ds = ds.sel(number=number).squeeze()

            data_series[i] = ds

    combined = xr.concat(data_series, dim='time')

    return combined


def get_data(nwp_path, variable, dates, grouping=None, number=None):
    """
    Open a set of daily grib files.
    :param nwp_path: str. Path to the grib files.
    :param variable: str. Variable to open.
    :param dates: list. Dates to open.
    :param grouping: str. Default=None. Format = frequency_method. frequency=('hourly', 'daily', 'monthly', yearly').
    method=('sum', 'mean', 'min', 'max')
    :param number: int. Default=None. Ensemble member number (Only for ERA20CM products)
    :return data: array. All the daily data in a singular array.
    """

    # List of al the grib files
    all_file_paths = [np.nan] * len(dates)

    for i, year in enumerate(dates):
        # GRIB filename
        file_name = (nwp_path + 'y_' + str(year) + '/' + str(year) + '_' + str(variable) + '.grib')
        # If the file exists, put the path in the list and save the correspondent date and hours
        if os.path.isfile(file_name):
            # Put the daily grib file name in the total file names array
            all_file_paths[i] = file_name
        else:
            print(file_name + ' does not exist')

    # Delete empty slots
    all_file_paths = [item for item in all_file_paths if not (pd.isnull(item)) is True]

    # Open the files
    if number is not None:
        data = open_data(all_file_paths, grouping, number)
    else:
        data = open_data(all_file_paths, grouping)

    return data


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


def get_gridpoint_series(data, lon, lat):
    """
    Find the data in a dataset gridpoint
    :param data: DataSet. Contains "latitude" and "longitude" as dimensions.
    :param lon: float.
    :param lat: float.
    :return data: DataSet. Original dataset in the nearest gridpoint to the selected latitude and longitude.
    """
    grid_latitudes = data['latitude'].values
    grid_longitudes = data['longitude'].values

    ilat, ilon = get_nearest_gridpoint(
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
        point_longitude=lon,
        point_latitude=lat
    )

    data = data.isel(latitude=ilat, longitude=ilon).squeeze()

    return data


def crop_domain(data, lat_min, lat_max, lon_min, lon_max):
    """
    Crop dataset domain.
    :param data: DataSet.
    :param lat_min: float.
    :param lat_max: float.
    :param lon_min: float.
    :param lon_max: float.
    :return data: DataSet. Original dataset cropped.
    """

    grid_latitudes = data['latitude'].values
    grid_longitudes = data['longitude'].values

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

    data = data.isel(latitude=slice(i_max_lat, i_min_lat), longitude=slice(i_min_lon, i_max_lon))

    return data


def reanalysis_ensemble_to_dataframe(path, dates, variable, lat, lon, grouping):
    variable_series = []

    for ensemble_number in range(9):
        ensemble_member = reanalysis_to_dataframe(path, dates, variable, lat, lon, grouping, ensemble_number)
        variable_series.append(ensemble_member)

    variable_series = pd.concat(variable_series, axis=1)
    variable_series.index = pd.to_datetime(variable_series.index)
    variable_series.to_csv('ERA20CM_' + variable + '.csv')

    return variable_series


def reanalysis_to_dataframe(path, dates, variable, lat, lon, grouping, ensemble_number=None):
    variable_series = []

    for year in tqdm.tqdm(dates, desc='   Opening grib files'):
        # Get the grib data for each year
        year_variable = get_data(
            path,
            variable,
            dates=[year],
            grouping=grouping,
            number=ensemble_number
        )

        # Get data in the grid point
        year_variable = get_gridpoint_series(
            year_variable,
            lat=lat,
            lon=lon
        )
        print(year_variable)
        year_variable = year_variable.to_dataframe()

        print(year_variable)
        # Get the netcdf name of the variable
        level, variable_reanalysis_code = variable.split('_')
        variable_code = [code for code, code_reanalysis in reanalysis_variables.items()
                         if variable_reanalysis_code in code_reanalysis][0]
        variable_netcdf_name = era_variable_names[variable_code]

        # Select only the variable values
        year_variable = year_variable[variable_netcdf_name]

        if ensemble_number is None:
            year_variable = year_variable.rename(variable_code)
        else:
            year_variable = year_variable.rename({variable_netcdf_name: variable_code + '_' + str(ensemble_number)})

        variable_series.append(year_variable)

    variable_series = pd.concat(variable_series, axis=0)

    # Save the dataframe
    variable_series.index = pd.to_datetime(variable_series.index)

    if ensemble_number is None:
        variable_series.to_csv('ERA20C_' + variable + '.csv')
    else:
        variable_series.to_csv('ERA20C_' + variable + '_' + str(ensemble_number) + '.csv')

    return variable_series


def get_reanalysis(variables, reanalysis_path, dates, lat, lon):
    precipitation = None
    temperature = None
    u_component = None
    v_component = None
    relative_humidity = None

    # Get reanalysis data in the gridpoint
    if 'PCNR' in variables:
        if os.path.isfile('ERA20CM_' + reanalysis_variables['PCNR'] + '.csv'):
            precipitation = pd.read_csv('ERA20CM_' + reanalysis_variables['PCNR'] + '.csv', index_col=0)
            precipitation.index = pd.to_datetime(precipitation.index)
        else:
            reanalysis_ensemble_to_dataframe(
                path=reanalysis_path,
                dates=dates,
                variable=reanalysis_variables['PCNR'],
                lat=lat,
                lon=lon,
                grouping='daily_sum'
            )
        precipitation.columns = [col + ' reanalysis' for col in precipitation.columns]
        precipitation = precipitation * 1000

        if os.path.isfile('ERA20C_' + reanalysis_variables['RHMA'] + '.csv'):
            relative_humidity = pd.read_csv('ERA20C_' + reanalysis_variables['RHMA'] + '.csv', index_col=0)
            relative_humidity.index = pd.to_datetime(relative_humidity.index)
        else:

            relative_humidity = reanalysis_to_dataframe(
                path=reanalysis_path,
                dates=dates,
                variable=reanalysis_variables['RHMA'],
                lat=lat,
                lon=lon,
                grouping=None
            )
        print(relative_humidity)
        relative_humidity = rascal.climate.Climatology(relative_humidity).climatological_variables()
        print(relative_humidity)
        relative_humidity.columns = [col + ' reanalysis' for col in relative_humidity.columns]
        print(relative_humidity)
        """
        if (os.path.isfile('ERA20C_' + reanalysis_variables['TDEW'] + '.csv') and
                os.path.isfile('ERA20C_' + reanalysis_variables['TMPA'] + '.csv')):

            temperature_dew = pd.read_csv('ERA20C_' + reanalysis_variables['TDEW'] + '.csv', index_col=0)
            temperature_dew.index = pd.to_datetime(temperature_dew.index)

            temperature = pd.read_csv('ERA20C_' + reanalysis_variables['TMPA'] + '.csv', index_col=0)
            temperature.index = pd.to_datetime(temperature.index)

        else:
            temperature_dew = reanalysis_to_dataframe(
                path=reanalysis_path,
                dates=dates,
                variable=reanalysis_variables['TDEW'],
                lat=lat,
                lon=lon,
                grouping=None
            )
            temperature = reanalysis_to_dataframe(
                path=reanalysis_path,
                dates=dates,
                variable=reanalysis_variables['TMPA'],
                lat=lat,
                lon=lon,
                grouping=None
            )

        df = pd.concat([temperature_dew, temperature], axis=1)
        autoval.utils.Preprocess(df).calculate_relative_humidity()
        relative_humidity = df['RHMA'].to_frame()
        relative_humidity.columns = [col + ' reanalysis' for col in relative_humidity.columns]
        """
    if 'TMPA' in variables:
        if os.path.isfile('ERA20C_' + reanalysis_variables['TMPA'] + '.csv'):
            temperature = pd.read_csv('ERA20C_' + reanalysis_variables['TMPA'] + '.csv', index_col=0)
            temperature.index = pd.to_datetime(temperature.index)
        else:
            temperature = reanalysis_to_dataframe(
                path=reanalysis_path,
                dates=dates,
                variable=reanalysis_variables['TMPA'],
                lat=lat,
                lon=lon,
                grouping=None
            )

        temperature = rascal.climate.Climatology(temperature).climatological_variables()
        temperature = temperature - 273.15
        temperature.columns = [col + ' reanalysis' for col in temperature.columns]

    if 'WSPD' in variables or 'WDIR' in variables:

        if os.path.isfile('ERA20C_' + reanalysis_variables['WSPD'][0] + '.csv'):
            u_component = pd.read_csv('ERA20C_' + reanalysis_variables['WSPD'][0] + '.csv', index_col=0)
            u_component.index = pd.to_datetime(u_component.index)
        else:
            u_component = reanalysis_to_dataframe(
                path=reanalysis_path,
                dates=dates,
                variable=reanalysis_variables['WSPD'][0],
                lat=lat,
                lon=lon,
                grouping='daily_mean'
            )

        if os.path.isfile('ERA20C_' + reanalysis_variables['WSPD'][1] + '.csv'):
            v_component = pd.read_csv('ERA20C_' + reanalysis_variables['WSPD'][1] + '.csv', index_col=0)
            v_component.index = pd.to_datetime(v_component.index)
        else:
            v_component = reanalysis_to_dataframe(
                path=reanalysis_path,
                dates=dates,
                variable=reanalysis_variables['WSPD'][1],
                lat=lat,
                lon=lon,
                grouping='daily_mean'
            )

    return precipitation, temperature, u_component, v_component, relative_humidity


def get_predictor(path, variable_names, years, latitude_limits, longitude_limits, grouping):
    """
    Get the predictor data (usually reanalysis) and concatenate data of different variables but same domain along the
    longitude axis.
    Useful for PCA of vectorial magnitudes.
    :param path: str. Path of the reanalysis data.
    :param variable_names: list. Reanalysis code name of the variables.
    :param years: list. Year to load.
    :param latitude_limits:
    :param longitude_limits:
    :param grouping:
    :return:
    """

    variables = []
    final_lon = 0

    for j, variable_name in enumerate(variable_names):

        variable = get_data(
            path,
            variable_name,
            dates=years,
            grouping=grouping
        )
        variable = crop_domain(
            variable,
            lat_min=latitude_limits[0],
            lat_max=latitude_limits[1],
            lon_min=longitude_limits[0],
            lon_max=longitude_limits[1]
        )

        original_name = [i for i in variable.data_vars][0]
        variable = variable.rename({original_name: 'z'})

        if j != 0:
            # Get the differences between the first longitude and the rest of the list.
            # Add 1 so the first element is not zero
            longitude_diffs = [lon - variable['longitude'].values[0] + 1 for lon in variable['longitude'].values]
            # Get the new longitudes to concatenate
            new_longitude = longitude_diffs + final_lon
            variable = variable.assign_coords(longitude=new_longitude)

        final_lon = variable['longitude'].values[-1]
        variables.append(variable)

    variables = xr.combine_by_coords(variables)
    variables["time"] = pd.to_datetime(variables["time"].values)

    return variables


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
