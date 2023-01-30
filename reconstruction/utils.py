import numpy as np
import os
import tqdm
import datetime
import pickle
import pandas as pd
import xarray as xr
import re
import itertools
import autoval.climate
import autoval.utils
import seaborn as sns
import matplotlib.pyplot as plt

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
    def __init__(self, path, variables, dates, lat, lon):
        self.variables = variables
        self.dates = dates
        self.lat = lat
        self.lon = lon

        self.precipitation, self.temperature, self.u, self.v, self.relative_humidity = \
            get_reanalysis(variables, path, dates, lat, lon)


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


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


def get_daily_stations(station_code: str, variable: str):
    """
    Get observed data from RMPNG format or AEMET format. Transform the data to daily resolution and get the relevant
    climatological variables.
    :param station_code: str. Name of the station.
    :param variable: str. Variable acronym.
    :return daily_climatological_variables: DataFrame.
    """

    data_path = '/home/alvaro/data/'
    observations_path = data_path + 'stations/rmpnsg/1h/'
    aemet_path = data_path + 'stations/rmpnsg/1d/'

    # Open all data from a station
    if 'PN' in station_code:
        observations = autoval.utils.open_observations(observations_path + station_code + '/', [variable])
        daily_climatological_variables = autoval.climate.Climatology(observations).climatological_variables()
    else:
        observations = open_aemet(aemet_path + station_code + '/', variable)
        # Get variables in daily resolution
        if variable == 'TMPA':
            daily_climatological_variables = observations.resample('D').mean()
        else:
            daily_climatological_variables = autoval.climate.Climatology(observations).climatological_variables()

    return daily_climatological_variables


def open_grib(grib_paths, grouping=None, number=None):
    """
    Combine a list of grib files in one DataArray.
    :param grib_paths: list. Paths of the grib file to open
    :param grouping: str. Default=None. Format = frequency_method. frequency=('hourly', 'daily', 'monthly', yearly').
    method=('sum', 'mean', 'min', 'max')
    :param number: int. Default=None. Ensemble member number (Only for ERA20CM products)
    :return combined: DataArray. All files concatenated in time
    """

    frequencies = {'hourly': "1H", 'daily': "1D", 'monthly': "1M", 'yearly': "1Y"}

    # Declare a list where each element of the list is one selected grib file in array format
    data_series = [0] * len(grib_paths)

    # Open each file and fill the list
    for i, grib_path in enumerate(grib_paths):

        # First check if the file exists
        if not os.path.isfile(grib_path):
            print('     The file ' + grib_path + ' does not exist')

        else:
            # Load to xarray
            ds = xr.load_dataset(grib_path, engine="cfgrib")
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


def get_grib(nwp_path, variable, dates, grouping=None, number=None):
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
        data = open_grib(all_file_paths, grouping, number)
    else:
        data = open_grib(all_file_paths, grouping)

    return data


def get_nearest_gridpoint(data, lon, lat):
    """
    Find the nearest grid point in a dataset
    :param data: DataSet. Contains "latitude" and "longitude" as dimensions.
    :param lon: float.
    :param lat: float.
    :return data: DataSet. Original dataset in the nearest gridpoint to the selected latitude and longitude.
    """
    nearest_latitude_distance = min(abs(data['latitude'].values - lat))
    nearest_longitude_distance = min(abs(data['longitude'].values - lon))

    ilat = list(abs(data['latitude'].values - lat)).index(nearest_latitude_distance)
    ilon = list(abs(data['longitude'].values - lon)).index(nearest_longitude_distance)

    return ilat, ilon


def get_gridpoint_series(data, lon, lat):
    """
    Find the data in a dataset gridpoint
    :param data: DataSet. Contains "latitude" and "longitude" as dimensions.
    :param lon: float.
    :param lat: float.
    :return data: DataSet. Original dataset in the nearest gridpoint to the selected latitude and longitude.
    """

    ilat, ilon = get_nearest_gridpoint(data, lon, lat)

    data = data.isel(latitude=ilat, longitude=ilon).squeeze()

    return data


def crop_domain(data, latitude_limits, longitude_limits):
    """
    Crop dataset domain.
    :param data: DataSet.
    :param latitude_limits: list = [minimum, maximum]
    :param longitude_limits:  list = [minimum, maximum]
    :return data: DataSet. Original dataset cropped.
    """

    min_lat, max_lat = latitude_limits
    min_lon, max_lon = longitude_limits

    i_min_lat, i_min_lon = get_nearest_gridpoint(data, min_lon, min_lat)
    i_max_lat, i_max_lon = get_nearest_gridpoint(data, max_lon, max_lat)

    data = data.isel(latitude=slice(i_min_lat, i_max_lat), longitude=slice(i_min_lon, i_max_lon))

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
        year_variable = get_grib(
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
        relative_humidity = autoval.climate.Climatology(relative_humidity).climatological_variables()
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

        temperature = autoval.climate.Climatology(temperature).climatological_variables()
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


def concatenate_reanalysis_data(path, variable_names, dates, latitude_limits, longitude_limits, grouping):
    """
    Get reanalysis data and concatenate data of different variables but same domain along the longitude axis.
    Useful for PCA of vectorial magnitudes.
    :param path: str. Path of the reanalysis data.
    :param variable_names: list. Reanalysis code name of the variables.
    :param dates: list. Year to load.
    :param latitude_limits:
    :param longitude_limits:
    :param grouping:
    :return:
    """

    variables = []
    final_lon = 0

    for j, variable_name in enumerate(variable_names):

        variable = get_grib(
            path,
            variable_name,
            dates=dates,
            grouping=grouping
        )
        variable = crop_domain(variable, latitude_limits=latitude_limits, longitude_limits=longitude_limits)

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
    humidity, precipitation = autoval.utils.get_common_index(humidity.to_frame(), precipitation.to_frame())

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
