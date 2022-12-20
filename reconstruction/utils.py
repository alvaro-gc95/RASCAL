import numpy as np
import os
import tqdm
import pygrib
import pandas as pd
import xarray as xr


def get_reanalysis_data(nwp_path, variable, dates, hours, mean_type=None):
    """
    Open a set of daily grib files.
    :param project: str. Project of the grib files.
    :param variable: str. Variable to open.
    :param dates: list. Dates to open.
    :param hours: int. Hours to get from each daily grib file.
    :param lats_idx: list. [initial_latitude, final_latitude] indexes of the latitude limits
    :param lons_idx: list. [initial_longitude, final_longitude] indexes of the longitude limits
    :param mean_type: str. Default=None. If 'daily mean' it takes the mean of the selected hours of the daily grib file
    :return data: array. All the daily data in a singular array.
    :return available_dates: list. List of dates with available data
    """

    # List of al the grib files
    all_file_paths = [np.nan] * len(dates)

    # Dates with available data
    available_dates = []

    for i in range(len(dates)):
        day = dates[i].day
        year = dates[i].year
        month = dates[i].month

        # grib file of the geopotential in the selected 'anomaly date'
        file_data = (str(nwp_path)
                     + 'y_' + str(year) + '/'
                     + str(variable) + '.grib')

        # If the file exists,put the path in the list and save the correspondent date and hours
        if os.path.isfile(file_data):
            # Put the daily grib file name in the total file names array
            all_file_paths[i] = file_data
            # Save the date if the grib file exists
            available_dates.extend([str(int(year)) + '-'
                                    + str(int(month)).zfill(2) + '-'
                                    + str(int(day)).zfill(2) + ' '
                                    + str(int(h)).zfill(2) + ':00' for h in hours])
        else:
            print(file_data + ' does not exist')

    all_file_paths = [item for item in all_file_paths if not (pd.isnull(item)) == True]
    print(len(all_file_paths))
    # Open the files
    data = open_grib(all_file_paths, hours=hours, mean_type=mean_type)

    if mean_type is None:
        data = np.reshape(data, (data.shape[0] * data[1], data.shape[2], data.shape[3]))

    return data, available_dates


def open_grib(grib_paths, hours, **kwargs):
    """
    Open an individual grib.
    :param grib_paths: str. Path of the grib file to open
    :param hours: int. Hours to get from the daily grib file
    :param kwargs:
                  - **lat (optional): list of index points of latitude points to crop from the original domain.
                                      It can be either only one point or two (lower and upper limit).
                                      If none is selected the full domain is used
                  - **lon (optional): list of index points of longitude points to crop from the original domain.
                                      It can be either only one point or two (lower and upper limit).
                                      If none is selected the full domain is used
                  - **mean type (optional): 'None' by default. If 'daily mean' it takes the mean of the selected hours
                                      of the daily grib file
    :return data: array. grib file in array format
    """

    # Declare a list where each element of the list is one selected grib file in array format
    data_series = [0] * len(grib_paths)

    # Open each file and fill the list
    for i in tqdm.tqdm(range(len(grib_paths)), desc='   Opening grib files'):

        # Path of the file
        grib_path = grib_paths[i]

        # First check if the file exists
        if not os.path.isfile(grib_path):
            print('     The file ' + grib_path + ' does not exist')

        else:

            ds = xr.load_dataset(grib_path, engine="cfgrib")


            """
            # Open grib file
            dataset = pygrib.open(grib_path)
            grb = dataset.select()
            dataset.close()

            # Transform the daily raw data into array
            all_data = np.array([g.values for g in grb])

            # Warnings of incomplete files or possible errors
            if np.isnan(all_data).any():
                print('     Warning: NaN in:' + str(grib_path))

            elif not all_data.shape[0] == 24:
                print('     Warning: Missing hours in:' + str(grib_path))

            elif os.stat(grib_path).st_size == 0:
                print('     Warning: Empty file: ' + str(grib_path))

            else:

                # Crop domain in latitude and longitude
                if 'lat' in kwargs.keys() and 'lon' in kwargs.keys():
                    # Check repeated values
                    kwargs['lat'] = list(dict.fromkeys(kwargs['lat']))
                    kwargs['lon'] = list(dict.fromkeys(kwargs['lon']))
                    # Select surface domain
                    if len(kwargs['lat']) == 2 and len(kwargs['lon']) == 2:
                        lon_ini = kwargs['lon'][0]  # Longitude lower limit
                        lon_end = kwargs['lon'][1]  # Longitude upper limit
                        lat_ini = kwargs['lat'][0]  # Latitude lower limit
                        lat_end = kwargs['lat'][1]  # Latitude upper limit
                        # Select hours
                        data = np.array(all_data[hours, lat_end:lat_ini + 1, lon_ini:lon_end + 1])

                    # Select latitude line domain
                    elif len(kwargs['lat']) == 2 and len(kwargs['lon']) == 1:
                        lon = kwargs['lon'][0]  # Longitude point
                        lat_ini = kwargs['lat'][0]  # Latitude lower limit
                        lat_end = kwargs['lat'][1]  # Latitude upper limit
                        # Select hours
                        data = np.array(all_data[hours, lat_end:lat_ini + 1, lon])
                    # Select longitude line domain
                    elif len(kwargs['lat']) == 1 and len(kwargs['lon']) == 2:
                        lat = kwargs['lat'][0]  # Latitude point
                        lon_ini = kwargs['lon'][0]  # Longitude lower limit
                        lon_end = kwargs['lon'][1]  # Longitude upper limit
                        # Select hours
                        data = np.array(all_data[hours, lat, lon_ini:lon_end + 1])

                    # Select point domain
                    elif len(kwargs['lat']) == 1 and len(kwargs['lon']) == 1:
                        lon = kwargs['lon'][0]  # Longitude point
                        lat = kwargs['lat'][0]  # Latitude point
                        # Select hours
                        data = np.array(all_data[hours, lat, lon])

                # Crop only in latitude
                elif 'lat' in kwargs.keys() and 'lon' not in kwargs.keys():

                    # Select surface domain
                    if len(kwargs['lat']) == 2:
                        lat_ini = kwargs['lat'][0]  # Latitude lower limit
                        lat_end = kwargs['lat'][1]  # Latitude upper limit
                        # Select hours
                        data = np.array(all_data[hours, lat_end:lat_ini + 1, :])

                    # Select longitude line domain
                    elif len(kwargs['lat']) == 1:
                        lat = kwargs['lat'][0]  # Latitude point
                        # Select hours
                        data = np.array(all_data[hours, lat, :])

                # Crop only in longitude
                elif 'lon' in kwargs.keys() and 'lat' not in kwargs.keys():

                    # Select surface domain
                    if len(kwargs['lon']) == 2:
                        lon_ini = kwargs['lon'][0]  # Longitude lower limit
                        lon_end = kwargs['lon'][1]  # Longitude upper limit
                        # Select hours
                        data = np.array(all_data[hours, :, lon_ini:lon_end + 1])

                    # Select latitude line domain
                    elif len(kwargs['lon']) == 1:
                        lon = kwargs['lon'][0]  # Latitude point
                        # Select hours
                        data = np.array(all_data[hours, :, lon])

                # Original domain
                else:
                    data = np.array(all_data[hours, :, :])

                # If mean type is selected, do the mean of the selected hours
                if 'mean_type' in kwargs.keys():
                    if kwargs['mean_type'] == 'daily':
                        data = np.nanmean(data, 0)

                # Add the file to the list of files
                data_series[i] = data
            """
    """
    # Re-format from list to array
    data_series = np.array(data_series)

    return data_series
    """