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

    for i, year in enumerate(dates):
        # grib file of the geopotential in the selected 'anomaly date'
        file_name = (nwp_path + 'y_' + str(year) + '/' + str(year) + '_' + str(variable) + '.grib')
        # If the file exists,put the path in the list and save the correspondent date and hours
        if os.path.isfile(file_name):
            # Put the daily grib file name in the total file names array
            all_file_paths[i] = file_name
        else:
            print(file_name + ' does not exist')

    # Delete empty slots
    all_file_paths = [item for item in all_file_paths if not (pd.isnull(item)) is True]

    # Open the files
    data = open_grib(all_file_paths, mean_type)

    return data


def open_grib(grib_paths, mean_type=None):
    """
    Combine a list of grib files in one DataArray.
    :param grib_paths: list. Paths of the grib file to open
    :return combines: DataArray. All files concatenated in time
    """

    # Declare a list where each element of the list is one selected grib file in array format
    data_series = [0] * len(grib_paths)

    # Open each file and fill the list
    for i, grib_path in tqdm.tqdm(enumerate(grib_paths), desc='   Opening grib files'):

        # First check if the file exists
        if not os.path.isfile(grib_path):
            print('     The file ' + grib_path + ' does not exist')

        else:
            ds = xr.load_dataset(grib_path, engine="cfgrib")
            if mean_type is None:
                pass
            else:
                ds = ds.resample(time=mean_type).mean()

            data_series[i] = ds

    combined = xr.concat(data_series, dim='time')

    return combined
