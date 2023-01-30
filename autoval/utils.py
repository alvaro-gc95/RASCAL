"""
Utils for AutoVal objects

Contact: alvaro@intermet.es
"""

import yaml
import os
import argparse
import itertools
import numpy as np
import pandas as pd
from datetime import datetime


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


class Configuration:
    """
    Store the configuration parameters
    """

    def __init__(self):
        config = open_yaml('../config.yaml')

        self.stations = config.get('stations')
        self.variables = config.get('variables')
        self.reference = config.get('reference')
        self.DateIni = datetime.strptime(config.get('DateIni'), '%d-%m-%Y')
        self.DateEnd = datetime.strptime(config.get('DateEnd'), '%d-%m-%Y')

    def time(self, freq):
        return pd.date_range(start=self.DateIni, end=self.DateEnd, freq=freq)

    def make_inputs(self):
        inputs = [None] * len(self.stations) * len(self.variables)
        for i, (station, variable) in enumerate(list(itertools.product(self.stations, self.variables))):
            inputs[i] = Inputs(station, self.reference, variable, self.DateIni, self.DateEnd)
        return inputs


class Inputs:
    """
    Store function inputs for individual stations and variables.
    """

    def __init__(self, station, reference, variable, date_ini, date_end):
        self.station = station
        self.reference = reference
        self.variable = variable
        self.DateIni = date_ini
        self.DateEnd = date_end


def open_yaml(yaml_path):
    """
    Read the configuration yaml file.
    :param yaml_path: str. Path of the yaml file
    :return configuration file: Object. Object containing the information of the configuration file.
    """
    # Check if the yaml exists
    if not os.path.exists(yaml_path):
        print(
            'WARNING: The configuration file ' + yaml_path + ' does not exist')
        exit()
    # Read data in ini
    with open(yaml_path, 'r') as stream:
        try:
            configuration_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return configuration_file


def get_inputs():
    """
    Get the input arguments from the terminal
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--station', type=str, help='Station of origin of the observed data')
    parser.add_argument('--variable', type=str, help='Observed variable acronym')
    parser.add_argument('--reference', type=str, help='Reference station')
    parser.add_argument('--DateIni', type=lambda d: datetime.strptime(d, '%d-%m-%Y'), help='Start date [DD-MM-YYYY]')
    parser.add_argument('--DateEnd', type=lambda d: datetime.strptime(d, '%d-%m-%Y'), help='End date [DD-MM-YYYY]')

    return parser.parse_args()


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
