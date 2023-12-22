import datetime
import itertools
import calendar
import os
import re

import numpy as np
import pandas as pd


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


def open_aemet(path, variable_name):
    """
    Open AEMET observations data format.
    :param path: str. Path of the file.
    :param variable_name: str.
    :return:
    """
    variable_acronyms = {
        'PCNR': 'Precipitacion',
        'TMPA': 'Temperatura',
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
        _, days_in_month = calendar.monthrange(variable_df['AÑO'].iloc[-1], variable_df['MES'].iloc[-1])
        final_date = datetime.datetime(variable_df['AÑO'].iloc[-1], variable_df['MES'].iloc[-1], days_in_month)
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