"""
Download reanalysis data. For now this works with the ECMWF reanalysis, the ERA20C and ERA20CM products

contact: alvaro@intermet.es
"""

import itertools
import os
import multiprocessing

from ecmwfapi import ECMWFDataServer

import antiser.utils

server = ECMWFDataServer()

config = antiser.utils.open_yaml('config.yaml')


def separate_variables_level_type():
    """
    Divide the variables in a list of single level variables and a dictionary of pressure level variables
    """

    pressure_level_variables = {}
    single_level_variables = []

    for sublist in config.get('reanalysis_variable_levels'):
        variable = sublist[0]
        level = sublist[1]
        if level == 'surf':
            single_level_variables.append(variable)
        else:
            pressure_level_variables[variable] = level

    return single_level_variables, pressure_level_variables


def get_downloader_inputs():
    """
    Get a list of inputs for the reanalysis downloader
    :return: inputs. List of variable codes, dates and level.
    """

    initial_year = config.get('initial_year')
    final_year = config.get('final_year')

    single_level_variables, pressure_level_variables = separate_variables_level_type()

    years = [str(y) for y in range(initial_year, final_year + 1)]

    pressure_level_inputs = list(itertools.product(pressure_level_variables.keys(), years, ['pl']))
    single_level_inputs = list(itertools.product(single_level_variables, years, ['sfc']))

    inputs = pressure_level_inputs + single_level_inputs

    return inputs


def get_era20c(inputs):
    """
    Download data from the ERA20C and ERA20CM reanalysis
    :param inputs: list. [variable, year, leveltype]
    """

    variable = inputs[0]
    year = inputs[1]
    leveltype = inputs[2]

    single_level_variables, pressure_level_variables = separate_variables_level_type()

    # Prepare the export file name and directory
    file_path = config.get('reanalysis_path') + '/y_' + str(year) + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if leveltype == 'pl':
        filename = (str(year) + '_' + pressure_level_variables[variable] + '_' + variable)
    elif leveltype == 'sfc':
        filename = (str(year) + '_SURF_' + variable)
    else:
        raise AttributeError(leveltype + ' is not a level type')

    # Model
    if variable == '228':
        parameters = {
            'dataset': "era20cm",
            'type': "fc",
            'number': "0/1/2/3/4/5/6/7/8/9",
            'stream': "enda",
            'step': "3",
            'class': "em",
            "expver": "1"
        }
    else:
        parameters = {
            'dataset': "era20c",
            'stream': "oper",
            'step': "0",
            'type': "an",
        }

    # File domain
    max_lat = config.get('reanalysis_max_lat')
    min_lat = config.get('reanalysis_min_lat')
    max_lon = config.get('reanalysis_max_lon')
    min_lon = config.get('reanalysis_min_lon')

    area = map(str, [max_lat, min_lon, min_lat, max_lon])
    area = '/'.join(area)
    parameters["area"] = area

    # File time
    if variable == '167' or variable == '168':
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
        parameters["class"] = "e2"
        parameters["expver"] = "1"
    else:
        time = "00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00"
    parameters["time"] = time
    parameters["date"] = str(year) + "-01-01/to/" + str(year) + "-12-31"

    # File grid
    parameters['grid'] = "0.75/0.75"

    # File variable
    parameters['param'] = variable
    parameters['levtype'] = leveltype
    if leveltype == 'pl':
        parameters['levelist'] = pressure_level_variables[variable]

    # Ouput path
    parameters['target'] = file_path + filename + '.grib'

    # Download file if it does not exist
    if not os.path.isfile(str(file_path) + str(filename) + '.grib'):
        server.retrieve(parameters)
    else:
        print(str(file_path) + str(filename) + '.grib already exists')


def request_reanalysis(dataset, parallelize=False):

    inputs = get_downloader_inputs()
    if parallelize:
        pool = multiprocessing.Pool()
        if dataset == 'era20c':
            pool.map(get_era20c, inputs)
        pool.close()
        pool.join()

    else:
        if dataset == 'era20c':
            for i in inputs:
                get_era20c(i)
