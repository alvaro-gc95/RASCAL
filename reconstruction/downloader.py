from ecmwfapi import ECMWFDataServer
import pandas as pd
import datetime
import itertools
import os
import multiprocessing

server = ECMWFDataServer()

path = '/home/alvaro/data/NWP/era20c/'

pressure_level_variables = {
    '129': '925',
    '131': '850',
    '132': '850',
    '130': '850',
    '157': '950'
}
single_level_variables = ["228", "165", "166", "246", "247", "167", "71.162", "72.162", "168"]


def get_downloader_inputs(initial_year, final_year):
    """
    Get a list of inputs for the reanalysis downloader
    :param initial_year: int.
    :param final_year: int.
    :return: inputs. List of variable codes, dates and level.
    """

    years = [str(y) for y in range(initial_year, final_year + 1)]

    pl_inputs = list(itertools.product(pressure_level_variables.keys(), years, ['pl']))
    sl_inputs = list(itertools.product(single_level_variables, years, ['sfc']))

    inputs = pl_inputs + sl_inputs

    return inputs


def get_era20c(inputs):

    variable = inputs[0]
    year = inputs[1]
    leveltype = inputs[2]

    file_path = path + '/y_' + str(year) + '/'

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if leveltype == 'pl':
        filename = (str(year) + '_' + str(pressure_level_variables[variable]) + '_' + variable)
    elif leveltype == 'sfc':
        filename = (str(year) + '_SURF_' + variable)
    else:
        raise AttributeError(leveltype + ' is not a level type')

    if not os.path.isfile(str(file_path) + str(filename) + '.grib'):
        if leveltype == 'pl':

            server.retrieve({
                'dataset': "era20c",
                'stream': "oper",
                'levtype': 'pl',
                'levelist': pressure_level_variables[variable],
                'time': "00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00",
                'date': str(year) + '0101/to/' + str(year) + '1231',
                'step': "0",
                'type': "an",
                'area': "80/-60/20/20",
                'grid': "0.75/0.75",
                'param': variable,
                'target': file_path + filename + '.grib'
            })

        elif leveltype == 'sfc':

            if variable == '228':
                server.retrieve({
                    "class": "em",
                    "dataset": "era20cm",
                    "date": str(year) + "-01-01/to/" + str(year) + "-12-31",
                    "expver": "1",
                    "levtype": "sfc",
                    "number": "0/1/2/3/4/5/6/7/8/9",
                    "param": "228.128",
                    "step": "3",
                    "stream": "enda",
                    "time": "00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00",
                    "type": "fc",
                    'area': "80/-60/20/20",
                    'grid': "0.75/0.75",
                    "target": file_path + filename + '.grib',
                })
            elif variable == '167' or variable == '168':
                server.retrieve({
                    "class": "e2",
                    "dataset": "era20c",
                    "date": str(year) + "-01-01/to/" + str(year) + "-12-31",
                    "expver": "1",
                    "levtype": "sfc",
                    "number": "0/1/2/3/4/5/6/7/8/9",
                    "param": variable,
                    "stream": "oper",
                    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
                    "type": "an",
                    'area': "80/-60/20/20",
                    'grid': "0.75/0.75",
                    "target": file_path + filename + '.grib',
                })

            else:
                server.retrieve({
                    'dataset': "era20c",
                    "class": "e2",
                    'stream': "oper",
                    'levtype': "sfc",
                    'time': "00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00",
                    'date': str(year) + '0101/to/' + str(year) + '1231',
                    'step': "0",
                    'type': "an",
                    'area': "80/-60/20/20",
                    'grid': "0.75/0.75",
                    'param': variable,
                    'target': file_path + filename + '.grib'
                })

    else:
        print(str(file_path) + str(filename) + '.grib already exists')
