from ecmwfapi import ECMWFDataServer
import pandas as pd
import datetime
import itertools
import os
import multiprocessing

server = ECMWFDataServer()

path = '/mnt/disco2/data/NWP/era20c/'

pressure_level_variables = {
    '129': '925',
    '130': '850',
    '131': '850',
    '132': '850',
    '157': '850'
}

single_level_variables = ['167', '168']

dates = pd.date_range(start=datetime.datetime(1900, 1, 1), end=datetime.datetime(2010, 12, 30))


def get_era20c(inputs):

    variable = inputs[0]
    year = inputs[1]
    leveltype = inputs[2]

    file_path = path + 'y_' + str(year) + '/'

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    filename = (str(year).zfill(4) + '_'
                + str(pressure_level_variables[variable]) + '_'
                + str(variable))

    if not os.path.isfile(str(file_path) + str(filename) + '.grib'):
        if leveltype == 'pl':
            server.retrieve({
                'dataset': "era20c",
                'stream': "oper",
                'levtype': 'pl',
                'levelist': pressure_level_variables[variable],
                'time': "00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00",
                'date': str(year) + '0101/to/' + str(year) +'1230',
                'step': "0",
                'type': "an",
                'area': "80/-60/20/20",
                'grid': "0.75/0.75",
                'param': variable,
                'target': file_path + filename + '.grib'
            })

        elif leveltype == 'sl':
            server.retrieve({
                'dataset': "era20c",
                'stream': "oper",
                'levtype': 'sl',
                'time': "00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00",
                'date': str(year) + '0101/to/' + str(year) +'1230',
                'step': "0",
                'type': "an",
                'area': "80/-60/20/20",
                'grid': "0.75/0.75",
                'param': variable,
                'target': file_path + filename + '.grib'
            })
    else:
        print(str(file_path) + str(filename) + '.grib already exists')
