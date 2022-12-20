from ecmwfapi import ECMWFDataServer
import pandas as pd
import datetime
import itertools
import os
import multiprocessing

server = ECMWFDataServer()

path = '/home/alvaro/data/NWP/era20c'

pressure_level_variables = {
    '129': '925',
    '130': '850',
    '131': '850',
    '157': '850'
}

single_level_variables = ['167', '168']

dates = pd.date_range(start=datetime.datetime(1900, 1, 1), end=datetime.datetime(2010, 12, 30))


def get_era20c(inputs):

    variable = inputs[0]
    date = inputs[1]
    leveltype = inputs[2]

    file_path = path + '/y_' + str(date.year) + '/m_' + str(date.month) + '/d_' + str(date.day) + '/'

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    filename = (str(date.year).zfill(4) + '_'
                + str(date.month).zfill(2) + '_'
                + str(date.day).zfill(2) + '_'
                + str(pressure_level_variables[variable]) + '_'
                + str(variable))

    if not os.path.isfile(str(file_path) + str(filename) + '.grib'):
        if leveltype == 'pl':
            server.retrieve({
                'dataset': "era20c",
                'stream': "oper",
                'levtype': 'pl',
                'levelist': pressure_level_variables[variable],
                'time': "00/to/21",
                'date': date.strftime('%Y%m%d'),
                'step': "0",
                'type': "an",
                'param': variable,
                'target': file_path + filename + '.grib'
            })

        elif leveltype == 'sl':
            server.retrieve({
                'dataset': "era20c",
                'stream': "oper",
                'levtype': 'sl',
                'time': "00/to/21",
                'date': date.strftime('%Y%m%d'),
                'step': "0",
                'type': "an",
                'param': variable,
                'target': file_path + filename + '.grib'
            })
    else:
        print(str(file_path) + str(filename) + '.grib already exists')
