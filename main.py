import multiprocessing
import itertools
import reconstruction.downloader as dw
import reconstruction.utils as ut
import datetime
import pandas as pd
import numpy as np

pressure_level_variables = {
    '129': '925',
    '130': '850',
    '131': '850',
    '157': '850'
}

single_level_variables = ['167', '168']

dates = pd.date_range(start=datetime.datetime(1900, 1, 1), end=datetime.datetime(2010, 12, 30))

pl_inputs = list(itertools.product(pressure_level_variables.keys(), dates, ['pl']))
sl_inputs = list(itertools.product(single_level_variables, dates, ['sl']))

inputs = pl_inputs + sl_inputs

if __name__ == '__main__':

    pool = multiprocessing.Pool()
    pool.map(dw.get_era20c, inputs)
    pool.close()
    pool.join()
    """
    geopotential, dates = ut.get_reanalysis_data('era20c', '925_129', dates=dates, hours=np.arange(0, 24, 3))
    print(geopotential.shape)
    print(len(dates))
    """
