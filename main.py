import multiprocessing
import itertools
import reconstruction.downloader as dw
import reconstruction.utils
import reconstruction.analogs
import datetime
import pandas as pd
import numpy as np
import autoval.utils
import autoval.climate
import matplotlib.pyplot as plt
import autoval.validation_tests

pressure_level_variables = {
    '129': '925',
    '130': '850',
    '131': '850',
    '157': '850'
}

single_level_variables = ['167', '168']

# dates = pd.date_range(start=datetime.datetime(1900, 1, 1), end=datetime.datetime(2010, 12, 30))

dates = [str(y) for y in range(1900, 2011)]

pl_inputs = list(itertools.product(pressure_level_variables.keys(), dates, ['pl']))
sl_inputs = list(itertools.product(single_level_variables, dates, ['sl']))

inputs = pl_inputs + sl_inputs

# Paths
data_path = '/mnt/disco2/data/'
era20c_path = data_path + 'NWP/era20c/'
era5_path = data_path + 'NWP/era5/60.0W20.0E20.0N80.0N/'
observations_path = data_path + 'stations/rmpnsg/1h/'

station = 'PN001002'

observed_variables = ['TMPA', 'PCNR', 'WSPD', 'WDIR']

if __name__ == '__main__':

    # Request reanalysis data
    # pool = multiprocessing.Pool()
    # pool.map(dw.get_era20c, inputs)
    # pool.close()
    # pool.join()

    # Open all data from a station
    observations = autoval.utils.open_observations(observations_path + station + '/', observed_variables)

    # Get variables in daily resolution
    daily_climatological_variables = autoval.climate.Climatology(observations).climatological_variables()

    # Get reanalysis data
    training_dates = pd.date_range(start=datetime.datetime(1980, 1, 1), end=datetime.datetime(2020, 12, 30))
    geopotential = reconstruction.utils.get_reanalysis_data(
        era20c_path,
        '925_129',
        dates=dates,
        hours=np.arange(0, 24, 3),
        mean_type="1D"
    )
    geopotential['z'].isel(time=0).plot()

    # Calculate PCs
    geopotential_anomalies = reconstruction.analogs.calculate_anomalies(geopotential, standardize=True)
    geopotential_anomalies['z'].isel(time=0).plot()

    # Find analogs
    eofs, pcs = reconstruction.analogs.get_pca(geopotential_anomalies['z'], ncomponents=3)
    for mode in range(3):
        eofs.isel(mode=mode).plot()
        plt.show()
    print(eofs)

    # Analyze differences
