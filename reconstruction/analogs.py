from eofs.xarray import Eof
import xarray as xr


def calculate_anomalies(data_array: xr.DataArray, standardize=False):
    """
    :param data_array: DataArray.
    :param standardize: bool. Default=False. If True divide the anomalies by its standard deviation.
    :return anomalies: DataArray.
    """
    mean = data_array.mean(dim='time')
    anomalies = data_array - mean

    if standardize:
        anomalies = anomalies / anomalies.std(dim='time')

    return anomalies


def get_pca(data_array, ncomponents, scaling=0):
    print(data_array)
    # Principal Components Analysis
    solver = Eof(data_array)
    # Get Principal Components
    pcs = solver.pcs(npcs=ncomponents, pcscaling=scaling)
    # Get Empirical Orthogonal Functions
    eofs = solver.eofs(neofs=ncomponents, eofscaling=scaling)
    return eofs, pcs


class Pca:
    def __init__(self, data_array, standardize, ncomponents, scaling):
        self.anomalies = calculate_anomalies(data_array, standardize=standardize)
        self.solver = get_pca(data_array, ncomponents, scaling)


def find_analogs(pcs):
    pass
