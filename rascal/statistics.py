"""
Statistical methods

Contact: alvaro@intermet.es
"""

import gc
import dask.array as da
import dask.dataframe as dd
from dask_ml.decomposition import PCA
import pandas as pd
from sklearn.linear_model import LinearRegression

import autoval.utils
import matplotlib.pyplot as plt


class DaskPCA:
    def __init__(self, data, n_components, mode, standardize=False):
        self.data = data
        self.n_components = n_components
        self.mode = mode
        self.anomaly = self.anomalies(standardize=standardize)
        self.eof, self.pc, self.explained_variance = self.calculate(mode=mode)

    @staticmethod
    def from_pandas_to_daskarray(df: pd.DataFrame, npartitions):
        df = autoval.utils.clean_dataset(df)
        daskarr = dd.from_pandas(df, npartitions=npartitions).to_dask_array()
        daskarr = daskarr.compute_chunk_sizes()

        return daskarr

    def anomalies(self, standardize=False):
        """
        Calculate anomalies of the field.
        :param standardize: bool, optional. (Default=False). If True standardize of the anomalies.
        """

        mean = self.data.mean()

        if standardize:
            std = self.data.std()
            anomaly = (self.data - mean) / std

        else:
            anomaly = self.data - mean

        return anomaly

    def calculate(self, mode):
        """
        Calculate the principal components, empirical orthogonal functions and explained variance ratio.
        :param mode: str. options = 'S' or 'T'. Mode of Analysis.
        :return eofs, pc, explained_variance: (DataFrame, DataFrame, list)
        """

        # Get Dask array of anomalies
        z = self.from_pandas_to_daskarray(self.anomaly, npartitions=3)

        # PCA mode
        if mode == 'T':
            z = z.transpose()
        elif mode == 'S':
            pass
        else:
            raise AttributeError(' Error: ' + mode + ' is not a PCA type')

        # Covariance matrix
        s = da.cov(z)

        # Get principal components
        pca = PCA(n_components=self.n_components)
        pca.fit(s)

        # Empirical Orthogonal Functions
        eofs = pd.DataFrame(pca.components_.transpose(),
                            index=self.anomaly.columns,
                            columns=['eof_' + str(c+1) for c in range(self.n_components)])

        # Loadings
        pc = self.anomaly.dot(eofs)
        pc.columns = ['pc_' + str(c+1) for c in range(self.n_components)]

        # Explained variance by each EOF
        explained_variance = list(map(lambda x: x*100, pca.explained_variance_ratio_))

        # Clear some memory
        del z, s
        gc.collect()

        return eofs, pc, explained_variance

    def regression(self):
        """
        Calculate the PCA regression and the rascal error
        """
        regression = self.pc.to_numpy().dot(self.eof.to_numpy().transpose())
        regression = pd.DataFrame(regression, index=self.pc.index, columns=self.eof.index)

        regression_error = self.anomaly - regression

        return regression, regression_error


def linear_regression(x: pd.DataFrame, y: pd.DataFrame):
    """
    Get linear regression using pandas and calculate the residuals of the predicted values
    """
    # Get only the common data
    x, y = autoval.utils.get_common_index(x, y)

    # Create object for the class
    linear_regressor = LinearRegression()

    # Perform linear regression
    linear_regressor.fit(x, y)
    residuals = x - linear_regressor.predict(x)

    return linear_regressor, residuals


def split_in_percentile_intervals(data, intervals):
    """

    :param data:
    :param intervals:
    :return:
    """

    split_data = []

    for column in data:

        variable = data[column]

        # Calculate percentile score thresholds
        score_thresholds = [variable.quantile(interval, interpolation='midpoint') for interval in intervals]
        if column == 'PCNR':
            print(variable)
            print(score_thresholds)
            print('------------------------------------')
        # Split data in percentile intervals
        split_variable = pd.DataFrame(index=variable.index)

        for i in range(len(intervals)):
            if i != len(intervals) - 1:
                # Name of the interval
                interval_label = column + ' ' + str(intervals[i]) + '-' + str(intervals[i + 1])
                interval_scores = variable.loc[(score_thresholds[i] <= variable) & (variable < score_thresholds[i+1])]

                split_variable[interval_label] = interval_scores

        split_data.append(split_variable)

    split_data = pd.concat(split_data, axis=1)

    return split_data


def compare_distributions(data1, data2):
    """
    Plot a comparision of distributions between two dataframes. Both have to have
    :param data1:
    :param data2:
    :return:
    """

    # Clean DataFrames of possible conflictive values and get only the common data
    data1, data2 = autoval.utils.get_common_index(data1, data2)

    if len(data1.columns) == len(data2.columns):
        ncols = len(data1.columns)
    else:
        raise AttributeError('Number of columns must be the same')

    fig = plt.figure()
    ax = fig.subplots(ncols)

    min1 = min(data1.min())
    min2 = min(data2.min())
    max1 = max(data1.max())
    max2 = max(data2.max())

    max_value = max([max1, max2])
    min_value = min([min1, min2])

    if ncols == 1:
        data1.hist(bins=50, ax=ax, color='blue', alpha=0.4)
        data2.hist(bins=50, ax=ax, color='red', alpha=0.4)
        ax.set_xlim(min_value, max_value)
        plt.legend()

    else:
        for i in range(ncols):
            data1[data1.columns[i]].hist(bins=50, ax=ax[i], color='blue', alpha=0.4, label=data1.columns[i])
            data2[data2.columns[i]].hist(bins=50, ax=ax[i], color='red', alpha=0.4, label=data2.columns[i])
            ax[i].set_xlim(min_value, max_value)
            plt.legend()

    plt.show()
