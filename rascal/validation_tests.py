"""
AutoVal 0.0.0

Pandas extension to do automatic meteorological data validation

Contact: alvaro@intermet.es
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rascal.climate
import rascal.utils
from rascal.utils import Climatology
import rascal.statistics

impossible_thresholds = {
    'TMPA': [-100, 100],
    'RHMA': [0, 100],
    'WSPD': [0, 200],
    'WDIR': [0, 360],
    'PCNR': [0, 1000],
    'RADS01': [0, 2000],
    'RADS02': [0, 2000],
    'RADL01': [0, 2000],
    'RADL02': [0, 2000]
}


@pd.api.extensions.register_dataframe_accessor("AutoVal")
class AutoValidation:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._variables = [c for c in pandas_obj.columns if c.split('_')[-1] not in ['IV', 'CC', 'TC', 'SC']]

    @staticmethod
    def _validate(obj):
        # Verify there is a column with at least one meteorological variable
        if not (set(obj.columns) & set(impossible_thresholds.keys())):
            raise AttributeError("Must have " + ', '.join(impossible_thresholds.keys()))

    def impossible_values(self, variables=None):
        """
        Label values outside an impossible threshold gap.
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Find the dates of the values above the upper impossible limit and below the lower impossible limit
        for variable in variables:
            self._obj[variable + '_' + str(impossible_thresholds[variable][0])] = impossible_thresholds[variable][0]
            self._obj[variable + '_' + str(impossible_thresholds[variable][1])] = impossible_thresholds[variable][1]

            self._obj = label_validation(
                self._obj,
                variables={variable: variable},
                thresholds=impossible_thresholds[variable],
                label='IV'
            )

            self._obj.drop([variable + '_' + str(impossible_thresholds[variable][0])], inplace=True, axis=1)
            self._obj.drop([variable + '_' + str(impossible_thresholds[variable][1])], inplace=True, axis=1)

        return self._obj

    def climatological_coherence(self, variables=None, percentiles=None, skip_labels=True):
        """
        Label values outside extreme climatological percentile values.
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Climatological percentile threshold to label data as suspicious
        if percentiles is None:
            percentiles = [0.01, 0.99]

        # Data to use to calculate the climatology
        if skip_labels:
            train_data = skip_label(self._obj, labels_to_skip=['IV'])
        else:
            train_data = skip_label(self._obj, labels_to_skip=[])

        # Climatology time series
        climatology = Climatology(train_data).daily_cycle(percentiles=percentiles, to_series=True)
        self._obj = pd.concat([self._obj, climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        self._obj = label_validation(
            self._obj,
            variables=dict(zip(variables, variables)),
            thresholds=percentiles,
            label='CC'
        )

        self._obj.drop(climatology.columns, axis=1, inplace=True)

        return self._obj

    def temporal_coherence(self, variables=None, percentiles=None, skip_labels=True):
        """
        Label values with suspicious time evolution (too abrupt or too constant changes)
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Climatological percentile threshold to label data as suspicious
        if percentiles is None:
            percentiles = [0.01, 0.99]

        # Get the variable delta for each time step
        delta_data = self._obj[variables].diff()
        delta_data.columns = [c + '_delta' for c in delta_data.columns]
        self._obj = pd.concat([self._obj, delta_data], axis=1)

        # Data to use to calculate the climatology
        if skip_labels:
            train_data = skip_label(self._obj[delta_data.columns], labels_to_skip=['IV'])
        else:
            train_data = skip_label(self._obj[delta_data.columns], labels_to_skip=[])

        # Climatology time series
        climatology = Climatology(train_data).daily_cycle(percentiles=percentiles, to_series=True)
        self._obj = pd.concat([self._obj, climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        self._obj = label_validation(
            self._obj,
            variables=dict(zip(sorted(delta_data.columns), sorted(variables))),
            thresholds=percentiles,
            label='TC'
        )

        self._obj.drop(climatology.columns, axis=1, inplace=True)
        self._obj.drop(delta_data.columns, axis=1, inplace=True)

        return self._obj

    def spatial_coherence(self, related_site, variables=None, min_corr=0.8, percentiles=None, skip_labels=True):
        """
        Calculate the regression with a highly correlated station and label values with residuals outside the selected
        percentile gap
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Climatological percentile threshold to label data as suspicious
        if percentiles is None:
            percentiles = [0.01, 0.99]

        # Data to use to calculate the climatology
        if skip_labels:
            train_data = skip_label(self._obj, labels_to_skip=['IV'])
        else:
            train_data = skip_label(self._obj, labels_to_skip=[])

        # Climatology of the regression residuals between observations and the reference data
        residuals_climatology, residuals = get_significant_residuals(train_data, related_site, min_corr, percentiles)
        self._obj = pd.concat([self._obj, residuals.astype('float64'), residuals_climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        self._obj = label_validation(
            self._obj,
            variables=dict(zip(sorted(residuals.columns), sorted(variables))),
            thresholds=percentiles,
            label='SC'
        )

        self._obj.drop(residuals_climatology.columns, axis=1, inplace=True)
        self._obj.drop(residuals.columns, axis=1, inplace=True)

        return self._obj

    def internal_coherence(self, percentile=None):
        """
        Find relationships between daily climatological variables and label days that deviates from the expected
        behaviour
        """

        # Climatological percentile threshold to label data as suspicious
        if percentile is None:
            percentile = 0.99

        # Get daily climatological variables
        daily_climatological_variables = Climatology(self._obj).climatological_variables()

        regression = pd.DataFrame(
            index=daily_climatological_variables.index,
            columns=daily_climatological_variables.columns
        )

        regression_error = pd.DataFrame(
            index=daily_climatological_variables.index,
            columns=daily_climatological_variables.columns
        )

        anomaly = pd.DataFrame(
            index=daily_climatological_variables.index,
            columns=daily_climatological_variables.columns
        )

        # Principal Components Analysis of daily variables for each month
        for month, monthly_df in daily_climatological_variables.groupby(daily_climatological_variables.index.month):
            # Get EOFs, PCAs and explained variance ratios
            pca = rascal.statistics.DaskPCA(monthly_df, n_components=3, mode='T', standardize=True)

            # Reconstruct the original time series with the PCA
            regression_month, regression_error_month = pca.regression()

            regression.loc[regression_month.index] = regression_month
            anomaly.loc[pca.anomaly.index] = pca.anomaly
            regression_error.loc[regression_error_month.index] = regression_error

        # Get the error of the rascal in hourly resolution
        regression_error = anomaly - regression
        regression_error = regression_error.where(regression_error > 0, np.nan)
        regression_error = regression_error.resample('H').ffill()

        original_variables = {
            'RADST': 'RADS01',
            'TAMP': 'TMPA',
            'TMEAN': 'TMPA',
            'PTOT': 'PCNR',
            'RHMEAN': 'RHMA',
            'VMEAN': 'WSPD'
        }

        # Label values with and error above the maximum percentile threshold
        for variable in daily_climatological_variables.columns:
            percentile_threshold = regression_error[variable].quantile(percentile)
            label_idx = regression_error[variable].loc[regression_error[variable] >= percentile_threshold].index
            label_idx = pd.to_datetime(label_idx)

            self._obj[original_variables[variable] + '_IC'] = 0
            self._obj[original_variables[variable] + '_IC'].loc[label_idx] = 1

        return self._obj

    def variance_test(self, variables=None, validation_window=None):
        """
        Label values within a selected window of time with an anomalous variance (too high or too low)
        """

        # Variables to validate
        if variables is None:
            variables = self._variables

        # Validation window size
        if validation_window is None:
            validation_window = '1D'

    def vplot(self, kind=None):

        fig = plt.figure()
        ax = fig.subplots(len(self._variables))

        if kind is None:
            kind = 'label_type'

        if kind == 'label_type':

            label_type_colors = {
                'IV': 'black',
                'CC': 'red',
                'SC': 'blue',
                'TC': 'green',
                'IC': 'yellow'
            }

            for i, variable in enumerate(self._variables):

                # Original data
                self._obj[variable].plot(ax=ax[i], color='grey')

                # Validation columns
                label_columns = [col for col in self._obj.columns if col not in self._variables and variable in col]

                for label, color in label_type_colors.items():
                    variable_label = [c for c in label_columns if label in c][0]
                    if len(self._obj[variable].loc[self._obj[variable_label] == 1]) > 0:
                        self._obj[variable].loc[self._obj[variable_label] == 1].plot(
                            ax=ax[i],
                            marker='o',
                            markersize=2,
                            color=color,
                            linewidth=0)

                ax[i].set_ylabel(variable)

        elif kind == 'label_count':

            label_number_colors = {
                1: 'blue',
                2: 'green',
                3: 'yellow',
                4: 'red'
            }

            for i, variable in enumerate(self._variables):

                # Original data
                self._obj[variable].plot(ax=ax[i], color='grey')

                # Validation columns
                label_columns = [col for col in self._obj.columns if col not in self._variables and variable in col]

                # Count the number of labels
                self._obj[variable + '_labels'] = self._obj[label_columns].sum(axis=1)

                # Plot points by changing the color depending on the number of suspect labels
                for n_labels, color in label_number_colors.items():
                    if len(self._obj[variable].loc[self._obj[variable + '_labels'] == n_labels]) > 0:
                        self._obj[variable].loc[self._obj[variable + '_labels'] == n_labels].plot(
                            ax=ax[i],
                            marker='o',
                            markersize=2,
                            color=color,
                            linewidth=0
                        )

                # Mark Impossible values
                if len(self._obj[variable].loc[self._obj[variable + '_IV'] == 1]) > 0:
                    self._obj[variable].loc[self._obj[variable + '_IV'] == 1].plot(
                        ax=ax[i],
                        marker='o',
                        markersize=2,
                        color='black',
                        linewidth=0
                    )

                ax[i].set_ylabel(variable)

                # Delete the label counter from the DataFrame
                self._obj.drop([variable + '_labels'], axis=1, inplace=True)

        return ax


def label_validation(df: pd.DataFrame, variables: dict, thresholds: (list, tuple), label: str):
    """
    Label a point if it is outside of a threshold gap.
    :param df: DataFrame.
    :param variables: dict. {labeling_variable: variable}.
    :param thresholds: list or tuple. validation gap.
    :param label: str. Name of the label.
    :return df: DataFrame. Original dataframe with the validation column added.
    """

    for labeling_variable, variable in variables.items():
        min_condition = df[labeling_variable] < df[labeling_variable + '_' + str(min(thresholds))]
        max_condition = df[labeling_variable] > df[labeling_variable + '_' + str(max(thresholds))]

        labeled_dates = df[variable].loc[min_condition | max_condition].index

        df[variable + '_' + label] = 0
        df.loc[labeled_dates, variable + '_' + label] = 1

    return df


def skip_label(df: pd.DataFrame, labels_to_skip: (list, tuple)):
    """
    Ignore data where the selected labels = 1.
    """
    columns_to_ignore = [c for c in df.columns if c.split('_')[-1] in labels_to_skip]
    label_columns = [c for c in df.columns if c.split('_')[-1] in ['IV', 'CC', 'SC', 'TC']]

    for col in columns_to_ignore:
        df = df[df[col] != 1]

    df.drop(label_columns, axis=1, inplace=True)

    return df


def get_significant_residuals(original: pd.DataFrame, reference: pd.DataFrame, correlation_threshold, percentiles):
    """
    """

    regression, residuals = Climatology(original).spatial_regression(reference)
    regression_series = rascal.utils.table_to_series(regression, original.index)

    correlation_columns = [c for c in regression_series.columns if 'correlation' in c]

    for col in correlation_columns:
        variable = col.split('_')[0]
        non_significant_dates = regression_series[col].loc[regression_series[col] < correlation_threshold].index
        residuals.loc[non_significant_dates, variable + '_residuals'] = np.nan

    residuals_climatology = Climatology(residuals.astype('float64')).daily_cycle(percentiles=percentiles,
                                                                                 to_series=True)

    return residuals_climatology, residuals
