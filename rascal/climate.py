"""
Calculate climatological aspects of a site

Contact: alvaro@intermet.es
"""

import itertools
import rascal.utils
import rascal.statistics

import numpy as np
import pandas as pd


class Climatology:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def climatological_variables(self):
        """
        Get relevant daily climatological variables as DataFrame: Maximum, minimum and mean temperature, maximum and
        mean wind velocity, total solar radiation, total precipitation.
        """

        daily_variables = pd.DataFrame()

        if 'TMPA' in self._obj.columns:
            tmax = self._obj['TMPA'].resample('D').max().rename('TMAX')
            tmin = self._obj['TMPA'].resample('D').min().rename('TMIN')
            tmean = self._obj['TMPA'].resample('D').mean().rename('TMEAN')
            tamp = abs(tmax - tmin)
            tamp = tamp.rename('TAMP')
            daily_variables = pd.concat([daily_variables, tmax, tmin, tmean], axis=1)

        if 'WSPD' in self._obj.columns:
            # vmax = self._obj['WSPD'].resample('D').max().rename('VMAX')
            vmean = self._obj['WSPD'].resample('D').mean().rename('VMEAN')
            # daily_variables = pd.concat([daily_variables, vmax, vmean], axis=1)
            daily_variables = pd.concat([daily_variables, vmean], axis=1)

        if 'RADS01' in self._obj.columns:
            rascal.utils.Preprocess(self._obj).clear_low_radiance()
            rads_total = self._obj['RADS01'].resample('D').sum().rename('RADST')
            rads_total = rads_total.where(rads_total > 0, np.nan)
            daily_variables = pd.concat([daily_variables, rads_total], axis=1)

        if 'PCNR' in self._obj.columns:
            ptot = self._obj['PCNR'].resample('D').sum().rename('PCNR')
            daily_variables = pd.concat([daily_variables, ptot], axis=1)

        if 'RHMA' in self._obj.columns:
            rhmax = self._obj['RHMA'].resample('D').max().rename('RHMA')
            daily_variables = pd.concat([daily_variables, rhmax], axis=1)

        daily_variables.index = pd.to_datetime(daily_variables.index)

        return daily_variables

    def daily_cycle(self, percentiles=None, to_series=False):
        """
        Calculate the percentiles of the monthly daily cycles
        """
        if percentiles is None:
            percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        columns = [v + '_' + str(p) for v, p in itertools.product(self._obj.columns, percentiles)]
        # Calculate the index of the monthly daily cycles (format = hour_month)
        idx = [str(h) + '_' + str(m) for h, m in itertools.product(range(0, 24), range(1, 13))]

        # Dataframe of climatological percentiles
        climatology_percentiles = pd.DataFrame(index=idx, columns=columns)

        # Calculate the climatological daily cycle for each month for each percentile
        for variable, percentile in itertools.product(self._obj.columns, percentiles):
            for month, month_dataset in self._obj.groupby(self._obj.index.month):
                monthly_climatology = month_dataset[variable].groupby(month_dataset.index.hour).quantile(percentile)
                for hour in monthly_climatology.index:
                    climatology_percentiles.loc[str(hour) + '_' + str(month), variable + '_' + str(percentile)] = \
                        monthly_climatology.loc[hour]

        # transform the monthly daily cycles to time series
        if to_series:
            return table_to_series(climatology_percentiles, self._obj.index)
        else:
            return climatology_percentiles

    def spatial_regression(self, related_site):
        """
        Get the correlation and linear regression with a reference station
        """

        # Calculate the index of tha monthly daily cycles (format = hour_month)
        idx = [str(h) + '_' + str(m) for h, m in itertools.product(range(0, 24), range(1, 13))]
        columns = [v + '_' + lr for v, lr in itertools.product(self._obj.columns, ['coef', 'intercept', 'correlation'])]
        # Dataframe of climatological percentiles
        regression = pd.DataFrame(index=idx, columns=columns)
        residuals = pd.DataFrame(index=self._obj.index, columns=[c + '_residuals' for c in self._obj.columns])

        # Group the data by month
        for month, month_dataset in self._obj.groupby(self._obj.index.month):
            # Fill the dataframe with the climatological daily cycle of each month
            for hour, hour_dataset in month_dataset.groupby(month_dataset.index.hour):

                # Select the data of the reference station by month and hour
                related_site_hm = related_site.loc[
                    (related_site.index.month == month) &
                    (related_site.index.hour == hour)
                ]
                hour_dataset = hour_dataset

                # Correlate the datasets
                correlation = related_site_hm.corrwith(hour_dataset)

                for variable in self._obj.columns:

                    linear_regressor, regr_res = rascal.statistics.linear_regression(
                        x=related_site_hm[variable].to_frame(),
                        y=hour_dataset[variable].to_frame()
                    )

                    # Save the coefficient, intercept, and correlation for the hour and month
                    regression.loc[str(hour) + '_' + str(month),
                                   variable + '_coef'] = np.squeeze(linear_regressor.coef_)
                    regression.loc[str(hour) + '_' + str(month),
                                   variable + '_intercept'] = np.squeeze(linear_regressor.intercept_)
                    regression.loc[str(hour) + '_' + str(month), variable + '_correlation'] = correlation[variable]

                    residuals.loc[regr_res.index, variable + '_residuals'] = regr_res[variable].values

        return regression, residuals

    def mcp(self):
        pass


def table_to_series(df: pd.DataFrame, new_index):
    """
    Transform a table of hourly values per month to a time series.
    """

    climatology_series = pd.DataFrame(index=new_index, columns=df.columns)

    for variable in df.columns:
        for month, month_dataset in climatology_series.groupby(climatology_series.index.month):
            for hour, hourly_dataset in month_dataset.groupby(month_dataset.index.hour):
                climatology_series.loc[hourly_dataset.index, variable] = df.loc[str(hour) + '_' + str(month), variable]

    return climatology_series
