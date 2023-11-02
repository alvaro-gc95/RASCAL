import os
import autoval.utils
import rascal.statistics

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.stats import percentileofscore
from sklearn.metrics import confusion_matrix

seasons = {
    1: 'DJF',
    2: 'DJF',
    3: 'MAM',
    4: 'MAM',
    5: 'MAM',
    6: 'JJA',
    7: 'JJA',
    8: 'JJA',
    9: 'SON',
    10: 'SON',
    11: 'SON',
    12: 'DJF'
}

variables = {
    'TMPA': ['TMAX', 'TMIN'],
    'PCNR': ['PCNR']
}

percentile_intervals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

retiro = {
    1900: 310,
    1901: 471,
    1902: 546,
    1903: 299,
    1904: 525,
    1905: 398,
    1906: 491,
    1907: 350,
    1908: 398,
    1909: 449,
    1910: 380,
    1911: 511,
    1912: 346,
    1913: 455,
    1914: 450,
    1915: 272,
    1916: 279,
    1917: np.nan,
    1918: np.nan,
    1919: 373,
    1920: 450,
    1921: 422,
    1922: 409,
    1923: 349,
    1924: 392,
    1925: 414,
    1926: 400,
    1927: 442,
    1928: 429,
    1929: 371,
    1930: 461,
    1931: 306,
    1932: 383,
    1933: 386,
    1934: 329,
    1935: 407,
    1936: 601,
    1937: 472,
    1938: 252,
    1939: 449,
    1940: 511,
    1941: 532,
    1942: 540,
    1943: 433,
    1944: 395,
    1945: 306,
    1946: 320,
    1947: 659,
    1948: 347,
    1949: 402,
    1950: 294,
    1951: 571,
    1952: 381,
    1953: 396,
    1954: 246,
    1955: 559,
    1956: 481,
    1957: 390,
    1958: 465,
    1959: 636,
    1960: 590,
    1961: 444,
    1962: 513,
    1963: 546,
    1964: 356,
    1965: 443,
    1966: 529,
    1967: 354,
    1968: 383,
    1969: 590,
    1970: 305,
    1971: 507,
    1972: 730,
    1973: 354,
    1974: 300,
    1975: 429,
    1976: 609,
    1977: 476,
    1978: 549,
    1979: 499,
    1980: 367,
    1981: 425,
    1982: 369,
    1983: 260,
    1984: 494,
    1985: 338,
    1986: 387,
    1987: 545,
    1988: 413,
    1989: 560,
    1990: 304,
    1991: 343,
    1992: 350,
    1993: 470,
    1994: 293,
    1995: 332,
    1996: 514,
    1997: 573,
    1998: 395,
    1999: 382,
    2000: 489,
    2001: 364,
    2002: 502,
    2003: 518,
    2004: 485,
    2005: 252,
    2006: 501,
    2007: 406,
    2008: 463,
    2009: 335,
    2010: 557,
    2011: 380,
    2012: np.nan
}
aemet = {
    1900: np.nan,
    1901: 700,
    1902: 680,
    1903: 575,
    1904: 610,
    1905: 578,
    1906: 620,
    1907: 630,
    1908: 620,
    1909: 680,
    1910: 685,
    1911: 680,
    1912: 550,
    1913: 650,
    1914: 680,
    1915: 800,
    1916: 680,
    1917: 600,
    1918: 620,
    1919: 800,
    1920: 680,
    1921: 550,
    1922: 660,
    1923: 670,
    1924: 630,
    1925: 680,
    1926: 670,
    1927: 700,
    1928: 680,
    1929: 650,
    1930: 750,
    1931: 640,
    1932: 780,
    1933: 700,
    1934: 580,
    1935: 590,
    1936: 900,
    1937: 850,
    1938: 500,
    1939: 700,
    1940: 720,
    1941: 840,
    1942: 680,
    1943: 600,
    1944: 550,
    1945: 500,
    1946: 680,
    1947: 780,
    1948: 600,
    1949: 590,
    1950: 550,
    1951: 800,
    1952: 680,
    1953: 500,
    1954: 510,
    1955: 680,
    1956: 750,
    1957: 640,
    1958: 680,
    1959: 750,
    1960: 920,
    1961: 700,
    1962: 760,
    1963: 890,
    1964: 580,
    1965: 680,
    1966: 750,
    1967: 580,
    1968: 660,
    1969: 700,
    1970: 860,
    1971: 620,
    1972: 750,
    1973: 820,
    1974: 580,
    1975: 600,
    1976: 720,
    1977: 780,
    1978: 730,
    1979: 820,
    1980: 600,
    1981: 580,
    1982: 610,
    1983: 590,
    1984: 700,
    1985: 600,
    1986: 610,
    1987: 710,
    1988: 630,
    1989: 710,
    1990: 580,
    1991: 590,
    1992: 600,
    1993: 600,
    1994: 580,
    1995: 600,
    1996: 850,
    1997: 800,
    1998: 540,
    1999: 640,
    2000: 700,
    2001: 670,
    2002: 710,
    2003: 720,
    2004: 550,
    2005: 500,
    2006: 570,
    2007: 580,
    2008: 710

}


def compare_method_skill(path, variable, methods, initial_year, final_year):
    """

    :param path:
    :param methods:
    :param initial_year:
    :param final_year:
    :return:
    """
    # Open observations
    observation = pd.read_csv(path + variable + '_observations_' + str(initial_year) + str(final_year) + '.csv',
                              index_col=0)
    observation.index = pd.to_datetime(observation.index)

    # Variables derived from the principal variable
    predicted_variables = variables[variable]

    # Skills to analyze
    skills = ['RMSE', 'MBE', 'Pearson', 'Spearman', 'Brier Score']

    for predicted_variable in predicted_variables:

        # Lists to fill and transform as dataframe by similarity method
        percentile_skills = []
        seasonal_skills = []
        quantiles = []
        reconstructions = []

        for similarity_method in methods:

            # Reconstruction
            reconstruction = pd.read_csv(path + variable + '_reconstruction_' + similarity_method + '_' +
                                         str(initial_year) + str(final_year) + '.csv', index_col=0)
            reconstruction.index = pd.to_datetime(reconstruction.index)

            # Get the relevant variables and its uncertainty bands
            reconstructed_variables_and_uncertainties = [col for col in reconstruction if predicted_variable in col]
            reconstruction_with_bands = reconstruction[reconstructed_variables_and_uncertainties]
            reconstruction_with_bands.columns = [col + ' ' + similarity_method for col in reconstruction_with_bands]
            reconstructions.append(reconstruction_with_bands)

            # Reconstruction without bands
            reconstruction = reconstruction[predicted_variable].to_frame()

            # Get quantiles of the values
            quantil = quantile_plot(predicted=reconstruction, observed=observation)
            quantil['method'] = similarity_method
            quantiles.append(quantil)

            # Bias by month and year
            get_month_year_bias(predicted=reconstruction, observed=observation)

            # Skill scores by season
            get_skill_by_season(predicted=reconstruction, observed=observation, similarity_method=similarity_method)

            # Skill scores by percentile
            get_skill_by_percentiles(predicted=reconstruction, observed=observation,
                                     similarity_method=similarity_method)

            # Files of skill score
            percentile_skill_files = [file for file in os.listdir('./output/') if
                                      'percentileskill' in file and '.csv' in file]
            seasonal_skill_files = [file for file in os.listdir('./output/') if
                                    'seasonalskill' in file and '.csv' in file]

            # Get skill scores
            for file in percentile_skill_files:
                if similarity_method in file and predicted_variable in file:
                    percentile_skill = pd.read_csv('./output/' + file)
                    percentile_skill.rename(columns={percentile_skill.columns[0]: "percentile"}, inplace=True)
                    percentile_skill['method'] = similarity_method
                    percentile_skills.append(percentile_skill)

            for file in seasonal_skill_files:
                if similarity_method in file and predicted_variable in file:
                    seasonal_skill = pd.read_csv('./output/' + file)
                    seasonal_skill.rename(columns={seasonal_skill.columns[0]: "season"}, inplace=True)
                    seasonal_skill['method'] = similarity_method
                    seasonal_skills.append(seasonal_skill)

        # Get the observations of the predicted variables only
        observation = observation[predicted_variable].to_frame()

        # Put the list of files in DataFrames for easier representation
        seasonal_skills = pd.concat(seasonal_skills, axis=0)
        percentile_skills = pd.concat(percentile_skills, axis=0)
        quantiles = pd.concat(quantiles, axis=0)
        reconstructions = pd.concat(reconstructions, axis=1)
        reconstructions = pd.concat([observation, reconstructions], axis=1)

        # Order the season column
        seasonal_skills['season'] = pd.Categorical(
            seasonal_skills['season'],
            categories=['DJF', 'MAM', 'JJA', 'SON'],
            ordered=True
        )

        # Context of the graphs
        sns.set_context("talk")
        sns.set_style("whitegrid")
        # Quantile - Quantile plot

        ax.set_title('Quantile-Quantile: ' + predicted_variable)
        plt.savefig('./plots/' + predicted_variable + '_qqplot.png')

        # Seasonal skills plots
        for skill in skills:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.subplots(1)
            sns.barplot(data=seasonal_skills, x="season", y=skill, hue="method", palette='muted', ax=ax)
            ax.grid(axis='y')
            if skill in ['Pearson', 'Spearman']:
                ax.set_ylim(-1, 1)
            if skill == 'Brier Score':
                ax.set_ylim(0, 1)
            plt.savefig('./plots/' + predicted_variable + '_seasonal_' + skill + '.png')

        # Percentile skills plots
        for skill in skills:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.subplots(1)
            pplot = sns.barplot(data=percentile_skills, x="percentile", y=skill, hue="method", palette='muted', ax=ax)
            ax.grid(axis='y')
            if skill in ['Pearson', 'Spearman']:
                ax.set_ylim(-1, 1)
            if skill == 'Brier Score':
                ax.set_ylim(0, 1)
            for item in pplot.get_xticklabels():
                item.set_rotation(90)
            plt.savefig('./plots/' + variable + '_percentile_' + skill + '.png')

        # recontructions = reconstructions.dropna()
        # reconstructions = reconstructions.resample('M').sum()
        # reconstructions = reconstructions.where(reconstructions > 0, np.nan)
        # reconstructions = reconstructions.rolling(10).mean()
        #

        # reconstructions['aemet'] = np.nan
        # reconstructions['retiro'] = np.nan
        #
        # print(np.array(list(aemet.values())))
        # reconstructions['aemet'].iloc[:len(aemet.keys())] = np.array(list(aemet.values()))
        # reconstructions['retiro'].iloc[:len(retiro.keys())] = np.array(list(retiro.values()))
        # print(reconstructions)

        reconstruction_anomalies = reconstructions.apply(lambda x: (x - x.mean()) / x.std())
        reconstruction_anomalies = reconstruction_anomalies.rolling(15).mean()

        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(20, 5))
        ax = fig.subplots(1)
        print(reconstructions)

        lines = sns.lineplot(data=reconstructions[['TMAX', 'TMAX percentiles']], ax=ax, palette=['black', 'tab:blue'])

        # lines = sns.lineplot(data=reconstructions, ax=ax)

        for line in lines.lines:
            line.set_linestyle('-')
        # for i, col in enumerate(reconstructions):
        #     if 'band' in col:
        #         lines.lines[i].set_linestyle("--")
        #     else:
        #         lines.lines[i].set_linestyle("-")

        # Plot uncertainty bands
        for similarity_method in methods:
            lower = predicted_variable + ' min band ' + similarity_method
            upper = predicted_variable + ' max band ' + similarity_method
            ax.fill_between(reconstructions.index, reconstructions[lower], reconstructions[upper], alpha=0.2)

        plt.savefig('./plots/' + predicted_variable + '_series.png')

        plt.show()


def get_month_year_bias(predicted, observed):
    # Check if the columns are the same
    common_cols = list(set(predicted.columns) & set(observed.columns))

    for col in common_cols:
        # Get common data
        observed_col, predicted_col = autoval.utils.get_common_index(
            observed[col].to_frame(),
            predicted[col].to_frame()
        )

        # Calculate the bias of each month
        bias = predicted_col - observed_col
        bias['month'] = bias.index.month
        sns.violinplot(data=bias, x="month", y=col)
        plt.show()
        fig = plt.figure()
        ax = fig.subplots(1)
        predicted_col.plot(ax=ax)
        observed_col.plot(ax=ax)
        bias.plot(ax=ax)
        plt.show()
        bias = bias.resample('M').mean()

        # Change the format of the dataframe to month-year
        bias['year'] = bias.index.year
        bias['month'] = bias.index.month
        bias = bias.reset_index()
        bias = bias.drop(['index'], axis=1)
        sns.set_style("whitegrid")

        f, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=bias, x='month', y=col)
        sns.despine(left=True)
        sns.despine(bottom=True)
        plt.show()

        f, ax = plt.subplots(figsize=(6, 9))
        sns.boxplot(data=bias, y='year', x=col, orient='h')
        sns.despine(left=True)
        sns.despine(bottom=True)
        plt.show()

        bias = bias.pivot(index='year', columns='month', values=col)

        bias_max = max(abs(bias).max())

        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        f, ax = plt.subplots(figsize=(6, 9))
        sns.heatmap(bias, linewidths=.5, ax=ax, cmap=cmap.reversed(), vmin=-bias_max, vmax=bias_max)
        plt.show()
        print(bias)

    return bias


def get_skill_by_percentiles(predicted, observed, similarity_method):
    # Check if the columns are the same
    common_cols = list(set(predicted.columns) & set(observed.columns))

    for col in common_cols:
        observed_col, predicted_col = autoval.utils.get_common_index(
            observed[col].to_frame(),
            predicted[col].to_frame()
        )

        observation_by_percentiles = rascal.statistics.split_in_percentile_intervals(
            observed_col,
            percentile_intervals
        )

        # Generate skill dataframe
        skills = pd.DataFrame(
            index=observation_by_percentiles.columns,
            columns=['RMSE', 'MBE', 'Pearson', 'Spearman', 'Brier Score']
        )

        for interval in observation_by_percentiles:
            observations_in_interval = observation_by_percentiles[interval]

            observation, prediction = autoval.utils.get_common_index(
                observations_in_interval.to_frame(),
                predicted_col
            )

            rmse = get_rmse(
                predicted=prediction.values,
                observed=observation.values
            )
            skills.loc[interval, 'RMSE'] = rmse

            mbe = get_mbe(
                predicted=prediction.values,
                observed=observation.values
            )
            skills['MBE'].loc[interval] = mbe

            prediction = prediction.squeeze()
            observation = observation.squeeze()
            # Join the DataFrames
            data = pd.concat([prediction, observation], axis=1)
            data.columns = ['rascal', 'observation']

            pearson = data.corr(method='pearson').values[1, 0]
            skills['Pearson'].loc[interval] = pearson

            spearman = data.corr(method='spearman').values[1, 0]
            skills['Spearman'].loc[interval] = spearman

            brier = get_bs(
                predicted=prediction.values,
                observed=observation.values,
                threshold=1
            )
            skills['Brier Score'].loc[interval] = brier

        skills.to_csv('./output/percentileskill_' + col + '_' + similarity_method + '.csv')


def get_skill_by_season(predicted, observed, similarity_method):
    # Check if the columns are the same
    common_cols = list(set(predicted.columns) & set(observed.columns))

    for col in common_cols:

        # Change names of the columns
        prediction = predicted[col]
        observation = observed[col]

        prediction, observation = autoval.utils.get_common_index(prediction.to_frame(), observation.to_frame())
        prediction = prediction.squeeze()
        observation = observation.squeeze()

        # Join the DataFrames
        data = pd.concat([prediction, observation], axis=1)
        data.columns = ['rascal', 'observation']

        # Divide the data by seasons
        data['season'] = [seasons[date.month] for date in data.index]

        # Name of the available seasons
        seasons_in_data = list(set(data['season'].values))

        # Generate skill dataframe
        skills = pd.DataFrame(index=seasons_in_data, columns=['RMSE', 'MBE', 'Pearson', 'Spearman', 'Brier Score'])

        # Calculate the climatology
        for season, seasonal_data in data.groupby(data['season']):
            # Drop the season column
            seasonal_data = seasonal_data.drop(['season'], axis=1)

            rmse = get_rmse(
                predicted=seasonal_data['rascal'].values,
                observed=seasonal_data['observation'].values
            )
            skills.loc[season, 'RMSE'] = rmse

            mbe = get_mbe(
                predicted=seasonal_data['rascal'].values,
                observed=seasonal_data['observation'].values
            )
            skills['MBE'].loc[season] = mbe

            pearson = seasonal_data.corr(method='pearson').values[1, 0]
            skills['Pearson'].loc[season] = pearson

            spearman = seasonal_data.corr(method='spearman').values[1, 0]
            skills['Spearman'].loc[season] = spearman

            brier = get_bs(
                predicted=seasonal_data['rascal'].values,
                observed=seasonal_data['observation'].values,
                threshold=1
            )
            skills['Brier Score'].loc[season] = brier

            # get_confusion_matrix(
            #     predicted=seasonal_data['rascal'].values,
            #     observed=seasonal_data['observation'].values,
            #     threshold=1
            # )

        skills.to_csv('./output/seasonalskill_' + col + '_' + similarity_method + '.csv')


def get_rmse(predicted, observed):
    """
    Root Mean squared error
    """
    return np.sqrt(np.mean((predicted - observed) ** 2))


def get_mbe(predicted, observed):
    """
    Mean Bias Error
    """
    return np.mean(predicted - observed)


def get_msess(predicted, observed):
    pass


def get_bs(predicted, observed, threshold):
    """
    Brier Score.
    """
    # Arrays of boolean variables.
    # Consider an event happening when the value of the variable is above a determined threshold
    predicted_bool = predicted.copy()
    observed_bool = observed.copy()

    # 1 if the event happens, 0 if not
    predicted_bool[predicted < threshold] = 0
    predicted_bool[predicted >= threshold] = 1

    observed_bool[observed < threshold] = 0
    observed_bool[observed >= threshold] = 1

    brier_score = np.mean((predicted_bool - observed_bool) ** 2)

    return brier_score


def get_confusion_matrix(predicted, observed, threshold):
    # Arrays of boolean variables.
    # Consider an event happening when the value of the variable is above a determined threshold
    predicted_bool = predicted.copy()
    observed_bool = observed.copy()

    # 1 if the event happens, 0 if not
    predicted_bool[predicted < threshold] = 0
    predicted_bool[predicted >= threshold] = 1

    observed_bool[observed < threshold] = 0
    observed_bool[observed >= threshold] = 1

    cf_matrix = confusion_matrix(observed_bool, predicted_bool)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.show()


def quantile_plot(predicted: pd.DataFrame, observed: pd.DataFrame):
    # Check if the columns are the same
    common_cols = list(set(predicted.columns) & set(observed.columns))

    for col in common_cols:
        # Get the common column
        prediction = predicted[col]
        observation = observed[col]

        prediction, observation = autoval.utils.get_common_index(prediction.to_frame(), observation.to_frame())

        prediction = prediction.squeeze()
        observation = observation.squeeze()

        # Get the quantile of each value
        prediction_quantiles = prediction.apply(lambda x: percentileofscore(prediction.values, score=x))
        observation_quantiles = observation.apply(lambda x: percentileofscore(observation.values, score=x))

        prediction_quantiles = prediction_quantiles.rename('quantile')
        observation_quantiles = observation_quantiles.rename('quantile')

        prediction = pd.concat([prediction, prediction_quantiles], axis=1)
        observation = pd.concat([observation, observation_quantiles], axis=1)

        # Reorder by quantile
        prediction = prediction.set_index('quantile')
        observation = observation.set_index('quantile')

        # Delete duplicate values
        prediction = prediction[~prediction.index.duplicated(keep='first')]
        observation = observation[~observation.index.duplicated(keep='first')]

        # Put in common dataframe
        df = pd.concat([prediction, observation], axis=1)
        df.columns = ['Reconstruction', 'Observation']

    return df


def plot_errors(errors, variables):
    error_types = ['RMSE', 'MBE', 'R²']

    for season in errors.index:

        seasonal_error = errors.loc[season]
        seasonal_error = seasonal_error.to_frame().transpose()
        for variable in variables:
            variable_to_plot = [col for col in seasonal_error if variable in col]
            variable_error = seasonal_error[variable_to_plot]
            fig = plt.figure()
            ax = fig.subplots(3)
            for i, error_type in enumerate(error_types):
                columns_to_plot = [col for col in variable_error.columns if error_type in col]
                seasonal_error_type = variable_error[columns_to_plot]
                seasonal_error_type.transpose().plot(ax=ax[i])
                ax[i].grid()
            plt.show()

    plt.show()


"""
Functions for the reconstruction's skill evaluation. These are used in the Jupyter Notebook 
'Reconstruction Validation.ipynb'
"""


class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd,
                 fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as fa
        import mpl_toolkits.axisartist.grid_finder as gf

        self.refstd = refstd  # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi / 2
        tlocs = np.arccos(rlocs)  # Conversion to polar angles
        gl1 = gf.FixedLocator(tlocs)  # Positions
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = fa.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()

        ax = fa.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")  # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)  # Unused

        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd ** 2 + rs ** 2 - 2 * self.refstd * rs * np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


def taylor_test(std_ref, models_skill):
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 10})
    dia = TaylorDiagram(std_ref, fig=fig, label='Observation', extend=True)
    dia.samplePoints[0].set_color('r')  # Mark reference point as a red star

    for i, (model, values) in enumerate(models_skill.iterrows()):

        # Uncomment to use the number of the model as the marker
        # marker = (f'{i + 1:03}')
        # marker = '$   ·' + marker + '$'

        marker = list(Line2D.markers.keys())[i]

        if "closest" in model:
            color = "tab:blue"
        elif "average" in model:
            color = "tab:red"
        elif "quantilemap" in model:
            color = "tab:green"
        else:
            color = "grey"

        dia.add_sample(values['std'], values['r2'],
                       marker=marker, ms=8, ls='',
                       label=model, color=color)

    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5')  # 5 levels in grey
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.0f')

    dia.add_grid()  # Add grid
    dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

    # Add a figure legend and title

    fig.legend(dia.samplePoints,
               [p.get_label() for p in dia.samplePoints],
               numpoints=1, prop=dict(size='small'), loc='upper right',
               bbox_to_anchor=(1.2, 1), ncol=int(np.ceil(len(models_skill) / 20)))
    fig.suptitle("Taylor diagram", size='x-large')  # Figure title

    return dia


def get_reconstructions(filepath: str, station_to_validate: str, variable_to_validate: str):
    file_paths = []
    for root, directories, files in os.walk(filepath):
        for file in files:
            reconstruction_information = file.split('.')[0].split('_')
            if len(reconstruction_information) == 3:
                station, variable, _ = reconstruction_information
                if station == station_to_validate and variable == variable_to_validate:
                    file_paths.append(root + "/" + file)
                else:
                    continue
            else:
                print("Warning: File naming not recognized. Must be 'station_variable_method.csv'")
    file_paths = sorted(file_paths)
    return file_paths


def get_hydrological_years(df):
    df["hydroyear"] = [date.year if date.month in [1, 2, 3, 4, 5, 6, 7, 8, 9] else date.year - 1 for date in df.index]
    return df


def get_ensemble(filepath, station_to_validate, variable_to_validate, freq, grouping):
    file_paths = get_reconstructions(filepath, station_to_validate, variable_to_validate)

    reconstruction_ensemble = []
    for file in file_paths:
        _, _, method = file.split('/')[-1].split('.')[0].split('_')
        reconstruction = pd.read_csv(file, index_col=0)
        reconstruction.index = pd.to_datetime(reconstruction.index)
        reconstruction.index.name = "time"
        if grouping == 'mean':
            reconstruction = reconstruction.resample(freq).mean()
        elif grouping == 'sum' or grouping == 'hydrosum' and freq != "1Y":
            reconstruction = reconstruction.resample(freq).sum()
        elif grouping == 'hydrosum' and freq == "1Y":
            reconstruction = get_hydrological_years(reconstruction)
            reconstruction = reconstruction.groupby("hydroyear").sum()
            reconstruction.index.name = "time"

        reconstruction["similarity_method"] = method
        reconstruction = reconstruction.reset_index()
        reconstruction_ensemble.append(reconstruction)

    reconstruction_ensemble = pd.concat(reconstruction_ensemble, axis=0)
    reconstruction_ensemble = reconstruction_ensemble.reset_index()

    return reconstruction_ensemble


def ensemble_to_dict(ensemble):
    """
    Get reconstruction ensemble DataFrame as a dictionary of each reconstruction
    """

    ensemble_dict = {}

    methods = sorted(list(set(ensemble["similarity_method"].values)))

    for method in methods:
        # Get individual reconstructions
        reconstruction = ensemble.loc[ensemble["similarity_method"] == method]
        reconstruction = reconstruction.set_index("time")

        ensemble_dict[method] = reconstruction

    return ensemble_dict



def clean_df(df):
    """
    Delete conflictive values from DataFrame (NaN or inf)
    :param df: DataFrame.
    :return df: DataFrame. Cleaned vesion of original df.
    """
    print(df)
    assert isinstance(df, pd.Series), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def get_common_data(x, y):
    """
    Reduce two DataFrames (x and y) to their common index elements
    """
    # Clean DataFrames of possible conflictive values
    # x = clean_df(x)
    # y = clean_df(y)
    common_idx = list(set(x.index).intersection(y.index))
    x = x.loc[common_idx]
    y = y.loc[common_idx]
    return x, y


def calculate_bias(observed, simulated):
    """
    Calculate bias
    :param observed: DataFrame
    :param simulated: DataFrame
    :return bias: DataFrame
    """
    # observed, simulated = get_common_data(observed, simulated)
    error = simulated - observed
    bias = error.mean()
    return bias


def calculate_rmse(observed, simulated):
    """
    Calculate Root Mean Squared Error
    :param observed: DataFrame
    :param simulated: DataFrame
    :return rmse: DataFrame
    """
    # observed, simulated = get_common_data(observed, simulated)
    squared_error = (simulated - observed) ** 2
    mse = squared_error.mean()
    rmse = np.sqrt(mse)
    return rmse


def calculate_correlation(observed, simulated):
    """
    Calculate correlation
    :param observed: DataFrame
    :param simulated: DataFrame
    :return correlation: DataFrame
    """
    observed = observed.dropna(axis=0)
    simulated = simulated.dropna(axis=0)
    observed, simulated = get_common_data(observed, simulated)
    correlation = simulated.corr(observed)
    return correlation


def calculate_ssmse(observed, simulated, reference):
    """
    MSE-based skill score
    """
    mse = calculate_mse(observed=observed, simulated=simulated)
    mse_r = calculate_mse(observed=observed, simulated=reference)
    ssmse = 1 - mse / mse_r
    return ssmse


def calculate_brier_score(observed, simulated, threshold):
    """
    Brier Score
    """


def calculate_contingency_table(observed, simulated, threshold, norm=False):
    """
    Contingency Table
    """
    # Above threshold
    observed_at = observed.where(observed >= threshold).dropna()
    simulated_at = simulated.where(simulated >= threshold).dropna()

    # Below threshold
    observed_bt = observed.where(observed < threshold).dropna()
    simulated_bt = simulated.where(simulated < threshold).dropna()

    true_positive = len(list(set(observed_at.index) & set(simulated_at.index)))
    true_negative = len(list(set(observed_bt.index) & set(simulated_bt.index)))

    false_positive = len(list(set(observed_bt.index) & set(simulated_at.index)))
    false_negative = len(list(set(observed_at.index) & set(simulated_bt.index)))

    contingency_table = pd.DataFrame(
        index=['Predicted True', 'Predicted False'],
        columns=['Observed True', 'Observed False']
    )

    contingency_table.loc['Predicted True', 'Observed True'] = true_positive
    contingency_table.loc['Predicted False', 'Observed False'] = true_negative
    contingency_table.loc['Predicted False', 'Observed True'] = false_negative
    contingency_table.loc['Predicted True', 'Observed False'] = false_positive

    if norm:
        n = sum(sum(contingency_table.values))
        contingency_table = contingency_table / n

    return contingency_table


def calculate_hss(observed, simulated, reference, threshold):
    """
    Heidke Skill Score
    This score goes from -inf to 1. If HSS is 1 the forecast is perfect, if its 0 there is no difference between the
    reference model and the forecast model. Good skill means HSS in the range (0, 1]
    """

    ss = calculate_contingency_table(observed=observed, simulated=simulated, threshold=threshold, norm=False)
    ss_r = calculate_contingency_table(observed=observed, simulated=reference, threshold=threshold, norm=False)

    n = sum(sum(ss.values))
    n_r = sum(sum(ss_r.values))

    r = (ss.loc['Predicted True', 'Observed True'] + ss.loc['Predicted False', 'Observed False']) / n
    r_r = (ss_r.loc['Predicted True', 'Observed True'] + ss_r.loc['Predicted False', 'Observed False']) / n_r

    if 1 - r_r == 0:
        hss = 9999
    else:
        hss = (r - r_r) / (1 - r_r)

    return hss




def get_skill(observation, simulations, variable):
    models_skill = pd.DataFrame(index=list(simulations.keys()), columns=['bias', 'rmse', 'r2', 'std'])

    for model, simulation in simulations.items():
        simulation['observation'] = observation

        model_std = np.nanstd(simulation[variable].values)
        model_rmse = calculate_rmse(simulation[variable], simulation['observation'])
        model_bias = calculate_bias(simulation[variable], simulation['observation'])
        model_correlation = calculate_correlation(simulation[variable], simulation['observation'])

        models_skill.loc[
            model, ['bias', 'rmse', 'r2', 'std']] = model_bias, model_rmse, model_correlation, model_std

    std_ref = observation.std().values[0]

    return std_ref, models_skill


def quantile_plots(predicted, observed):

    # Check if the columns are the same
    common_cols = list(set(predicted.columns) & set(observed.columns))

    for col in common_cols:
        # Get the common column
        prediction = predicted[col]
        observation = observed[col]

        prediction, observation = get_common_data(prediction.to_frame(), observation.to_frame())

        prediction = prediction.squeeze()
        observation = observation.squeeze()

        # Get the quantile of each value
        prediction_quantiles = prediction.apply(lambda x: percentileofscore(prediction.values, score=x))
        observation_quantiles = observation.apply(lambda x: percentileofscore(observation.values, score=x))

        prediction_quantiles = prediction_quantiles.rename('quantile')
        observation_quantiles = observation_quantiles.rename('quantile')

        prediction = pd.concat([prediction, prediction_quantiles], axis=1)
        observation = pd.concat([observation, observation_quantiles], axis=1)

        # Reorder by quantile
        prediction = prediction.set_index('quantile')
        observation = observation.set_index('quantile')

        # Delete duplicate values
        prediction = prediction[~prediction.index.duplicated(keep='first')]
        observation = observation[~observation.index.duplicated(keep='first')]

        # Put in common dataframe
        df = pd.concat([prediction, observation], axis=1)
        df.columns = ['Reconstruction', 'Observation']

    return df