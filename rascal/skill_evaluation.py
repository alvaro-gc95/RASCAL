import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import percentileofscore
import rascal.statistics

import autoval.utils
import seaborn as sns
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

        #recontructions = reconstructions.dropna()
        #reconstructions = reconstructions.resample('M').sum()
        #reconstructions = reconstructions.where(reconstructions > 0, np.nan)
        #reconstructions = reconstructions.rolling(10).mean()
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
