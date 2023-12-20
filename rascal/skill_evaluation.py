import os
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
    'PCP': ['PCP']
}

percentile_intervals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


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

        # ax.set_aspect("equal")

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
    fig = plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 14})
    dia = TaylorDiagram(std_ref, fig=fig, label='Observation', extend=False)
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
               bbox_to_anchor=(1.5, 0.8), ncol=int(np.ceil(len(models_skill) / 20)))
    # fig.suptitle("Taylor diagram", size='x-large')  # Figure title

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


def calculate_mbe(observed, simulated):
    """
    Mean Squared Error
    """
    observed, simulated = get_common_data(observed, simulated)
    be = observed - simulated
    mbe = be.mean()
    return mbe


def calculate_mse(observed, simulated):
    """
    Mean Squared Error
    """
    observed, simulated = get_common_data(observed, simulated)
    se = (observed - simulated)**2
    mse = se.mean()
    return mse


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
    # Arrays of boolean variables.
    # Consider an event happening when the value of the variable is above a determined threshold
    predicted_bool = simulated.copy()
    observed_bool = observed.copy()

    # 1 if the event happens, 0 if not
    predicted_bool[simulated < threshold] = 0
    predicted_bool[simulated >= threshold] = 1

    observed_bool[observed < threshold] = 0
    observed_bool[observed >= threshold] = 1

    brier_score = np.mean((predicted_bool - observed_bool) ** 2)

    return brier_score


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


def get_skill(observation, simulations, variable, **kwargs):
    """
     Generate a pd.DataFrame with the table of skills of various simulations. The skill metrics are:
        - Mean Bias Error (bias)
        - Root Mean Squared Error (rmse)
        - Correlation Coefficient (r2)
        - Standard Deviation (std)
        - MSE-based Skill Score (ssmse)
        - Heidke Skill Score (hss)
        - Brier Score (bs)
    :param observation: pd.DataFrame
    :param simulations: pd.DataFrame
    :param variable: str
    :param kwargs:
        - reference_variable: pd.DataFrame. Time series of a reference model to compare when calculating SSMSE and HSS.
        - threshold. float. Necessary only for HSS and BS
    :return:
        - std_ref: Standard deviation of the observations
        - models_skill: Table of each skill score for each simulation.
    """

    if "reference_model" in list(kwargs.keys()):
        if "threshold" not in list(kwargs.keys()):
            print("WARNING: A threshold is needed to calculate HSS and BS")
            models_skill = pd.DataFrame(
                index=list(simulations.keys()),
                columns=["bias", "rmse", "r2", "std", "ssmse"]
            )
            ssmse = True
            hss = False
            bs = False

        else:
            models_skill = pd.DataFrame(
                index=list(simulations.keys()),
                columns=["bias", "rmse", "r2", "std", "ssmse", "hss", "bs"]
            )
            ssmse = True
            hss = True
            bs = True
    else:
        print("WARNING: A reference model is needed to calculate HSS and SSMSE")
        if "threshold" not in list(kwargs.keys()):
            print("WARNING: A threshold is needed to calculate HSS and BS")
            models_skill = pd.DataFrame(
                index=list(simulations.keys()),
                columns=['bias', 'rmse', 'r2', 'std']
            )
            ssmse = False
            hss = False
            bs = False

        else:
            models_skill = pd.DataFrame(
                index=list(simulations.keys()),
                columns=['bias', 'rmse', 'r2', 'std']
            )
            ssmse = False
            hss = False
            bs = True

    for model, simulation in simulations.items():
        simulation['observation'] = observation.copy()

        model_std = np.nanstd(simulation[variable].values)
        model_rmse = calculate_rmse(simulation[variable], simulation['observation'])
        model_bias = calculate_bias(simulation[variable], simulation['observation'])
        model_correlation = calculate_correlation(simulation[variable], simulation['observation'])
        models_skill.loc[
            model, ['bias', 'rmse', 'r2', 'std']] = model_bias, model_rmse, model_correlation, model_std

        if ssmse:
            model_ssmse = calculate_ssmse(
                observed=simulation["observation"],
                simulated=simulation[variable],
                reference=kwargs["reference_model"][variable]
            )
            models_skill.loc[model, "ssmse"] = model_ssmse
        if hss:
            model_hss = calculate_hss(
                observed=simulation["observation"],
                simulated=simulation[variable],
                reference=kwargs["reference_model"][variable],
                threshold=kwargs["threshold"]
            )
            models_skill.loc[model, "hss"] = model_hss
        if bs:
            model_bs = calculate_brier_score(
                observed=simulation["observation"],
                simulated=simulation[variable],
                threshold=kwargs["threshold"]
            )
            models_skill.loc[model, "bs"] = model_bs

    std_ref = observation.std().values[0]

    return std_ref, models_skill


def get_month_year_bias(predicted, observed):
    # Check if the columns are the same
    common_cols = list(set(predicted.columns) & set(observed.columns))

    for col in common_cols:
        # Get common data
        observed_col, predicted_col = get_common_data(
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
        observed_col, predicted_col = get_common_data(
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

            observation, prediction = get_common_data(
                observations_in_interval.to_frame(),
                predicted_col
            )

            rmse = calculate_rmse(
                simulated=prediction.values,
                observed=observation.values
            )
            skills.loc[interval, 'RMSE'] = rmse

            mbe = calculate_mbe(
                simulated=prediction.values,
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

            brier = calculate_brier_score(
                simulated=prediction.values,
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

        prediction, observation = get_common_data(prediction.to_frame(), observation.to_frame())
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

            rmse = calculate_rmse(
                simulated=seasonal_data['rascal'].values,
                observed=seasonal_data['observation'].values
            )
            skills.loc[season, 'RMSE'] = rmse

            mbe = calculate_mbe(
                simulated=seasonal_data['rascal'].values,
                observed=seasonal_data['observation'].values
            )
            skills['MBE'].loc[season] = mbe

            pearson = seasonal_data.corr(method='pearson').values[1, 0]
            skills['Pearson'].loc[season] = pearson

            spearman = seasonal_data.corr(method='spearman').values[1, 0]
            skills['Spearman'].loc[season] = spearman

            brier = calculate_brier_score(
                simulated=seasonal_data['rascal'].values,
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


def quantile_plots(predicted, observed):
    """
    Quantile-Quantile Table. Calculate the quantile of each value of the prediction and the simulation. Then compare
    the value of each quantile between prediction and observation.
    :param predicted: pd.DataFrame
    :param observed: pd.DataFrame
    :return equivalent_quantiles: pd.DataFrame
    """

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
        equivalent_quantiles = pd.concat([prediction, observation], axis=1)
        equivalent_quantiles.columns = ['Predicted', 'Observed']

    return equivalent_quantiles
