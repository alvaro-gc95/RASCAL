import os
import itertools
import rascal.utils
import rascal.analogs

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import cycle
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


class RSkill:

    def __init__(
            self,
            observations=None,
            reconstructions=None,
            reanalysis=None,
            data=None
    ):

        if observations is not None:
            self.observations = observations.add_suffix("_observation")
        else:
            self.observations = data[[col for col in data.columns if "_observation" in col]]

        if reconstructions is not None:
            self.reconstructions = reconstructions.add_suffix("_reconstructions")
        else:
            self.reconstructions = data[[col for col in data.columns if "_reconstructions" in col]]

        if reanalysis is not None:
            self.reanalysis = reanalysis.add_suffix("_reanalysis")
        else:
            self.reanalysis = data[[col for col in data.columns if "_reanalysis" in col]]

        if data is None:
            self.data = pd.concat([self.observations, self.reanalysis, self.reconstructions], axis=1)
        else:
            self.data = data

    def resample(self, freq: str, grouping: str, hydroyear=False, skipna=False):
        """
        Resample the dataset containing observations, reconstructions and reanalysis data.
        :param freq: New sampling frequency.
        :param grouping: Options="mean", "median" or "sum"
        :param hydroyear: Default=False. If True, when the resampling frequency is "1Y" it takes hydrological years
        :param skipna: Default=False. If True ignore NaNs.
        (from October to September) instead of natural years
        """

        if hydroyear and "1Y" in grouping:
            resampled_df = get_hydrological_years(self.data)
            resampled_df = resampled_df.groupby("hydroyear")
        else:
            resampled_df = self.data.resample(freq)

        idx = resampled_df.indices
        if grouping == 'mean':
            resampled_df = [x[1].mean(skipna=skipna) for x in resampled_df]
        elif grouping == "median":
            resampled_df = [x[1].median(skipna=skipna) for x in resampled_df]
        elif grouping == 'sum':
            resampled_df = [x[1].sum(skipna=skipna) for x in resampled_df]
        else:
            print("ERROR: grouping '" + grouping + "' does not exist")
        resampled_df = pd.DataFrame(resampled_df, idx)

        resampled_df.index.name = "time"

        return RSkill(data=resampled_df)

    def plotseries(self, color=None, start=None, end=None, methods=None):

        if color is None:
            color = {
                "reanalysis": "y",
                "observations": "black",
                "quantilemap": "tab:green",
                "average": "tab:red",
                "closest": "tab:blue"
            }

        if methods is None:
            methods_full_name = self.reconstructions.columns
        else:
            methods_full_name = []
            for c, m in itertools.product(self.reconstructions.columns, methods):
                if m in c:
                    methods_full_name.append(c)

        plt.rcParams.update({'font.size': 22})

        fig = plt.figure(figsize=(21, 8))
        ax = fig.subplots()

        linestyles = ['-', '--', '-.', ':']
        linestyles_cycle = cycle(linestyles)

        # Reconstructions
        previous_method = "None"
        for method in methods_full_name:

            actual_method = ''.join((x for x in method.split("_")[1] if not x.isdigit()))

            if previous_method != actual_method:
                linestyles_cycle = cycle(linestyles)

            reconstruction = self.reconstructions[method]
            label = method.split("_")[1]
            rec_color = [c for c in list(color.keys()) if c in method][0]
            reconstruction.plot(
                ax=ax,
                label=label,
                color=color[rec_color],
                linestyle=next(linestyles_cycle),
                linewidth=3
            )
            previous_method = actual_method

        # Observations
        self.observations[self.observations.columns[0]].plot(
            ax=ax,
            color=color["observations"],
            label="Observations",
            linewidth=3,
            marker='o'
        )

        # Reanalysis
        for member in self.reanalysis.columns:
            if member.split("_")[1] == "mean":
                alpha = 1
                lw = 3
            else:
                alpha = 0.4
                lw = 1

            if len(self.reanalysis.columns) > 1:
                label = "Reanalysis ens. " + member.split("_")[1]
            else:
                label = "Reanalysis"
                lw = 3
                alpha = 1
            self.reanalysis[member].plot(ax=ax, color=color["reanalysis"], label=label, alpha=alpha, linewidth=lw)

        ax.legend(loc='upper right', ncol=1, bbox_to_anchor=[1.31, 1.03])
        ax.set_xlim(start, end)
        ax.grid()

        return fig, ax

    def skill(self, reference=None, threshold=None):

        if reference is not None:
            reference_model = self.reanalysis[reference]
        else:
            reference_model = None

        observation_std, reconstruction_skill_table = get_skill(
            observation=self.observations,
            simulations=self.reconstructions,
            reference_model=reference_model,
            threshold=threshold
        )

        _, reanalysis_skill_table = get_skill(
            observation=self.observations,
            simulations=self.reanalysis,
            reference_model=reference_model,
            threshold=threshold
        )

        skill_table = pd.concat([reconstruction_skill_table, reanalysis_skill_table], axis=0)
        return observation_std, skill_table

    def taylor(self):
        observation_std, skill_table = self.skill()
        fig, ax = taylor_test(std_ref=observation_std, models_skill=skill_table)
        return fig, ax

    def annual_cycle(self, grouping=None, color=None):
        """
        cycle_type: str. Options=["monthly", "annual"]
        freq: str. time resolution of the data in the cycle.
        """

        grouped_data = self.data.groupby(self.data.index.month)

        if grouping == "sum":
            grouped_data = grouped_data.sum()
        elif grouping == "mean":
            grouped_data = grouped_data.mean()
        elif grouping == "median":
            grouped_data = grouped_data.median()
        elif grouping == "std":
            grouped_data = grouped_data.std()
        else:
            print("Grouping '" + grouping + "' does not exist")
            exit()

        plt.rcParams.update({'font.size': 15})

        fig = plt.figure(figsize=(6, 6))
        ax = fig.subplots()

        if color is None:
            color = {
                "reanalysis": "y",
                "observation": "black",
                "quantilemap": "tab:green",
                "average": "tab:red",
                "closest": "tab:blue"
            }

        linestyles = ['-', '--', '-.', ':']
        linestyles_cycle = cycle(linestyles)

        n_members = len([c for c in grouped_data.columns if "reanalysis" in c])

        # Reconstructions
        previous_method = "None"
        for method in grouped_data.columns:

            actual_method = ''.join((x for x in method.split("_")[1] if not x.isdigit()))

            if previous_method != actual_method:
                linestyles_cycle = cycle(linestyles)

            lw = 2
            alpha = 1
            marker = None

            # Create label
            if "reconstruction" in method:
                label = method.split("_")[1]

            elif "reanalysis" in method:
                if method.split("_")[1] != "mean":
                    alpha = 0.4
                    lw = 1.5
                if n_members > 1:
                    label = "Reanalysis ens. " + method.split("_")[1]
                else:
                    alpha = 1
                    label = "Reanalysis"

            elif "observation" in method:
                marker = 'o'
                label = "Observation"

            else:
                label = method

            seasonality = grouped_data[method]

            rec_color = [c for c in list(color.keys()) if c in method][0]
            seasonality.plot(
                ax=ax,
                label=label,
                color=color[rec_color],
                linestyle=next(linestyles_cycle),
                linewidth=lw,
                alpha=alpha,
                marker=marker,
            )
            previous_method = actual_method

        ax.legend(loc="upper right", bbox_to_anchor=[1.6, 1.0], ncol=1, fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xlim(1, 12)
        ax.grid()
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

        return fig, ax

    def qqplot(self):

        fig = plt.figure(figsize=(8, 6))
        ax = fig.subplots()

        markers = [".", "o", "<", ">", "^", "v", "8", "s", "p", "P", "*", "h", "H", "X", "d", "D"]
        markers_cycled = itertools.cycle(markers)

        for model in self.reconstructions.columns:

            reconstruction = self.reconstructions[model].to_frame()

            quantiles = rascal.skill_evaluation.get_equivalent_quantile(
                predicted=reconstruction,
                observed=self.observations
            )

            if "closest" in model:
                color = "tab:blue"
            elif "average" in model:
                color = "tab:red"
            elif "quantilemap" in model:
                color = "tab:green"
            else:
                color = "y"

            label = model.split("_")[1]

            sns.scatterplot(
                data=quantiles,
                x='Predicted',
                y='Observed',
                ax=ax,
                alpha=0.6,
                color=color,
                label=label,
                marker=next(markers_cycled)
            )

        markers_cycled = itertools.cycle(markers)
        for member in self.reanalysis.columns:
            reanalysis_member = self.reanalysis[member].to_frame()

            reanalysis_quantiles = rascal.skill_evaluation.get_equivalent_quantile(
                predicted=reanalysis_member,
                observed=self.observations
            )

            if member.split("_")[1] == "mean":
                alpha = 0.6
            else:
                alpha = 0.2

            if len(self.reanalysis.columns) > 1:
                label = "Reanalysis ens. " + member.split("_")[1]
            else:
                label = "Reanalysis"
                alpha = 0.6

            sns.scatterplot(
                data=reanalysis_quantiles,
                x='Predicted',
                y='Observed',
                ax=ax,
                alpha=alpha,
                color="y",
                label=label,
                marker=next(markers_cycled)
            )
        ax.axline([0, 0], [1, 1], color='grey')
        ax.set_aspect('equal')
        ax.grid()

        ax.legend(
            loc="upper right",
            bbox_to_anchor=[1.6, 1.0],
            ncol=int(np.ceil(len(self.data.columns) / 20)),
            fontsize=12
        )

        min_value = self.observations.min().values[0]
        max_value = self.observations.max().values[0]

        ax.set_ylim(min_value, max_value)
        ax.set_xlim(min_value, max_value)

        return fig, ax


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

    markers = [".", "o", "<", ">", "^", "v", "8", "s", "p", "P", "*", "h", "H", "X", "d", "D"]
    markers_cycled = itertools.cycle(markers)
    # markers = list(Line2D.markers.keys())
    # useless_markers = [",", "1", "2", "3", "4", "+", "x", "|", "_"]
    # useless_markers.extend(list(range(12)))

    # for um in useless_markers:
    #     markers.remove(um)

    n_members = len([m for m in models_skill.index if "reanalysis" in m])

    for i, (model, values) in enumerate(models_skill.iterrows()):

        # Uncomment to use the number of the model as the marker
        # marker = (f'{i + 1:03}')
        # marker = '$   ·' + marker + '$'

        # marker = markers[i]

        marker = next(markers_cycled)

        # Select color
        if "closest" in model:
            color = "tab:blue"
        elif "average" in model:
            color = "tab:red"
        elif "quantilemap" in model:
            color = "tab:green"
        else:
            color = "y"

        # Select transparency
        if "reanalysis" in model and "mean" not in model and n_members > 1:
            alpha = 0.2
        else:
            alpha = 0.5

        # Create label
        if "reconstruction" in model:
            label = model.split("_")[1]
        elif "reanalysis" in model:

            if n_members > 1:
                label = "Reanalysis ens. " + model.split("_")[1]
            else:
                label = "Reanalysis"
        else:
            label = model

        dia.add_sample(values['std'], values['r2'],
                       marker=marker, ms=8, ls='',
                       label=label, color=color, markeredgecolor=color, alpha=alpha)

    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5')  # 5 levels in grey
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.0f')

    dia.add_grid()  # Add grid
    dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

    # Add a figure legend and title
    fig.legend(dia.samplePoints,
               [p.get_label() for p in dia.samplePoints],
               numpoints=1, prop=dict(size='small'), loc='upper right',
               bbox_to_anchor=(1.5, 1), ncol=int(np.ceil(len(models_skill) / 20)))

    # fig.suptitle("Taylor diagram", size='x-large')  # Figure title

    return fig, dia


def get_reconstructions_paths(filepath: str, station_to_validate: str, variable_to_validate: str):
    """

    :param filepath:
    :param station_to_validate:
    :param variable_to_validate:
    :return:
    """
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


def get_reanalysis_in_gridpoint(
        path: str,
        initial_year: int,
        final_year: int,
        variable: str,
        grouping: str,
        grid_point: dict,
        file_format=".grib",
        ensemble_member=None
):
    """
    Get reanalysis data in a single gridpoint.
    WARNING!: The temperature is converted from Kelvin to Celsius and the precipitation from meters to mm
    :param path: reanalysis files path
    :param initial_year: time series initial year
    :param final_year: time series final year
    :param variable: variable to retrieve
    :param grouping: 'hour(optional)_frequency_grouping'
    :param grid_point: {"lat": Latitude decimal coordinate, "lon": Longitude decimal coordinate}
    :param file_format: File extension. Default=".grib". Options=(".grib", ".nc")
    :param ensemble_member: int. Default=None. Ensemble member to select when using ensemble data, like forecasts.
    """
    # Years of the reanalysis data
    years = [str(y) for y in range(initial_year, final_year + 1)]

    # Get the file paths of the reanalysis variable
    reanalysis_files = rascal.utils.get_files(
        nwp_path=path,
        variables=variable,
        dates=years,
        file_format=file_format
    )

    # Get the variable as a time series in the gridpoint of the station
    reanalysis = rascal.analogs.Predictor(
        paths=reanalysis_files,
        grouping=grouping,
        lat_min=grid_point["lat"],
        lat_max=grid_point["lat"],
        lon_min=grid_point["lon"],
        lon_max=grid_point["lon"],
        mosaic=False,
        number=ensemble_member
    )
    reanalysis_data = reanalysis.data.drop_vars(["latitude", "longitude"]).to_dataframe()

    # Change variable name to RASCAL common acronym
    if ensemble_member is None:
        reanalysis_data.columns = [variable]
    else:
        reanalysis_data.columns = [variable + "_" + str(ensemble_member)]

    # Change units
    # Kelvin to Celsius
    if variable in ["TMAX", "TMIN", "TMEAN", "TMPA"]:
        reanalysis_data = reanalysis_data - 273.1
    # m to mm
    elif variable == "PCP":
        reanalysis_data = reanalysis_data * 1000

    return reanalysis_data


def get_reanalysis_ensemble(df, variable_to_validate, freq, grouping):
    """
    Process reanalysis resamples when dealing with ensembles. This work for a multiindexed dataframe or with a
    multiple column dataframe.
    :param df: pd.DataFrame.
    :param variable_to_validate: str.
    :param freq: str.
    :param grouping: str. Options=['mean', 'sum', 'hydrosum']
    :return resampled_df: pd.DataFrame.
    """
    if isinstance(df.index, pd.MultiIndex):

        unstacked_df = df.unstack()

        if grouping == "mean":
            resampled_df = unstacked_df.resample(freq).mean()
        elif grouping == 'sum' or grouping == 'hydrosum' and freq != "1Y":
            resampled_df = unstacked_df.resample(freq).sum()
        elif grouping == 'hydrosum' and freq == "1Y":
            unstacked_df = rascal.skill_evaluation.get_hydrological_years(unstacked_df)
            resampled_df = unstacked_df.groupby("hydroyear").sum()
            resampled_df.index.name = "time"
        else:
            print("ERROR: grouping '" + grouping + "' does not exist")

        ensemble_mean = resampled_df.mean(axis=1).to_frame()
        ensemble_mean.columns = ["_".join([variable_to_validate, "ensemble", "mean"])]

        resampled_df.columns = resampled_df.columns.droplevel(0)
        resampled_df.columns = [
            "_".join([variable_to_validate, str(col)]) for col in resampled_df.columns
        ]

        resampled_df = pd.concat([resampled_df, ensemble_mean], axis=1)

    else:
        if grouping == "mean":
            resampled_df = df.resample(freq).mean()
        elif grouping == 'sum' or grouping == 'hydrosum' and freq != "1Y":
            resampled_df = df.resample(freq).sum()
        elif grouping == 'hydrosum' and freq == "1Y":
            unstacked_df = rascal.skill_evaluation.get_hydrological_years(df)
            resampled_df = unstacked_df.groupby("hydroyear").sum()
            resampled_df.index.name = "time"
        else:
            print("ERROR: grouping '" + grouping + "' does not exist")

        if len(df.columns) > 1:
            ensemble_mean = resampled_df.mean(axis=1).to_frame()
            ensemble_mean.columns = ["_".join([variable_to_validate, "mean"])]
            resampled_df = pd.concat([resampled_df, ensemble_mean], axis=1)

    return resampled_df


def get_reconstruction_ensemble(filepath, station_to_validate, variable_to_validate, freq, grouping):
    file_paths = get_reconstructions_paths(filepath, station_to_validate, variable_to_validate)

    reconstruction_ensemble = []

    for file in file_paths:

        # Reconstruction method
        _, _, method = file.split('/')[-1].split('.')[0].split('_')
        # Reconstruction data
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

        # This part seems a bit unpractical with the exception of some seaborn utilities, a more practical alternative
        # is to concatenate each reconstruction as a different column in the dataframe
        # reconstruction["similarity_method"] = method
        # reconstruction = reconstruction.reset_index()
        # reconstruction_ensemble.append(reconstruction)

        reconstruction.columns = ["_".join([col, method]) for col in reconstruction.columns]
        reconstruction_ensemble.append(reconstruction)

    # reconstruction_ensemble = pd.concat(reconstruction_ensemble, axis=0)
    # reconstruction_ensemble = reconstruction_ensemble.reset_index()
    # reconstruction_ensemble = reconstruction_ensemble.drop(columns="index")

    reconstruction_ensemble = pd.concat(reconstruction_ensemble, axis=1)
    reconstruction_ensemble = delete_bands(reconstruction_ensemble)

    return reconstruction_ensemble


def delete_bands(df):
    """
    Delete maximum and minimum values bands of the reconstruction dataframes
    """
    band_columns = [col for col in df.columns if "band" in col]
    df = df.drop(columns=band_columns)
    return df


def ensemble_to_dict(ensemble, ensemble_col):
    """
    Get reconstruction ensemble DataFrame as a dictionary of each reconstruction
    """

    ensemble_dict = {}

    members = sorted(list(set(np.squeeze(ensemble[ensemble_col].values))))

    for member in members:
        # Get individual reconstructions
        ensemble_member = ensemble[ensemble[ensemble_col] == member]
        ensemble_member = ensemble_member.set_index("time")

        ensemble_dict[member] = ensemble_member

    return ensemble_dict


def clean_df(df):
    """
    Delete conflictive values from DataFrame (NaN or inf)
    :param df: DataFrame.
    :return df: DataFrame. Cleaned vesion of original df.
    """
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
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
    se = (observed - simulated) ** 2
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


def get_skill(observation, simulations, **kwargs):
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
    :param kwargs:
        - reference_variable: pd.DataFrame. Time series of a reference model to compare when calculating SSMSE and HSS.
        - threshold. float. Necessary only for HSS and BS
    :return:
        - std_ref: Standard deviation of the observations
        - models_skill: Table of each skill score for each simulation.
    """

    if "reference_model" in list(kwargs.keys()) and kwargs["reference_model"] is not None:
        if "threshold" not in list(kwargs.keys()) or kwargs["threshold"] is None:
            models_skill = pd.DataFrame(
                index=simulations.columns,
                columns=["bias", "rmse", "r2", "std", "ssmse"]
            )
            ssmse = True
            hss = False
            bs = False

        else:
            models_skill = pd.DataFrame(
                index=simulations.columns,
                columns=["bias", "rmse", "r2", "std", "ssmse", "hss", "bs"]
            )
            ssmse = True
            hss = True
            bs = True
    else:
        if "threshold" not in list(kwargs.keys()):
            models_skill = pd.DataFrame(
                index=simulations.columns,
                columns=['bias', 'rmse', 'r2', 'std']
            )
            ssmse = False
            hss = False
            bs = False

        else:
            models_skill = pd.DataFrame(
                index=simulations.columns,
                columns=['bias', 'rmse', 'r2', 'std']
            )
            ssmse = False
            hss = False
            bs = True

    for col in simulations.columns:
        simulation = simulations[col].to_frame().copy()
        simulation['observation'] = observation.copy()

        model_std = np.nanstd(simulation[col].values)
        model_rmse = calculate_rmse(simulation[col], simulation['observation'])
        model_bias = calculate_bias(simulation[col], simulation['observation'])
        model_correlation = calculate_correlation(simulation[col], simulation['observation'])
        models_skill.loc[
            col, ['bias', 'rmse', 'r2', 'std']
        ] = model_bias, model_rmse, model_correlation, model_std

        if ssmse:
            model_ssmse = calculate_ssmse(
                observed=simulation["observation"],
                simulated=simulation[col],
                reference=kwargs["reference_model"]
            )
            models_skill.loc[col, "ssmse"] = model_ssmse
        if hss:
            model_hss = calculate_hss(
                observed=simulation["observation"],
                simulated=simulation[col],
                reference=kwargs["reference_model"],
                threshold=kwargs["threshold"]
            )
            models_skill.loc[col, "hss"] = model_hss
        if bs:
            model_bs = calculate_brier_score(
                observed=simulation["observation"],
                simulated=simulation[col],
                threshold=kwargs["threshold"]
            )
            models_skill.loc[col, "bs"] = model_bs

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


def get_equivalent_quantile(predicted: pd.DataFrame, observed: pd.DataFrame):
    """
    Quantile-Quantile Table. Calculate the quantile of each value of the prediction and the simulation. Then compare
    the value of each quantile between prediction and observation.
    :param predicted: pd.DataFrame
    :param observed: pd.DataFrame
    :return equivalent_quantiles: pd.DataFrame
    """

    prediction, observation = get_common_data(clean_df(predicted), clean_df(observed))

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


def get_days_above_threshold(df, threshold, inverse=False):
    """
    Label day as 1 when a certain threshold is surpassed, and zero if not. The threshold value is not included.
    :param df: pd.DataFrame
    :param threshold: float.
    :param inverse: bool. Default=False. If True it returns days below threshold.
    """

    days_above_threshold = df.copy()
    for col in df.columns:
        if inverse:
            days_above_threshold[col][df[col] > threshold] = 0
            days_above_threshold[col][df[col] <= threshold] = 1
        else:
            days_above_threshold[col][df[col] > threshold] = 1
            days_above_threshold[col][df[col] <= threshold] = 0

    return days_above_threshold
