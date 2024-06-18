"""
########################################################################################################################
# ------------------------ RASCAL (Reconstruction by AnalogS of ClimatologicAL time series) -------------------------- #
########################################################################################################################
Version 1.0.0
Contact: alvaro@intermet.es

Multiple Runs Example

This is an example of how to use RASCAL. This script might be useful as a template to run multiple reconstructions with
different parameters for various variables and stations. Some extra steps are added in order to save time when using it
multiple times, for example, saving the predictors or the pcs as temporal files to avoid recalculating some common
intermediate results in each run.

"""

import os
import pickle
import datetime
import itertools

import rascal.utils
import rascal.analysis

import pandas as pd

from rascal.analogs import Station, Predictor, Analogs

# Open configuration
config = rascal.utils.open_yaml('config.yaml')

"""
------------------------------------------------------------------------------------------------------------------------
System Parameters ------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
# Paths
observations_path = config.get("observations_path")
reanalysis_path = config.get("reanalysis_path")
tmp_path = config.get("temporal_files_path")
output_path = config.get("output_path")


"""
------------------------------------------------------------------------------------------------------------------------
Reconstruction Parameters ----------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
initial_year = config.get('initial_year')
final_year = config.get('final_year')
years = [str(y) for y in range(initial_year, final_year + 1)]

# Variable to reconstruct
variables = config.get("variables")

# Training period
training_start = config.get('training_start')
training_end = config.get('training_end')
training_dates = pd.date_range(
    start=datetime.datetime(training_start[0], training_start[1], training_start[2]),
    end=datetime.datetime(training_end[0], training_end[1], training_end[2]),
    freq='1D'
)

# Reconstruction period
test_start = config.get('test_start')
test_end = config.get('test_end')
test_dates = pd.date_range(
    start=datetime.datetime(test_start[0], test_start[1], test_start[2]),
    end=datetime.datetime(test_end[0], test_end[1], test_end[2]),
    freq='1D'
)

# Station information
stations = config.get("stations")


"""
------------------------------------------------------------------------------------------------------------------------
Predictor Parameters ---------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
# Predictor domain
predictor_lat_min = config.get("predictor_lat_min")
predictor_lat_max = config.get("predictor_lat_max")
predictor_lon_min = config.get("predictor_lon_min")
predictor_lon_max = config.get("predictor_lon_max")

# Dictionaries of predictor variable acronyms for each predictand
predictors_for_variable = config.get("predictor_for_variable")
mapping_variables_for_variable = config.get("mapping_variables_for_variable")

# Principal predictor grouping
predictor_grouping = config.get("predictor_grouping")
# Grouping per variable (For the secondary predictor when using the Quantile Map method)
grouping_per_variable = config.get("grouping_per_variable")

# Overwrite Predictor object pickle
overwrite_predictor = config.get("overwrite_predictor")

"""
------------------------------------------------------------------------------------------------------------------------
Principal Component Analysis Parameters --------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

seasons = config.get("seasons")
pca_scaling = config.get("pca_scaling")
n_components = config.get("n_components")
standardize_anomalies = config.get("standardize_anomalies")

# Overwrite Predictor object pickle
overwrite_pcs = config.get("overwrite_pcs")

# Projection Years
projection_data_path = config.get("reanalysis_path")
projection_start = datetime.datetime(2012, 1, 1, 0, 0, 0)
projection_end = datetime.datetime(2012, 3, 31, 0, 0, 0)
projection_dates = pd.date_range(start=projection_start, end=projection_end, freq="1D")
projection_years = sorted(set([str(d.year) for d in projection_dates]))

"""
------------------------------------------------------------------------------------------------------------------------
Analog Method Parameters -----------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

similarity_methods = config.get("similarity_methods")
pool_sizes = config.get("analog_pool_size")
sample_sizes = config.get("weighted_mean_sample_size")

"""
------------------------------------------------------------------------------------------------------------------------
Multiple Runs Main Function --------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':

    for station_code, variable in itertools.product(stations, variables):

        # --------------------------------------------------------------------------------------------------------------
        # 1) Get historical record -------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # This is an utility that works if the directory in the path contains a meta.csv file with columns:
        #     - [code, name, latitude, longitude, altitude]
        # The get data method gets daily quantities of the selected variable. Example:
        #     - variable = TMPA (Hourly Temperature) -> .get_data("TMPA") retrieves "TMAX", "TMIN" and "TMEAN"
        # The file in the directory must be named as the variable to get, ex: TMPA.csv

        # The use of this object is optional, opening a pd.DataFrame of the historical data also works, but a latitude
        # and longitude of the station is needed in order to obtain a time series in the gridpoint for the secondary
        # predictor when using the Quantile Map method. Then in section 2) station.latitude and station.longitude
        # must be substituted by the value of the coordinates of the station.

        station = Station(path=observations_path + station_code + '/')
        station_data = station.get_data(variable=variable)

        # --------------------------------------------------------------------------------------------------------------
        # 2) Get reanalysis data ---------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        predictor_variables = predictors_for_variable[variable]
        mapping_variables = mapping_variables_for_variable[variable]

        predictor_filename = tmp_path + "-".join(
            ["and".join(predictor_variables), predictor_grouping, "predictor.pkl"]
        )
        pcs_filename = tmp_path + "-".join(
            ["and".join(predictor_variables), str(n_components).zfill(2), str(pca_scaling), "PCS.pkl"]
        )
        predictor_exists = os.path.exists(predictor_filename)
        pcs_exists = os.path.exists(pcs_filename)

        # Get predictors
        if overwrite_predictor or not predictor_exists:
            # Get file paths
            predictor_files = rascal.utils.get_files(
                nwp_path=reanalysis_path,
                variables=predictor_variables,
                dates=years,
                file_format=".nc")

            # Generate Predictor
            predictors = Predictor(
                paths=predictor_files,
                grouping=predictor_grouping,
                lat_min=predictor_lat_min,
                lat_max=predictor_lat_max,
                lon_min=predictor_lon_min,
                lon_max=predictor_lon_max,
                mosaic=True,
                number=None
            )
            rascal.utils.save_object(predictors, predictor_filename)

        else:
            with open(predictor_filename, 'rb') as inp:
                predictors = pickle.load(inp)

        # Get mapping variables
        if "quantilemap" in similarity_methods:

            mapping_years = years.copy()
            mapping_years.extend(projection_years)

            # Get file paths
            mapping_variable_files = rascal.utils.get_files(
                nwp_path=reanalysis_path,
                variables=mapping_variables,
                dates=mapping_years,
                file_format=".nc")

            # Generate Predictor
            mapping_variable = Predictor(
                paths=mapping_variable_files,
                grouping=grouping_per_variable[variable],
                lat_min=station.latitude,
                lat_max=station.latitude,
                lon_min=station.longitude,
                lon_max=station.longitude,
                mosaic=False,
                number=None
            )
            if len(mapping_variables) > 1:
                mapping_variable.module()

        # Get predictor to project
        projection_files = rascal.utils.get_files(
            nwp_path=projection_data_path,
            variables=predictor_variables,
            dates=projection_years,
            file_format=".nc")
        to_project = Predictor(
            paths=projection_files,
            grouping=predictor_grouping,
            lat_min=predictor_lat_min,
            lat_max=predictor_lat_max,
            lon_min=predictor_lon_min,
            lon_max=predictor_lon_max,
            mosaic=True,
            number=None
        )
        to_project.data = to_project.data.sel(time=projection_dates)

        # --------------------------------------------------------------------------------------------------------------
        # 3) Get Principal Components of the reanalysis data -----------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # To save time, check if the Principal component analysis is already done with the same predictors, components
        # and scaling. If a pickle containing the pcs is found, then retrieve it, if not it recalculates the PCs.
        # Is possible to overwrite an existing PCs file if overwrite_pcs == True.
        # WARNING! The parameter overwrite_pcs is not the same as the overwrite input in the pcs method. The "overwrite"
        # input is to overwrite another pickle file that contains all the variables of the PC Analysis (EoFs, loadings
        # ...). The overwrite input avoids to recalculate the PCs, but it does not avoid having to load the predictor
        # field to Predictor.
        # overwrite_pcs == True allows to use pre-calculated PCS to save a significant amount of time in each
        # reconstruction

        if not pcs_exists or overwrite_pcs:
            predictor_pcs = predictors.pcs(
                npcs=n_components,
                seasons=seasons,
                standardize=standardize_anomalies,
                pcscaling=pca_scaling,
                overwrite=True,
                path="./tmp/",
                training=[d for d in training_dates],
                project=to_project.data
            )
            rascal.utils.save_object(predictor_pcs, pcs_filename)
        else:
            with open(pcs_filename, 'rb') as inp:
                predictor_pcs = pickle.load(inp)

        # --------------------------------------------------------------------------------------------------------------
        # 4) Take an analog pool for each day to reconstruct, and select an analog for each similarity method ----------
        # --------------------------------------------------------------------------------------------------------------
        # Create an output directory for the predictors and PC scaling used
        output_directory = (
            output_path + variable + "/" +
            "-".join(["and".join(predictor_variables), "and".join(mapping_variables), str(pca_scaling)])
        )
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Generate Reconstructions for each method and pool size
        for method in similarity_methods:

            # This method does not depend on the size of the pool
            if method == "closest":

                analogs = Analogs(pcs=predictor_pcs, observations=station_data, dates=to_project.data["time"].values)
                reconstruction = analogs.reconstruct(
                    pool_size=min(pool_sizes),
                    method=method,
                )

                reconstruction_filename = '_'.join([
                    station_code,
                    variable,
                    method,
                ])
                reconstruction.to_csv(output_directory + "/" + reconstruction_filename + ".csv")
                print(" Saved: " + reconstruction_filename)

            # This method depends only on the number of analogs to average
            elif method == "average":

                for sample_size in sample_sizes:
                    analogs = Analogs(pcs=predictor_pcs, observations=station_data, dates=to_project.data["time"].values)
                    reconstruction = analogs.reconstruct(
                        pool_size=min(pool_sizes),
                        method=method,
                        sample_size=sample_size,
                    )

                    reconstruction_filename = '_'.join([
                        station_code,
                        variable,
                        method + str(sample_size).zfill(2),
                    ])

                    reconstruction.to_csv(output_directory + "/" + reconstruction_filename + ".csv")
                    print(" Saved: " + reconstruction_filename)

            # This method, since it uses the quantiles of the secondary predictor distribution, is sensible to the
            # pool size
            elif method == "quantilemap":

                for pool_size in pool_sizes:
                    analogs = Analogs(pcs=predictor_pcs, observations=station_data, dates=to_project.data["time"].values)
                    reconstruction = analogs.reconstruct(
                        pool_size=pool_size,
                        method=method,
                        mapping_variable=mapping_variable
                    )

                    reconstruction_filename = '_'.join([
                        station_code,
                        variable,
                        method + str(pool_size).zfill(3),
                    ])
                    reconstruction.to_csv(output_directory + "/" + reconstruction_filename + ".csv")
                    print(" Saved: " + reconstruction_filename)

            else:
                print("ERROR: Method " + method + " does not exist")



