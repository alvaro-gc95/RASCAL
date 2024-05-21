Code Overview: Modules
=======================

analogs.py
------------

.. py:module:: analogs

   :synopsis: This module contain the principal classes and functions to make the time series preprocessing and reconstructions





   The ``rascal.analogs.Station()`` class stores station metadata (code, name, altitude, longitude and latitude) and calculate daily time series.

   .. py:class:: rascal.analogs.Station(path)

      Stores station metadata (code, name, altitude, longitude and latitude) and calculate daily time series.

      :param path: Path of the directory that contains the observations.
      :type path: str
   
      .. py:attribute:: path
   
         Path of the directory that contains the observations
   
         :type: str
   
      .. py:attribute:: meta
      
         DataFrame with the metadata of the station (code, name, altitude, longitude and latitude) 
   
         :type: pd.DataFrame
   
      .. py:attribute:: longitude
      
         Longitude of the station
   
         :type: float
   
      .. py:attribute:: latitude
   
         Latitude of the station
   
         :type: float
   
      .. py:attribute:: altitude
      
         Elevation of the station
      
         :type: float
      
      .. py:method:: get_data(variable, [skipna=True])
      
         Get the daily time series of the ``variable``
   
         :param variable: variable name.
         :type variable: str
         :param skipna: skipna when resampling to daily frequency.
         :type skipna: bool
         :return: data
         :rtype: pd.DataFrame
      
      .. py:method:: get_gridpoint(grid_latitudes, grid_longitudes)






   The ``rascal.analogs.Predictor()`` class stores the predictor data and Principal Component Analysis results:

   .. py:class:: rascal.analogs.Predictor(paths, grouping, lat_min, lat_max, lon_min, lon_max, [mosaic=True], [number=None])

      Predictor class. This contains data about the predictor variable to use for the reconstruction.

    
      :param path: Paths of the grib file to open.
      :param grouping: Method of grouping the data, str format = "frequency_method"
      
         - frequency=("hourly", "daily", "monthly", "yearly")
         - method=("mean", "max", "min", "sum")
         
      :param lat_min: Predictor field minimum latitude
      :param lat_max: Predictor field maximum latitude
      :param lon_min: Predictor field minimum longitude
      :param lon_max: Predictor field maximum longitude
      :param mosaic: if True apply ``.to_mosaic()`` method
      :param number: Ensemble member number
   
      :type path: list[str]
      :type grouping: str or None
      :type lat_min: float
      :type lat_max: float
      :type lon_min: float
      :type lon_max: float
      :type mosaic: bool or None
      :type number: int or None
                     
   
      .. py:attribute:: data
   
         :type: xr.Dataset
   
      .. py:method:: crop(lat_min, lat_max, lon_min, lon_max)
   
         Crop the domain of the dataframe
         
         :param lat_min: New minimum latitude
         :param lat_max: New maximum latitude
         :param lon_min: New minimum longitude
         :param lon_max: New maximum longitude
         :type lat_min: float
         :type lat_max: float
         :type lon_min: float
         :type lon_max: float
   
      .. py:method:: to_mosaic()
      
         To use various simultaneous predictors or a vectorial variable, concatenate the variables along the longitude
         axis to obtain a single compound variable, easier to use when performing PCA.
   
         :return: compound_predictor
         :rtype: xr.Dataset

      .. py:method:: module()
   
         Get the module of the predictor variables as if they were components of a vector.
         
         :return: self
         :rtype: Predictor

   
      .. py:method:: anomalies([seasons], [standardize], [mean_period])
      
         Calculate seasonal anomalies of the field. The definition of season is flexible, being only a list of months
         contained within it.
         
         :param seasons: Months of the season. Default = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
         :param standardize: Standardize anomalies. Default = True
         :param mean_period: Dates to use as mean climatology to calculate the anomalies.
         :type seasons: list[list[int]] or None
         :type standardize: bool or None
         :type mean_period: list[pd.DatetimeIndex] or None
         
         :return: anomalies (dims = [time, latitude, longitude, season])
         :rtype: xr.Dataset
   
      .. py:method:: pcs(path, npcs, [seasons], [standardize], [pcscaling], [overwrite], [training], [project])

         Perform Principal Component Analysis. To save computation time, the PCA object can be saved as a pickle, so
         the analysis does not have to be performed every time.
         
         :param path: Path to save the PCA results
         :param npcs: Number of components.
         :param seasons: List of list of months of every season.
         :param standardize: If True, the anomalies used in the PCA are standardized.
         :param pcscaling: Set the scaling of the PCs used to compute covariance. The following values are accepted:
         
             - 0 : Un-scaled PCs.
             - 1 : PCs are scaled to unit variance (divided by the square-root of their eigenvalue) (default).
             - 2 : PCs are multiplied by the square-root of their eigenvalue.
             
         :param overwrite: Default = False. If True recalculate the PCA and overwrite the pickle with the PCA
         :param training: Dates to use for calculating the PCA
         :param project: Data to project onto the calculated PCA results.

         :type path: str
         :type npcs: int
         :type seasons: list[list[int]] or None
         :type standardize: bool or None
         :type pcscaling: int or None
         :type overwrite: bool or None
         :type training: list[pd.DatetimeIndex] or None
         :type project: xr.Dataset or None




   The ``rascal.analogs.Analogs()`` get the pool of analog days and reconstruct the time series:

   .. py:class:: rascal.analogs.Analogs(pcs, dates, observations)

      Predictor class. This contains data about the predictor variable to use for the reconstruction.

      :param path: Optional "kind" of ingredients.
      :type path: list[str] or None
   
      .. py:method:: get_pool(size, [vw_size], [vw_type], [distance])
      
         Get the pool of ``size`` closest neighbors to each day
         
        :param size: Number of neighbors in the pool.
        :param vw_size: Validation window size. How many data points around each point is ignored to validate the
            reconstruction.
        :param vw_type: Type of validation window. Options:
        
            - forward: The original date is the last date of the window.
            - backward: The original date is the firs date of the window.
            - centered: The original date is in the center of the window.
            
        :param distance: Metric to determine the distance between points in the PCs space. Options:
        
            - euclidean
            - mahalanobis (Wishlist)
            
        :return: ``(analog_dates, analog_distances)``, dates of the analogs in the pool for each day, and distances in the PCs space of each
            
        :type size: int
        :type vw_size: int or None
        :type vw_type: str or None
        :type distance: str or None
        :rtype: (pd.DataFrame, pd.DataFrame)
        
   
      .. py:method:: reconstruct([pool_size], [method], [sample_size], [mapping_variable], [vw_size], [vw_type], [distance])
          
         Reconstruct a time series using the analog pool for each day.
         
         :param pool_size: Size of the analog pool for each day.
         :param method: Similarity method to select the best analog of the pool. Options are:
         
            - 'closest': (Selected by default) Select the closest analog in the PCs space
            - 'average': Calculate the weighted average of the 'sample_size' closest analogs in the PCs space.
            - 'quantilemap': Select the analog that represent the same quantile in the observations pool that another mapping variable.
            
         :param sample_size: Number of analogs to average in the 'average' method
         :param mapping_variable: Time series of a variable to use as mapping in 'quantilemap'
         :param vw_size: Validation window size. How many data points around each point is ignored to validate the reconstruction.
         :param vw_type: Type of validation window. Options:
         
            - forward: The original date is the last date of the window.
            - backward: The original date is the firs date of the window.
            - centered: The original date is in the center of the window.
         
         :param distance: Metric to determine the distance between points in the PCs space. Options:
         
            - euclidean
            - mahalanobis (Wishlist)
            
         :type pool_size: int or None
         :type method: str or None
         :type sample_size: int or None
         :type mapping_variable: Predictor or None
         :type vw_size: int or None
         :type vw_type: str or None
         :type distance: str or None
         
         :return: reconstruction
         :rtype: pd.DataFrame
   

analysis.py
------------

.. py:module:: analysis

   :synopsis: This module contain the principal classes and functions analyze the skill and validate the reconstructions
   
   
   You can use the ``rascal.analysis.RSkill()`` class to validate and analyze the skill of the reconstructions:

   .. py:class:: rascal.analysis.RSkill([observations], [reconstructions], [reanalysis], [data])
      
      Predictor class. This contains data about the predictor variable to use for the reconstruction.
      
      :param observations: Obstervations time series
      :type observations: pd.DataFrame or None
      :param reconstructions: Reconstructions time series
      :type reconstructions: pd.DataFrame or None
      :param reanalysis: Reanalysis time series
      :type reanalysis: pd.DataFrame or None
      :param data: All data joined (observations, reconstructions, reanalysis)
      :type data: pd.DataFrame or None
      
      .. py:attribute:: observations
      
         Obstervations time series
      
         :type: pd.DataFrame
      
      .. py:attribute:: reconstructions
      
         Reconstructions time series
         
         :type: pd.DataFrame
      
      .. py:attribute:: reanalysis
      
         Reanalysis time series
      
         :type: pd.DataFrame
      
      .. py:attribute:: data
      
         All data joined (observations, reconstructions, reanalysis) concatenated in the columns axis
      
         :type: pd.DataFrame
      
      .. py:method:: resample(freq, grouping, [hydroyear], [skipna])
      
         Resample the dataset containing observations, reconstructions and reanalysis data.
         
         :param freq: New sampling frequency.
         :param grouping: Options="mean", "median" or "sum"
         :param hydroyear: Default=False. If True, when the resampling frequency is "1Y" it takes hydrological years (from October to September) instead of natural years
         :param skipna: Default=False. If True ignore NaNs. 
         
         :type freq: str
         :type grouping: str
         :type hydroyear: bool or None
         :type skipna: bool or None
         
         :return: RSkill with resampled data
         :rtype: RSkill
        
   
      .. py:method:: plotseries([color], [start], [end], [methods])
      
         Plot the time series of the reconstructions with the reanalysis and observations series
      
         :param color: dict of which color to use (values) with each dataset (keys)
         :param start: Start date of the plot
         :param end: End date of the plot
         :param methods: Reconstruction methods to plot
         
         :type color: dict or None
         :type start: Datetime or None
         :type end: Datetime or None
         :type methods: list[str] or None
   
      .. py:method:: skill([reference=None], [threshold=None])
      
        Generate a pd.DataFrame with the table of skills of various simulations. The skill metrics are:
        
            - Mean Bias Error (bias)
            - Root Mean Squared Error (rmse)
            - Correlation Coefficient (r2)
            - Standard Deviation (std)
            - MSE-based Skill Score (ssmse)
            - Heidke Skill Score (hss)
            - Brier Score (bs)

         :param reference: Time series of a reference model to compare when calculating SSMSE and HSS.
         :param threshold: Threshold to use when computing the HSS and BS
         
         :type referece: pd.DataFrame or None
         :type threshold: float or None
   
         :return: ``(observation_std, skill_table)``, Standard deviation of the observations and table of each skill score for each simulation.
         :rtype: (float, pd.DataFrame)  
         
         
      .. py:method:: taylor()
      
         Calls ``.skill()`` method and computes the Taylor diagram
         
         :return: fig, ax
   
      .. py:method:: annual_cycle([grouping], [color])
      
         Plot the annual cycle of the reconstructions, reanalysis and observations
      
         :param grouping: (Default="mean") Monthly grouping to plot in the cylce. Options=("sum", "mean", "median", "std")
         :param color: dict of which color to use (values) with each dataset (keys)
         :type grouping: str or None
         :type color: dict or None
   
      .. py:method:: qqplot()
      
         Quantile-Quantile plot



indices.py
------------

.. py:module:: indices

   :synopsis: This module contain the principal classes and functions to calculate relevant climatic indices
   
   
   You can use the ``rascal.indices.CIndex()`` class to retrieve relevant climatic indices based on:
   Data, C. (2009). Guidelines on analysis of extremes in a changing climate in support of informed decisions for adaptation. World Meteorological Organization.



   .. py:class:: rascal.analysis.CIndex(df)


      :param df: Time series containing the relevant variables for the index calculation.
      :type df: pd.DataFrame
   
      .. py:method:: fd()
   
         Count of days where TN (daily minimum temperature) < 0°C
         Let TNij be the daily minimum temperature on day i in period j. Count the number of days where TNij < 0°C.
         
         :return: idx
         :rtype: pd.DataFrame
      
      .. py:method:: su() 
   
         Count of days where TX (daily maximum temperature) > 25°C
         Let TXij be the daily maximum temperature on day i in period j. Count the number of days where TXij > 25°C.
         
         :return: idx
         :rtype: pd.DataFrame
   
      .. py:method:: id()
   
         Count of days where TX < 0°C
         Let TXij be the daily maximum temperature on day i in period j. Count the number of days where TXij < 0°C.
         
         :return: idx
         :rtype: pd.DataFrame
      
      .. py:method:: tr()
   
         Count of days where TN > 20°C
         Let TNij be the daily minimum temperature on day i in period j. Count the number of days where TNij > 20°C.
         
         :return: idx
         :rtype: pd.DataFrame
      
      .. py:method:: gsl()
   
         Annual count of days between first span of at least six days where TG (daily mean temperature) > 5°C and first span
         in second half of the year of at least six days where TG < 5°C.
         Let TGij be the daily mean temperature on day i in period j. Count the annual (1 Jan to 31 Dec in
         Northern Hemisphere, 1 July to 30 June in Southern Hemisphere) number of days between the first
         occurrence of at least six consecutive days where TGij > 5°C and the first occurrence after 1 July (1 Jan
         in Southern Hemisphere) of at least six consecutive days where TGij < 5°C.
         
         :return: idx
         :rtype: pd.DataFrame
      
      .. py:method:: txx() 
   
         Monthly maximum value of daily maximum temperature:
         Let TXik be the daily maximum temperature on day i in month k. The maximum daily maximum
         temperature is then TXx = max (TXik).
         
         :return: idx
         :rtype: pd.DataFrame
      
      .. py:method:: tnx()
   
         Monthly maximum value of daily minimum temperature:
         Let TNik be the daily minium temperature on day i in month k. The maximum daily minimum temperature
         is then TNx = max (TNik).
         
         :return: idx
         :rtype: pd.DataFrame
      
      .. py:method:: txn()
   
         Monthly minimum value of daily maximum temperature:
         Let TXik be the daily maximum temperature on day i in month k. The minimum daily maximum
         temperature is then TXn = min (TXik)
         
         :return: idx
         :rtype: pd.DataFrame
      
      .. py:method:: tnn()
   
         Monthly minimum value of daily minimum temperature:
         Let TNik be the daily minimum temperature on day i in month k. The minimum daily minimum
         temperature is then TNn = min (TNik)
         
         :return: idx
         :rtype: pd.DataFrame
      
      .. py:method:: tn10p()
   
         Count of days where TN < 10th percentile
         Let TNij be the daily minimum temperature on day i in period j and let TNin10 be the calendar day 10th
         percentile of daily minimum temperature calculated for a five-day window centred on each calendar day
         in the base period n (1961-1990). Count the number of days where TNij < TNin10.
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: tx10p()
 
         Count of days where TX < 10th percentile
         Let TXij be the daily maximum temperature on day i in period j and let TXin10 be the calendar day 10th
         percentile of daily maximum temperature calculated for a five-day window centred on each calendar day
         in the base period n (1961-1990). Count the number of days where TXij < TXin10
         
         :return: idx
         :rtype: pd.DataFrame
    
      .. py:method:: tn90p()
   
         Count of days where TN > 90th percentile
         Let TNij be the daily minimum temperature on day i in period j and let TNin90 be the calendar day 90th
         percentile of daily minimum temperature calculated for a five-day window centred on each calendar day
         in the base period n (1961-1990). Count the number of days where TNij > TNin90
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: tx90p()
   
         Count of days where TX > 90th percentile
         Let TXij be the daily maximum temperature on day i in period j and let TXin90 be the calendar day 90th
         percentile of daily maximum temperature calculated for a five-day window centred on each calendar day
         in the base period n (1961-1990). Count the number of days where TXij > TXin90.
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: wsdi()

         Count of days in a span of at least six days where TX > 90th percentile
         Let TXij be the daily maximum temperature on day i in period j and let TXin90 be the calendar day 90th
         percentile of daily maximum temperature calculated for a five-day window centred on each calendar day
         in the base period n (1961-1990). Count the number of days where, in intervals of at least six
         consecutive days TXij > TXin90.
         
         :return: idx
         :rtype: pd.DataFrame
    
      .. py:method:: csdi()
   
         Count of days in a span of at least six days where TN > 10th percentile
         Let TNij be the daily minimum temperature on day i in period j and let TNin10 be the calendar day 10th
         percentile of daily minimum temperature calculated for a five-day window centred on each calendar day
         in the base period n (1961-1990). Count the number of days where, in intervals of at least six
         consecutive days TNij < TNin10.
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: dtr()

         Mean difference between TX and TN (°C)
         Let TXij and TNij be the daily maximum and minium temperature on day i in period j. If I represents the
         total number of days in j then the mean diurnal temperature range in period j DTRj = sum (TXij - TNij) / I.
         
         :return: idx
         :rtype: pd.DataFrame
    
      .. py:method:: rx1day()
   
         Highest precipitation amount in one-day period
         Let RRij be the daily precipitation amount on day i in period j. The maximum one-day value for period j is
         RX1dayj = max (RRij).
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: rx5day()
   
         Highest precipitation amount in five-day period
         Let RRkj be the precipitation amount for the five-day interval k in period j, where k is defined by the last
         day. The maximum five-day values for period j are RX5dayj = max (RRkj)
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: sdii()

         Mean precipitation amount on a wet day
         Let RRij be the daily precipitation amount on wet day w (RR ≥ 1 mm) in period j. If W represents the
         number of wet days in j then the simple precipitation intensity index SDIIj = sum (RRwj) / W.
         
         :return: idx
         :rtype: pd.DataFrame
    
      .. py:method:: r10mm()
   
         Count of days where RR (daily precipitation amount) ≥ 10 mm
         Let RRij be the daily precipitation amount on day i in period j. Count the number of days where RRij ≥ 10 mm.
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: r20mm()
   
         Count of days where RR ≥ 20 mm
         Let RRij be the daily precipitation amount on day i in period j. Count the number of days where RRij ≥ 20 mm.
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: rnnmm(threshold)
      
         :param threshold: Precipitation threshold
         :type threshold: float
   
         Count of days where RR ≥ user-defined threshold in mm
         Let RRij be the daily precipitation amount on day i in period j. Count the number of days where RRij ≥ nn mm.
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: cdd()
   
         Maximum length of dry spell (RR < 1 mm)
         Let RRij be the daily precipitation amount on day i in period j. Count the largest number of consecutive
         days where RRij < 1 mm.
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: cwd()
   
         Maximum length of wet spell (RR ≥ 1 mm)
         Let RRij be the daily precipitation amount on day i in period j. Count the largest number of consecutive
         days where RRij ≥ 1 mm
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: r95ptot()
   
         Precipitation due to very wet days (> 95th percentile)
         Let RRwj be the daily precipitation amount on a wet day w (RR ≥ 1 mm) in period j and let RRwn95 be
         the 95th percentile of precipitation on wet days in the base period n (1961-1990). Then R95pTOTj = sum (RRwj),
         where RRwj > RRwn95.
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: r99ptot()
   
         Precipitation due to extremely wet days (> 99th percentile)
         Let RRwj be the daily precipitation amount on a wet day w (RR ≥ 1 mm) in period j and let RRwn99 be
         the 99th percentile of precipitation on wet days in the base period n (1961-1990). Then R99pTOTj = sum (RRwj),
         where RRwj > RRwn99
         
         :return: idx
         :rtype: pd.DataFrame

      .. py:method:: prcptot()
   
         Total precipitation in wet days (> 1 mm)
         Let RRwj be the daily precipitation amount on a wet day w (RR ≥ 1 mm) in period j. Then PRCPTOTj = sum (RRwj)
         
         :return: idx
         :rtype: pd.DataFrame


