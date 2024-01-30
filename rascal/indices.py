"""
Relevant climatic indices based on:
Data, C. (2009). Guidelines on analysis of extremes in a changing climate in support
of informed decisions for adaptation. World Meteorological Organization.
contact:alvaro@intermet.es
"""
import datetime
import itertools
import pandas as pd
from typing import Callable, Any


class CIndex:
    def __init__(self, df):
        self.variables = df

    def fd(self):
        variable = "TMIN"
        idx = contains_variable(df=self.variables, var=variable, foo=get_frost_days)
        idx = rename_var(idx, var=variable, new_var="FD")
        return idx

    def su(self):
        variable = "TMAX"
        idx = contains_variable(df=self.variables, var=variable, foo=get_summer_days)
        idx = rename_var(idx, var=variable, new_var="SU")
        return idx

    def id(self):
        variable = "TMAX"
        idx = contains_variable(df=self.variables, var=variable, foo=get_icing_days)
        idx = rename_var(idx, var=variable, new_var="ID")
        return idx

    def tr(self):
        variable = "TMIN"
        idx = contains_variable(df=self.variables, var=variable, foo=get_tropical_nigths)
        idx = rename_var(idx, var=variable, new_var="TR")
        return idx

    def gsl(self):
        variable = "TMEAN"
        idx = contains_variable(df=self.variables, var=variable, foo=get_growing_season_length)
        idx = rename_var(idx, var=variable, new_var="GSL")
        return idx

    def txx(self):
        variable = "TMAX"
        idx = contains_variable(df=self.variables, var=variable, foo=get_txx)
        idx = rename_var(idx, var=variable, new_var="TXx")
        return idx

    def tnx(self):
        variable = "TMIN"
        idx = contains_variable(df=self.variables, var=variable, foo=get_tnx)
        idx = rename_var(idx, var=variable, new_var="TNx")
        return idx

    def txn(self):
        variable = "TMAX"
        idx = contains_variable(df=self.variables, var=variable, foo=get_txn)
        idx = rename_var(idx, var=variable, new_var="TXn")
        return idx

    def tnn(self):
        variable = "TMIN"
        idx = contains_variable(df=self.variables, var=variable, foo=get_tnn)
        idx = rename_var(idx, var=variable, new_var="TNn")
        return idx

    def tn10p(self):
        variable = "TMIN"
        idx = contains_variable(df=self.variables, var=variable, foo=get_cold_nigths)
        idx = rename_var(idx, var=variable, new_var="TN10p")
        return idx

    def tx10p(self):
        variable = "TMAX"
        idx = contains_variable(df=self.variables, var=variable, foo=get_cold_day_times)
        idx = rename_var(idx, var=variable, new_var="TX10p")
        return idx

    def tn90p(self):
        variable = "TMIN"
        idx = contains_variable(df=self.variables, var=variable, foo=get_warm_nights)
        idx = rename_var(idx, var=variable, new_var="TN90p")
        return idx

    def tx90p(self):
        variable = "TMAX"
        idx = contains_variable(df=self.variables, var=variable, foo=get_warm_day_times)
        idx = rename_var(idx, var=variable, new_var="TX90p")
        return idx

    def wsdi(self):
        variable = "TMAX"
        idx = contains_variable(df=self.variables, var=variable, foo=get_warm_spell_duration_index)
        idx = rename_var(idx, var=variable, new_var="TR")
        return idx

    def csdi(self):
        variable = "TMIN"
        idx = contains_variable(df=self.variables, var=variable, foo=get_cold_spell_duration_index)
        idx = rename_var(idx, var=variable, new_var="CDSI")
        return idx

    def dtr(self):
        variable = "TAMP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_diurnal_temperature_range)
        idx = rename_var(idx, var=variable, new_var="DTR")
        return idx

    def rx1day(self):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_maximum_one_day_precipitation)
        idx = rename_var(idx, var=variable, new_var="RX1day")
        return idx

    def rx5day(self):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_maximum_five_day_precipitation)
        idx = rename_var(idx, var=variable, new_var="RX5day")
        return idx

    def sdii(self):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_simple_daily_intensity_index)
        idx = rename_var(idx, var=variable, new_var="SDII")
        return idx

    def r10mm(self):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_heavy_precipitation_days)
        idx = rename_var(idx, var=variable, new_var="R10mm")
        return idx

    def r20mm(self):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_very_heavy_precipitation_days)
        idx = rename_var(idx, var=variable, new_var="R20mm")
        return idx

    def rnnmm(self, threshold):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_rnnmm, threshold=threshold)
        idx = rename_var(idx, var=variable, new_var="Rnnmm")
        return idx

    def cdd(self):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_consecutive_dry_days)
        idx = rename_var(idx, var=variable, new_var="CDD")
        return idx

    def cwd(self):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_consecutive_wet_days)
        idx = rename_var(idx, var=variable, new_var="CWD")
        return idx

    def r95ptot(self):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_precipitation_due_to_very_wet_days)
        idx = rename_var(idx, var=variable, new_var="R95pTOT")
        return idx

    def r99ptot(self):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_precipitation_due_to_extremely_wet_days)
        idx = rename_var(idx, var=variable, new_var="R99pTOT")
        return idx

    def prcptot(self):
        variable = "PCP"
        idx = contains_variable(df=self.variables, var=variable, foo=get_total_precipitation_in_wet_days)
        idx = rename_var(idx, var=variable, new_var="PRCTOT")
        return idx


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


def contains_variable(df: pd.DataFrame, var: str, foo: Callable[..., Any], threshold=None):
    """
    Execute function to a df if the df DataFrame contains var in its columns
    """
    if any(var in c for c in df.columns):
        if "threshold" in foo.__code__.co_varnames:
            return foo(df, threshold)
        else:
            return foo(df)
    else:
        print("Error: " + foo.__name__ + " requires " + var)


def rename_var(df, var, new_var):
    new_cols = [c.replace(var, new_var) for c in df.columns]
    df.columns = new_cols
    return df


def get_frost_days(df):
    """
    Count of days where TN (daily minimum temperature) < 0°C
    Let TNij be the daily minimum temperature on day i in period j. Count the number of days where TNij < 0°C.
    """
    fd = get_days_above_threshold(df, threshold=0, inverse=True)
    return fd


def get_summer_days(df):
    """
    Count of days where TX (daily maximum temperature) > 25°C
    Let TXij be the daily maximum temperature on day i in period j. Count the number of days where TXij > 25°C.
    """
    su = get_days_above_threshold(df, threshold=25, inverse=False)
    return su


def get_icing_days(df):
    """
    Count of days where TX < 0°C
    Let TXij be the daily maximum temperature on day i in period j. Count the number of days where TXij < 0°C.
    """
    idays = get_days_above_threshold(df, threshold=0, inverse=True)
    return idays


def get_tropical_nigths(df):
    """
    Count of days where TN > 20°C
    Let TNij be the daily minimum temperature on day i in period j. Count the number of days where TNij >
    20°C.
    """
    tn = get_days_above_threshold(df, threshold=20, inverse=False)
    return tn


def get_growing_season_length(df):
    """
    Annual count of days between first span of at least six days where TG (daily mean temperature) > 5°C and first span
    in second half of the year of at least six days where TG < 5°C.
    Let TGij be the daily mean temperature on day i in period j. Count the annual (1 Jan to 31 Dec in
    Northern Hemisphere, 1 July to 30 June in Southern Hemisphere) number of days between the first
    occurrence of at least six consecutive days where TGij > 5°C and the first occurrence after 1 July (1 Jan
    in Southern Hemisphere) of at least six consecutive days where TGij < 5°C.
    """
    pass


def get_txx(df):
    """
    Monthly maximum value of daily maximum temperature:
    Let TXik be the daily maximum temperature on day i in month k. The maximum daily maximum
    temperature is then TXx = max (TXik).
    """
    txx = df.resample("1m").max()
    return txx


def get_tnx(df):
    """
    Monthly maximum value of daily minimum temperature:
    Let TNik be the daily minium temperature on day i in month k. The maximum daily minimum temperature
    is then TNx = max (TNik).
    """
    tnx = df.resample("1m").max()
    return tnx


def get_txn(df):
    """
    Monthly minimum value of daily maximum temperature:
    Let TXik be the daily maximum temperature on day i in month k. The minimum daily maximum
    temperature is then TXn = min (TXik)
    """
    txn = df.resample("1m").min()
    return txn


def get_tnn(df):
    """
    Monthly minimum value of daily minimum temperature:
    Let TNik be the daily minimum temperature on day i in month k. The minimum daily minimum
    temperature is then TNn = min (TNik)
    """
    tnn = df.resample("1m").max()
    return tnn


def get_cold_nigths(df):
    """
    Count of days where TN < 10th percentile
    Let TNij be the daily minimum temperature on day i in period j and let TNin10 be the calendar day 10th
    percentile of daily minimum temperature calculated for a five-day window centred on each calendar day
    in the base period n (1961-1990). Count the number of days where TNij < TNin10.
    """

    centered_min = df.rolling(window=5, center=True).min()
    tn10p = centered_min.groupby(centered_min.index.date).quantile(0.1).values[0]
    pass


def get_cold_day_times(df):
    """
    Count of days where TX < 10th percentile
    Let TXij be the daily maximum temperature on day i in period j and let TXin10 be the calendar day 10th
    percentile of daily maximum temperature calculated for a five-day window centred on each calendar day
    in the base period n (1961-1990). Count the number of days where TXij < TXin10
    """
    pass


def get_warm_nights(df):
    """
    Count of days where TN > 90th percentile
    Let TNij be the daily minimum temperature on day i in period j and let TNin90 be the calendar day 90th
    percentile of daily minimum temperature calculated for a five-day window centred on each calendar day
    in the base period n (1961-1990). Count the number of days where TNij > TNin90
    """
    pass


def get_warm_day_times(df):
    """
    Count of days where TX > 90th percentile
    Let TXij be the daily maximum temperature on day i in period j and let TXin90 be the calendar day 90th
    percentile of daily maximum temperature calculated for a five-day window centred on each calendar day
    in the base period n (1961-1990). Count the number of days where TXij > TXin90.
    """
    pass


def get_warm_spell_duration_index(df):
    """
    Count of days in a span of at least six days where TX > 90th percentile
    Let TXij be the daily maximum temperature on day i in period j and let TXin90 be the calendar day 90th
    percentile of daily maximum temperature calculated for a five-day window centred on each calendar day
    in the base period n (1961-1990). Count the number of days where, in intervals of at least six
    consecutive days TXij > TXin90.
    """
    pass


def get_cold_spell_duration_index(df):
    """
    Count of days in a span of at least six days where TN > 10th percentile
    Let TNij be the daily minimum temperature on day i in period j and let TNin10 be the calendar day 10th
    percentile of daily minimum temperature calculated for a five-day window centred on each calendar day
    in the base period n (1961-1990). Count the number of days where, in intervals of at least six
    consecutive days TNij < TNin10.
    """
    pass


def get_diurnal_temperature_range(df):
    """
    Mean difference between TX and TN (°C)
    Let TXij and TNij be the daily maximum and minium temperature on day i in period j. If I represents the
    total number of days in j then the mean diurnal temperature range in period j DTRj = sum (TXij - TNij) / I.
    """
    pass


def get_maximum_one_day_precipitation(df):
    """
    Highest precipitation amount in one-day period
    Let RRij be the daily precipitation amount on day i in period j. The maximum one-day value for period j is
    RX1dayj = max (RRij).
    """
    pass


def get_maximum_five_day_precipitation(df):
    """
    Highest precipitation amount in five-day period
    Let RRkj be the precipitation amount for the five-day interval k in period j, where k is defined by the last
    day. The maximum five-day values for period j are RX5dayj = max (RRkj)
    """
    pass


def get_simple_daily_intensity_index(df):
    """
    Mean precipitation amount on a wet day
    Let RRij be the daily precipitation amount on wet day w (RR ≥ 1 mm) in period j. If W represents the
    number of wet days in j then the simple precipitation intensity index SDIIj = sum (RRwj) / W.
    """
    pass


def get_heavy_precipitation_days(df):
    """
    Count of days where RR (daily precipitation amount) ≥ 10 mm
    Let RRij be the daily precipitation amount on day i in period j. Count the number of days where RRij ≥ 10 mm.
    """
    r10mm = get_days_above_threshold(df=df, threshold=9.9, inverse=False)
    return r10mm


def get_very_heavy_precipitation_days(df):
    """
    Count of days where RR ≥ 20 mm
    Let RRij be the daily precipitation amount on day i in period j. Count the number of days where RRij ≥ 20 mm.
    """
    r10mm = get_days_above_threshold(df=df, threshold=19.9, inverse=False)
    return r10mm


def get_rnnmm(df, threshold):
    """
    Count of days where RR ≥ user-defined threshold in mm
    Let RRij be the daily precipitation amount on day i in period j. Count the number of days where RRij ≥ nn mm.
    """
    rnnmm = get_days_above_threshold(df=df, threshold=threshold, inverse=False)
    return rnnmm


def get_consecutive_dry_days(df):
    """
    Maximum length of dry spell (RR < 1 mm)
    Let RRij be the daily precipitation amount on day i in period j. Count the largest number of consecutive
    days where RRij < 1 mm.
    """
    dry_days = get_days_above_threshold(df=df, threshold=1, inverse=True)
    cols = dry_days.columns
    dry_days = dry_days.squeeze()
    consecutive_days = dry_days * (dry_days.groupby((dry_days != dry_days.shift()).cumsum()).cumcount() + 1)
    consecutive_days = consecutive_days.to_frame()
    consecutive_days.columns = cols
    consecutive_days = consecutive_days.resample("1m").max()

    return consecutive_days


def get_consecutive_wet_days(df):
    """
    Maximum length of wet spell (RR ≥ 1 mm)
    Let RRij be the daily precipitation amount on day i in period j. Count the largest number of consecutive
    days where RRij ≥ 1 mm
    """
    wet_days = get_days_above_threshold(df=df, threshold=0.9, inverse=False)
    cols = wet_days.columns
    dry_days = wet_days.squeeze()
    consecutive_days = dry_days * (dry_days.groupby((dry_days != dry_days.shift()).cumsum()).cumcount() + 1)
    consecutive_days = consecutive_days.to_frame()
    consecutive_days.columns = cols
    consecutive_days = consecutive_days.resample("1m").max()
    return consecutive_days


def get_precipitation_due_to_very_wet_days(df):
    """
    Precipitation due to very wet days (> 95th percentile)
    Let RRwj be the daily precipitation amount on a wet day w (RR ≥ 1 mm) in period j and let RRwn95 be
    the 95th percentile of precipitation on wet days in the base period n (1961-1990). Then R95pTOTj = sum (RRwj),
    where RRwj > RRwn95.
    """
    wet_days = get_days_above_threshold(df, threshold=0.9, inverse=False)
    base_period = df[(df.index.year >= 1961) & (df.index.year <= 1990)]
    p95 = base_period.quantile(0.95).values[0]
    days_above_p95 = get_days_above_threshold(df, threshold=p95, inverse=False)
    days_above_p95 = days_above_p95 + wet_days
    days_above_p95 = days_above_p95.applymap(lambda x: 1 if x == 2 else 0)

    return days_above_p95


def get_precipitation_due_to_extremely_wet_days(df):
    """
    Precipitation due to extremely wet days (> 99th percentile)
    Let RRwj be the daily precipitation amount on a wet day w (RR ≥ 1 mm) in period j and let RRwn99 be
    the 99th percentile of precipitation on wet days in the base period n (1961-1990). Then R99pTOTj = sum (RRwj),
    where RRwj > RRwn99
    """
    wet_days = get_days_above_threshold(df, threshold=0.9, inverse=False)
    base_period = df[(df.index.year >= 1961) & (df.index.year <= 1990)]
    p99 = base_period.quantile(0.99).values[0]
    days_above_p99 = get_days_above_threshold(df, threshold=p99, inverse=False)
    days_above_p99 = days_above_p99 + wet_days
    days_above_p99 = days_above_p99.applymap(lambda x: 1 if x == 2 else 0)

    return days_above_p99


def get_total_precipitation_in_wet_days(df):
    """
    Total precipitation in wet days (> 1 mm)
    Let RRwj be the daily precipitation amount on a wet day w (RR ≥ 1 mm) in period j. Then PRCPTOTj = sum (RRwj)
    """
    pass
