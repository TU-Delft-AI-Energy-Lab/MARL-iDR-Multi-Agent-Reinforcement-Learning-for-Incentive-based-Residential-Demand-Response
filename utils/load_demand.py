import datetime
import pandas as pd
import numpy as np


def load_requests():
    df = pd.read_csv('data/15minute_data_austin_fixed_consumption.csv', parse_dates=['time'], index_col=['time'])
    return df


def load_day(df, day, max_steps):
    minutes = max_steps * 15
    time_delta = pd.to_timedelta(minutes, 'm')
    start_date = datetime.datetime.strptime('{} {}'.format(day, 2018), '%j %Y')
    end_date = start_date + time_delta
    df = df.loc[(df.index >= start_date) & (df.index < end_date)]
    return df


def get_device_demands(df, agent_ids, day, timestep):
    minutes = timestep * 15
    time_delta = pd.to_timedelta(minutes, 'm')
    start_date = datetime.datetime.strptime('{} {}'.format(day, 2018), '%j %Y')
    time = start_date + time_delta
    df = df.loc[(df['dataid'].isin(agent_ids)) & (df.index == time)]
    return df


def get_peak_demand(df):
    df = df.groupby(pd.Grouper(freq='15Min')).sum()
    return df['total'].max()


def load_baselines():
    return np.load('data/baselines_regr_temp_correction.npy')
