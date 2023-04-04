import datetime
import pandas as pd
import numpy as np
from scipy.stats import linregress

AGENT_IDS = [661, 1642, 2335, 2361, 2818, 3039, 3456, 3538, 4031, 4373, 4767, 5746, 6139, 7536, 7719, 7800, 7901, 7951, 8156, 8386, 8565, 9019, 9160, 9922, 9278]
NON_AC = ['car1', 'clotheswasher1', 'dishwasher1', 'dry1', 'waterheater1', 'non-shiftable']
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df_temperature = pd.read_csv('data/outdoor_temperatures_noaa.csv',
                             delim_whitespace=True,
                             parse_dates=[['LST_DATE', 'LST_TIME']],
                             index_col=['LST_DATE_LST_TIME'],
                             usecols=['LST_DATE', 'LST_TIME', 'T_HR_AVG'])
print(df_temperature.describe)

# df_load = pd.read_csv('pecan_street_data/15minute_data_austin_processed_08_04.csv',
df_load = pd.read_csv('data/15minute_data_austin_fixed_consumption.csv',
                      parse_dates=['time'],
                      index_col=['time'])
print(df_load.describe)

start, end, steps = 182, 365, 96
baselines = np.zeros((len(AGENT_IDS), len(range(start, end)), steps))
for id, agent in enumerate(AGENT_IDS):
    df_filter = df_load.loc[df_load['dataid'] == agent]
    df_load_resampled = df_filter.resample('H').max()
    start_date = datetime.datetime(2018, 1, 1)
    temperatures = []
    loads = []

    for day in range(1, 364):
        for step in range(24):
            offset = day * 24 + step
            time_delta = pd.to_timedelta(offset, 'h')
            current_time = start_date + time_delta
            temperature = df_temperature.loc[current_time]['T_HR_AVG']
            if temperature == -9999:
                df_temperature.loc[current_time]['T_HR_AVG'] = df_temperature.loc[current_time - pd.to_timedelta(1, 'h')]['T_HR_AVG']
            load = df_load_resampled.loc[current_time]['air']
            if load > 0 and temperature != -9999:
                temperatures.append(temperature)
                loads.append(load)

    if temperatures and loads:
        slope, intercept, _, _, _ = linregress(temperatures, loads)
    else:
        slope, intercept = 0, 0

    for day in range(start, end):
        for step in range(steps):

            # Select last 10 same weekdays at the same moment
            time_delta = datetime.timedelta(minutes=step*15)
            start_date = datetime.datetime.strptime('{} {}'.format(day, 2018), '%j %Y')
            time = start_date + time_delta
            time_h = start_date + datetime.timedelta(hours=int(step*0.25))
            similar = [time - datetime.timedelta(days=7 * i) for i in range(1, 11)]
            df_load_similar = df_filter.loc[similar]
            similar_rounded = [ix.round('1h') for ix in df_load_similar.index]

            # Filter the 5 days with the largest demand
            # TODO exclude holidays
            df_load_similar = df_load_similar.nlargest(5, 'total')
            baseline_total = df_load_similar['total'].mean()

            # Take average of those 5 moments as the baseline
            df_temperature_similar = df_temperature.loc[similar_rounded]
            avg_temp = df_temperature_similar['T_HR_AVG'].mean()
            current_temp = df_temperature.loc[time_h]['T_HR_AVG']
            temp_diff = current_temp - avg_temp

            baseline_ac = slope * temp_diff
            # baseline = baseline_ac + baseline_total # with temperature correction
            baseline = baseline_total               # without temperature correction
            baselines[id][day-start][step] = baseline

            print(agent, day, step, baseline)

# np.save('pecan_street_data/baselines_regr_temp_correction.npy', baselines)
np.save('data/baselines_regr_temp_correction_new.npy', baselines)
# np.save('pecan_street_data/baselines_regr.npy', baselines)
