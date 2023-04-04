import datetime
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This constant limits the number of rows read in from the big CSV file.
# Set to None if you want to read the whole thing
LIMIT = None
pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 100)
devices_excl_other = ['car1', 'air1', 'clotheswasher1', 'dishwasher1', 'drye1', 'dryg1', 'waterheater1']
# devices = ['car1', 'air1', 'clotheswasher1', 'dishwasher1', 'drye1', 'dryg1', 'waterheater1', 'non-shiftable', 'solar']
devices = ['car1', 'air1', 'clotheswasher1', 'dishwasher1', 'drye1', 'dryg1', 'waterheater1']
include = ['dataid', 'grid', 'car1', 'air1', 'clotheswasher1', 'dishwasher1', 'drye1', 'dryg1', 'waterheater1', 'solar', 'non-shiftable', 'total']
drop = ['dataid', 'leg1v', 'leg2v', 'grid', 'solar']

# read the 15 minute data file for Austin
df = pd.read_csv('data/15minute_data_austin_fixed_consumption.csv',
                       engine='python', encoding="ISO-8859-1", parse_dates=['time'], index_col=['time'], nrows=LIMIT)
print(df.describe)

# Filter Household
# incl_dataid = 661
# df = df.loc[df['dataid'] == incl_dataid]
# excl_dataid = 9019
# df = df.loc[df['dataid'] != excl_dataid]
print(df.max())

# Filter dates
for day in range(182, 200):
    # day = random.randint(1, 365)
    start_date = datetime.datetime.strptime('{} {} {}'.format(day, 2018, 0), '%j %Y %H')
    end_date = datetime.datetime.strptime('{} {} {}'.format(day + 1, 2018, 12), '%j %Y %H')
    # start_date = datetime.datetime(2018, 10, 16)
    # end_date = datetime.datetime(2018, 10, 17)
    df_filter = df.loc[(df.index >= start_date) & (df.index < end_date)]

    # group the data by time or date and take the mean of those
    # df.index = df.reset_index()['time'].apply(lambda x: x - pd.Timestamp(x.date()))
    # df = df.groupby(pd.Grouper(freq='M')).max()
    # y = df.groupby(['dataid']).max()
    # print(y.describe)

    # convert from kW to kWh
    # df['total_kwh'] = df['total'].apply(lambda x: x)

    # Plot boxplot for device
    # threshold = 0.01
    # x = df.apply(lambda l: np.where(l < threshold, np.nan, l))
    # print(x[devices].describe())
    # fig = plt.figure(figsize=(8,6))
    # ax = fig.gca()
    # x.boxplot(column=devices, ax=ax)

    # create the plot
    # df = df.drop(drop, 'columns')
    # df = df.dropna('columns', thresh=1)
    # Use seaborn style defaults and set the default figure size
    sns.set(rc={'figure.figsize': (11, 4)})
    # solar_plot = df_filter[devices + ['total', 'total_incl_solar']].plot(linewidth=0.5, marker='.')
    solar_plot = df_filter[devices + ['total']].plot(linewidth=0.5, marker='.')
    # solar_plot = df_filter[devices].plot(linewidth=0.5, marker='.')
    solar_plot.set_xlabel('Date')
    solar_plot.set_ylabel('Grid Usage kW')

    # Plot hist
    # plt.hist(x['clotheswasher1'].to_numpy(), bins=50)

    # display the plot
    plt.title('Major consumers')
    plt.ylabel('Power consumnption (KW)')
    plt.show()

print('done')
