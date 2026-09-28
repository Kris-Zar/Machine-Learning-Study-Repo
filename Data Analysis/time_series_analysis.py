
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv('stock_data.csv',parse_dates=True, index_col='Date')
data.head()

data.drop(columns='Unnamed: 0', inplace =True)
data.head()

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.lineplot(data=data    , x='Date', y='High', label='High Price', color='blue')

plt.xlabel('Date')
plt.ylabel('High')
plt.title('Share Highest Price Over Time')

plt.show()

data_resampled = data.resample('ME').mean(numeric_only=True)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.lineplot(data=data_resampled    , x='Date', y='High', label='Average Prize Monthly', color='green')

plt.xlabel('Date')
plt.ylabel('High')
plt.title('Share Average  Price Over Time')

plt.show()

if 'Date' not in data.columns:
  print("Already in the index or does not exist in the dataframe")
else:
  data.set_index('Date')

plt.figure(figsize=(12,6))
plot_acf(data['High'],lags = 40)
plt.xlabel('Lag')
plt.ylabel('AutoCorrelation')
plt.title('Autocorrelation Function')
plt.show()

result = adfuller(data['High'])
print("ADF Statistics",result[0])
print("p-Value",result[1])
print("Critical Value",result[2])

from matplotlib.lines import lineStyles
data['High_diff'] = data['High'].diff()

plt.figure(figsize=(12,6))
plt.plot(data['High'], label='Original',color='blue')
plt.plot(data['High_diff'], label='Differenced',linestyle='-',color='green') # Changed to High_diff
plt.legend()
plt.title('Original VS Differenced')
plt.show()

window_size = 120
data['high_smoothed'] = data['High'].rolling(window=window_size).mean()

plt.figure(figsize=(12, 6))

plt.plot(data['High'], label='Original High', color='blue')
plt.plot(data['high_smoothed'], label=f'Moving Average (Window={window_size})', linestyle='--', color='orange')

plt.xlabel('Date')
plt.ylabel('High')
plt.title('Original vs Moving Average')
plt.legend()
plt.show()

df_combined = pd.concat([data['High'], data['High_diff']], axis=1)

print(df_combined.head())
data.dropna(subset=['High_diff'], inplace=True)
data['High_diff'].head()

result = adfuller(data['High_diff'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[3])