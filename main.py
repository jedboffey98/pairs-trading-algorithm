# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import math

import alpaca_trade_api as alpaca
import numpy
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import seaborn
from pandas import Series
import statsmodels.api as stat
from statsmodels.tsa.stattools import adfuller, coint
from datetime import datetime, timezone, timedelta
import rfc3339 as rfc
from bs4 import BeautifulSoup

api = alpaca.REST('PKDE3BTUG52I1WUW98H7', '4uGnwRbXhhTawwuUnZ7DdK8VjeWYSHkZOtS6fcFM',
                  'https://paper-api.alpaca.markets')

AGE_THRESHOLD = 10
LONG_ENTRY_THRESHOLD = -1
LONG_STOPLOSS_THRESHOLD = -2
LONG_EXIT_THRESHOLD = 0
SHORT_ENTRY_THRESHOLD = 1
SHORT_STOPLOSS_THRESHOLD = 2
SHORT_EXIT_THRESHOLD = 0
P_VALUE_THRESHOLD = 0.01


def zscore(series):
    return (series - series.mean()) / np.std(series)


def find_cointegrated_pairs(data):
    data = data.head(500)
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            print(f'Regressing on {keys[i], keys[j]}')
            S1 = data[keys[i]].dropna()
            S2 = data[keys[j]].dropna()
            if len(S1) == len(S2) and len(S1) != 0:
                result = coint(S1, S2)
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                if pvalue < P_VALUE_THRESHOLD:
                    pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


def trade(S1, S2, window1, window2):
    ratios = S1 / S2
    ma1 = ratios.rolling(window=window1,
                         center=False).mean()
    ma2 = ratios.rolling(window=window2,
                         center=False).mean()
    std = ratios.rolling(window=window2,
                         center=False).std()

    s1_vol20 = S1.rolling(window=20, center=False).std()
    s2_vol20 = S2.rolling(window=20, center=False).std()

    rolling_zscore = (ma1 - ma2) / std

    money = [100]
    age = 0
    aged = False
    existing_pos = 0
    print(len(ratios))
    for i in range(len(ratios)):
        print(rolling_zscore[i])
        if existing_pos != 0:
            ret_1 = (S1[i] - S1[i - 1]) / S1[i - 1]
            ret_2 = (S2[i] - S2[i - 1]) / S2[i - 1]

            s1_unadj_wt = max(s1_vol20[i], s2_vol20[i]) / s1_vol20[i]
            s2_unadj_wt = max(s1_vol20[i], s2_vol20[i]) / s2_vol20[i]

            sum_wt = s1_unadj_wt + s2_unadj_wt
            s1_wt = s1_unadj_wt / sum_wt
            s2_wt = s2_unadj_wt / sum_wt

            f_ret = ret_1 * s1_wt - ret_2 * s2_wt
            fret1_z5 = 1 + (existing_pos * f_ret)
            if math.isnan(fret1_z5):
                print(f'{ret_1} {ret_2}')
                fret1_z5 = 1
            new_money = money[-1] * fret1_z5
            money.append(new_money)

            if abs(rolling_zscore[i]) > 2:
                existing_pos = 0
                age = 0
        else:
            money.append(money[-1])

        if LONG_ENTRY_THRESHOLD > rolling_zscore[i] > LONG_STOPLOSS_THRESHOLD:
            existing_pos = 1
            age += 1
        elif SHORT_STOPLOSS_THRESHOLD > rolling_zscore[i] > SHORT_ENTRY_THRESHOLD:
            existing_pos = -1
            age += 1
        # Clear positions if the z-score between -.5 and .5
        elif (existing_pos == -1 and rolling_zscore[i] < SHORT_EXIT_THRESHOLD) or (
                existing_pos == 1 and rolling_zscore[i] > LONG_EXIT_THRESHOLD):
            existing_pos = 0
            age = 0
            aged = False

        if age > AGE_THRESHOLD:
            existing_pos = 0
            age = 0
            aged = True

    print(f'{S1.name}:{S2.name} made {money[-1]}')
    return money


def bucket_by_comparable():
    URL = 'https://en.wikipedia.org/wiki/S%26P_100'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    tbl = soup.find("table", {"id": "constituents"})
    df = pd.read_html(str(tbl))[0]

    concat_df = pd.DataFrame()
    count = 1
    total = len(df['Symbol'])
    NY = 'America/New_York'
    end = pd.Timestamp('2020-01-01', tz=NY).isoformat()
    dfs = []
    for symbol in df['Symbol']:
        print(f'Getting data {symbol} ({count}/{total})')
        URL = f'https://www.marketwatch.com/investing/stock/{symbol}'
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        competitors = soup.find("div", {"class": "element element--table Competitors"})
        if competitors == None:
            continue
        urls = competitors.find_all(href=True)
        comps = [symbol]
        for link in urls:
            if 'https://www.marketwatch.com/investing/stock/' in str(link):
                split1 = str(link).split('https://www.marketwatch.com/investing/stock/')
                split2 = split1[1].split('?')
                comps.append(split2[0].upper())
                print(split2[0])

        df = api.get_barset(comps, 'day', limit=1000, end=end).df
        adj_df = pd.DataFrame()
        print(df)
        for comp in comps:
            print(comp)
            adj_df = pd.concat([adj_df, df[comp.upper()]['close'].rename(comp.upper())], axis=1)
            if comp.upper() not in concat_df:
                concat_df = pd.concat([concat_df, df[comp.upper()]['close'].rename(comp.upper())], axis=1)
        count += 1
        print(concat_df)

        dfs.append(adj_df)
    concat_df.to_csv('bucket_comp_data.csv')
    return dfs, concat_df


def all_spx():
    fetch = input("Pull data again? y/n: ")
    if fetch == 'y':
        URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        tbl = soup.find("table", {"id": "constituents"})
        spx_df = pd.read_html(str(tbl))[0]
        spx_tickers = spx_df['Symbol']
        print(spx_tickers)

        NY = 'America/New_York'
        end = pd.Timestamp('2020-01-01', tz=NY).isoformat()
        return_df = pd.DataFrame()
        for ticker in spx_tickers.values:
            print(ticker)
            ticker_df = api.get_barset(ticker, 'day', limit=1000, end=end).df
            return_df = pd.concat([return_df, ticker_df[ticker.upper()]['close'].rename(ticker.upper())], axis=1)

        return_df.to_csv('comp_data.csv')
    else:
        return_df = pandas.read_csv('comp_data.csv', index_col=0)

    return return_df, return_df


if __name__ == '__main__':
    method = input("All S&P500 (1) or bucket by comparable (2)? ")
    if method == "2":
        dfs, concat_df = bucket_by_comparable()
    elif method == "1":
        dfs, concat_df = all_spx()
        dfs = [dfs]

    print(concat_df)
    calculate_pairs = input("Calculate pairs (will re-calculate pairs & trades based on different potentially "
                            "Marketwatch comparisons..)? ")
    all_pairs = []
    if calculate_pairs == 'y':
        for frame in dfs:
            scores, pvalues, pairs = find_cointegrated_pairs(frame)

            print(pairs)
            for pair in pairs:
                all_pairs.append(pair)
        pairs_df = pd.DataFrame(all_pairs)
        pairs_df.to_csv(f'pairs_method_{method}.csv')
        total_money = None
        count = 0
        for pair in pairs_df.values:
            print(pair)
            left = pair[0]
            right = pair[1]
            money = trade(concat_df.tail(500)[left], concat_df.tail(500)[right], 60, 5)
            if total_money is None:
                total_money = np.array(money)
                count += 1
            else:
                count += 1
                total_money = np.add(total_money, money)

        print(total_money)
        pairs_df.to_csv(f'pairs_method_{method}.csv')
        numpy.savetxt(f'final_{method}.csv', total_money, delimiter=',')
        print("written")
    else:
        if method == '1':
            pairs_df = pandas.read_csv(f'pairs_method_{method}.csv', index_col=0, header=0)
            for pair in pairs_df.values:
                print(pair)
                left = pair[0]
                right = pair[1]
                money = trade(concat_df.tail(500)[left], concat_df.tail(500)[right], 60, 5)
                total_money = None
                count = 0
                if total_money is None:
                    total_money = np.array(money)
                    count += 1
                else:
                    count += 1
                    total_money = np.add(total_money, money)

            pairs_df.to_csv(f'pairs_method_{method}.csv')
            numpy.savetxt(f'final_{method}.csv', total_money, delimiter=',')
            print("written")
        else:
            total_money_df = pandas.read_csv(f'final_{method}.csv', header=None)
            total_money = total_money_df.values
            pairs_df = pandas.read_csv(f'pairs_method_{method}.csv')

    dates = pd.to_datetime(concat_df.tail(501).index)
    count = total_money[0] / 100
    plt.plot(dates, total_money / count)
    plt.setp(plt.gca().xaxis.get_majorticklabels(),
             'rotation', 45)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.ylabel('Percentage Equity')
    plt.show()
