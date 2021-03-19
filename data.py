
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Multivariate Linear Regression with Symbolic Regressors and L1L2 Regularization            -- #
# -- for future prices prediction, the case for UsdMxn                                                   -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: IFFranciscoME                                                                               -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/A3_Regresion_Simbolica                                 -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import numpy as np
from os import listdir, path
from os.path import isfile, join
import quandl as quandl


# authentication for quandl
quandl.ApiConfig.api_key = "4WLCk2ZUAMAPPyowvAYr"

# -- ------------------------------------------------------------------- Obtain, clean and reformat Data -- #
# -- Historical prices of E-Micro Eur/Usd Futures M6E1 (Front Month)

m6e1 = quandl.get('CHRIS/CME_m6e1', start_date='2018-08-31', end_date='2020-08-31')
m6e1 = m6e1.rename(columns={'Previous Day Open Interest': 'OpenInterest', 'Last': 'Close'})
m6e1 = m6e1[['Open', 'High', 'Low', 'Close', 'OpenInterest', 'Volume']]
m6e1.columns = ['open', 'high', 'low', 'close', 'openinterest', 'volume']
m6e1.reset_index(inplace=True, drop=False)
m6e1 = m6e1.rename(columns={'Date': 'timestamp'}, inplace=False)

# ---------------------------------------------------------------------------- Historical Prices Reading -- #

# the price in the file is expressed as the USD to purchase one MXN
# if is needed to convert to the inverse, the MXN to purchase one USD, uncomment the following line
mode = 'MXN_USD'

# path in order to read files
main_path = 'files/daily/'
abspath_f = path.abspath(main_path)
files_f = sorted([f for f in listdir(abspath_f) if isfile(join(abspath_f, f))])
price_data = {}

# iterative data reading
for file_f in files_f:
    data_f = pd.read_csv(main_path + file_f)
    data_f['timestamp'] = pd.to_datetime(data_f['timestamp'])

    # swap since original is wrong
    high = data_f['high'].copy()
    low = data_f['low'].copy()
    data_f['high'] = low
    data_f['low'] = high

    if mode == 'MXN_USD':
        data_f['open'] = round(1/data_f['open'], 5)
        data_f['high'] = round(1/data_f['high'], 5)
        data_f['low'] = round(1/data_f['low'], 5)
        data_f['close'] = round(1/data_f['close'], 5)

    data_f = data_f[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    years_f = set([str(datadate.year) for datadate in list(data_f['timestamp'])])
    for year_f in years_f:
        price_data['MP_D_' + year_f] = data_f

# Train data from 2011 to 2018
all_data = pd.concat([price_data[list(price_data.keys())[0]], price_data[list(price_data.keys())[1]],
                      price_data[list(price_data.keys())[2]], price_data[list(price_data.keys())[3]],
                      price_data[list(price_data.keys())[4]], price_data[list(price_data.keys())[5]],
                      price_data[list(price_data.keys())[6]], price_data[list(price_data.keys())[7]],
                      price_data[list(price_data.keys())[8]], price_data[list(price_data.keys())[9]]])


# historical data 
#ohlc_data = {'train1': train_data1, 'train2': train_data2 , 'train3': train_data3, 'test': test_data}

# --------------------------------------------------------------------- Parameters for Symbolic Features -- #
# --------------------------------------------------------------------- -------------------------------- -- #

symbolic_params = {'functions': ["sub", "add", 'inv', 'mul', 'div', 'abs', 'log','max','min'],
                   'population': 2000, 'tournament': 20, 'hof': 20, 'generations': 5, 'n_features':10,
                   'init_depth': (4,12), 'init_method': 'half and half', 'parsimony': 0, 'constants': None,
                   'metric': 'pearson', 'metric_goal': 0.7,
                   'prob_cross': 0.4, 'prob_mutation_subtree': 0.3,
                   'prob_mutation_hoist': 0.1, 'prob_mutation_point': 0.2,
                   'verbose': True, 'parallelization': True, 'warm_start': True}

# ------------------------------------------------------------------------ Hyperparameters for the Model -- #
# ------------------------------------------------------------------------ ----------------------------- -- #

# 100 different values for each parameter
params = {'ratio': np.arange(0, 1, 0.01), 'c': np.arange(0, 2, 0.02)}
params_reg = {'ratio': np.arange(0, 1, 0.01), 'c': np.arange(.2, 1, 0.2)}
