
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
