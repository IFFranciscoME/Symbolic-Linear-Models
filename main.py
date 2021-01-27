
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Multivariate Linear Regression with Symbolic Regressors and L1L2 Regularization            -- #
# -- for future prices prediction, the case for UsdMxn                                                   -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: IFFranciscoME                                                                               -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/A3_Regresion_Simbolica                                 -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import numpy as np
import functions as fn
from data import ohlc_data, params
from visualizations import vs

pd.set_option('display.max_rows', None)                   # sin limite de renglones maximos
pd.set_option('display.max_columns', None)                # sin limite de columnas maximas
pd.set_option('display.width', None)                      # sin limite el ancho del display
pd.set_option('display.expand_frame_repr', False)         # visualizar todas las columnas

# -------------------------------------------------------------------------------------------- Read data -- #

# train data (2011 to 2018) - Use any number of times
data = ohlc_data['train']

# test data (2019) - Use max 10 times
# data = ohlc_data['train']

# validation data (2020) - Use 1 time at the very end
# data = ohlc_data['val']

# -- ------------------------------------------------------------------------- Exploratory Data Analysis -- #

# Description table
# data.describe()

# OHLC plot
p_theme = {'color_1': '#ABABAB', 'color_2': '#ABABAB', 'color_3': '#ABABAB', 'font_color_1': '#ABABAB',
           'font_size_1': 12, 'font_size_2': 16}
p_dims = {'width': 1450, 'height': 800}
p_vlines = [data['timestamp'].head(1), data['timestamp'].tail(1)]
p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

# cargar funcion importada desde GIST de visualizaciones
ohlc = vs['g_ohlc'](p_ohlc=data, p_theme=p_theme, p_dims=p_dims, p_vlines=p_vlines, p_labels=p_labels)

# mostrar plot
# ohlc.show()

# -- ------------------------------------------------------------------------------- Feature Engineering -- #

# Data scaling
data = fn.data_trans(p_data=data, p_trans='Robust')

# Feature engineering (autoregressive and hadamard functions)
data = fn.features(p_data=data, p_nmax=7)

# rearange of data for regression model
data_reg = {'x_data': data[data.columns[3:-1]], 'y_data': data[data.columns[1]]}

# rearange of data for classification model
data_cla = {'x_data': data[data.columns[3:-1]], 'y_data': data[data.columns[2]]}
# -- ---------------------------------------------------------------------------------- Feature analysis -- #

# matriz de correlacion
cor_mat = data.iloc[:, 1:].corr()

# -- ---------------------------------------------------------------------------------------- Models fit -- #
# -- With autoregressive and hadamard features

# -- Ordinary Least Squares Model (Regression)

# without regularization
data_op = fn.optimization(p_data=data_reg, p_type='regression', p_params=params,
                          p_model='ols', p_iter=1000)

# with elastic net regularization
data_op = fn.optimization(p_data=data_reg, p_type='regression', p_params=params,
                          p_model='ols_en', p_iter=1000)

# -- Logistic Regression Model (Classification)

# without regularization
data_op = fn.optimization(p_data=data_cla, p_type='classification', p_params=params,
                          p_model='logistic', p_iter=500)

# with elastic net regularization
data_op = fn.optimization(p_data=data_cla, p_type='classification', p_params=params,
                          p_model='logistic_en', p_iter=500)

# -- ------------------------------------------------------------------------------- Features simbolicos -- #

# semilla para siempre obtener el mismo resultado
# np.random.seed(123)

# Generacion de muchos features formado con variables simbolica
# symbolic, table = fn.symbolic_features(p_x=features.iloc[:, 3:], p_y=features.iloc[:, 1])
# nuevos_features = pd.DataFrame(symbolic['fit'], index=features.index)

# -- ---------------------------------------------------------------------------------------- Models fit -- #
# -- With autoregressive and hadamard features and symbolic features

# -- Ordinary Least Squares Model (Regression)

# without regularization

# with elastic net regularization

# -- Logistic Regression Model (Classification)

# without regularization

# with elastic net regularization
