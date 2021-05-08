
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
import matplotlib. pyplot as plt
import functions as fn
from data import params, params_reg, all_data
from visualizations import vs
import visualizations as vis

# pd.set_option('display.max_rows', None)                   # sin limite de renglones maximos
# pd.set_option('display.max_columns', None)                # sin limite de columnas maximas
# pd.set_option('display.width', None)                      # sin limite el ancho del display
# pd.set_option('display.expand_frame_repr', False)         # visualizar todas las columnas

# -------------------------------------------------------------------------------------------- Read data -- #
# train data (2011 to 2019) - Use any number of times
#data = ohlc_data['train']

# test data (2019) - Use max 10 times
# data = ohlc_data['train']

# validation data (2020) - Use 1 time at the very end
# data = ohlc_data['val']

# -- ------------------------------------------------------------------------- Exploratory Data Analysis -- #

# Description table
# data.describe()

# OHLC plot
#p_theme = {'color_1': '#ABABAB', 'color_2': '#ABABAB', 'color_3': '#ABABAB', 'font_color_1': '#ABABAB',
 #          'font_size_1': 12, 'font_size_2': 16}
#p_dims = {'width': 1450, 'height': 800}
#p_vlines = [data['timestamp'].head(1), data['timestamp'].tail(1)]
#p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

# cargar funcion importada desde GIST de visualizaciones
#ohlc = vs['g_ohlc'](p_ohlc=data, p_theme=p_theme, p_dims=p_dims, p_vlines=p_vlines, p_labels=p_labels)

# mostrar plot
# ohlc.show()

# -- ------------------------------------------------------------------------------- Feature Engineering -- #

# Feature engineering (autoregressive and hadamard functions)
data_t, data_y, data_x, features_names = fn.features(p_data=all_data, p_nmax=7)

# Data scaling
data_x = fn.data_trans(p_data=data_x, p_trans='Rubust')

var = fn.variables(data_y, data_x, N=4)
busqueda= fn.busqueda_en_train(var, n=4)
# Rearange of data for regression model

# Rearange of data for classification model
data_cla = {'x_data': data_x, 'y_data': data_y['co_d']}

# -- ---------------------------------------------------------------------------------- Feature analysis -- #

# matriz de correlacion
cor_mat = data_x.corr()

# optimization
#data_op = fn.optimization(p_data=data_cla, p_type='classification', p_params=params,
                          #p_model='logistic', p_iter=500)

# in sample results

# out of sample results

# -- with elastic net regularization

# optimization
#data_op = fn.optimization(p_data=data_cla, p_type='classification', p_params=params,
                          #p_model='logistic_en', p_iter=500)
