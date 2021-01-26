
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
from data import ohlc_data
from visualizations import vs

pd.set_option('display.max_rows', None)                   # sin limite de renglones maximos
pd.set_option('display.max_columns', None)                # sin limite de columnas maximas
pd.set_option('display.width', None)                      # sin limite el ancho del display
pd.set_option('display.expand_frame_repr', False)         # visualizar todas las columnas

# ------------------------------------------------------------------------------------------ Obtain data -- #

# m6e : Micro Eur/Usd Future: https://www.cmegroup.com/trading/fx/e-micros/e-micro-euro.html
# obtained with Quandl / Chris free data set. Gathers the data from webscraping cmegroup's future data
# data = m6e1.tail(160)
# data.tail()

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

# ingenieria de caracteristicas con variable endogena
features = fn.f_features(p_data=data, p_nmax=7)

# Conjunto de entrenamiento

# -- ---------------------------------------------------------------------------------- Feature analysis -- #

# matriz de correlacion
cor_mat = features.iloc[:, 1:].corr()

# -- ---------------------------------------------------------------------------------------- Models fit -- #

# Multple linear regression model
lm_model = fn.mult_regression(p_x=features.iloc[:, 3:], p_y=features.iloc[:, 1])
lm_model_reg = fn.mult_reg_l1l2(p_x=features.iloc[:, 3:], p_y=features.iloc[:, 1], p_alpha=1e-2, p_iter=1e6,
                                l1_ratio=.25)
# -- ------------------------------------------------------------------------------- Features simbolicos -- #

# semilla para siempre obtener el mismo resultado
np.random.seed(879)
# Generacion de muchos features formado con variables simbolica
symbolic, table = fn.symbolic_features(p_x=features.iloc[:, 3:], p_y=features.iloc[:, 1])
nuevos_features = pd.DataFrame(symbolic['fit'], index=features.index)

# -- ---------------------------------------------------------------------------------------- Models fit -- #

# Multple linear regression model
selected, af, rf=fn.optimizacion(nuevos_features, features)
#.02 alpha 1 ratio

print('Modelo Lineal 1: rss: ', lm_model['rss'])
print('Modelo Lineal 1: score: ', lm_model['score'])
# RSS of the model with all the variables
print('Modelo Lineal simbolico y con regularización: rss: ', selected['elasticnet']['rss'])
print('Modelo Lineal simbolico y con regularización: score: ', selected['elasticnet']['score'])
