
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
from data import ohlc_data, params
from visualizations import vs
import visualizations as vis

# pd.set_option('display.max_rows', None)                   # sin limite de renglones maximos
# pd.set_option('display.max_columns', None)                # sin limite de columnas maximas
# pd.set_option('display.width', None)                      # sin limite el ancho del display
# pd.set_option('display.expand_frame_repr', False)         # visualizar todas las columnas

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

# Feature engineering (autoregressive and hadamard functions)
data, features_names = fn.features(p_data=data, p_nmax=7)

# Datetime
data_t = data[data.columns[0]]
# Target variables (co: regression, co_d: classification)
data_y = data[list(data.columns[1:3])]
# Autoregressive and hadamard features
data_x = data[list(data.columns[3:])]

# Data scaling
data_x = fn.data_trans(p_data=data_x, p_trans='Robust')

# Rearange of data for regression model
data_reg = {'x_data': data_x, 'y_data': data_y['co']}

# Rearange of data for classification model
data_cla = {'x_data': data_x, 'y_data': data_y['co_d']}

# -- ---------------------------------------------------------------------------------- Feature analysis -- #

# matriz de correlacion
cor_mat = data_x.corr()

# -- ---------------------------------------------------------------------------------------- Models fit -- #
# -- With autoregressive and hadamard features

# -- ---------------------------------------------------- Regression Model: Ordinary Least Squares Model -- # 

# -- without regularization

# in sample results
reg_1 = fn.ols_reg(p_data=data_reg, p_params=params, p_model="ols", p_iter=1000)
residuales_1 = reg_1['results']['y_data']-reg_1['results']['y_data_p']
hetero_1 = fn.check_hetero(residuales_1)
vis.residual(residuales=residuales_1)
# out of sample results

# -- with elasticnet regularization

# optimization
data_op = fn.optimization(p_data=data_reg, p_type='regression', p_params=params,
                          p_model='ols_en', p_iter=100000)
# in sample results
new_params = {"ratio": data_op["population"][0][0], "c": data_op["population"][0][1]}
reg_2 = fn.ols_reg(p_data=data_reg, p_params=new_params, p_model="ols_en", p_iter=100000)
residuales_2 = reg_2['results']['y_data']-reg_2['results']['y_data_p']
hetero_2 = fn.check_hetero(residuales_2)
vis.residual(residuales=residuales_2)

#.850748
#1400437493.505029
# out of sample results

# -- --------------------------------------------------- Classification Model: Logistic Regression Model -- # 

# -- without regularization

# optimization
#data_op = fn.optimization(p_data=data_cla, p_type='classification', p_params=params,
                          #p_model='logistic', p_iter=500)

# in sample results

# out of sample results

# -- with elastic net regularization

# optimization
#data_op = fn.optimization(p_data=data_cla, p_type='classification', p_params=params,
                          #p_model='logistic_en', p_iter=500)

# in sample results

# out of sample results

# -- --------------------------------------------------------------------------------- Symbolic Features -- #

# semilla para siempre obtener el mismo resultado
np.random.seed(546)

# Symbolic features generation
symbolic, table = fn.symbolic_features(p_x=data_reg['x_data'], p_y=data_reg['y_data'])
x_simbolic = pd.DataFrame(symbolic['fit'], index=data_reg['x_data'].index)
data_reg_sim = {'x_data': x_simbolic, 'y_data': data_y['co']}
#corre = vis.correlation(pd.concat([x_simbolic.iloc[:, -15:], data_y['co']], axis=1))
#plt.show()
# -- ---------------------------------------------------------------------------------------- Models fit -- #
# -- With autoregressive and hadamard features and symbolic features

# -- Ordinary Least Squares Model (Regression)

# without regularization
reg_5 = fn.ols_reg(p_data=data_reg_sim, p_params=params, p_model="ols", p_iter=10000)
residuales_5 = reg_5['results']['y_data']-reg_5['results']['y_data_p']
hetero_5 = fn.check_hetero(residuales_5)
vis.residual(residuales=residuales_5)

# with elastic net regularization

data_op = fn.optimization(p_data=data_reg_sim, p_type='regression', p_params=params,
                          p_model='ols_en', p_iter=100000)
# in sample results
new_params = {"ratio": data_op["population"][0][0], "c": data_op["population"][0][1]}
reg_6 = fn.ols_reg(p_data=data_reg_sim, p_params=new_params, p_model="ols_en", p_iter=10000)
residuales_6 = reg_6['results']['y_data']-reg_6['results']['y_data_p']
hetero_6 = fn.check_hetero(residuales_6)
vis.residual(residuales=residuales_6)

# -- Logistic Regression Model (Classification)

# without regularization

# with elastic net regularization
