
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
data_x = fn.data_trans(p_data=data_x, p_trans='Robust')

var = fn.variables(data_y, data_x, N=3)
busqueda= fn.busqueda_en_train(var, n=3)
# Rearange of data for regression model

# Rearange of data for classification model
data_cla = {'x_data': data_x, 'y_data': data_y['co_d']}

# -- ---------------------------------------------------------------------------------- Feature analysis -- #

# matriz de correlacion
cor_mat = data_x.corr()

# -- ---------------------------------------------------------------------------------------- Models fit -- #
# -- With autoregressive and hadamard features

# -- ---------------------------------------------------- Regression Model: Ordinary Least Squares Model -- # 

# -- without regularization


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

# Symbolic features generation
symbolic, table = fn.symbolic_features(p_x=data_reg1['x_data'], p_y=data_reg1['y_data'])
x_simbolic = pd.DataFrame(symbolic['fit'], index=data_reg1['x_data'].index)
data_reg_sim = {'x_data': x_simbolic, 'y_data': data_y['co']}
#corre = vis.correlation(pd.concat([x_simbolic.iloc[:, -15:], data_y['co']], axis=1))
#plt.show()
reg_51 = fn.ols_reg(p_data=data_reg_sim, p_params=params_reg, p_model="ols", p_iter=10000)
residuales_51 = reg_51['results']['y_data']-reg_51['results']['y_data_p']
hetero_51 = fn.check_hetero(residuales_51)
vis.residual(residuales=residuales_51)
# with elastic net regularization
data_op = fn.optimization(p_data=data_reg_sim, p_type='regression', p_params=params_reg,
                          p_model='ols_en', p_iter=100000)
# in sample results
new_params = {"ratio": data_op["population"][0][0], "c": data_op["population"][0][1]}
reg_61 = fn.ols_reg(p_data=data_reg_sim, p_params=new_params, p_model="ols_en", p_iter=10000)
residuales_61 = reg_61['results']['y_data']-reg_61['results']['y_data_p']
hetero_61 = fn.check_hetero(residuales_61)
vis.residual(residuales=residuales_61)


symbolic, table = fn.symbolic_features(p_x=data_reg2['x_data'], p_y=data_reg2['y_data'])
x_simbolic = pd.DataFrame(symbolic['fit'], index=data_reg2['x_data'].index)
data_reg_sim = {'x_data': x_simbolic, 'y_data': data_y['co']}
#corre = vis.correlation(pd.concat([x_simbolic.iloc[:, -15:], data_y['co']], axis=1))
#plt.show()
reg_52 = fn.ols_reg(p_data=data_reg_sim, p_params=params_reg, p_model="ols", p_iter=10000)
residuales_52 = reg_52['results']['y_data']-reg_52['results']['y_data_p']
hetero_52 = fn.check_hetero(residuales_52)
vis.residual(residuales=residuales_52)
# with elastic net regularization
data_op = fn.optimization(p_data=data_reg_sim, p_type='regression', p_params=params_reg,
                          p_model='ols_en', p_iter=100000)
# in sample results
new_params = {"ratio": data_op["population"][0][0], "c": data_op["population"][0][1]}
reg_62 = fn.ols_reg(p_data=data_reg_sim, p_params=new_params, p_model="ols_en", p_iter=10000)
residuales_62 = reg_62['results']['y_data']-reg_62['results']['y_data_p']
hetero_62 = fn.check_hetero(residuales_62)
vis.residual(residuales=residuales_62)


symbolic, table = fn.symbolic_features(p_x=data_reg3['x_data'], p_y=data_reg3['y_data'])
x_simbolic = pd.DataFrame(symbolic['fit'], index=data_reg3['x_data'].index)
data_reg_sim = {'x_data': x_simbolic, 'y_data': data_y['co']}
#corre = vis.correlation(pd.concat([x_simbolic.iloc[:, -15:], data_y['co']], axis=1))
#plt.show()
reg_53 = fn.ols_reg(p_data=data_reg_sim, p_params=params_reg, p_model="ols", p_iter=10000)
residuales_53 = reg_53['results']['y_data']-reg_53['results']['y_data_p']
hetero_53 = fn.check_hetero(residuales_53)
vis.residual(residuales=residuales_53)
# with elastic net regularization
data_op = fn.optimization(p_data=data_reg_sim, p_type='regression', p_params=params_reg,
                          p_model='ols_en', p_iter=100000)
# in sample results
new_params = {"ratio": data_op["population"][0][0], "c": data_op["population"][0][1]}
reg_63 = fn.ols_reg(p_data=data_reg_sim, p_params=new_params, p_model="ols_en", p_iter=10000)
residuales_63 = reg_63['results']['y_data']-reg_63['results']['y_data_p']
hetero_63 = fn.check_hetero(residuales_63)
vis.residual(residuales=residuales_63)
# -- ---------------------------------------------------------------------------------------- Models fit -- #
# -- With autoregressive and hadamard features and symbolic features

# -- Logistic Regression Model (Classification)

# without regularization

# with elastic net regularization
