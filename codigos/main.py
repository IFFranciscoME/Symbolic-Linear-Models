
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Multivariate Linear Symbolic Regression with Regularization using Elastic Net              -- #
# -- for future prices prediction, the case for UsdMxn                                                   -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: IFFranciscoME                                                                               -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/A3_Regresion_Simbolica                                 -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import sympy as sp
import pandas as pd
import numpy as np
import codigos.functions as fn
from codigos.data import m6e1
from codigos.visualizations import vs

pd.set_option('display.max_rows', None)                   # sin limite de renglones maximos
pd.set_option('display.max_columns', None)                # sin limite de columnas maximas
pd.set_option('display.width', None)                      # sin limite el ancho del display
pd.set_option('display.expand_frame_repr', False)         # visualizar todas las columnas

# ------------------------------------------------------------------------------------------ Obtain data -- #

# m6e : Micro Eur/Usd Future: https://www.cmegroup.com/trading/fx/e-micros/e-micro-euro.html
# obtained with Quandl / Chris free data set. Gathers the data from webscraping cmegroup's future data
data = m6e1.tail(160)
data.tail()

# -- ------------------------------------------------------------------------- Exploratory Data Analysis -- #

# Description table
data.describe()

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
data_features = fn.f_features(p_data=data)

# muestra de los features
data_features.head()
print(type(data_features))

# -- ---------------------------------------------------------------------------------- Feature analysis -- #

# matriz de correlacion
cor_mat = data_features.iloc[:, 1:-1].corr()

# -- ---------------------------------------------------------------------------- TimeSeries Block Split -- #

# opciones para definir los bloques
split = {'a': {'1sem'}, 'b': {'2sem'}, 'c': {'3sem'}, 'd': {'4sem'}, 'model': {'changepoint'}}

# funcion para definir bloques
bloques = fn.f_tsbs(p_data=data_features)

# -- ----------------------------------------------------------------------------------------- Model fit -- #
alphas = [1e-5, 1e-3, 1e-2, 1, 1e2, 1e3, 1e5]
models = fn.mult_regression(p_x=data_features.iloc[:, 3:-1],
                            p_y=data_features.iloc[:, 1],
                            p_alpha=alphas[1], p_iter=1e6)

# -- ------------------------------------------------------------------------------- Features simbolicos -- #

# semilla para siempre obtener el mismo resultado
np.random.seed(455)

# Generacion de un feature formado con variable simbolica
symbolic = fn.symbolic_regression(p_x=data_features.iloc[:, 3:-1], p_y=data_features.iloc[:, 1])

# convertir a str el resultado
texto = symbolic.__str__()

# declaracion de operaciones simbolicas
op_sim = {
    'sub': lambda x, y: x - y,
    'div': lambda x, y: x / y,
    'mul': lambda x, y: x * y,
    'add': lambda x, y: x + y,
    'inv': lambda x: x**(-1)
}

# este es un ejemplo de como declarar cada variable como simbolica
# sustituir una variable simbolica en la expresion
# tener un objeto tipo simbolico sympy

# expresion simbolica en formato de texto
exp_sim = str(sp.sympify(texto, locals=op_sim, evaluate=True))

# evaluar la variable con el valor que se desea para agregar resultado numerico de la
# expresion simbolica en el cuadro de features
# data_features.eval("gplearn={}".format(exp_sim), inplace=True)

data_features["gp1"] = (data_features["lag_ol_4"]-data_features["ma_ol_3"]) / data_features["lag_ho_4"]
data_features["gp2"] = data_features["lag_ho_7"]*data_features["ma_hl_5"]*data_features["ma_oi_5"]*data_features["ma_ol_3"]
data_features["gp3"] = (data_features["lag_ol_5"]+data_features["ma_ho_6"]-data_features["ma_oi_6"]) * data_features["ma_hl_5"]
data_features["gp4"] = (data_features["ma_ho_6"]-data_features["ma_oi_9"]) / data_features["lag_ol_5"]

s1 = "(lag_ol_4 - ma_ol_3)/(lag_ho_4)"
s2 = data_features["lag_ho_7"]*data_features["ma_hl_5"]*data_features["ma_oi_5"]*data_features["ma_ol_3"]
s3 = (data_features["lag_ol_5"]+data_features["ma_ho_6"]-data_features["ma_oi_6"]) * data_features["ma_hl_5"]
s4 = (data_features["ma_ho_6"]-data_features["ma_oi_9"]) / data_features["lag_ol_5"]

# PROCESO: buscar el menor score en la funcion simbolic regresor modificando los procentajes de
# mutación, de stopping criteria, y la semilla: encontrado fue aprox .305
# dividir para obtener las variables (4 se me hizo el numero óptimo al ver exp_sim)
# se logró reducir hasta 62,134 el rss
# observación: "abs" y "sen" no estan en op_sim, no se están tomando en cuenta por lo que veo

# escribir nuevos features en un excel para proceso de ajuste de modelo en R
data_features.to_csv('simbolic_features.csv')

# reajustar modelos
models2 = fn.mult_regression(p_x=data_features.iloc[:, 3:-1],
                             p_y=data_features.iloc[:, 1],
                             p_alpha=alphas[1], p_iter=1e6)

# -- PASOS a seguir
# 1ero - Regresion simbolica para obtener regresores "Individuales", Hacer esto manualmente.
# (Ingenieria de variables simbolicas a traves un proceso de separacion de componentes de una regresion
# simbolica con gplearn).
# Una prueba semi/manual en python con un modelo simple, ridge, lasso y elasticnet.
# revisar como separar las subcomponentes del resultado de la regreion simbolica sistematicamente.
# ma_hl_5*(lag_ol_5 + ma_ho_6 - ma_oi_6)*(lag_ol_6 + 0.074 + ma_hl_3*(lag_ol_4 - ma_ol_3)/lag_ho_4)
# paralelizar proces de uso de gplearn para generar unos regresores muy buenos y correlacionados con
# la variable a explicar.
# 2do - Un muy buen proceso de ajuste de un modelo de Regresion lineal.
# 3ero - Un muy buen proceso de ajuste de un modelo de Regresion.

# -- Elementos de automatizacion de extraccion de features
# separar el feature general en N sub-ecuaciones

# evaluar cada una de las N sub-equaciones para generar N nuevos features
# -- Bajo el supuesto de haber separado los n = 4 sub ecuaciones

# Otro experimento es hacer el producto hadamard con features en el mismo punto del tiempo (periodo)
# ej: lag_hl_1 (hadamard) ma_hl_1
