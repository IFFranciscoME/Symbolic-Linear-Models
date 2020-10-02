
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Multivariate Linear Symbolic Regression with Regularization using Elastic Net              -- #
# -- for future prices prediction, the case for UsdMxn                                                   -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: IFFranciscoME                                                                               -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/A3_Regresion_Simbolica                                 -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from statsmodels.tsa.api import acf, pacf              # funciones de econometria
from sklearn.preprocessing import StandardScaler       # estandarizacion de variables
from gplearn.genetic import SymbolicRegressor          # regresion simbolica
import gplearn as gpl
import graphviz as graphviz

# ---------------------------------------------------------------------------------- Feature Engineering -- #
# --------------------------------------------------------------------------------------------------------- #

def f_features(p_data):
    """
    Parameters
    ----------
    p_data

    """

    # reasignar datos
    data = p_data.copy()

    # pips descontados al cierre
    data['co'] = (data['close'] - data['open'])*10000

    # pips descontados alcistas
    data['ho'] = (data['high'] - data['open'])*10000
    
    # pips descontados bajistas
    data['ol'] = (data['open'] - data['low'])*10000
    
    # pips descontados en total (medida de volatilidad)
    data['hl'] = (data['high'] - data['low'])*10000
    
    # clase a predecir
    data['co_d'] = [1 if i > 0 else 0 for i in list(data['co'])]
    
    # funciones de ACF y PACF para determinar ancho de ventana historica
    data_acf = acf(data['co'].dropna(), nlags=12, fft=True)
    data_pac = pacf(data['co'].dropna(), nlags=12)
    sig = round(1.96/np.sqrt(len(data['co'])), 4)
    
    # componentes AR y MA
    maxs = list(set(list(np.where((data_pac > sig) | (data_pac < -sig))[0]) +
                    list(np.where((data_acf > sig) | (data_acf < -sig))[0])))
    # encontrar la componente maxima como indicativo de informacion historica autorelacionada
    max_n = maxs[np.argmax(maxs)]
    
    # condicion arbitraria: 5 resagos minimos para calcular variables moviles
    if max_n <= 2:
        max_n = 5
    
    # ciclo para calcular N features con logica de "Ventanas de tamaño n"
    for n in range(0, max_n):
    
        # resago n de oi
        data['lag_oi_' + str(n + 1)] = data['openinterest'].shift(n + 1)
    
        # resago n de ol
        data['lag_ol_' + str(n + 1)] = data['ol'].shift(n + 1)
    
        # resago n de ho
        data['lag_ho_' + str(n + 1)] = data['ho'].shift(n + 1)
    
        # resago n de hl
        data['lag_ho_' + str(n + 1)] = data['hl'].shift(n + 1)
    
        # promedio movil de open-high de ventana n
        data['ma_oi_' + str(n + 2)] = data['openinterest'].rolling(n + 2).mean()
    
        # promedio movil de open-high de ventana n
        data['ma_ol_' + str(n + 2)] = data['ol'].rolling(n + 2).mean()
    
        # promedio movil de ventana n
        data['ma_ho_' + str(n + 2)] = data['ho'].rolling(n + 2).mean()
    
        # promedio movil de ventana n
        data['ma_hl_' + str(n + 2)] = data['hl'].rolling(n + 2).mean()
    
    # asignar timestamp como index
    data.index = pd.to_datetime(data.index)
    # quitar columnas no necesarias para modelos de ML
    r_features = data.drop(['open', 'high', 'low', 'close',
                            'hl', 'ol', 'ho', 'openinterest', 'volume'], axis=1)
    # borrar columnas donde exista solo NAs
    r_features = r_features.dropna(axis='columns', how='all')
    # borrar renglones donde exista algun NA
    r_features = r_features.dropna(axis='rows')
    # convertir a numeros tipo float las columnas
    r_features.iloc[:, 2:] = r_features.iloc[:, 2:].astype(float)

    # estandarizacion de todas las variables independientes
    lista = r_features[list(r_features.columns[2:])]

    # armar objeto de salida
    r_features[list(r_features.columns[2:])] = StandardScaler().fit_transform(lista)
    # reformatear columna de variable binaria a 0 y 1
    r_features['co_d'] = [0 if i <= 0 else 1 for i in r_features['co_d']]
    # resetear index
    r_features.reset_index(inplace=True, drop=True)

    return r_features


# --------------------------------------------------------- MODEL: Multivariate Linear Regression Models -- #
# --------------------------------------------------------------------------------------------------------- #

def mult_regression(p_x, p_y, p_alpha, p_iter):
    """
    Funcion para ajustar varios modelos lineales

    Parameters
    ----------

    p_x: pd.DataFrame
        with regressors or predictor variables
        p_x = data_features.iloc[0:30, 3:]

    p_y: pd.DataFrame
        with variable to predict
        p_y = data_features.iloc[0:30, 1]

    p_alpha: float
        alpha for the models
        p_alpha = alphas[1e-3]

    p_iter: int
        Number of iterations until stop the model fit process
        p_iter = 1e6

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

    """

    # Fit LINEAR regression
    linreg = LinearRegression(normalize=False, fit_intercept=False)
    linreg.fit(p_x, p_y)
    y_p_linear = linreg.predict(p_x)

    # Fit RIDGE regression
    ridgereg = Ridge(alpha=p_alpha, normalize=False, max_iter=p_iter, fit_intercept=False)
    ridgereg.fit(p_x, p_y)
    y_p_ridge = ridgereg.predict(p_x)

    # Fit LASSO regression
    lassoreg = Lasso(alpha=p_alpha, normalize=False, max_iter=p_iter, fit_intercept=False)
    lassoreg.fit(p_x, p_y)
    y_p_lasso = lassoreg.predict(p_x)

    # Fit ElasticNet regression
    enetreg = ElasticNet(alpha=p_alpha, normalize=False, max_iter=p_iter, l1_ratio=0.5, fit_intercept=False)
    enetreg.fit(p_x, p_y)
    y_p_enet = enetreg.predict(p_x)

    # Return the result of the model
    r_models = {'linear': {'rss': sum((y_p_linear - p_y) ** 2),
                           'intercept': linreg.intercept_,
                           'coef': linreg.coef_},
                'rige': {'rss': sum((y_p_ridge - p_y) ** 2),
                         'intercept': ridgereg.intercept_,
                         'coef': ridgereg.coef_},
                'lasso': {'rss': sum((y_p_lasso - p_y) ** 2),
                          'intercept': lassoreg.intercept_,
                          'coef': lassoreg.coef_},
                'elasticnet': {'rss': sum((y_p_enet - p_y) ** 2),
                               'intercept': enetreg.intercept_,
                               'coef': enetreg.coef_}
                }

    return r_models


# ------------------------------------------------------------------------------- TimeSeries Block Split -- #
# --------------------------------------------------------------------------------------------------------- #

def f_tsbs(p_data):
    """
    TimeSeries Block Split tecnique for crossvalidation a time series based model

    Parameters
    ----------
    p_data: pd.DataFrame
        con los datos a dividir
        p_data = data_features

    Returns
    -------

    """
    return 1


def _rss(y, y_pred, w):
    diffs = (y-y_pred)**2
    return np.sum(diffs)


def symbolic_regression(p_x, p_y):
    """
        Funcion para crear regresores no lineales

        Parameters
        ----------

        p_x: pd.DataFrame
            with regressors or predictor variables
            p_x = data_features.iloc[0:30, 3:]

        p_y: pd.DataFrame
            with variable to predict
            p_y = data_features.iloc[0:30, 1]
    Returns
    -------
    score_gp: float
        error of prediction
    """
    rss = gpl.fitness.make_fitness(_rss, greater_is_better=False)
    est_gp = SymbolicRegressor(function_set=["sub", "add", 'inv', 'mul', 'div', 'sqrt'], feature_names=p_x.columns,
                               stopping_criteria=500, metric=rss,
                               p_crossover=0.5, p_subtree_mutation=0.15, p_hoist_mutation=0.05,
                               p_point_mutation=0.3, verbose=1, random_state=None, n_jobs=-1, warm_start=True)
    est_gp.fit(p_x, p_y)                 # (con train)
    score_gp = est_gp.score(p_x, p_y)    # (con test)
    print(score_gp)
    dot_data = est_gp._program.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph.render('tree.gv', view=True)
    return est_gp._program, graph
