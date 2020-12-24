
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Multivariate Linear Regression with Symbolic Regressors and L1L2 Regularization            -- #
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
from sklearn.preprocessing import StandardScaler       # estandarizacion de variables
from sklearn.metrics import (accuracy_score, precision_score, recall_score)
from sklearn import svm
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer          # regresion simbolica
import gplearn as gpl


# ---------------------------------------------------------------------------------- Feature Engineering -- #
# --------------------------------------------------------------------------------------------------------- #

def f_features(p_data, p_nmax):

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

    # ciclo para calcular N features con logica de "Ventanas de tamaño n"
    for n in range(0, p_nmax):
    
        # rezago n de Open Interest
        data['lag_oi_' + str(n + 1)] = data['openinterest'].shift(n + 1)
    
        # rezago n de Open - Low
        data['lag_ol_' + str(n + 1)] = data['ol'].shift(n + 1)
    
        # rezago n de High - Open
        data['lag_ho_' + str(n + 1)] = data['ho'].shift(n + 1)
    
        # rezago n de High - Low
        data['lag_hl_' + str(n + 1)] = data['hl'].shift(n + 1)
    
        # promedio movil de open-high de ventana n
        data['ma_oi_' + str(n + 2)] = data['openinterest'].rolling(n + 2).mean()
    
        # promedio movil de open-high de ventana n
        data['ma_ol_' + str(n + 2)] = data['ol'].rolling(n + 2).mean()
    
        # promedio movil de ventana n
        data['ma_ho_' + str(n + 2)] = data['ho'].rolling(n + 2).mean()
    
        # promedio movil de ventana n
        data['ma_hl_' + str(n + 2)] = data['hl'].rolling(n + 2).mean()

        # hadamard product of previously generated features
        list_hadamard = [data['lag_oi_' + str(n + 1)], data['lag_ol_' + str(n + 1)],
                         data['lag_ho_' + str(n + 1)], data['lag_hl_' + str(n + 1)]]

        for x in list_hadamard:
            data['had_'+'lag_oi_' + str(n + 1) + '_' + 'ma_oi_' + str(n + 2)] = x*data['ma_oi_' + str(n + 2)]
            data['had_'+'lag_oi_' + str(n + 1) + '_' + 'ma_ol_' + str(n + 2)] = x*data['ma_ol_' + str(n + 2)]
            data['had_'+'lag_oi_' + str(n + 1) + '_' + 'ma_ho_' + str(n + 2)] = x*data['ma_ho_' + str(n + 2)]
            data['had_'+'lag_oi_' + str(n + 1) + '_' + 'ma_hl_' + str(n + 2)] = x*data['ma_hl_' + str(n + 2)]

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


# ---------------------------------------------------------- MODEL: Multivariate Linear Regression Model -- #
# --------------------------------------------------------------------------------------------------------- #

def mult_regression(p_x, p_y):
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

    Returns
    -------
    r_models: dict
        Diccionario con modelos ajustados

    """

    # Fit LINEAR regression
    linreg = LinearRegression(normalize=False, fit_intercept=False)
    xtrain, xtest, ytrain, ytest = train_test_split(p_x, p_y, test_size=.2, shuffle=False)
    linreg.fit(xtrain, ytrain)
    y_p_linear = linreg.predict(xtest)
    y_p_score = linreg.score(xtest, ytest)

    # Return the result of the model
    linear_model = {'rss': np.round(sum((y_p_linear - ytest) ** 2), 4),
                    'predict': y_p_linear,
                    'model': linreg,
                    'intercept': linreg.intercept_,
                    'coef': linreg.coef_,
                    'score': np.round(y_p_score, 4)}
    return linear_model


# -------------------------------- MODEL: Multivariate Linear Regression Models with L1L2 regularization -- #
# --------------------------------------------------------------------------------------------------------- #

def mult_reg_l1l2(p_x, p_y, p_alpha, p_iter):
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
    xtrain, xtest, ytrain, ytest = train_test_split(p_x, p_y, test_size=.2, shuffle=False)
    # Fit RIDGE regression
    ridgereg = Ridge(alpha=p_alpha, normalize=False, max_iter=p_iter, fit_intercept=False)
    ridgereg.fit(xtrain, ytrain)
    y_p_ridge = ridgereg.predict(xtest)

    # Fit LASSO regression
    lassoreg = Lasso(alpha=p_alpha, normalize=False, max_iter=p_iter, fit_intercept=False)
    lassoreg.fit(xtrain, ytrain)
    y_p_lasso = lassoreg.predict(xtest)

    # Fit ElasticNet regression
    enetreg = ElasticNet(alpha=p_alpha, normalize=False, max_iter=p_iter, l1_ratio=0.5, fit_intercept=False)
    enetreg.fit(xtrain, ytrain)
    y_p_enet = enetreg.predict(xtest)

    # RSS = residual sum of squares

    # Return the result of the model
    r_models = {"summary": {"Ridge rss": sum((y_p_ridge - ytest) ** 2),
                            "lasso rss": sum((y_p_lasso - ytest) ** 2),
                            "elasticnet rss": sum((y_p_enet - ytest) ** 2)},
                'rige': {'rss': sum((y_p_ridge - ytest) ** 2),
                         'predict': y_p_ridge,
                         'model': ridgereg,
                         'intercept': ridgereg.intercept_,
                         'coef': ridgereg.coef_},
                'lasso': {'rss': sum((y_p_lasso - ytest) ** 2),
                          'predict': y_p_lasso,
                          'model': lassoreg,
                          'intercept': lassoreg.intercept_,
                          'coef': lassoreg.coef_},
                'elasticnet': {'rss': sum((y_p_enet - ytest) ** 2),
                               'predict': y_p_enet,
                               'model': enetreg,
                               'intercept': enetreg.intercept_,
                               'coef': enetreg.coef_}
                }

    return r_models


# ------------------------------------------------------------------ MODEL: Symbolic Features Generation -- #
# --------------------------------------------------------------------------------------------------------- #

def symbolic_features(p_x, p_y):
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

    # -- metric to measure performance
    def _rss(y, y_pred, w):
        diffs = (y - y_pred) ** 2
        return np.sum(diffs)

    rss = gpl.fitness.make_fitness(_rss, greater_is_better=False)

    model = SymbolicTransformer(function_set=["sub", "add", 'inv', 'mul', 'div', 'abs', 'log',
                                              'min', 'max'],
                                population_size=1000, hall_of_fame=100, n_components=10,
                                generations=20, tournament_size=20,  stopping_criteria=.05,
                                const_range=None, init_method='half and half', init_depth=(4, 12),
                                metric='pearson', parsimony_coefficient=0.001,
                                p_crossover=0.4, p_subtree_mutation=0.1, p_hoist_mutation=0.2,
                                p_point_mutation=0.1, p_point_replace=.05,
                                verbose=1, random_state=None, n_jobs=-1, feature_names=p_x.columns,
                                warm_start=True)
    xtrain, xtest, ytrain, ytest = train_test_split(p_x, p_y, test_size=.2, shuffle=False)
    model.fit_transform(xtrain, ytrain)
    model_params = model.get_params()
    gp_features = model.transform(p_x)
    model_fit = np.hstack((p_x, gp_features))
    results = {'fit': model_fit, 'params': model_params, 'model': model}
    best_p = model._best_programs
    best_p_dict={}
    for p in best_p:
        factor_name = 'alpha_'+str(best_p.index(p)+1)
        best_p_dict[factor_name] = {'fitness': p.fitness_, "expression": str(p), 'depth': p.depth_,
                                    "length": p.length_}
    best_p_dict = pd.DataFrame(best_p_dict).T
    best_p_dict = best_p_dict.sort_values(by="fitness")
    return results,best_p_dict
