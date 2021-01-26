
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
import random
from sympy import *

from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from gplearn.genetic import SymbolicTransformer
import gplearn as gpl

from deap import base, creator, tools, algorithms


# ---------------------------------------------------------------------------------- Data Transformation -- #
# ---------------------------------------------------------------------------------- ------------------- -- #

def data_trans(p_data, p_trans):
    """
    Scale the data according to the choosen function, scales all the columns in the input data

    Parameters
    ----------
    p_trans: str
        Standard: Standardize features by removing the mean and scaling to unit variance
        Robust: Scale features using statistics that are robust to outliers
        MaxAbs: Scale each feature by its maximum absolute value

    p_datos: pd.DataFrame
        with data to be transformed

    Returns
    -------
    p_datos: pd.DataFrame
        with output data transformed

    References
    ----------
    
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

    """

    if p_trans == 'Standard':

        # Standardize features by removing the mean and scaling to unit variance
        lista = p_data[list(p_data.columns[1:])]
        p_data[list(p_data.columns[1:])] = StandardScaler().fit_transform(lista)

    elif p_trans == 'Robust':

        # Scale features using statistics that are robust to outliers
        lista = p_data[list(p_data.columns[1:])]
        p_data[list(p_data.columns[1:])] = RobustScaler().fit_transform(lista)

    elif p_trans == 'MaxAbs':

        # Scale each feature by its maximum absolute value
        lista = p_data[list(p_data.columns[1:])]
        p_data[list(p_data.columns[1:])] = MaxAbsScaler().fit_transform(lista)

    return p_data


# ---------------------------------------------------------------------------------- Feature Engineering -- #
# --------------------------------------------------------------------------------------------------------- #

def features(p_data, p_nmax):

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
        data['lag_vl_' + str(n + 1)] = data['volume'].shift(n + 1)
    
        # rezago n de Open - Low
        data['lag_ol_' + str(n + 1)] = data['ol'].shift(n + 1)
    
        # rezago n de High - Open
        data['lag_ho_' + str(n + 1)] = data['ho'].shift(n + 1)
    
        # rezago n de High - Low
        data['lag_hl_' + str(n + 1)] = data['hl'].shift(n + 1)
    
        # promedio movil de open-high de ventana n
        data['ma_vl_' + str(n + 2)] = data['volume'].rolling(n + 2).mean()
    
        # promedio movil de open-high de ventana n
        data['ma_ol_' + str(n + 2)] = data['ol'].rolling(n + 2).mean()
    
        # promedio movil de ventana n
        data['ma_ho_' + str(n + 2)] = data['ho'].rolling(n + 2).mean()
    
        # promedio movil de ventana n
        data['ma_hl_' + str(n + 2)] = data['hl'].rolling(n + 2).mean()

        # hadamard product of previously generated features
        list_hadamard = [data['lag_vl_' + str(n + 1)], data['lag_ol_' + str(n + 1)],
                         data['lag_ho_' + str(n + 1)], data['lag_hl_' + str(n + 1)]]

        for x in list_hadamard:
            data['had_'+'lag_vl_' + str(n + 1) + '_' + 'ma_vl_' + str(n + 2)] = x*data['ma_vl_' + str(n + 2)]
            data['had_'+'lag_vl_' + str(n + 1) + '_' + 'ma_ol_' + str(n + 2)] = x*data['ma_ol_' + str(n + 2)]
            data['had_'+'lag_vl_' + str(n + 1) + '_' + 'ma_ho_' + str(n + 2)] = x*data['ma_ho_' + str(n + 2)]
            data['had_'+'lag_vl_' + str(n + 1) + '_' + 'ma_hl_' + str(n + 2)] = x*data['ma_hl_' + str(n + 2)]

    # asignar timestamp como index
    data.index = pd.to_datetime(data.index)
    # quitar columnas no necesarias para modelos de ML
    r_features = data.drop(['open', 'high', 'low', 'close', 'hl', 'ol', 'ho', 'volume'], axis=1)
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


# --------------------------------------------------- Logistic Regression with ElasticNet Regularization -- #
# --------------------------------------------------------------------------------------------------------- #

def logistic_reg(p_data, p_model, p_params, p_iter):

    """
    Logistic Regression with and without elastic net regularization for classification

    Parameters
    ----------

    p_data: pd.DataFrame
        with x and y data
        'x_train': explanatory variables in training set
        'y_train': target variable in training set (numeric for regression, binary for classification)
        'x_test': explanatory variables in test set
        'y_test': target variable in test set (numeric for regression, binary for classification)

    p_model: str
        'logistic': logistic regression without regularization
        'logistic_en': logistic regression with elastic net regularization

    p_params: dict
        any parameter that is needed for the model.

    p_iter: int
        number of iterations to perform

    Returns
    -------

        dict = {'rss': Residual Sum of Squares, 
                'predict': model predictions,
                'model': model object,
                'intercept': intercept information,
                'coef': model coefficients,
                'score': fitness score (regression: R2, classification: Accuracy and AUC)}

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    """

    # Logistic regression without regularization
    if p_model == 'logistic':

        # Model specification
        logistic_model = LogisticRegression(l1_ratio=0, C=p_params['c'], tol=1e-3, penalty='none', 
                                            olver='saga', multi_class='ovr', n_jobs=-1,
                                            max_iter=p_iter, fit_intercept=False, random_state=123)

        # Model fit
        logistic_model.fit(p_data['x_train'], p_data['y_train'])

        # Model predict
        y_logistic_model = logistic_model.predict(p_data['x_train'])

        # Model results
        return {'rss': sum((y_logistic_model - p_data['y_train']) ** 2),
                'predict': y_logistic_model,
                'model': logistic_model,
                'intercept': logistic_model.intercept_,
                'coef': logistic_model.coef_,
                'score': 'accuracy, auc'}

    # Logistic regression with elastic net regularization
    elif p_model == 'logistic_en':

        # Model specification
        logistic_en_model = LogisticRegression(l1_ratio=p_params['ratio'], C=p_params['c'], tol=1e-3,
                                  penalty='elasticnet', solver='saga', multi_class='ovr', n_jobs=-1,
                                  max_iter=p_iter, fit_intercept=False, random_state=123)
        
        # Model fit
        logistic_en_model.fit(p_data['x_train'], p_data['y_train'])

        # Model predict
        y_logistic_en_model = logistic_en_model.predict(p_data['x_train'])

        # Model results
        return {'rss': sum((y_logistic_en_model - p_data['y_train']) ** 2),
                'predict': y_logistic_en_model,
                'model': logistic_en_model,
                'intercept': logistic_en_model.intercept_,
                'coef': logistic_en_model.coef_,
                'score': 'accuracy, auc'}

    else:
        return 'error'


# ------------------------------------------------------------ OLS Regression Models with Regularization -- #
# --------------------------------------------------------------------------------------------------------- #

def ols_reg(p_data, p_model, p_params, p_iter):
    """
    Ordinary Least Squares regression with and without elastic net regularization for regression

    Parameters
    ----------

    p_data: pd.DataFrame
        with x and y data
        'x_train': explanatory variables in training set
        'y_train': target variable in training set (numeric for regression, binary for classification)
        'x_test': explanatory variables in test set
        'y_test': target variable in test set (numeric for regression, binary for classification)

    p_model: str
        'ols': Ordinary Least Squares regression without regularization
        'ols_en': Ordinary Least Squares regression with elastic net regularization

    p_params: dict
        any parameter that is needed for the model.

    p_iter: int
        number of iterations to perform

    Returns
    -------

        dict = {'rss': Residual Sum of Squares, 
                'predict': model predictions,
                'model': model object,
                'intercept': intercept information,
                'coef': model coefficients,
                'score': fitness score (regression: R2, classification: Accuracy and AUC)}

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
      
    """

    # Ordinary Least Squares regression without regularization
    if p_model == 'ols':

        # Model specification
        ols_model = LinearRegression(normalize=False, fit_intercept=False)

        # Model fit
        ols_model.fit(p_data['x_train'], p_data['y_train'])

        # Model predict
        y_ols_model = ols_model.predict(p_data['x_train'])

        # Model results
        return {'rss': sum((y_ols_model - p_data['y_train']) ** 2),
                'predict': y_ols_model,
                'model': ols_model,
                'intercept': ols_model.intercept_,
                'coef': ols_model.coef_,
                'score': r2_score(p_data['y_train'], y_ols_model)}

    # Ordinary Least Squares regression without regularization
    elif p_model == 'ols_en':

        # Model specification
        ols_en_model = ElasticNet(alpha=1, normalize=False, max_iter=p_iter, l1_ratio=p_params['ratio'],    
                                  it_intercept=False)

        # Model fit
        ols_en_model.fit(p_data['x_train'], p_data['y_train'])

        # Model predict
        y_ols_en_model = ols_en_model.predict(p_data['x_train'])

        # Model results
        return {'rss': sum((y_ols_en_model - p_data['y_train']) ** 2),
                'predict': y_ols_en_model,
                'model': ols_en_model,
                'intercept': ols_en_model.intercept_,
                'coef': ols_en_model.coef_,
                'score': r2_score(p_data['y_train'], y_ols_en_model)}

    else:
        return 'error'


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

    model = SymbolicTransformer(function_set=["sub", "add", 'inv', 'mul', 'div', 'abs', 'log', 'max', 'min'],
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

    return results, best_p_dict


# -------------------------------------------------------------------------- FUNCTION: Genetic Algorithm -- #
# ------------------------------------------------------- ------------------------------------------------- #

def optimization(p_data, p_model, p_params):
    """
    optimization with genetic algorithms

    Parameters
    ----------
    p_data:  pd.DataFrame

    p_model: str

    p_params: dict


    Returns
    ----------


    References
    ----------
    https://deap.readthedocs.io

    """

    # initialize genetic algorithm object
    creator.create("FitnessMax_en", base.Fitness, weights=(1.0,))
    creator.create("Individual_en", list, fitness=creator.FitnessMax_en)
    toolbox_en = base.Toolbox()

    # define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
    toolbox_en.register("attr_ratio", random.choice, p_model['params']['ratio'])
    toolbox_en.register("attr_c", random.choice, p_model['params']['c'])

    # This is the order in which genes will be combined to create a chromosome
    toolbox_en.register("Individual_en", tools.initCycle, creator.Individual_en,
                        (toolbox_en.attr_ratio, toolbox_en.attr_c), n=1)

    # population definition
    toolbox_en.register("population", tools.initRepeat, list, toolbox_en.Individual_en)

    # -------------------------------------------------------------- funcion de mutacion para LS SVM -- #
    def mutate_en(individual):

        # select which parameter to mutate
        gene = random.randint(0, len(p_model['params']) - 1)

        if gene == 0:
            individual[0] = random.choice(p_model['params']['ratio'])
        elif gene == 1:
            individual[1] = random.choice(p_model['params']['c'])

        return individual,

    # --------------------------------------------------- funcion de evaluacion para OLS Elastic Net -- #
    def evaluate_en(eva_individual):

        # output of genetic algorithm
        chromosome = {'ratio': eva_individual[0], 'c': eva_individual[1]}

        
        if p_model == 'classification':

            # model results
            model = logistic_reg(p_data=p_data, p_params=chromosome)

            # True positives in train data
            train_tp = model['results']['matrix']['train'][0, 0]
            # True negatives in train data
            train_tn = model['results']['matrix']['train'][1, 1]
            # Model accuracy
            train_fit = (train_tp + train_tn) / len(model['results']['data']['train'])

            # True positives in test data
            test_tp = model['results']['matrix']['test'][0, 0]
            # True negatives in test data
            test_tn = model['results']['matrix']['test'][1, 1]
            # Model accuracy
            test_fit = (test_tp + test_tn) / len(model['results']['data']['test'])

            # Fitness measure
            model_fit = np.mean([train_fit, test_fit])

            return model_fit,
                
        elif p_model == 'regression':
            
            model_fit = 1
            # aqui va para el caso de regresion 

            return model_fit,

    toolbox_en.register("mate", tools.cxOnePoint)
    toolbox_en.register("mutate", mutate_en)
    toolbox_en.register("select", tools.selTournament, tournsize=10)
    toolbox_en.register("evaluate", evaluate_en)

    population_size = 50
    crossover_probability = 0.8
    mutation_probability = 0.1
    number_of_generations = 4

    en_pop = toolbox_en.population(n=population_size)
    en_hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Genetic Algorithm Implementation
    en_pop, en_log = algorithms.eaSimple(population=en_pop, toolbox=toolbox_en, stats=stats,
                                            cxpb=crossover_probability, mutpb=mutation_probability,
                                            ngen=number_of_generations, halloffame=en_hof, verbose=True)

    # transform the deap objects into list so it can be serialized and stored with pickle
    en_pop = [list(pop) for pop in list(en_pop)]
    en_log = [list(log) for log in list(en_log)]
    en_hof = [list(hof) for hof in list(en_hof)]

    return {'population': en_pop, 'logs': en_log, 'hof': en_hof}
