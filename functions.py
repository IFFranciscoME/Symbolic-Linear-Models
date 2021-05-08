
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
import data as dt
import numpy as np
import pandas as pd
import random
from sympy import *
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import visualizations as vis
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

    # keep the column names
    cols = list(p_data.columns)

    if p_trans == 'Standard':
        
        # Standardize features by removing the mean and scaling to unit variance
        p_data = pd.DataFrame(StandardScaler().fit_transform(p_data))
        p_data.columns = cols

    elif p_trans == 'Robust':

        # Scale features using statistics that are robust to outliers
        p_data = pd.DataFrame(RobustScaler().fit_transform(p_data))
        p_data.columns = cols

    elif p_trans == 'MaxAbs':

        # Scale each feature by its maximum absolute value
        p_data = pd.DataFrame(MaxAbsScaler().fit_transform(p_data))
        p_data.columns = cols

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

    # columns names modification for numeric values (only x_features)
    features_names = r_features.columns
    names = np.arange(3, len(r_features.iloc[3,]), 1)
    str_names = [str(i) for i in names]
    r_features.columns = features_names[0:3].to_list() + str_names

    data_t = r_features[r_features.columns[0]]
    data_t.drop(data_t.tail(1).index, inplace=True)
    # Target variables (co: regression, co_d: classification)
    data_y = r_features[list(r_features.columns[1:3])].shift(1)
    data_y.drop(data_y.head(1).index, inplace=True)
    data_y = data_y.reset_index()
    # Autoregressive and hadamard features
    data_x = r_features[list(r_features.columns[3:])]
    data_x = data_x.drop(data_x.tail(1).index)
    return data_t, data_y, data_x, features_names.to_list()


# -- --------------------------------------------------------------------- Metrics for Model Performance -- # 
# -- --------------------------------------------------------------------- ----------------------------- -- #

def model_metrics(p_model, p_data, p_type):
    """
    
    Parameters
    ----------
    p_model: str
        string with the name of the model
    
    p_data: dict
        With x_data, y_data keys
    
    p_type: str
        the type of model: 'regression', 'classification'
   
    Returns
    -------
    r_model_metrics

    References
    ----------


    """

    # metrics for a classification model
    if p_type == 'classification':

        # fitted train values
        y_model_data = p_model.predict(p_data['x_data'])
        p_y_result_data = pd.DataFrame({'y_data': p_data['y_data'], 'y_data_pred': y_model_data})

        # Confussion matrix
        cm_data = confusion_matrix(p_y_result_data['y_data'], p_y_result_data['y_data_pred'])
        # Probabilities of class in train data
        probs_data = p_model.predict_proba(p_data['x_data'])
        # in case of a nan, replace it with zero (to prevent errors)
        probs_data = np.nan_to_num(probs_data)
        
        # Accuracy rate
        acc_data = round(accuracy_score(list(p_data['y_data']), y_model_data), 4)
        # False Positive Rate, True Positive Rate, Thresholds
        fpr_data, tpr_data, thresholds = roc_curve(list(p_data['y_data']), probs_data[:, 1], pos_label=1)
        # Area Under the Curve (ROC) for train data
        auc_data = round(roc_auc_score(list(p_data['y_data']), probs_data[:, 1]), 4)

        # Return the result of the model
        return {'model': p_model,
                'results': {'x_data': p_data['x_data'], 'y_data': p_data['y_data'], 'y_data_p': y_model_data},

                'metrics': {'acc': acc_data, 'tpr': tpr_data, 'fpr': fpr_data, 'probs': probs_data,
                'auc': auc_data, 'matrix': cm_data}}

    # metrics for a regression model
    elif p_type == 'regression':

        # Model predict
        y_model_data = p_model.predict(p_data['x_data'])

        # Return the result of the model
        return {'model': p_model,
                'results': {'x_data': p_data['x_data'], 'y_data': p_data['y_data'], 'y_data_p': y_model_data},

                'metrics': {'rss': sum((p_data['y_data']-y_model_data) ** 2), 'coef': p_model.coef_,
                            'r2': r2_score(p_data['y_data'], y_model_data), 'intercept': p_model.intercept_}}

    else:
        return 'error: invalid type of model'


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

    # dummy variable
    model = ''

    # Logistic regression without regularization
    if p_model == 'logistic':

        # Model specification
        model = LogisticRegression(tol=1e-3, 
                                   solver='saga', multi_class='ovr', n_jobs=-1,
                                   max_iter=p_iter, fit_intercept=False, random_state=123)
        
    # Logistic regression with elastic net regularization
    elif p_model == 'logistic_en':

        # Model specification
        model = LogisticRegression(l1_ratio=p_params['ratio'], C=p_params['c'], tol=1e-3,
                                   penalty='elasticnet', solver='saga', multi_class='ovr', n_jobs=-1,
                                   max_iter=p_iter, fit_intercept=False, random_state=123)

    # report error
    else:
        return 'error: invalid model'
    
    # Model fit
    model.fit(p_data['x_data'], p_data['y_data'])

    # performance metrics of the model
    return model_metrics(p_model=model, p_data=p_data, p_type='classification')


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

    # dummy variable
    model = ''

    # Ordinary Least Squares regression without regularization
    if p_model == 'ols':

        # Model specification
        model = LinearRegression(normalize=False, fit_intercept=True)

    # Ordinary Least Squares regression with regularization
    elif p_model == 'ols_en':

        # Model specification
        model = ElasticNet(alpha=p_params['c'], normalize=False, max_iter=p_iter, l1_ratio=p_params['ratio'],
                           fit_intercept=True)

    # error in model key
    else:
        return 'error: invalid model key'

    # Model fit
    model.fit(p_data['x_data'], p_data['y_data'])

    # performance metrics of the model
    return model_metrics(p_model=model, p_data=p_data, p_type='regression')


# ------------------------------------------------------------------------- Symbolic Features Generation -- #
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
    model = SymbolicTransformer(function_set=["sub", "add", 'inv', 'mul', 'div', 'abs', 'log'],
                                population_size=5000, hall_of_fame=20, n_components=10,
                                tournament_size=20,
                                generations=5, init_depth=(4, 8), init_method='half and half',
                                parsimony_coefficient=0.1, const_range=None, metric='pearson',
                                stopping_criteria=0.65, p_crossover=0.4, p_subtree_mutation=0.3,
                                p_hoist_mutation=0.1, p_point_mutation=0.2, verbose=True,
                                warm_start=True, n_jobs=-1, feature_names=p_x.columns)
    model.fit_transform(p_x, p_y)
    model_params = model.get_params()
    gp_features = model.transform(p_x)

    model_fit = np.hstack((p_x, gp_features))
    results = {'fit': model_fit, 'params': model_params, 'model': model, "features": gp_features}
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

def optimization(p_data, p_model, p_type, p_params, p_iter):
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

    # Delete, if exists, genetic algorithm functional classes
    try:
        del creator.FitnessMax_en
        del creator.Individual_en
    except AttributeError:
        pass

    # Initialize genetic algorithm object
    creator.create("FitnessMax_en", base.Fitness, weights=(1.0,))
    creator.create("Individual_en", list, fitness=creator.FitnessMax_en)
    toolbox_en = base.Toolbox()

    # Define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
    toolbox_en.register("attr_ratio", random.choice, p_params['ratio'])
    toolbox_en.register("attr_c", random.choice, p_params['c'])

    # This is the order in which genes will be combined to create a chromosome
    toolbox_en.register("Individual_en", tools.initCycle, creator.Individual_en,
                        (toolbox_en.attr_ratio, toolbox_en.attr_c), n=1)

    # Population definition
    toolbox_en.register("population", tools.initRepeat, list, toolbox_en.Individual_en)

    # -------------------------------------------------------------------------------- Mutation function -- #
    def mutate_en(individual):

        # select which parameter to mutate
        gene = random.randint(0, len(p_params) - 1)

        if gene == 0:
            individual[0] = random.choice(p_params['ratio'])
        elif gene == 1:
            individual[1] = random.choice(p_params['c'])

        return individual,

    # ------------------------------------------------------------------------------ Evaluation function -- #
    def evaluate_en(eva_individual):

        # output of genetic algorithm
        chromosome = {'ratio': eva_individual[0], 'c': eva_individual[1]}

        # evaluation with fitness metric for classification model
        if p_type == 'classification':

            # model results
            model = logistic_reg(p_data=p_data, p_params=chromosome, p_model=p_model, p_iter=p_iter)

            # fitness measure
            model_fit = model['metrics']['auc']

            # always return a tupple
            return model_fit,

        # evaluation with fitness metric for regression model
        elif p_type == 'regression':
            
           # model results
            model = ols_reg(p_data=p_data, p_params=chromosome, p_model=p_model, p_iter=p_iter)

            # Fitness measure
            model_fit = model['metrics']['r2']

            # always return a tupple
            return model_fit,
        
        # error in type of model
        else:
            return 'error: invalid type of model'

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


# Prueba de heterodasticidad de los residuos


def check_hetero(param_data):
    """
    Funcion que verifica si los residuos de una estimacion son heterosedasticos
    Parameters
    ---------
    param_data: DataFrame: DataFrame que contiene residuales
    ---------
    norm: boolean: indica true si los datos presentan heterodasticidad, o false si no la presentan.
    Debuggin
    ---------
    check_hetero(datos_residuales)
    """
    # arch test
    heterosced = het_arch(param_data)
    alpha = .05  # intervalo de 95% de confianza
    # si p-value menor a alpha se concluye que no hay heterodasticidad
    heter = True if heterosced[1] > alpha else False
    return heter


def variables(data_y, data_x, N):
    frames_x = np.array_split(data_x, N)
    frames_y = np.array_split(data_y, N)
    dict_all = {}
    for frame in range(0, N):
        x_train, x_test, y_train, y_test = train_test_split(frames_x[frame], frames_y[frame],
                                                            test_size=.2, shuffle=False)
        dict_all["frame"+str(frame + 1)] = [x_train, x_test, y_train, y_test]
    return dict_all


def actual_test(dict_split, params, definition,reg_type):
    data_reg = {'x_data': dict_split[0], 'y_data': dict_split[2]["co"]}
    reg = ols_reg(p_data=data_reg, p_params=params, p_model=reg_type, p_iter=100000)
    pred_test = reg["model"].predict(dict_split[1])
    residuales = reg['results']['y_data'] - reg['results']['y_data_p']
    #Pruebas de residuales
    vis.residual(residuales=residuales)
    vis.histograma(residuales)
    #heterocedasticidad
    hetero = check_hetero(residuales)
    #jungbox
    ljung = acorr_ljungbox(residuales, lags=7,return_df=True)
    #normality
    name = ["Jarque-Bera", "Chi2 two tail prob", "Skew","Kurtosis"]
    test = sms.jarque_bera(residuales)
    jarquebera = lzip(name,test)
    rss = sum((dict_split[3]["co"]-pred_test) ** 2)
    return rss, data_reg, reg, definition, hetero, ljung, jarquebera


def busqueda_en_train(dict_all, n):
    best_of_each = []
    for split in range(0, n):
        dict_split = dict_all["frame" + str(split + 1)]

        #normal
        rss, data_reg, model, definition, hetero, jlung, jarquebera = actual_test(dict_split,
                                                                                  dt.params_reg, "normal", "ols")
        best_of_each.append([definition, rss, split, hetero, jlung, jarquebera])
        #best_of_each.append([definition, rss, split, hetero, jlung, model])
        #regularización
        #data_op = optimization(p_data=data_reg, p_type='regression', p_params=dt.params_reg,
         #                      p_model='ols_en', p_iter=100000)
        #new_params = {"ratio": data_op["population"][0][0], "c": data_op["population"][0][1]}
        #rss, data_reg, model, definition, hetero, jlung = actual_test(dict_split, new_params, "normal + regularizacion", "ols_en")
        #best_of_each.append([definition, rss, split, hetero, jlung, model])
        #simbolica + regularización
        #np.random.seed(546)
        #symbolic, table = symbolic_features(p_x=data_reg['x_data'], p_y=data_reg['y_data'])
        #dict_split[0] = pd.DataFrame(symbolic['fit'], index=data_reg['x_data'].index)
        #xtest = pd.DataFrame(np.hstack((dict_split[1], symbolic["model"].transform(dict_split[1]))), index=dict_split[1].index)
        #dict_split[1] = xtest
       # data_reg_sim = {'x_data': dict_split[0], 'y_data': dict_split[2]["co"]}
        # with elastic net regularization
        #data_op = optimization(p_data=data_reg_sim, p_type='regression', p_params=dt.params_reg,
         #                      p_model='ols_en', p_iter=100000)
        # in sample results
        #new_params = {"ratio": data_op["population"][0][0], "c": data_op["population"][0][1]}
        #rss, data_reg, model, definition, hetero, jlung = actual_test(dict_split, new_params, "simbolica + regularización", "ols_en")
        #best_of_each.append([definition, rss, split, hetero, jlung, model])
    return best_of_each
