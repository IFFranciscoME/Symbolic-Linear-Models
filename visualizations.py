
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Multiple Linear Regression Regularization with Elastic Net for future prices prediction    -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: IFFranciscoME                                                                               -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/FuturesElasticNet                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import requests
import plotly.io as pio
pio.renderers.default = "browser"

# ----------------------------------------------------------------- GIST 1: Data Visualization in Python -- #
base = "https://gist.githubusercontent.com/IFFranciscoME/9b7b842ed368e9118c5ef09bb9b228cf/raw/"
gist = "f39623784704f556a49486bae2e9335f6d6da9e7/gist_p_data_vis"
vs = {}
exec(requests.get(base).content, vs)
