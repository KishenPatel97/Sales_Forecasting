import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from math import ceil
from itertools import cycle
from itertools import product
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from xgboost import plot_importance

import time
import sys
import gc
import pickle

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


# Down casts the data entries from int64 to int32 and float64 to float32
# This reduces the size of the records by almost half. (From 134mb to 61mb)
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


# Import and clean data (importing csv into pandas)
# Read in .csv files into pandas data frames
#train = pd.read_csv('sales_train.csv')
#test = pd.read_csv('test.csv').set_index('ID')
#submission = pd.read_csv('sample_submission.csv')
#items = pd.read_csv('items.csv')
#item_cats = pd.read_csv('item_categories.csv')
#shops = pd.read_csv('shops.csv')


# Calls the downcasting function
#train = downcast_dtypes(train)
#test = downcast_dtypes(test)
#submission = downcast_dtypes(submission)
#items = downcast_dtypes(items)
#item_cats = downcast_dtypes(item_cats)
#shops = downcast_dtypes(shops)

#train = train.merge(items, on='item_id')

# Removes outliers from train
#train = train[train.item_price < 90000]
#train = train[train.item_cnt_day < 999]

# replaces the negative price item with the median item_price of all items with the id of 2973 and in shop id 32
#median = train[(train.shop_id == 32) & (train.item_id == 2973) & (train.date_block_num == 4) & (
#            train.item_price > 0)].item_price.median()


app = dash.Dash(__name__)

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Sales Forecasting", style={'text-align': 'center'}),

    html.Div(['Shop ID: ',
              dcc.Input(id="input_shop_id", value='initial value', type="number", placeholder="Shop ID")]),
    html.Br(),
    html.Div(id='shop_id_output'),
    html.Br(),

    html.Div(['Item ID: ',
              dcc.Input(id="input_item_id", value='initial value', type="number", placeholder="Item ID")]),
    html.Br(),
    html.Div(id='item_id_output'),
    html.Br(),

    #dcc.Graph(id='shop_item_sales_graph', figure={})
])

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='shop_id_output', component_property='children')],
    [Output(component_id='item_id_output', component_property='children')],
    [Input("input_shop_id", "value")],
    [Input("input_item_id", "value")],

)
def update_outputs(input_shop_id, input_item_id):
    return ['Output Shop ID: {}'.format(input_shop_id),
            'Output Item ID: {}'.format(input_item_id)]
    #return " | ".join( (str(val) for val in vals if val))



# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)