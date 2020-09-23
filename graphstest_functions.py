import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from joblib import dump, load

from scipy import stats


def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


class PredictionModel:
    def __init__(self):
        self.import_data()
        self.down_cast()
        self.clean_df()
        self.clean_lag()

    def downcast_dtypes(self, df):
        float_cols = [c for c in df if df[c].dtype == "float64"]
        int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
        df[float_cols] = df[float_cols].astype(np.float32)
        df[int_cols] = df[int_cols].astype(np.int16)
        return df

    def import_data(self):
        self.train = pd.read_csv('sales_train.csv')
        self.items = pd.read_csv('items.csv')
        self.item_cats = pd.read_csv('item_categories.csv')
        self.items_t = pd.read_csv('items_translated_text.csv')
        self.train_lag = pd.read_csv('month_lag_grouped.csv')
        self.train_lag_new = pd.read_csv('new_month_group.csv')

    def down_cast(self):
        self.train = self.downcast_dtypes(self.train)
        self.items = self.downcast_dtypes(self.items)
        self.train_lag = self.downcast_dtypes(self.train_lag)
        self.item_cats = self.downcast_dtypes(self.item_cats)
        self.train_lag_new = self.downcast_dtypes(self.train_lag_new)

    def clean_lag(self):
        self.train_lag_new = self.train_lag_new.dropna()
        #self.train_lag_new.drop(labels=['date_block_num'], axis=1, inplace=True)

        self.train_lag_new = self.train_lag_new[self.train_lag_new.item_price < 90000]
        self.train_lag_new = self.train_lag_new[self.train_lag_new.item_cnt_day < 999]

        # replaces the negative price item with the median item_price of all items with the id of 2973 and in shop id 32
        median = self.train_lag_new[(self.train_lag_new.shop_id == 32) & (self.train_lag_new.item_id == 2973) & (self.train_lag_new.date_block_num == 4) & (
                        self.train_lag_new.item_price > 0)].item_price.median()
        self.train_lag_new.loc[self.train_lag_new.item_price < 0, 'item_price'] = median

    def one_hot_encode_lag(self):
        self.train_lag_new['date_block_num'] = [('month ' + str(i)) for i in self.train_lag_new['date_block_num']]
        self.train_lag_new['shop_id'] = [('shop ' + str(i)) for i in self.train_lag_new['shop_id']]
        self.train_lag_new['item_category_id'] = [('item_category ' + str(i)) for i in
                                                  self.train_lag_new['item_category_id']]
        self.train_lag_new['item_id'] = [('item ' + str(i)) for i in self.train_lag_new['item_id']]

    def run_lag_model(self):
        self.x_lag = self.train_lag_new.iloc[:, :-1].values
        self.y_lag = self.train_lag_new.iloc[:, -1].values
        self.ct_lag = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1, 2, 3, 4])], remainder='passthrough')
        self.x_lag = self.ct_lag.fit_transform(self.x_lag)
        self.X_train_lag, self.X_test_lag, self.Y_train_lag, self.Y_test_lag \
            = train_test_split(self.x_lag, self.y_lag, test_size=0.2,random_state=0)
        self.regressor_lag = LinearRegression()
        self.regressor_lag.fit(self.X_train_lag, self.Y_train_lag)

    def clean_df(self):
        self.train = self.train.merge(self.items, on='item_id')
        self.train = self.train.drop(columns='item_name')
        self.train['date'] = pd.to_datetime(self.train['date'], format='%d.%m.%Y')

        # Removes outliers from train
        self.train = self.train[self.train.item_price < 90000]
        self.train = self.train[self.train.item_cnt_day < 999]

        median = self.train[
            (self.train.shop_id == 32) & (self.train.item_id == 2973) & (self.train.date_block_num == 4) & (
                    self.train.item_price > 0)].item_price.median()
        self.train.loc[self.train.item_price < 0, 'item_price'] = median

        train_cnt = self.train['item_cnt_day']
        self.train.drop(labels=['item_cnt_day'], axis=1, inplace=True)
        self.train.insert(6, 'item_cnt_day', train_cnt)

        self.train = pd.DataFrame(
            self.train.groupby(['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price'])
            ['item_cnt_day'].sum().reset_index())
        self.train.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)

        self.train['item_cnt_month'] = (self.train['item_cnt_month']
                                        .fillna(0)
                                        .clip(0, 20)  # NB clip target here
                                        .astype(np.float16))

        self.train_grouped_month = self.train.copy()

    def one_hot_encode(self):
        # Changes numerical, categorical features into strings to properly be represented as categorical in onehotencoding
        # nominal intergers can not be converted to binary encoding, convert to string
        self.train['date_block_num'] = [('month ' + str(i)) for i in self.train['date_block_num']]
        self.train['shop_id'] = [('shop ' + str(i)) for i in self.train['shop_id']]
        self.train['item_category_id'] = [('item_category ' + str(i)) for i in self.train['item_category_id']]
        self.train['item_id'] = [('item ' + str(i)) for i in self.train['item_id']]

    def run_model(self):
        self.x = self.train.iloc[:, :-1].values
        self.y = self.train.iloc[:, -1].values
        self.ct = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1, 2, 3])], remainder='passthrough')
        self.x = self.ct.fit_transform(self.x)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=0)
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.Y_train)

    def get_z_list(self, shop_id_num, item_id_num, month):
        self.item_cat = self.items.loc[self.items['item_id'] == item_id_num, ['item_category_id']].values[0][0]
        # item_cat = items[items['item_id'] == item_id_num]['item_category_id'].values
        self.prices = self.train.loc[self.train['item_id'] == 'item ' + str(item_id_num), ['item_price']].values
        self.price = (stats.mode(self.prices))[0][0][0]

        return ['month ' + str(month), 'shop ' + str(shop_id_num), 'item_category ' + str(self.item_cat),
                'item ' + str(item_id_num), self.price]

    def get_z_list_lag(self, shop_id_num, item_id_num):
        # item_cat = items.loc[items['item_id'] == item_id_num, ['item_category_id']].values[0][0]
        self.item_cat = self.items[self.items['item_id'] == item_id_num]['item_category_id'].values
        self.prices = self.train.loc[self.train['item_id'] == item_id_num, ['item_price']].values
        self.price = (stats.mode(self.prices))[0][0][0]
        self.date_num = 34
        new_pd = self.train_lag_new.loc[self.train_lag_new['date_block_num'] == self.date_num - 1].loc[
            self.train_lag_new['shop_id'] == shop_id_num].loc[self.train_lag_new['item_id'] == item_id_num]
        new_pd2 = self.train_lag_new.loc[self.train_lag_new['date_block_num'] == date_num - 2].loc[
            self.train_lag_new['shop_id'] == shop_id_num].loc[self.train_lag_new['item_id'] == item_id_num]
        new_pd3 = self.train_lag_new.loc[self.train_lag_new['date_block_num'] == date_num - 3].loc[
            self.train_lag_new['shop_id'] == shop_id_num].loc[self.train_lag_new['item_id'] == item_id_num]
        new_pd4 = self.train_lag_new.loc[self.train_lag_new['date_block_num'] == date_num - 4].loc[
            self.train_lag_new['shop_id'] == shop_id_num].loc[self.train_lag_new['item_id'] == item_id_num]
        new_pd5 = self.train_lag_new.loc[self.train_lag_new['date_block_num'] == date_num - 5].loc[
            self.train_lag_new['shop_id'] == shop_id_num].loc[self.train_lag_new['item_id'] == item_id_num]
        # print(len(new_pd['date_block_num']))
        if len(new_pd['shop_id']) > 0:
            mon1 = self.train_lag_new['item_cnt_day'][new_pd.index[0]]
        else:
            mon1 = 0

        if len(new_pd2['shop_id']) > 0:
            mon2 = self.train_lag_new['item_cnt_day'][new_pd2.index[0]]
        else:
            mon2 = 0

        if len(new_pd3['shop_id']) > 0:
            mon3 = self.train_lag_new['item_cnt_day'][new_pd3.index[0]]
        else:
            mon3 = 0

        if len(new_pd4['shop_id']) > 0:
            mon4 = self.train_lag_new['item_cnt_day'][new_pd4.index[0]]
        else:
            mon4 = 0

        if len(new_pd5['shop_id']) > 0:
            mon5 = self.train_lag_new['item_cnt_day'][new_pd5.index[0]]
        else:
            mon5 = 0

        self.z_lag = ['november', 'shop ' + str(shop_id_num), 'item_category ' + str(self.item_cat), 'item ' + str(item_id_num),
             self.price, mon1, mon2, mon3, mon4, mon5]



    def predict_month(self, shop_id_num, item_id_num, month):
        z = self.get_z_list(shop_id_num, item_id_num, month)

        # z = ['month 34', 'shop 55', 'item_category 76', 'item 492', 600.0]
        z = np.array(z, dtype=object).reshape(1, -1)
        z = self.ct.transform(z)
        z_pred = self.regressor.predict(z)

        return round(z_pred[0], 3)

    def create_one_shop_one_item_df(self, itemid, shopid):
        self.one_shop_df = self.train_grouped_month[self.train_grouped_month['shop_id'] == shopid]
        self.one_shop_one_item_df = self.one_shop_df[self.one_shop_df['item_id'] == itemid]


    def create_one_shop_df(self, shopid):
        self.one_shop_df = self.train_grouped_month[self.train_grouped_month['shop_id'] == shopid]

    def create_3d_scatter_fig(self):
        self.fig = px.scatter_3d(self.one_shop_one_item_df, x='date_block_num', y='item_price', z='item_cnt_month', color='item_price')

    def get_translated_name(self, itemid):
        self.t_name = self.items_t[self.items_t['item_id'] == itemid]['english_name']

    def get_valid_item_list(self, shopid):
        self.one_shop = self.train_grouped_month[self.train_grouped_month['shop_id'] == shopid]
        self.valid_items = list(self.one_shop.item_id.unique())

    def get_valid_shops_list(self):
        self.valid_shops = list(self.train_grouped_month.shop_id.unique())

    def convert_list_to_options_dict_items(self):
        self.list_of_dicts_items = []

        for item in self.valid_items:
            temp_dict = {'label': item, 'value': item}
            self.list_of_dicts_items.append(temp_dict)

    def convert_list_to_options_dict_shops(self):
        self.list_of_dicts_shops = []

        for item in self.valid_shops:
            temp_dict = {'label': item, 'value': item}
            self.list_of_dicts_shops.append(temp_dict)


sample_model = PredictionModel()

sample_model.one_hot_encode_lag()
sample_model.run_lag_model()


print(sample_model.train.head(5))

#sample_model.one_hot_encode()
#sample_model.run_model()

sample_shop_id = 55
sample_item_id = 492

sample_model.create_one_shop_df(sample_shop_id)
sample_model.create_one_shop_one_item_df(sample_shop_id, sample_item_id)
sample_model.create_3d_scatter_fig()

# gets the list of items that are sold in a particular shop to put into the drop down menu
sample_model.get_valid_item_list(sample_shop_id)
sample_model.convert_list_to_options_dict_items()

# gets a list of the valid shop_id's to display in the drop down menu
sample_model.get_valid_shops_list()
sample_model.convert_list_to_options_dict_shops()


# basic stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


# App layout
app.layout = html.Div([

    html.H1("Sales Forecasting", style={'text-align': 'center'}),
    html.H2("Enter Shop ID and Item ID to see next months predicted sales count for that item.",
            style={'text-align': 'center'}),

    # contains the graph
    html.Div([
        dcc.Graph(
            id='3d-scatter',
            figure={})]),
    html.Br(),

    html.H2([
        'Predicted Sales for Next Month: ',
        html.H2(
            id='prediction_count')], style={'text-align': 'center'}),

    # contains the shop id drop down menu
    html.Div(['Shop ID: ',
              dcc.Dropdown(
                  id='shop-dropdown',
                  options=sample_model.list_of_dicts_shops,
                  placeholder="Select a Shop ID (0-60)"
              )
              ]),

    # contains the item id drop down menu
    html.Div(['Item ID: ',
              dcc.Dropdown(
                  id='item-dropdown',
                  options=sample_model.list_of_dicts_items,
                  placeholder="Select an Item ID"
              )
              ]),

    # displays the item name
    html.Div([
        'Item Name: ',
        html.Div(
            id='item_name')]),

    # contains the submit button
    html.Div(html.Button(id='submit-button-state', n_clicks=0, children='Show Sales Graph')),
    html.Br(),

    html.Div(html.Button(id='predict-button-state', n_clicks=0, children='Predict Next Month')),
    html.Br(),

])

# Connects the selected shop_id to the item_id drop down with valid item_id's
@app.callback(
    Output(component_id='item-dropdown', component_property='options'),
    Input('shop-dropdown', 'value')
)
def update_dropdown_option(shop_id_from_dropdown):
    sample_model.get_valid_item_list(shop_id_from_dropdown)
    sample_model.convert_list_to_options_dict_items()
    return sample_model.list_of_dicts_items


# Connect the Plotly graphs with Dash drop down Components
@app.callback(
    [Output(component_id='3d-scatter', component_property='figure'),
     Output(component_id='item_name', component_property='children')],
    [Input('submit-button-state', 'n_clicks')],
    [State("shop-dropdown", "value"),
     State("item-dropdown", "value")]
)
def update_graph(n_clicks, input_shop_id, input_item_id):
    sample_model.create_one_shop_one_item_df(input_item_id, input_shop_id)
    sample_model.create_3d_scatter_fig()
    sample_model.get_translated_name(input_item_id)
    return sample_model.fig, 'Item Name: '.join(sample_model.t_name)


@app.callback(
    Output(component_id='prediction_count', component_property='children'),
    [Input('predict-button-state', 'n_clicks')],
    [State("shop-dropdown", "value"),
     State("item-dropdown", "value")]
)
def predict(n_clicks, input_shop_id, input_item_id):
    sample_model.get_z_list_lag(input_shop_id, input_item_id)
    return sample_model.z_lag


# runs the whole thing
if __name__ == '__main__':
    app.run_server(debug=True)
