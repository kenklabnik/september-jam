# Import necessary libraries
import re
import os
import random
from decimal import Decimal, localcontext, ROUND_DOWN
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from scipy import stats as st
from scipy import stats as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# color constants
TITLE_COLOR = "#ffee51"
TEXT_COLOR = "#ddcc40"
BOX_COLOR = "#323255"

# style constants
HEADER_STYLE = {
     "color": TITLE_COLOR
}

SUBHEADER_STYLE = {
    "color": TITLE_COLOR,
    "margin": "10px"
}

TITLE_CARD_STYLE = {
    "background-color": BOX_COLOR,
    "rober-radius": "12px",
    "box-shadow": "0 4px 8px rgba(0,0,0,0.1)",
    "padding": "15px",        
}

TEXT_BODY_STYLE = {
    "color": TEXT_COLOR,
    "text-padding": "20px"
}

DROPDOWN_STYLE = {
    'background-color': '#f8f8f0',
    'color': '#1c1c2e',
    'border': '1px solid #2dd4bf',
    'border-radius': '4px',
    'padding': '5px'
}

# function for truncating floats later, from StackOverflow:
def truncate(number, places):
    if not isinstance(places, int):
        raise ValueError("Decimal places must be an integer.")
    if places < 1:
        raise ValueError("Decimal places must be at least 1.")
    # If you want to truncate to 0 decimal places, just do int(number).

    with localcontext() as context:
        context.rounding = ROUND_DOWN
        exponent = Decimal(str(10 ** - places))
        return Decimal(str(number)).quantize(exponent).to_eng_string()
    
# default countries for sample line graph
countries = ["Finland", "India", "United States", "Afghanistan"]
    
# load dataset
data = pd.read_csv("World-happiness-report-updated_2024.csv",  encoding="latin1")

# TODO preprocess data

# instantiate Plotly Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

# app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H1("HAPPINESS REPORT",
                            className="text-center mt-4",
                            style=HEADER_STYLE),
                    html.P(
                        """
                        Ever wondered which countries are the happiest? Explore the World Happiness Report,
                        which contains data tracking the happiness of people from various countries from 2005-2023,
                        and see for yourself in this Plotly Dash-powered dashboard!
                        """,
                        className="text-start fs-5", style=TEXT_BODY_STYLE
                    )
                ]),
            style=TITLE_CARD_STYLE, id="title-card")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H2("Happiness of Select Countries Over Time", className="text-center", style=SUBHEADER_STYLE),
            dcc.Dropdown(
                id="sample-country-dropdown",
                options=[{'label': country, 'value': country} for country in sorted(data['Country name'].unique())],
                value=countries,
                multi=True,
                placeholder="Or pick your own countries here...",
                style=DROPDOWN_STYLE
            ),
            dcc.Graph(id="sample-country-graph")
        ])
    ])
]) #end of Container

# app reactivity logic
# TODO any number of graphs we can come up with

# callback that updates "Happiness of Select Countries Over Time" graph
@app.callback(
    Output("sample-country-graph", "figure"),
    Input("sample-country-dropdown", "value")
)
def update_sample_graph(countries):
    if not countries:
        return px.line(title="Select at least one country.")
    
    sample_countries = data[data["Country name"].isin(countries)]

    displayed = (
        sample_countries
        .groupby(['year', 'Country name'])['Life Ladder']
        .mean()
        .reset_index()
    )

    fig = px.line(
        displayed,
        x='year',
        y='Life Ladder',
        color='Country name',
        title="Happiness Trends Over Time",
        labels={'Life Ladder': 'Life Ladder Score', 'year': 'Year', 'Country name': "Country"}
    )

    return fig






# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=7124)