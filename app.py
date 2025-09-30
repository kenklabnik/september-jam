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

# constants
TITLE_COLOR = "#ffee51"
TEXT_COLOR = "#ddcc40"
BOX_COLOR = "#323255"

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
                            style={"color": TITLE_COLOR}),
                    html.P(
                        """
                        Ever wondered which countries are the happiest? Explore the World Happiness Report,
                        which contains data tracking the happiness of people from various countries from 2005-2023,
                        and see for yourself in this Plotly Dash-powered dashboard!
                        """,
                        className="text-start fs-5", style={"color": TEXT_COLOR, "text-padding": '40px'}
                    )
                ]),
            style={
            "background-color": BOX_COLOR,
            "rober-radius": "12px",
            "box-shadow": "0 4px 8px rgba(0,0,0,0.1)",
            "padding": "15px",
            })
        ])
    ])
])

# app reactivity logic
# TODO

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=7124)