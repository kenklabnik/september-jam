# Import necessary libraries

from decimal import Decimal, localcontext, ROUND_DOWN
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
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

ROW_PADDING_STYLE = {
    "padding": "8px"
}

GRAPH_HEIGHT = 600
GRAPH_WIDTH = 900

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

# list of features for dropdown
feature_list = ['Log GDP per capita',
    'Social support', 'Healthy life expectancy at birth',
    'Freedom to make life choices', 'Generosity',
    'Perceptions of corruption', 'Positive affect', 'Negative affect', 'Life Ladder']
    
# load dataset
data = pd.read_csv("World-happiness-report-updated_2024.csv",  encoding="latin1")

# standard-scaled version of data for charts where that's appropriate
scaler = StandardScaler()
scaled_data = pd.DataFrame(
    scaler.fit_transform(data[feature_list]),
    columns=feature_list,
    index=data.index)
scaled_data['Country name'] = data['Country name']
scaled_data['year'] = data['year']

# create correlation heatmap for later display
def display_corr_heatmap():
    corr = data[["Life Ladder", "Log GDP per capita", "Social support",
           "Healthy life expectancy at birth", "Freedom to make life choices",
           "Generosity", "Perceptions of corruption",
           "Positive affect", "Negative affect"]].corr()
    
    corr_truncated = corr.map(lambda x: truncate(x, 2))
    
    return px.imshow(corr_truncated, text_auto=True, aspect="auto")

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
    ],  id="title-display-area",
        style=ROW_PADDING_STYLE
    ),
    dbc.Row([
        dbc.Col([
            html.H2("Happiness of Select Countries Over Time", className="text-center", style=SUBHEADER_STYLE),
            dcc.Dropdown(
                id="sample-country-dropdown",
                options=[{'label': country, 'value': country} for country in sorted(data['Country name'].unique())],
                value=countries,
                multi=True,
                placeholder="Pick your countries here",
                style=DROPDOWN_STYLE
            ),
            dcc.Graph(id="sample-country-graph")
        ])
        ],  id="sample-country-display-area",
        style=ROW_PADDING_STYLE
    ),
    dbc.Row([
        dbc.Col([
            html.H2("Full Stats By Country", className="text-center", style=SUBHEADER_STYLE),
            html.P("""Note that these stats are standardized for a better viewing experience, which means the values were compressed so that the 
                   mean is zero and the standard deviation is one. This is most obvious with GDP and life expectancy information.""",
                   className="text-start fs-5", style=TEXT_BODY_STYLE),
            dcc.Dropdown(
                id="full-stats-country-dropdown",
                options=[{'label': country, 'value': country} for country in sorted(data['Country name'].unique())],
                value="Canada",
                placeholder="Pick a country",
                style=DROPDOWN_STYLE
            ),
            dcc.Dropdown(
                id="full-stats-features-dropdown",
                options=[{'label': feature, 'value': feature} for feature in feature_list],
                value=feature_list,
                multi=True,
                placeholder="Pick features to investigate",
                style=DROPDOWN_STYLE
            ),
            dcc.Graph(id="full-stats-graph")
        ])
    ],  id="full-stats-display-area",
        style=ROW_PADDING_STYLE
        ),
    dbc.Row([
        dbc.Col([
            html.H2("Correlation Heatmap", className="text-center", style=SUBHEADER_STYLE),
            html.P("""
                   See the degree to which different features are correlated here. A 1 indicates a full 1:1 correlation, a -1 indicates inverse correlation, 
                   a 0 indicates no correlation, and decimal values indicate correlations that are weak when close to 0 and strong when far from 0.
                   """,
                   className="text-start fs-5", style=TEXT_BODY_STYLE),
            #nested column helps sort out a default size issue
            dbc.Col([
                dcc.Graph(id="correlation-heatmap",
                figure = display_corr_heatmap())
            ])
        ])
    ],  id="corr-heatmap-display-area",
        style=ROW_PADDING_STYLE
    ),
    dbc.Row([
        dbc.Col([
            html.H2("Feature Importance", className="text-center", style=SUBHEADER_STYLE),
            html.P("""
                    We trained some machine learning models on this data, using pre-2019 data as our training features and target and post-2019 data as our test 
                   features and target. Our best model, a CatBoost gradient boosting regressor, achieved a mean absolute error of about 0.32. This chart shows its SHAP 
                   values. SHAP is "Shapley Additive Explanations", a metric that scores which features contributed most to the model's predictions. 
                   """, className="text-start fs-5", style=TEXT_BODY_STYLE)
        ]),
        dbc.Col([
            html.Img(src=app.get_asset_url('catboost_shap_graph.png'), style={"text-align": "center"})
        ])
    ],  id="shap-display-area",
        style=ROW_PADDING_STYLE)
],
    fluid=True
) #end of Container

# app reactivity logic

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

    fig.update_layout(
        height=GRAPH_HEIGHT,
        width=GRAPH_WIDTH
    )

    return fig

# callback that updates "Full Country Stats" graph
@app.callback(
    Output("full-stats-graph", "figure"),
    Input("full-stats-country-dropdown", "value"),
    Input("full-stats-features-dropdown", "value")
)
def update_full_stats_graph(country, features):
    if not country:
        return px.line(title="Select a country.")
    
    sample_country = scaled_data[scaled_data["Country name"].isin([country])]   

    fig = px.line(
        sample_country,
        x='year',
        y=features,
        title="Happiness Trends Over Time: {}".format(country)
    )

    fig.update_layout(
        height=GRAPH_HEIGHT,
        width=GRAPH_WIDTH
    )

    return fig

# if you are looking for the correlation heatmap logic, it has been moved to above the app layout since a callback is not required
# same with the feature importance graph, which is a static image

# for Render
server = app.server

# Run the app
if __name__ == '__main__':
    #for local hosting
    #app.run(debug=True, port=7124)

    #for Render hosting
    app.run_server(debug=False)