import datetime
import argparse
import dateutil.parser
import sys
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import requests

base_url = 'http://127.0.0.1:8082/api'

date = datetime.datetime.now()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

r = requests.get(base_url + '/races?limit=100&date=' + date.strftime('%Y-%m-%d'))

races = r.json().get('results')

def race_title(race):
    date = dateutil.parser.isoparse(race['start_at'])
    return 'R' + str(race['session']['num']) + 'C' + str(race['num']) + ' ' + date.strftime('%H:%M') + ' ' + race['sub_category']

options = [{'label': race_title(r), 'value': r['id']} for r in races]

app.layout = html.Div(children=[
    dcc.Dropdown(id='race-dropdown', options=options),

    html.P(id='placeholder'),

    dcc.Graph(
        id='odds-graph'
    ),

    dcc.Interval(
        id='interval-component',
        interval=30*1000, # in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('odds-graph', 'figure'), [Input('race-dropdown', 'value'), Input('interval-component', 'n_intervals'),])
def update_fig(race=None, interval=None):

    if race is None:
        return {}

    r = requests.get(base_url + '/odds?limit=1000&race=' + str(race))
    odds = r.json().get('results')

    if len(odds) == 0:
        return {}

    data = [{'num': o['player']['num'], 'ts': dateutil.parser.isoparse(o['date']), 'value': o['value']} for o in odds if o['value']<100]
    df = pd.DataFrame(data)

    fig = px.line(df, x="ts", y="value", color="num")

    return fig

app.run_server(debug=False, port=8050)
