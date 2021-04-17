import argparse
import datetime
import sys
import os
import django

sys.path.append(os.path.abspath(os.path.join('.')))

print(sys.path)

#os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cataclop.settings")

from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

from cataclop.core.models import Odds, Race

parser = argparse.ArgumentParser()
parser.add_argument('R', type=int, help='Race session number (R)')
parser.add_argument('C', type=int, help='Race number (C)')

args = parser.parse_args()

date = datetime.now()

try:
    race = Race.objects.get(start_at__date=date.strftime('%Y-%m-%d'), session__num=args.R, num=args.C)
except ObjectDoesNotExist:
    sys.exit()

odds = Odds.objects.filter(player__race=race)

data = [{'num': o.num, 'ts': o.date, 'value': o.value} for o in odds]
df = pd.DataFrame(data)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

fig = px.line(df, x="ts", y="value", color="num")

app.layout = html.Div(children=[
    html.H1(children=str(race)),

    html.Div(children='''
        Evolution of players odds
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

def run():
    app.run_server(debug=True)