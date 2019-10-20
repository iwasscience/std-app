
from dash.dependencies import Input
from dash.dependencies import Output
from dash.dependencies import Input, Output, State

import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from myproject import app

import pandas as pd
import numpy as np

from .ai import clf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

df = pd.read_csv('myproject/dashboard/student_data.csv')

subjects = ['math score', 'reading score', 'writing score']

def register_callback(app):

    @app.callback(Output('hist-score', 'figure'),
                  [Input('hist-dropdown', 'value')])
    def plot_hist(feature):

        fig = {'data': [

            go.Histogram(x=df[df['gender'] == gender][feature],
                         opacity=0.9, name=gender)
            for gender in df['gender'].unique()

        ],

            'layout':


                go.Layout(
            title='{} distribution by gender'.format(feature),
            xaxis={'title': feature},
            yaxis={'title': 'Count'},
            barmode='overlay',
            hovermode='closest'

        )



        }

        return fig


    @app.callback(Output('scatter-score', 'figure'),
                  [Input('xaxis-scatter', 'value'),
                   Input('yaxis-scatter', 'value')])
    def plot_scatter(x_feature, y_feature):

        fig = {'data': [

            go.Scatter(

                x=df[x_feature],
                y=df[y_feature],
                opacity=0.9,
                mode='markers',
                marker=dict(
                    color='rgba(135, 206, 250, 0.5)',
                    size=14,
                    line=dict(
                        color='MediumPurple',
                        width=2
                    )
                )
            )





        ], 'layout':

            go.Layout(
            title='{} by {}'.format(x_feature, y_feature),
                xaxis={'title': x_feature},
                yaxis={'title': y_feature},
                hovermode='closest')
        }

        return fig


    @app.callback(
        Output('predict-result', 'children'),
        [Input('predict-button', 'n_clicks')],
        [State('gender-check', 'value'),
         State('parent-dropdown', 'value'),
         State('lunch-dropdown', 'value'),
         State('test-dropdown', 'value')]
    )
    def get_prediction(n_clicks, gender_check, parent_degree, lunch_type, test_attend):

        # return '{},{},{},{}'.format(gender_check, parent_degree, lunch_type, test_attend)

        results = []

        if gender_check == 'male':
            results.append(1)
        else:
            results.append(0)

        degrees = ['parental level of education_some high school',
                   'parental level of education_high school',
                   "parental level of education_bachelor's degree",
                   "parental level of education_master's degree",
                   "parental level of education_some college"]

        for degree in degrees:
            if parent_degree == degree:
                results.append(1)
            else:
                results.append(0)

        if lunch_type == 'standard':
            results.append(1)
        else:
            results.append(0)

        if test_attend == 'not attended':
            results.append(1)
        else:
            results.append(0)

        prediction = clf.predict([results])

        return 'Model Prediction: The Student will perform {}.'.format(prediction[0])
