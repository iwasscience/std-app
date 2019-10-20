import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from myproject import app
import numpy as np

### ai ###

import pandas as pd
# for upload
import base64

# Model Imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

###

#datenbank importieren um die im dash layout zu benutzen? (forms etc)

#kek = html.Div('omegalul')

# np.random.seed(42)
# random_x = np.random.randint(1,101,100)
# random_y = np.random.randint(1,101,100)
#
#
# kek = html.Div([
#
#     html.Div([
#
#         dcc.Input(id='number-in',value=1, style={'fontSize':24}),
#         html.Button(id='submit-button', n_clicks=0, children='Submit Here', style={'fonzSize':24}),
#         html.H1(id='number-out')
#
#     ]),
#
#     html.Div([
#
#         dcc.Input(id='number-in_2',value=1, style={'fontSize':24}),
#         html.Button(id='submit-button_2', n_clicks=0, children='Submit Here', style={'fonzSize':24}),
#         html.H1(id='number-out_2')
#
#     ])
#
# ])


# ---------- AB HIER TEST AI --------- #

#df = pd.read_csv('~/Desktop/dash_directory/baby_steps/student_data.csv')


# Model
# ------------------------------------------------------------------------------------------------------------------------------------------------------- #
df = pd.read_csv('myproject/dashboard/student_data.csv')

# Preprocessing

df['total score'] = df['math score'] + df['reading score'] + df['writing score']

average = df['total score'].mean()
# No one scored exacly the mean, so we can leave the 'equal' operation out.
df['performance'] = np.where(df['total score'] > average, 'above average', 'below average')
df[['total score', 'performance']].head()

# One Hot Encoding predictors

df_encoded = pd.get_dummies(df, columns=['gender',
                                         'parental level of education',
                                         'lunch', 'test preparation course'], drop_first=True)

# Train Test Split

X = df_encoded.drop(['math score', 'reading score', 'writing score', 'total score',
                     'performance', 'race/ethnicity'], axis=1).values  # add values to transform df to np array
y = df['performance'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# RGB SVM + GSCV for Hyperparameter Tuning

# RBF SVM
svm = SVC(kernel='rbf')

# GridSearchCV
parameters = {'C': [0.1, 1, 10], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
clf = GridSearchCV(svm, parameters, cv=5)
clf.fit(X_train, y_train)

# Dashboard
# ------------------------------------------------------------------------------------------------------------------------------------------------------- #

subjects = ['math score', 'reading score', 'writing score']

kek = html.Div([

    html.H1('Student Performance  Dashboard',
            style={'padding': '10px', 'text-align': 'center',
                   'background': '#D3D3D3'}),

    html.Div([

        dcc.Dropdown(id='hist-dropdown',
                     options=[{'label': subject, 'value': subject} for subject in subjects],
                     placeholder='Select Feature Hist')

    ], style={'width': '40%', 'marginTop': '20px',
              'display': 'inline-block', 'marginRight': '100px',
              'marginLeft': '60px'}),

    html.Div([

        dcc.Dropdown(id='xaxis-scatter',
                     options=[{'label': subject, 'value': subject} for subject in subjects],
                     placeholder='Select Feature X-Axis')

    ], style={'width': '20%', 'marginTop': '20px', 'display': 'inline-block', 'marginRight': '10px'}),

    html.Div([

        dcc.Dropdown(id='yaxis-scatter',
                     options=[{'label': subject, 'value': subject} for subject in subjects],
                     placeholder='Select Feature Y-Axis')

    ], style={'width': '20%', 'marginTop': '20px', 'display': 'inline-block'}),


    html.Div([

        dcc.Graph(id='hist-score',
                  figure={'data': [
                      go.Histogram(x=df[df['gender'] == gender]['math score'],
                                   opacity=0.9, name=gender)
                      for gender in df['gender'].unique()

                  ],

                      'layout': go.Layout(

                      title=('score distribution by gender'),
                      xaxis={'title': 'Subject'},
                      yaxis={'title': 'Count'},
                      barmode='overlay',
                      hovermode='closest'
                  )

                  })

    ], style={'width': '49%', 'display': 'inline-block'}),

    html.Div([

        dcc.Graph(id='scatter-score',
                  figure={'data': [

                      go.Scatter(

                          x=df['math score'],
                          y=df['reading score'],
                          opacity=0.9,
                          mode='markers',
                          marker=dict(
                              size=14,
                              symbol='pentagon',
                              color='rgb(0,191,255)',
                              line=dict(width=2)
                          )



                      )

                  ],

                      'layout': go.Layout(

                      title=('subject by subject scatter plot')
                  )

                  })

    ], style={'width': '49%', 'display': 'inline-block'}),

    html.H2('Performance Prediction',
            style={'padding': '10px', 'text-align': 'center',
                   'background': '#D3D3D3'}),


    html.Div([

        #html.H3('Select your parental level of education', style={'marginLeft': '10px'}),

        dcc.Dropdown(id='parent-dropdown',
                     options=[
                         {'label': 'None', 'value': 'parental level of education_some high school'},
                         {'label': 'High School', 'value': 'parental level of education_high school'},
                         {'label': "Bachelor's degree",
                             'value': "parental level of education_bachelor's degree"},
                         {'label': "Master's Degree", 'value': "parental level of education_master's degree"},
                         {'label': 'Some College Degree', 'value': "some high school"},

                     ],
                     placeholder='Select a degree'
                     )

    ], style={'width': '20%', 'display': 'inline-block', 'marginTop': '10px'}),


    html.Div([


        dcc.Dropdown(id='lunch-dropdown',
                     options=[
                         {'label': 'Standard', 'value': 'standard'},
                         {'label': 'Free/Reduced', 'value': 'free/reduced'},


                     ],
                     placeholder='Select college lunch type'
                     )

    ], style={'width': '20%', 'display': 'inline-block', 'marginTop': '10px', 'marginLeft': '10px'}),

    html.Div([


        dcc.Dropdown(id='test-dropdown',
                     options=[
                         {'label': 'Attended', 'value': 'attended'},
                         {'label': 'not Attended', 'value': 'not attended'},


                     ],
                     placeholder='Select attendancy'
                     )

    ], style={'width': '20%', 'display': 'inline-block', 'marginTop': '10px', 'marginLeft': '10px'}),


    html.Div([


        dcc.RadioItems(id='gender-check',
                       options=[
                           {'label': 'Female', 'value': 'female'},
                           {'label': 'Male', 'value': 'male'},
                       ],
                       value='female',
                       )

    ],
        style={'display': 'inline-block',
               'marginLeft': '20px', 'width': '5%'}
    ),

    html.Div([

        html.Button(id='predict-button',
                    n_clicks=0,
                    children='Predict',
                    style={'fontSize': 24, 'marginTop': '20px'})

    ], style={'display': 'inline-block', 'width': '20px', 'marginLeft': '10px'}),

    html.Div([

        html.H3(id='predict-result')

    ], style={'marginTop': '20px'}),

    html.Div([

        dcc.Upload(id='upload-data',
                   children=html.Div([
                       'Drag and Drop'
                   ]))

    ])



])
