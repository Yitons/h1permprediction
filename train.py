#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:03:17 2020

@author: xuel12
"""

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
from navbar import Navbar
import pickle
import pandas as pd
from constants import US_STATE_ABBREV
from sklearn.linear_model import LogisticRegression
import constants

base_path = constants.BASE_PATH
input_dir = constants.INPUT_DIR
header_dir = constants.HEADER_DIR
temp_dir = constants.TEMP_DIR
nav = Navbar()


def UsertrainH1B(filename):
    df = pd.read_csv(temp_dir+filename+'.csv', engine = 'python')
    df["PW_UNIT_OF_PAY"] = df["PW_UNIT_OF_PAY"].replace(constants.unit_map)
    selected_variables = ['CASE_STATUS',
                          'EMPLOYER_STATE',
                          'WORKSITE_STATE',
                          'JOB_CATEGORY',
                          'JOB_LEVEL',
                          'FULL_TIME_POSITION',
                          'PW_UNIT_OF_PAY',
                          'PW_WAGE_LEVEL',
                          'H-1B_DEPENDENT',
                          'WILLFUL_VIOLATOR']
    df = df[selected_variables]
    cate_column_name = [
        'EMPLOYER_STATE',
        'WORKSITE_STATE',
        'JOB_CATEGORY',
        'JOB_LEVEL',
        'FULL_TIME_POSITION',
        'PW_UNIT_OF_PAY',
        'PW_WAGE_LEVEL'
        , 'H-1B_DEPENDENT',
        'WILLFUL_VIOLATOR']
    data = pd.get_dummies(df, columns=cate_column_name)
    data = data.reset_index(drop=True)
    X_train = data.drop(['CASE_STATUS'], axis=1)
    y_train = data['CASE_STATUS']
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pickle_out = open(temp_dir + "H1B_USER_MODEL.pickle", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()

def USertrainPERM(filename):
    perm = pd.read_csv(temp_dir + filename + '.csv', engine='python')
    perm["JOB_INFO_WORK_STATE"] = perm["JOB_INFO_WORK_STATE"].map( US_STATE_ABBREV)
    perm["EMPLOYER_STATE"] = perm["EMPLOYER_STATE"].map( US_STATE_ABBREV)
    perm = perm.fillna("Unknown")
    select_columns = [
    "CASE_STATUS", "REFILE",
    "FW_OWNERSHIP_INTEREST","PW_LEVEL_9089",
    "JOB_INFO_WORK_STATE" , "JOB_INFO_EDUCATION" , "JOB_INFO_TRAINING",
    "JOB_INFO_ALT_FIELD" ,"JOB_INFO_JOB_REQ_NORMAL" ,"JOB_INFO_FOREIGN_LANG_REQ",
    "JOB_INFO_COMBO_OCCUPATION",
    "RECR_INFO_COLL_UNIV_TEACHER",
    "FW_INFO_BIRTH_COUNTRY", "CLASS_OF_ADMISSION",
    "FW_INFO_TRAINING_COMP" ]
    perm = perm[select_columns]
    cate_column_name = [
    "REFILE",
    "FW_OWNERSHIP_INTEREST","PW_LEVEL_9089",
    "JOB_INFO_WORK_STATE" , "JOB_INFO_EDUCATION" , "JOB_INFO_TRAINING",
    "JOB_INFO_ALT_FIELD" ,"JOB_INFO_JOB_REQ_NORMAL" ,"JOB_INFO_FOREIGN_LANG_REQ",
    "JOB_INFO_COMBO_OCCUPATION",
    "RECR_INFO_COLL_UNIV_TEACHER",
    "FW_INFO_BIRTH_COUNTRY", "CLASS_OF_ADMISSION",
    "FW_INFO_TRAINING_COMP" ]
    data = pd.get_dummies(perm, columns=cate_column_name)
    data = data.reset_index(drop=True)

    X_train = data.drop(['CASE_STATUS'], axis=1)
    y_train = data['CASE_STATUS']

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pickle_out = open(temp_dir + "PERM_USER_MODEL.pickle", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()

upload = dbc.Container([
        html.Div([
            # Specify directory,
            html.H4("Please specify the base directory"),
            dbc.Input(id="input-on-submit", placeholder=base_path, value=base_path, type="text"),
            html.Br(),
        ]),
        
        html.Div([
        # upload a new dataset out of default directory
        html.H4("Upload a new dataset"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
            style={
                "width": "60%",
                "height": "40px",
                "lineHeight": "40px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            max_size=-1,
            multiple=True,
        ),
        
        # File list
        html.Div([
            html.H4("File List"),
            html.Ul(id="file-list")
        ], style={'font-size': '12px',}
        ),
    ]),
])

body = dbc.Container(
    [

        # Uploading all files
        html.H4("Process dataset"),
        html.Div(
            [
                dbc.Button("Start/Stop processing", id="submit-data", n_clicks=0),
                dbc.Spinner(html.Div(id="submiting-data")),
            ]
        ),
        
        # Create Div to place a invisible element inside
        html.Div([
            dcc.Input(id='upload-status', value = 'stop'),
            dcc.Input(id = 'csvreader-status',value = -1),
            dcc.Input(id='combinecsv-status', value = 'wait'),
        ], style={'display': 'none'}
        ),

        # status indicators
        html.Div([        
            daq.Indicator(id='start-indicator',label="Files Uploaded",value=True,color='grey'),
        ], style={'width': '30%', 'display': 'inline-block'}
        ),
        html.Div([
            daq.Indicator(id='xlsx2csv-indicator',label="Parsing Files",value=True,color='grey'),
        ], style={'width': '30%', 'display': 'inline-block'}
        ),
        html.Div([        
            daq.Indicator(id='csvcombine-indicator',label="Combining Data",value=True,color='grey'),
        ], style={'width': '30%', 'display': 'inline-block'}
        ),

         
        # dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
        dbc.Progress(id="progress"),       
        html.Div(id='parsing status', children='wait for input data'),  # add a section to store and display output
        html.Br(),
  
    ],
    className="mt-4",
)

train = dbc.Container([
    html.Div([
        # training
        html.H4("Train dataset"),
        html.Div(
            [
                dbc.Button("Start/stop training", id="submit-training", n_clicks=0),
                dbc.Spinner(html.Div(id="submiting-training")),
            ]
        ),
        daq.Indicator(id='train-indicator',label="Training Done",value=True,color='grey'),
        html.Br(),
    ])
])

def Training():
    layout = html.Div(
    [
         nav,
         upload,
         body,
         train

    ],
    )
    return layout
