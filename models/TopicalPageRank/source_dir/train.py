from __future__ import absolute_import

import sys
import time
import os
import argparse


import json
import pke 
import pandas as pd


def train(params_json, train_channel):

    # read mode parameters from json string
    model_params = json.load(params_json)
    # split df in single news
    data_news = pd.read_csv(os.path.join(train_channel, model_params['train_df']))
    print(data_news.head())

    for index, row in data_news.iterrows():
        file = open(train_channel+'/'+str(index)+".txt","w") 
        file.write(row["claim_text"]) 
        file.close() 

    # location for storing the trained model.
    model_dir = os.environ['SM_MODEL_DIR']

    pke.utils.compute_lda_model(train_channel, model_dir+'/'+model_params['model_name']+'-model', n_topics=model_params['n_topics'], extension='txt', language=model_params['language'], normalization=model_params['normalization'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # hyperparameters and other parameters are passed in a json-string
    parser.add_argument('--params_json', type=str)

    # This is a way to pass additional arguments when running as a script
    # and use sagemaker-containers defaults to set their values when not specified.
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    
    train(args.params_json, args.train)
 
