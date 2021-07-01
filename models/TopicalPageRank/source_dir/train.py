from __future__ import absolute_import

import sys
import time
import os
import argparse


import json
import pke 
import pandas as pd


def train(model_name, n_topics, n_datasets, language, normalization, train_df, train_channel):

    # split df in single news
    data_news_all = pd.read_csv(os.path.join(train_channel, train_df))
    if len(data_news_all) < n_datasets:
        data_news = data_news_all
    else:
        data_news = data_news_all.sample(n_datasets)
        data_news.reset_index(drop=True, inplace=True)
    print(data_news.head())

    for index, row in data_news.iterrows():
        file = open(train_channel+'/'+str(index)+".txt","w") 
        file.write(row["title"]) 
        file.close() 

    # location for storing the trained model.
    model_dir = os.environ['SM_MODEL_DIR']

    pke.utils.compute_lda_model(train_channel, model_dir+'/'+model_name+'-model', n_topics=n_topics, extension='txt', language=language, normalization=normalization)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # required hyperparameters and other parameters
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--n_topics', type=int, default=100)
    parser.add_argument('--n_datasets', type=int, default=5000)
    parser.add_argument('--language', type=str, default="de")
    parser.add_argument('--normalization', type=str, default="None")
    parser.add_argument('--train_df', type=str, default="news_de.csv")

    # This is a way to pass additional arguments when running as a script
    # and use sagemaker-containers defaults to set their values when not specified.
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # parameters which are used only for inference
    parser.add_argument('--grammar', type=str)
    parser.add_argument('--max_count', type=int)
    parser.add_argument('--window', type=int)

    args = parser.parse_args()
    
    train(args.model_name, args.n_topics, args.n_datasets, args.language, args.normalization, args.train_df, args.train)
 
