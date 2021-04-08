from __future__ import absolute_import

import sys
import time
import os
import argparse


import json
import pandas as pd
from nltk.corpus import stopwords
import string
import gensim
import pickle


def text_preprocess(text):
    text = text.replace("5G", "fuenfg")
    text = text.replace("5g", "fuenfg")
    tokens = gensim.utils.simple_preprocess(text)
    for ind, token in enumerate(tokens):
        if token == "fuenfg":
            tokens[ind] = "5G"
    return tokens

def train(model_name, language, vector_size, min_count, epochs, train_df, train_channel):

    # split df in single news
    df_factchecks = pd.read_csv(os.path.join(train_channel, train_df))

    # location for storing the trained model.
    model_dir = os.environ['SM_MODEL_DIR']

    # prepare data for training
    stoplist = list(string.punctuation)
    stoplist += stopwords.words(language)
    documents_train = []
    for i, row in df_factchecks.iterrows():
        if 'claim_text' in row:
            tokens = text_preprocess(row["claim_text"])
            # Remove stop words
            words = [w for w in tokens if not w in stoplist]
            # For training data, add tags
            documents_train.append(gensim.models.doc2vec.TaggedDocument(words, [i]))

    model = gensim.models.doc2vec.Doc2Vec(  vector_size=vector_size, 
                                            min_count=min_count, 
                                            epochs=epochs)
    model.build_vocab(documents_train)
    model.train(documents_train, total_examples=model.corpus_count, epochs=model.epochs)
    # save_model(model, doc2vec_models_prefix+LC)
    # save the model
    with open(os.path.join(model_dir, model_name+'-model'), 'wb') as out:
        pickle.dump(model, out)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # required hyperparameters and other parameters
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--language', type=str, default="de")
    parser.add_argument('--vector_size', type=int, default=40)
    parser.add_argument('--min_count', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_df', type=str, default="factchecks_de.csv")

    # This is a way to pass additional arguments when running as a script
    # and use sagemaker-containers defaults to set their values when not specified.
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # parameters which are used only for inference

    args = parser.parse_args()
    
    train(args.model_name, args.language, args.vector_size, args.min_count, args.epochs, args.train_df, args.train)
 
