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
    text = text.lower()
    text = text.replace("5g", "fuenfg")
    text = text.replace("co2", "cozwei")
    tokens = gensim.utils.simple_preprocess(text)
    for ind, token in enumerate(tokens):
        if token == "fuenfg":
            tokens[ind] = "5g"
        if token == "cozwei":
            tokens[ind] = "co2"
    return tokens

def train(model_name, language, vector_size, min_count, epochs, train_df, taxonomy_uri, train_channel):

    # split df in single news
    df_factchecks = pd.read_csv(os.path.join(train_channel, train_df))

    # location for storing the trained model.
    model_dir = os.environ['SM_MODEL_DIR']

    # prepare data for training
    stoplist = list(string.punctuation)
    stoplist += stopwords.words(language)
    stoplist += ['archiviert', 'archiviertes', 'angeblich', 'angebliche', 'facebook', 'seien', 'sei', 'facebookpost', 'behauptung', 'sozialen', 'netzwerken', 'heißt', 'verbreitet', 'mögliche', 'höher', 'wort', 'teils', 'kaum', 'lassen', 'ersten', 'heraus', 'vergleich', 'simpsons', 'behauptet', 'etwa', 'worden', 'immer', 'post', 'sehen', 'kursiert', 'geteilt', 'hätten', 'sollen', 'zeigen', 'derzeit', 'seit', 'wurde', 'schon', 'mehr', 'zwei', 'gibt', 'dabei', 'steht', 'zeigt', 'sic', 'wegen', 'viele', 'netz', 'posting', 'video', 'gesagt', 'internet', 'artikel', 'nutzer', 'jahr', 'beitrag', 'macht', 'sharepic', 'gebe', 'zusammenhang', 'dafür', 'text', 'ab', 'jahren', 'kursieren', 'mann', 'frau', 'überschrift', 'laut', 'seite', 'de', 'zeige', 'wer', 'demnach', 'ende', 'prozent', 'wurden', 'mehrere', 'zudem', 'darin', 'suggeriert', 'zahlen', 'beleg', 'millionen', 'denen', 'beim', 'müssen', 'bereits', 'drei', 'darauf', 'online', 'jahre', 'geht', 'august', 'mehreren', 'beispiel', 'bekommen', 'welt', 'behauptungen', 'neue', 'land', 'stadt', 'oktober', 'erklärt', 'gefährlich', 'sogar', 'belegen', 'gar', 'heute', 'webseite', 'könne', 'schreibt', 'angebliches', 'mal', 'aktuell', 'angeblichen', 'behaupten', 'eindämmung', 'zufolge','jedoch', 'aussage', 'zugeschrieben', 'geld', 'eindruck', 'positiv', 'daten','zahl', 'berichtet', 'märz', 'davon', 'november', 'neben', 'bestätigt', 'leben', 'weniger', 'http', 'neuen', 'schutz', 'aktuellen', 'gab', 'halten', 'oft', 'vermeintliche', 'ganz', 'anfang', 'tag', 'aussagen', 'könnten', 'darunter', 'dezember', 'grund', 'erhalten', 'kommt', 'logo', 'unterstellt', 'erweckt', 'erst', 'wochen', 'gegeben', 'daher', 'zeit', 'gut', 'tage', 'sowie', 'rund', 'gestellt', 'screenshot', 'mitarbeiter', 'user', 'zweiten', 'april', 'geben', 'grafik', 'videos', 'fordert', 'häufig', 'außerdem','lautet', 'beiträgen', 'vermeintlichen', 'finden', 'gemacht', 'stellt', 'posts', 'personen', 'berichten', 'angegeben', 'verbreiten', 'arzt', 'präsident', 'bevölkerung', 'infektion', 'com', 'ländern', 'präsidenten', 'krise', 'bürger', 'rede', 'berichten', 'angegeben', 'verbreiten', 'fall', 'dpaq', 'runde', 'soziale', 'gebracht', 'worte', 'quelle', 'bringen', 'lesen', 'lange', 'tatsächlich', 'erneut', 'statt', 'september', 'weltweit', 'vielen', 'januar', 'nachdem', 'warnt', 'große', 'versucht', 'beweise', 'teilen', 'hingegen', 'juli', 'zusammen', 'luft', 'schreiben', 'wissen', 'per', 'monaten', 'beweis', 'anhand', 'dürfen', 'vermeintlich', 'twitter', 'blog', 'falsch', 'mitte', 'aufschrift', 'februar', 'trägt', 'kurz', 'cookies', 'browser']
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
    # save the model
    with open(os.path.join(model_dir, model_name+'-model'), 'wb') as out:
        pickle.dump(model, out)

    # download taxonomy
    os.chdir('/tmp')
    taxonomy_file_name = "taxonomy.json"
    destbucket, destkey = taxonomy_uri.split('/',2)[-1].split('/',1)
    s3_client.download_file(taxonomy_file_name, destbucket, destkey)
    # read taxonomy
    with open(taxonomy_file_name, "r") as f:
        taxonomy_json = json.load(f)
    # report similarities for tag terms
    report = {}
    report["similarity-threshold"] = taxonomy_json["similarity-threshold"]
    # create list of terms already considered in taxonomy
    term_list = []
    for category in taxonomy_json:
        for tag in taxonomy_json[category]:
            if (category == "similarity-threshold") or (category == "excluded-terms"):
                continue
            for term in taxonomy_json[category][tag]:
                if term not in term_list:
                    term_list.append(term)
    # find most similar words for tag terms
    for category in taxonomy_json:
        if (category == "similarity-threshold") or (category == "excluded-terms"):
            continue
        for tag in taxonomy_json[category]:
            tag_terms = {}
            for term in taxonomy_json[category][tag]:
                tokens = text_preprocess(term)
                for token in tokens:
                    if token in model.wv.vocab:
                        token_similarities = {}
                        tuple_list = model.wv.most_similar(token)
                        for similarity_tuple in tuple_list:
                            # Is word considered in tagging detection?
                            if similarity_tuple[0] in taxonomy_json["excluded-terms"]:
                                continue
                            # Is word already used as tag term?
                            if similarity_tuple[0] in term_list:
                                continue
                            token_similarities[similarity_tuple[0]] = similarity_tuple[1]
                        tag_terms[token] = token_similarities
            report[tag] = tag_terms
    # write taxonomy
    report_file_name = "taxonomy_report.json"
    with open(report_file_name, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    reportkey = '/'.join(destkey.split('/')[0:-1])
    reportkey = os.path.join(reportkey, report_file_name)

    print("Uploading taxonomy report into bucket {} with key {}".format(destbucket, reportkey))
    s3.upload_file(report_file_name, destbucket, reportkey)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # required hyperparameters and other parameters
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--language', type=str, default="de")
    parser.add_argument('--vector_size', type=int, default=40)
    parser.add_argument('--min_count', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_df', type=str, default="news_de.csv")
    parser.add_argument('--taxonomy_uri', type=str, default="s3://factchecks-dev/tagging/category-tag-terms-de.json")

    # This is a way to pass additional arguments when running as a script
    # and use sagemaker-containers defaults to set their values when not specified.
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # parameters which are used only for inference

    args = parser.parse_args()
    
    train(args.model_name, args.language, args.vector_size, args.min_count, args.epochs, args.train_df, args.taxonomy_uri, args.train)
 
