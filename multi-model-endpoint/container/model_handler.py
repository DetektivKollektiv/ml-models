"""
ModelHandler defines a model handler for load and inference requests for factcheck models
"""
import pke
from nltk.corpus import stopwords
import gensim
import glob
import json
import logging
import os
import re
import string
from io import StringIO
import pickle

import pandas as pd
import numpy as np

class ModelHandler(object):
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.model = None
        self.model_type = None
        self.model_params = None

    def get_model_files_prefix(self, model_dir):
        """
        Get the model prefix name for the model artifacts (dependent on the model).

        :param model_dir: Path to the directory with model artifacts
        :return: prefix string for model artifact files
        """
        sym_file_suffix = "-params.json"
        checkpoint_prefix_regex = "{}/*{}".format(model_dir, sym_file_suffix) # Ex output: /opt/ml/models/resnet-18/model/*-symbol.json
        checkpoint_prefix_filename = glob.glob(checkpoint_prefix_regex)[0] # Ex output: /opt/ml/models/resnet-18/model/resnet18-symbol.json
        checkpoint_prefix = os.path.basename(checkpoint_prefix_filename).split(sym_file_suffix)[0] # Ex output: resnet18
        logging.info("Prefix for the model artifacts: {}".format(checkpoint_prefix))
        return checkpoint_prefix

    def read_model_params(self, model_dir, checkpoint_prefix):
        """
        Get the model params and return the list

        :param model_dir: Path to the directory with model artifacts
        :param checkpoint_prefix: Model files prefix name
        """
        params_file_path = os.path.join(model_dir, "{}-{}".format(checkpoint_prefix, "params.json"))
        if not os.path.isfile(params_file_path):
            raise RuntimeError("Missing {} file.".format(params_file_path))

        with open(params_file_path) as f:
            self.model_params = json.load(f)

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir") 

        # get the type of model
        checkpoint_prefix = self.get_model_files_prefix(model_dir)
        self.model_type = checkpoint_prefix
        # read the model parameters
        self.read_model_params(model_dir, checkpoint_prefix)

        # Load model
        try:
            if self.model_type == "TopicalPageRank" or self.model_type == "DocSim":
                self.model = os.path.join(model_dir, "{}-{}".format(checkpoint_prefix, "model")) # path to model
                if not os.path.isfile(self.model):
                    raise RuntimeError("Missing {} file.".format(self.model))
            else:
                logging.error("Model {} not supported!".format(self.model_type))
                raise RuntimeError("Model {} not supported!".format(self.model_type))
         
        except Exception as e:
            logging.error("Exception: {}".format(e))
            raise MemoryError

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        if self.model_type == "TopicalPageRank" or self.model_type == "DocSim":
            # Return the input text from the request
            text_list = []
            for idx, data in enumerate(request):
                # Read the bytearray from the input
                text = data.get('body').decode("utf-8") 
                text_list.append(text)
            return text_list
        else:
            logging.error("Model {} not supported!".format(self.model_type))
            raise RuntimeError("Model {} not supported!".format(self.model_type))

    def text_preprocess(self, text):
        text = text.replace("5G", "fuenfg")
        text = text.replace("5g", "fuenfg")
        tokens = gensim.utils.simple_preprocess(text)
        for ind, token in enumerate(tokens):
            if token == "fuenfg":
                tokens[ind] = "5G"
        return tokens

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in NDArray
        """
        stoplist = list(string.punctuation)
        language = self.model_params['language']
        if language == "de":
            language = "german"
        stoplist += stopwords.words(language)
        stoplist += ['archiviert', 'archiviertes', 'angeblich', 'angebliche', 'facebook', 'seien', 'sei', 'facebookpost', 'behauptung', 'sozialen', 'netzwerken', 'heißt', 'verbreitet', 'mögliche', 'höher', 'wort', 'teils', 'kaum', 'lassen', 'ersten', 'heraus', 'vergleich', 'simpsons', 'behauptet', 'etwa', 'worden', 'immer', 'post', 'sehen', 'kursiert', 'geteilt', 'hätten', 'sollen', 'zeigen', 'derzeit', 'seit', 'wurde', 'schon', 'mehr', 'zwei', 'gibt', 'dabei', 'steht', 'zeigt', 'sic', 'wegen', 'viele', 'netz', 'posting', 'video', 'gesagt', 'internet', 'artikel', 'nutzer', 'jahr', 'beitrag', 'macht', 'sharepic', 'gebe', 'zusammenhang', 'dafür', 'text', 'ab', 'jahren', 'kursieren', 'mann', 'frau', 'überschrift', 'laut', 'seite', 'de', 'zeige', 'wer', 'demnach', 'ende', 'prozent', 'wurden', 'mehrere', 'zudem', 'darin', 'suggeriert', 'zahlen', 'beleg', 'millionen', 'denen', 'beim', 'müssen', 'bereits', 'drei', 'darauf', 'online', 'jahre', 'geht', 'august', 'mehreren', 'beispiel', 'bekommen', 'welt', 'behauptungen', 'neue', 'land', 'stadt', 'oktober', 'erklärt', 'gefährlich', 'sogar', 'belegen', 'gar', 'heute', 'webseite', 'könne', 'schreibt', 'angebliches', 'mal', 'aktuell', 'angeblichen', 'behaupten', 'eindämmung', 'zufolge','jedoch', 'aussage', 'zugeschrieben', 'geld', 'eindruck', 'positiv', 'daten','zahl', 'berichtet', 'märz', 'davon', 'november', 'neben', 'bestätigt', 'leben', 'weniger', 'http', 'neuen', 'schutz', 'aktuellen', 'gab', 'halten', 'oft', 'vermeintliche', 'ganz', 'anfang', 'tag', 'aussagen', 'könnten', 'darunter', 'dezember', 'grund', 'erhalten', 'kommt', 'logo', 'unterstellt', 'erweckt', 'erst', 'wochen', 'gegeben', 'daher', 'zeit', 'gut', 'tage', 'sowie', 'rund', 'gestellt', 'screenshot', 'mitarbeiter', 'user', 'zweiten', 'april', 'geben', 'grafik', 'videos', 'fordert', 'häufig', 'außerdem','lautet', 'beiträgen', 'vermeintlichen', 'finden', 'gemacht', 'stellt', 'posts', 'personen', 'berichten', 'angegeben', 'verbreiten', 'arzt', 'präsident', 'bevölkerung', 'infektion', 'com', 'ländern', 'präsidenten', 'krise', 'bürger', 'rede', 'berichten', 'angegeben', 'verbreiten', 'fall', 'dpaq', 'runde', 'soziale', 'gebracht', 'worte', 'quelle', 'bringen', 'lesen', 'lange', 'tatsächlich', 'erneut', 'statt', 'september', 'weltweit', 'vielen', 'januar', 'nachdem', 'warnt', 'große', 'versucht', 'beweise', 'teilen', 'hingegen', 'juli', 'zusammen', 'luft', 'schreiben', 'wissen', 'per', 'monaten', 'beweis', 'anhand', 'dürfen', 'vermeintlich', 'twitter', 'blog', 'falsch', 'mitte', 'aufschrift', 'februar', 'trägt', 'kurz']

        # Do some inference call to engine here and return output
        if self.model_type == "TopicalPageRank":
            pos = {'NOUN', 'PROPN', 'ADJ'} # the valid Part-of-Speeches to occur in the graph, e.g. {'NOUN', 'PROPN', 'ADJ'}
            grammar = self.model_params['grammar'] # the grammar for selecting the keyphrase candidates, e.g. "NP: {<ADJ>*<NOUN|PROPN>}"
            language = self.model_params['language'] # e.g. 'de'
            normalization = self.model_params['normalization'] # word normalization method, e.g. ‘stemming’
            window = self.model_params['window'] # edges connecting two words occurring in a window are weighted by co-occurrence counts, e.g. 10
            max_count = self.model_params['max_count'] # maximal count of highest scored keyphrases, which are returned
            logging.info("TopicalPageRank model_input: {}".format(model_input))
            # 1. create a TopicalPageRank extractor.
            extractor = pke.unsupervised.TopicalPageRank()
            phrases_list = []
            for text_input in model_input:
                # 2. load the input text
                extractor.load_document(input=text_input,
                                        language=language,
                                        normalization=normalization)            
                # 3. select the noun phrases as keyphrase candidates.
                extractor.candidate_selection(grammar=grammar)
                # 4. weight the keyphrase candidates using Single Topical PageRank.
                #    Builds a word-graph in which edges connecting two words occurring
                #    in a window are weighted by co-occurrence counts.
                extractor.candidate_weighting(window=window,
                                            pos=pos,
                                            lda_model=self.model,
                                            stoplist=stoplist)
                # 5. get the highest scored candidates as keyphrases
                keyphrases = extractor.get_n_best(n=max_count)
                logging.info("text_input: {}. keyphrases: {}".format(text_input, keyphrases))
                phrases_list.append(keyphrases)
            return phrases_list
        elif self.model_type == "DocSim":
            # load model
            with open(self.model, 'rb') as inp:
                model = pickle.load(inp)
            inference = []
            logging.info("DocSim model_input: {}".format(model_input))
            for text_input in model_input:
                logging.info("text_input: {}".format(text_input))
                # read string into dataframe
                df = pd.read_csv(StringIO(text_input), header=None)
                similarities = []
                for i, row in df.iterrows():
                    logging.info("{}. row: {}".format(i, row))
                    # prepare first document
                    logging.info("row.iloc[0]: {}".format(row.iloc[0]))
                    tokens = self.text_preprocess(row.iloc[0])
                    # Remove stop words
                    words1 = [w for w in tokens if not w in stoplist and w in model.wv.key_to_index]
                    logging.info("words1: {}".format(words1))
                    # prepare first document
                    logging.info("row.iloc[1]: {}".format(row.iloc[1]))
                    tokens = gensim.utils.simple_preprocess(row.iloc[1])
                    # Remove stop words
                    words2 = [w for w in tokens if not w in stoplist and w in model.wv.key_to_index]
                    logging.info("words2: {}".format(words2))
                    if (len(words1) == 0) or (len(words2) == 0):
                        similarities.append("0.00")
                        logging.warning("Word list is empty!")
                    else:
                        similarities.append(str(model.wv.n_similarity(words1, words2)))
                    logging.info("similarities: {}".format(similarities))
                inference.append(similarities)
                logging.info("inference: {}".format(inference))
            return inference
        else:
            logging.error("Model {} not supported!".format(self.model_type))
            raise RuntimeError("Model {} not supported!".format(self.model_type))

    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        if self.model_type == "TopicalPageRank" or self.model_type == "DocSim":
            return inference_output
        else:
            logging.error("Model {} not supported!".format(self.model_type))
            raise RuntimeError("Model {} not supported!".format(self.model_type))
        
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
 
