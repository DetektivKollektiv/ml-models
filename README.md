# Amazon SageMaker Multi-Model Endpoint for searching factchecks

With Amazon SageMaker multi-model endpoints, customers can create an endpoint that seamlessly hosts up to thousands of models. These endpoints are well suited to use cases where any one of a large number of models, which can be served from a common inference container, needs to be invokable on-demand and where it is acceptable for infrequently invoked models to incur some additional latency. For applications which require consistently low inference latency, a traditional endpoint is still the best choice.

At a high level, Amazon SageMaker manages the loading and unloading of models for a multi-model endpoint, as they are needed. When an invocation request is made for a particular model, Amazon SageMaker routes the request to an instance assigned to that model, downloads the model artifacts from S3 onto that instance, and initiates loading of the model into the memory of the container. As soon as the loading is complete, Amazon SageMaker performs the requested invocation and returns the result. If the model is already loaded in memory on the selected instance, the downloading and loading steps are skipped and the invocation is performed immediately.

For the inference container to serve multiple models in a multi-model endpoint, it must implement additional APIs in order to load, list, get, unload and invoke specific models. 

## Introduction to Multi Model Server (MMS)

[Multi Model Server](https://github.com/awslabs/multi-model-server) is an open source framework for serving machine learning models. It provides the HTTP frontend and model management capabilities required by multi-model endpoints to host multiple models within a single container, load models into and unload models out of the container dynamically, and performing inference on a specified loaded model.

MMS supports a pluggable custom backend handler where you can implement your own algorithm.

## ModelHandler defines a model handler for load and inference requests for the different models

Each supported model need to be considered in model_handler.py.

The model artifacts include among others a json file, e.g. TopicalPageRank-params.json which provides parameters to be used in the predictions.

### TopicalPageRank artifacts
The artifacts comprise the files:
TopicalPageRank-params.json
TopicalPageRank-model

### DocSim artifacts
The artifacts comprise the files:
DocSim-params.json
DocSim-model

# Training of models

## models_to_be_trained.json
In this file you can configure which models shall be trained.

## pipeline.yml
For training of the models access is required to the corresponding buckets, allow access on these buckets in S3Policy.

## Models
Supported models needs a training script train.py, inputData.json for specifying the location of training data and a json file specifying the parameters required for training and inference.
Be aware that training of each model should not last more than one hour as otherwise the code pipeline would fail.
Example of TopicalPageRank-params.json:
{
    "model_name": "TopicalPageRank",
    "grammar": "NP: {<ADJ>*<NOUN|PROPN>}", 
    "language": "de", 
    "normalization": "None",
    "n_topics": "100",
    "window": "10", 
    "max_count": "10",
    "train_df": "factchecks_de.csv"
}