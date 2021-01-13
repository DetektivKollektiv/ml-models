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

Example of TopicalPageRank-params.json:
{
    "language": "de", 
    "grammar": "NP: {<ADJ>*<NOUN|PROPN>}",
    "normalization": "stemming",
    "window": 10,
    "max_count": 5
}
