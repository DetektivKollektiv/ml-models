import argparse
import json
import os
import sys
import time

import boto3
import sagemaker
from sagemaker.workflow.airflow import training_config


def get_training_params(
    model_name,
    job_id,
    role,
    image_uri,
    training_uri,
    output_uri,
    hyperparameters,
):
    # Create the estimator
    estimator = sagemaker.estimator.Estimator(
        image_uri,
        role,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=output_uri,
    )
    # Set the hyperparameters
    estimator.set_hyperparameters(**hyperparameters)

    # Specify the data source
    s3_input_train = sagemaker.inputs.TrainingInput(
        s3_data=training_uri, content_type="csv"
    )
    data = {"train": s3_input_train}

    # Get the training request
    request = training_config(estimator, inputs=data, job_name=job_id)
    return {
        "Parameters": {
            "ModelName": model_name,
            "TrainJobId": job_id,
            "TrainJobRequest": json.dumps(request),
        }
    }
    
def get_training_image(region=None):
    region = region or boto3.Session().region_name
    return sagemaker.image_uris.retrieve(
        region=region, framework="xgboost", version="1.0-1"
    )

def get_endpoint_params(model_name, role, image_uri, stage):
    return {
        "Parameters": {
            "ImageRepoUri": image_uri,
            "ModelName": model_name,
            "ModelsPrefix": model_name+"-"+stage,
            "MLOpsRoleArn": role,
            "Stage": stage,
        }
    }

def get_models():
    with open("models/models_to_be_trained.json", "r") as f:
        models_json = json.load(f)
    models = []
    for model in models_json:
        if models_json[model]:
            models.append(model)
    return models

def get_image_uri(dir):
    with open(os.path.join(dir, "container/imageDetail.json"), "r") as f:
        image_json = json.load(f)
        image_uri = image_json["imageURI"]
    print("{} image uri: {}".format(dir, image_uri))
    return image_uri

def get_pipeline_id(pipeline_name):
    # Get pipeline execution id
    codepipeline = boto3.client("codepipeline")
    response = codepipeline.get_pipeline_state(name=pipeline_name)
    return response["stageStates"][0]["latestExecution"]["pipelineExecutionId"]

def get_trial(model_name, job_id):
    return {
        "ExperimentName": model_name,
        "TrialName": job_id,
    }

def main(
    pipeline_name,
    model_name,
    role,
    data_bucket,
    data_dir,
    endpoint_dir,
    training_dir,
    output_dir,
    stage,
):
    # Get the job id and source revisions
    job_id = get_pipeline_id(pipeline_name)
    print("job id: {}".format(job_id))
    output_uri = "s3://{0}/{1}".format(data_bucket, model_name)

    # Load the endpoint image uri and input data config
    endpoint_image_uri = get_image_uri(endpoint_dir)

    # Load the training image uri and input data config
    training_image_uri = get_image_uri(training_dir)

    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Read a list of models to be trained
    models = get_models()

    # Create trials for all models
    trials = {}
    training_jobs = {}
    for model in models:
        model_dir = os.path.join("models", model)
        with open(os.path.join(model_dir, "inputData.json"), "r") as f:
            input_data = json.load(f)
            training_uri = input_data["Training"]["Uri"]
            training_file = input_data["Training"]["file_name"]
            print("Train model {} with data {} in {}".format(model, training_uri, training_file))

        # Configure experiments and trials
        trials[model] = get_trial(model, job_id)

        # Configure training requests
        with open(os.path.join(model_dir, model+"-params.json"), "r") as f:
            hyperparameters = json.load(f)
        training_jobs[model] = get_training_params(
                model,
                job_id,
                role,
                training_image_uri,
                training_uri,
                output_uri,
                hyperparameters,
            )

    # Write experiment and trial configs
    with open(os.path.join(output_dir, "trials.json"), "w") as f:
        json.dump(trials, f)

    # Write the training request
    with open(os.path.join(output_dir, "training-jobs.json"), "w") as f:
        json.dump(training_jobs, f)

    # Write the dev & prod params for CFN
    with open(os.path.join(output_dir, "deploy-endpoint.json"), "w") as f:
        params = get_endpoint_params(model_name, role, endpoint_image_uri, stage)
        json.dump(params, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--data-bucket", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--endpoint-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True)
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    main(**args)
