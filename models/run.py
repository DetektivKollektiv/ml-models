import argparse
import json
import os
import sys
import time

import boto3
import sagemaker
from sagemaker.workflow.airflow import training_config


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


def main(
    pipeline_name,
    model_name,
    role,
    data_bucket,
    data_dir,
    factchecks_prefix,
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

    with open(os.path.join(data_dir, "inputData.json"), "r") as f:
        input_data = json.load(f)
        training_uri = input_data["TrainingUri"]
        validation_uri = input_data["ValidationUri"]
        baseline_uri = input_data["BaselineUri"]
        print(
            "training uri: {}\nvalidation uri: {}\n baseline uri: {}".format(
                training_uri, validation_uri, baseline_uri
            )
        )

    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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
    parser.add_argument("--factchecks-prefix", required=True)
    parser.add_argument("--endpoint-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True)
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    main(**args)
