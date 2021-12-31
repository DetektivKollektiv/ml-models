import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

sm_runtime = boto3.client("sagemaker-runtime")
s3 = boto3.client('s3')

file_name = "models-id.json"
bucket = os.environ["BUCKET"]
key = file_name


def lambda_handler(event, context):
    logger.debug("event %s", json.dumps(event))
    endpoint_name = os.environ["ENDPOINT_NAME"]
    logger.info("api for endpoint %s", endpoint_name)
    model_id = os.environ["DEFAULT_ID"]
    logger.info("default id for models %s", model_id)

    # Get posted body and content type
    content_type = event["headers"].get("Content-Type", "text/csv")
    if content_type.startswith("text/csv"):
        payload = event["body"]
    elif content_type.startswith("application/json"):
        payload = json.loads(event["body"])
    else:
        message = "bad content type: {}".format(content_type)
        logger.error()
        return {"statusCode": 500, "message": message}

    os.chdir('/tmp')
    s3.download_file(bucket, key, file_name)
    with open(file_name, "r") as f:
        models_json = json.load(f)
    # Get target model and its id
    target_model = event['pathParameters']['model_name']
    if target_model in models_json:
        model = models_json[target_model]
    else:
        model = target_model+"-"+model_id

    logger.info("content type: %s size: %d", content_type, len(payload))

    try:
        # Invoke the endpoint with full multi-line payload
        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=payload,
            ContentType=content_type,
            TargetModel=model+".tar.gz",
            Accept="application/json",
        )
        # Return predictions as JSON dictionary instead of CSV text
        predictions = response["Body"].read().decode("utf-8")
        logger.debug("Model: %s", model)
        logger.debug("Predictions %s", json.dumps(predictions))
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": content_type,
                "X-SageMaker-Endpoint": endpoint_name,
            },
            "body": predictions,
        }
    except ClientError as e:
        logger.error(
            "Unexpected sagemaker error: {}".format(e.response["Error"]["Message"])
        )
        logger.error(e)
        return {"statusCode": 500, "message": "Unexpected sagemaker error"}

# update the ids of models in case a new model was trained
def update_modelId(event, context):
    model = event["model"]
    id = event["id"]
    model_id = os.environ["DEFAULT_ID"]

    # download file with model ids
    os.chdir('/tmp')
    s3.download_file(bucket, key, file_name)
    with open(file_name, "r") as f:
        models_json = json.load(f)

    if model in models_json:
        model_id = models_json[model]
        logger.info("model {} has the current id {}.".format(model, model_id))
    else:
        logger.info("model {} has the default id {}".format(model, model_id))

    models_json[model] = id
    logger.info("model {} has the new id {}.".format(model, id))
    with open(file_name, "w") as f:
        json.dump(models_json, f)
    s3.upload_file(file_name, bucket, key)

