import boto3
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

cloudformation = boto3.resource('cloudformation')
lb = boto3.client('lambda')


# update stack with training jobs to start retraining of models
def lambda_handler(event, context):
    logger.debug("event %s", json.dumps(event))
    stack_name = os.environ["STACK_NAME"]
    logger.info("update training stack %s", stack_name)

    # use the current datetime as suffix for the trainingjobs
    model_id = str(datetime.now())
    payload_dict = {"trial_id": model_id}
    payload_str = json.dumps(payload_dict)

    response = lb.invoke(
        FunctionName = os.environ["LAMBDA_NAME"],
        Payload = payload_str.encode()
    )

    stack = cloudformation.Stack(stack_name)
    response = stack.update(
        UsePreviousTemplate=True,
        Parameters=[
            {
                'ParameterKey': 'ModelId',
                'ParameterValue': model_id,
            },
        ]
    )
