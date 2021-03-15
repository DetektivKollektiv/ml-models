import argparse
import json
import os
import sys
from datetime import datetime
import tarfile
import shutil 

import boto3
import sagemaker
from sagemaker.workflow.airflow import training_config

sagemaker_session = sagemaker.session.Session()
bucket = sagemaker_session.default_bucket()

def create_tar_file(source_files, filename):
    with tarfile.open(filename, mode="w:gz") as t:
        for sf in source_files:
            # Add all files into the root of the directory structure of the tar
            t.add(sf, arcname=os.path.basename(sf))
    return filename
    
# JSON encode hyperparameters.
def json_encode_hyperparameters(hyperparameters):
    return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}

def get_training_request(
    model_name,
    model_id,
    stage,
    role,
    image_uri,
    training_uri,
    hyperparameters,
):
    model_uri = "s3://{0}/{1}/{2}".format(bucket, stage, model_name)

    # include location of tarfile and name of training script
    hyperparameters["sagemaker_program"] = "train.py"
    hyperparameters["sagemaker_submit_directory"] = model_uri+"/code/train.tar.gz"
    params = json_encode_hyperparameters(hyperparameters)

    # Create the estimator
    estimator = sagemaker.estimator.Estimator(
        image_uri,
        role,
        train_instance_count=1,
        train_instance_type="ml.c5.xlarge",
        base_job_name = model_name,
        output_path = model_uri+"/model",
        hyperparameters=params
    )
    # Specify the data source
    s3_input_train = sagemaker.inputs.TrainingInput(
        s3_data=training_uri, content_type="csv"
    )
    data = {"train": s3_input_train}

    # Get the training request
    request = training_config(estimator, inputs=data, job_name=get_training_job_name(model_name, model_id))
    return json.dumps(request)
    
def get_endpoint_params(model_name, role, image_uri, stage, training_requests, model_id):
    model_location = {}
    for model in training_requests:
        request = json.loads(training_requests[model])
        model_location[model] = request["OutputDataConfig"]["S3OutputPath"]+"/"+get_training_job_name(model, model_id)+"/output"
    return {
        "Parameters": {
            "ImageRepoUri": image_uri,
            "ModelName": model_name,
            "ModelsPrefix": stage,
            "MLOpsRoleArn": role,
            "ModelLocations": json.dumps(model_location),
            "Stage": stage,
            "ModelId": model_id
        }
    }

def get_trial_name(model_name, job_id):
    return model_name+"-"+job_id

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

def get_training_job_name(model_name, model_id):
    return model_name+"-"+model_id

def get_custom_resource_params(model_name, stage):
    return {
        "Parameters": {
            "ModelName": model_name,
            "Stage": stage,
            "TrainingJobStackName": model_name+"-training-job-"+stage,
            "SMexperimentLambda": model_name+"-create-sm-experiment-"+stage
        }
    }

def main(
    pipeline_name,
    model_name,
    role,
    endpoint_dir,
    training_dir,
    output_dir,
    stage,
):
    # Get the job id and source revisions
    job_id = get_pipeline_id(pipeline_name)
    print("job id: {}".format(job_id))

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
    experiments = []

    # Create training requests for all models
    training_requests = {}

    # use the current datetime as identifier for new models
    model_id = str(datetime.now())
    model_id = model_id.replace(":", "")
    model_id = model_id.replace(".", "")
    model_id = model_id.replace(" ", "-")
    # Write training job template
    training_template = "Description: Create training jobs\n" \
                        "\n" \
                        "Parameters:\n" \
                        "  ModelId:\n" \
                        "    Type: String\n" \
                        "    Description: model id used as suffix for training jobs\n" \
                        "    Default: "+model_id+"\n" \
                        "Resources:\n"
    model_id_sub = "${ModelId}"

    for model in models:
        model_dir = os.path.join("models", model)
        with open(os.path.join(model_dir, "inputData.json"), "r") as f:
            input_data = json.load(f)
            training_uri = input_data["Training"]["Uri"]
            training_uri = training_uri.replace("STAGE", stage)
            training_file = input_data["Training"]["file_name"]
            print("Train model {} with data {} in {}".format(model, training_uri, training_file))
            # create tar file with training script
            tar_file = os.path.join(model_dir, "train.tar.gz")
            create_tar_file([os.path.join(model_dir, "source_dir/train.py")], tar_file)
            # upload tar file to S3
            sources = sagemaker_session.upload_data(tar_file, bucket, stage + '/' + model + '/code')
            print(sources)
            # delete tar file after uploading
            try:
                os.remove(tar_file)
            except OSError:
                pass

        # Configure experiments and trials
        experiments.append(model)

        # Configure training requests
        params_file = os.path.join(model_dir, model+"-params.json")
        shutil.copyfile(params_file, "assets/"+model+"-params.json")
        with open(os.path.join(params_file), "r") as f:
            hyperparameters = json.load(f)
        training_requests[model] = get_training_request(
                model,
                model_id_sub,
                stage,
                role,
                training_image_uri,
                training_uri,
                hyperparameters,
            )
        # Upload params-file
        params_location = sagemaker_session.upload_data(params_file, bucket, stage + '/' + model + '/params')
        print("Parameter-file uploaded to {}".format(params_location))

        # create Cloudformation template for training jobs
        training_template +=    '   {}Job:\n'.format(model)
        training_template +=    '       Type: Custom::TrainingJob\n' \
                                '       Properties:\n' \
                                '           ServiceToken: !Sub "arn:aws:lambda:${AWS::Region}:${AWS::AccountId}:function:'+model_name+'-cfn-training-job-'+stage+'"\n'
        training_template +=    '           TrainingJobName: !Sub '+get_training_job_name(model, model_id_sub)+'\n'
        training_template +=    '           TrainingJobRequest: !Sub \''+training_requests[model]+'\'\n'
        training_template +=    '           ExperimentName: {}'.format(model)+'\n'
        training_template +=    '           TrialName: !Sub '+get_trial_name(model, model_id_sub)+'\n\n'

    # Write experiment and trial configs
    trials = {}
    trials["Models"] = experiments
    trials["TrialID"] = model_id
    with open(os.path.join(output_dir, "trials.json"), "w") as f:
        json.dump(trials, f)

    # Write the training template
    with open(os.path.join(output_dir, "training-job.yml"), "w") as f:
        f.write(training_template)

    # Write the dev & prod params for template-custom-resource.yml
    with open(os.path.join(output_dir, "template-custom-resource.json"), "w") as f:
        params = get_custom_resource_params(model_name, stage)
        json.dump(params, f)

    # Write the dev & prod params for CFN
    with open(os.path.join(output_dir, "deploy-endpoint.json"), "w") as f:
        params = get_endpoint_params(model_name, role, endpoint_image_uri, stage, training_requests, model_id)
        json.dump(params, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--endpoint-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True)
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    main(**args)
