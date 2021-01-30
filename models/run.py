import argparse
import json
import os
import sys
import time
import tarfile

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
    
def get_training_request(
    model_name,
    job_id,
    role,
    image_uri,
    training_uri,
    hyperparameters,
):
    model_uri = "s3://{0}/{1}".format(bucket, model_name)

    # include location of tarfile and name of training script
    hyperparameters["sagemaker_program"] = "train.py"
    hyperparameters["sagemaker_submit_directory"] = model_uri+"/code"

    # Create the estimator
    estimator = sagemaker.estimator.Estimator(
        image_uri,
        role,
        train_instance_count=1,
        train_instance_type="ml.m5.large",
        base_job_name = model_name,
        output_path = model_uri+"/model",
        hyperparameters=hyperparameters
    )
    # Specify the data source
    s3_input_train = sagemaker.inputs.TrainingInput(
        s3_data=training_uri, content_type="csv"
    )
    data = {"train": s3_input_train}

    # Get the training request
    request = training_config(estimator, inputs=data, job_name=job_id)
    return json.dumps(request)
    
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
    trials = {}
    # Write training job template
    training_template = "Description: Wait for training jobs\n" \
                        "Parameters:\n" \
                        "   TrainJobId:\n" \
                        "       Type: String\n" \
                        "       Description: Id of the Codepipeline + SagemakerJobs\n" \
                        "\n" \
                        "Resources:\n"

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
            sources = sagemaker_session.upload_data(tar_file, bucket, model + '/code')
            print(sources)
            # delete tar file after uploading
            try:
                os.remove(tar_file)
            except OSError:
                pass

        # Configure experiments and trials
        trials[model] = get_trial(model, job_id)

        # Configure training requests
        with open(os.path.join(model_dir, model+"-params.json"), "r") as f:
            hyperparameters = json.load(f)
        training_request = get_training_request(
                model,
                job_id,
                role,
                training_image_uri,
                training_uri,
                hyperparameters,
            )

        # create Cloudformation template for training jobs
        training_template +=    '   {}TrainingJob:\n'.format(model)
        training_template +=    '       Type: Custom::TrainingJob\n' \
                                '       Properties:\n' \
                                '           ServiceToken: !Sub "arn:aws:lambda:${AWS::Region}:${AWS::AccountId}:function:sagemaker-cfn-training-job"\n'
        training_template +=    '           TrainingJobName: !Sub mlops-'+model+'-'+job_id+'\n'
        training_template +=    '           TrainingJobRequest: '+training_request+'\n'
        training_template +=    '           ExperimentName: {}'.format(model)+'\n'
        training_template +=    '           TrialName: '+job_id+'\n\n'

    # Write experiment and trial configs
    with open(os.path.join(output_dir, "trials.json"), "w") as f:
        json.dump(trials, f)

    # Write the training template
    with open(os.path.join(output_dir, "training-job.yml"), "w") as f:
        f.write(training_template)

    # Write the dev & prod params for CFN
    with open(os.path.join(output_dir, "deploy-endpoint.json"), "w") as f:
        params = get_endpoint_params(model_name, role, endpoint_image_uri, stage)
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
