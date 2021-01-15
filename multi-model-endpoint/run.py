import argparse
import json
import os
import sys
import time


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


def main(
    model_name,
    role,
    ecr_dir,
    output_dir,
    stage,
):
    # Load the image uri and input data config
    with open(os.path.join(ecr_dir, "imageDetail.json"), "r") as f:
        image_json = json.load(f)
        print(image_json)
        image_uri = image_json["ImageURI"]
    print("image uri: {}".format(image_uri))

    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Write the dev & prod params for CFN
    with open(os.path.join(output_dir, "deploy-endpoint.json"), "w") as f:
        params = get_endpoint_params(model_name, role, image_uri, stage)
        json.dump(params, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--ecr-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True)
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    main(**args)
