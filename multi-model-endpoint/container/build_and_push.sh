%%sh

# The name of our algorithm
algorithm_name=$MODEL_NAME

# cd container

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-eu-central-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
# $(aws ecr get-login --region ${region} --no-include-email)
aws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin ${fullname}


# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -q -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}:${STAGE}
docker push ${fullname}
