#!/bin/bash
# Build and push WhisperX pipeline Docker image to ECR

set -e

AWS_REGION="us-east-2"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="whisperx-pipeline"
IMAGE_TAG="${1:-latest}"

echo "=== Building WhisperX Pipeline Docker Image ==="
echo "Account: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo "Tag: $IMAGE_TAG"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Create repo if it doesn't exist
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

# Build
cd "$(dirname "$0")/.."
docker build -f docker/whisperx/Dockerfile -t $ECR_REPO:$IMAGE_TAG .

# Tag and push
docker tag $ECR_REPO:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

echo ""
echo "=== Image pushed: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG ==="
