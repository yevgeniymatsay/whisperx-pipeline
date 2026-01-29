#!/usr/bin/env python3
"""One-time setup for AWS Batch infrastructure in us-east-1.

This script sets up the necessary AWS resources to run WhisperX jobs in us-east-1
while waiting for GPU quota increase in us-east-2.

Usage:
    python scripts/setup_batch_us_east_1.py
    python scripts/setup_batch_us_east_1.py --skip-ecr  # Skip ECR mirror step
"""

import argparse
import time
import boto3
from botocore.exceptions import ClientError

ACCOUNT_ID = "864981718771"
SOURCE_REGION = "us-east-2"
TARGET_REGION = "us-east-1"
ECR_REPO = "whisperx-pipeline"
SECRET_NAME = "hf-token"
LOG_GROUP = "/aws/batch/whisperx-pipeline"


def create_ecr_repository():
    """Create ECR repository in us-east-1 if it doesn't exist."""
    print("\n=== Step 1: Creating ECR Repository ===")
    ecr = boto3.client("ecr", region_name=TARGET_REGION)

    try:
        ecr.describe_repositories(repositoryNames=[ECR_REPO])
        print(f"  ECR repository '{ECR_REPO}' already exists in {TARGET_REGION}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            print(f"  Creating ECR repository '{ECR_REPO}' in {TARGET_REGION}...")
            ecr.create_repository(repositoryName=ECR_REPO)
            print(f"  Created ECR repository '{ECR_REPO}'")
            return True
        raise


def print_ecr_mirror_instructions():
    """Print instructions for mirroring the Docker image."""
    print("\n=== Step 2: Mirror Docker Image ===")
    print("  Run the following commands to mirror the image:")
    print()
    print(f"  # Login to both ECR registries")
    print(f"  aws ecr get-login-password --region {SOURCE_REGION} | docker login --username AWS --password-stdin {ACCOUNT_ID}.dkr.ecr.{SOURCE_REGION}.amazonaws.com")
    print(f"  aws ecr get-login-password --region {TARGET_REGION} | docker login --username AWS --password-stdin {ACCOUNT_ID}.dkr.ecr.{TARGET_REGION}.amazonaws.com")
    print()
    print(f"  # Pull, tag, and push")
    print(f"  docker pull {ACCOUNT_ID}.dkr.ecr.{SOURCE_REGION}.amazonaws.com/{ECR_REPO}:latest")
    print(f"  docker tag {ACCOUNT_ID}.dkr.ecr.{SOURCE_REGION}.amazonaws.com/{ECR_REPO}:latest {ACCOUNT_ID}.dkr.ecr.{TARGET_REGION}.amazonaws.com/{ECR_REPO}:latest")
    print(f"  docker push {ACCOUNT_ID}.dkr.ecr.{TARGET_REGION}.amazonaws.com/{ECR_REPO}:latest")
    print()


def check_ecr_image_exists():
    """Check if the Docker image exists in us-east-1 ECR."""
    ecr = boto3.client("ecr", region_name=TARGET_REGION)
    try:
        response = ecr.describe_images(
            repositoryName=ECR_REPO, imageIds=[{"imageTag": "latest"}]
        )
        return len(response.get("imageDetails", [])) > 0
    except ClientError:
        return False


def copy_secret():
    """Copy HF_TOKEN secret from us-east-2 to us-east-1."""
    print("\n=== Step 3: Copying HF_TOKEN Secret ===")
    sm_source = boto3.client("secretsmanager", region_name=SOURCE_REGION)
    sm_target = boto3.client("secretsmanager", region_name=TARGET_REGION)

    # Check if secret already exists in target
    try:
        sm_target.describe_secret(SecretId=SECRET_NAME)
        print(f"  Secret '{SECRET_NAME}' already exists in {TARGET_REGION}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    # Get secret value from source
    try:
        response = sm_source.get_secret_value(SecretId=SECRET_NAME)
        secret_value = response["SecretString"]
    except ClientError as e:
        print(f"  Error: Could not read secret from {SOURCE_REGION}: {e}")
        return False

    # Create secret in target
    print(f"  Creating secret '{SECRET_NAME}' in {TARGET_REGION}...")
    sm_target.create_secret(Name=SECRET_NAME, SecretString=secret_value)
    print(f"  Created secret '{SECRET_NAME}'")
    return True


def create_log_group():
    """Create CloudWatch log group in us-east-1."""
    print("\n=== Step 4: Creating CloudWatch Log Group ===")
    logs = boto3.client("logs", region_name=TARGET_REGION)

    try:
        logs.describe_log_groups(logGroupNamePrefix=LOG_GROUP)
        response = logs.describe_log_groups(logGroupNamePrefix=LOG_GROUP)
        if any(lg["logGroupName"] == LOG_GROUP for lg in response.get("logGroups", [])):
            print(f"  Log group '{LOG_GROUP}' already exists in {TARGET_REGION}")
            return True
    except ClientError:
        pass

    print(f"  Creating log group '{LOG_GROUP}' in {TARGET_REGION}...")
    logs.create_log_group(logGroupName=LOG_GROUP)
    print(f"  Created log group '{LOG_GROUP}'")
    return True


def get_default_vpc_resources():
    """Get default VPC subnets and security group."""
    print("\n=== Step 5: Getting Default VPC Resources ===")
    ec2 = boto3.client("ec2", region_name=TARGET_REGION)

    # Get default VPC
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        print("  Error: No default VPC found in {TARGET_REGION}")
        return None, None
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    print(f"  Found default VPC: {vpc_id}")

    # Get subnets in default VPC
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    subnet_ids = [s["SubnetId"] for s in subnets["Subnets"]]
    print(f"  Found {len(subnet_ids)} subnets: {', '.join(subnet_ids)}")

    # Get default security group
    sgs = ec2.describe_security_groups(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "group-name", "Values": ["default"]},
        ]
    )
    if not sgs["SecurityGroups"]:
        print("  Error: No default security group found")
        return subnet_ids, None
    sg_id = sgs["SecurityGroups"][0]["GroupId"]
    print(f"  Found default security group: {sg_id}")

    return subnet_ids, [sg_id]


def create_compute_environment(subnet_ids: list, security_group_ids: list):
    """Create ON-DEMAND compute environment."""
    print("\n=== Step 6: Creating Compute Environment ===")
    batch = boto3.client("batch", region_name=TARGET_REGION)

    compute_env_name = "whisperx-gpu-ondemand"

    # Check if already exists
    try:
        response = batch.describe_compute_environments(
            computeEnvironments=[compute_env_name]
        )
        if response["computeEnvironments"]:
            status = response["computeEnvironments"][0]["status"]
            print(f"  Compute environment '{compute_env_name}' already exists (status: {status})")
            return compute_env_name
    except ClientError:
        pass

    print(f"  Creating compute environment '{compute_env_name}'...")
    batch.create_compute_environment(
        computeEnvironmentName=compute_env_name,
        type="MANAGED",
        state="ENABLED",
        computeResources={
            "type": "EC2",
            "allocationStrategy": "BEST_FIT_PROGRESSIVE",
            "minvCpus": 0,
            "maxvCpus": 16,
            "instanceTypes": ["g5.xlarge"],
            "subnets": subnet_ids,
            "securityGroupIds": security_group_ids,
            "instanceRole": f"arn:aws:iam::{ACCOUNT_ID}:instance-profile/ecsInstanceRole",
        },
        serviceRole=f"arn:aws:iam::{ACCOUNT_ID}:role/aws-service-role/batch.amazonaws.com/AWSServiceRoleForBatch",
    )
    print(f"  Created compute environment '{compute_env_name}'")
    return compute_env_name


def wait_for_compute_environment(compute_env_name: str, timeout: int = 300):
    """Wait for compute environment to reach VALID status."""
    print(f"\n=== Step 7: Waiting for Compute Environment ===")
    batch = boto3.client("batch", region_name=TARGET_REGION)

    start_time = time.time()
    while time.time() - start_time < timeout:
        response = batch.describe_compute_environments(
            computeEnvironments=[compute_env_name]
        )
        if response["computeEnvironments"]:
            status = response["computeEnvironments"][0]["status"]
            print(f"  Status: {status}", end="\r")
            if status == "VALID":
                print(f"  Compute environment is VALID")
                return True
            if status == "INVALID":
                print(f"\n  Error: Compute environment is INVALID")
                reason = response["computeEnvironments"][0].get("statusReason", "Unknown")
                print(f"  Reason: {reason}")
                return False
        time.sleep(10)

    print(f"\n  Timeout waiting for compute environment")
    return False


def create_job_queue(compute_env_name: str):
    """Create job queue."""
    print("\n=== Step 8: Creating Job Queue ===")
    batch = boto3.client("batch", region_name=TARGET_REGION)

    queue_name = "whisperx-transcription-queue"

    # Check if already exists
    try:
        response = batch.describe_job_queues(jobQueues=[queue_name])
        if response["jobQueues"]:
            status = response["jobQueues"][0]["status"]
            print(f"  Job queue '{queue_name}' already exists (status: {status})")
            return queue_name
    except ClientError:
        pass

    print(f"  Creating job queue '{queue_name}'...")
    batch.create_job_queue(
        jobQueueName=queue_name,
        state="ENABLED",
        priority=1,
        computeEnvironmentOrder=[{"order": 1, "computeEnvironment": compute_env_name}],
    )
    print(f"  Created job queue '{queue_name}'")
    return queue_name


def wait_for_job_queue(queue_name: str, timeout: int = 300):
    """Wait for job queue to reach VALID status."""
    print(f"\n=== Step 9: Waiting for Job Queue ===")
    batch = boto3.client("batch", region_name=TARGET_REGION)

    start_time = time.time()
    while time.time() - start_time < timeout:
        response = batch.describe_job_queues(jobQueues=[queue_name])
        if response["jobQueues"]:
            status = response["jobQueues"][0]["status"]
            print(f"  Status: {status}", end="\r")
            if status == "VALID":
                print(f"  Job queue is VALID")
                return True
            if status == "INVALID":
                print(f"\n  Error: Job queue is INVALID")
                reason = response["jobQueues"][0].get("statusReason", "Unknown")
                print(f"  Reason: {reason}")
                return False
        time.sleep(10)

    print(f"\n  Timeout waiting for job queue")
    return False


def register_job_definition():
    """Register job definition."""
    print("\n=== Step 10: Registering Job Definition ===")
    batch = boto3.client("batch", region_name=TARGET_REGION)

    job_def_name = "whisperx-pipeline"

    # Get secret ARN with version suffix
    sm = boto3.client("secretsmanager", region_name=TARGET_REGION)
    secret_arn = sm.describe_secret(SecretId=SECRET_NAME)["ARN"]

    print(f"  Registering job definition '{job_def_name}'...")
    execution_role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/ecsTaskExecutionRole"
    response = batch.register_job_definition(
        jobDefinitionName=job_def_name,
        type="container",
        containerProperties={
            "image": f"{ACCOUNT_ID}.dkr.ecr.{TARGET_REGION}.amazonaws.com/{ECR_REPO}:latest",
            "resourceRequirements": [
                {"type": "GPU", "value": "1"},
                {"type": "VCPU", "value": "4"},
                {"type": "MEMORY", "value": "16384"},
            ],
            "executionRoleArn": execution_role_arn,
            "secrets": [{"name": "HF_TOKEN", "valueFrom": secret_arn}],
            "environment": [
                {"name": "S3_BUCKET", "value": f"rezora-whisperx-us-east-1-{ACCOUNT_ID}"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": LOG_GROUP,
                    "awslogs-region": TARGET_REGION,
                    "awslogs-stream-prefix": "whisperx",
                },
            },
            "command": ["--video-id", "Ref::video_id", "--run-id", "Ref::run_id"],
        },
        timeout={"attemptDurationSeconds": 7200},
        retryStrategy={
            "attempts": 2,
            "evaluateOnExit": [
                {"onExitCode": "137", "action": "RETRY"},
                {"onExitCode": "1", "action": "EXIT"},
            ],
        },
    )
    revision = response["revision"]
    print(f"  Registered job definition '{job_def_name}:{revision}'")
    return f"{job_def_name}:{revision}"


def main():
    parser = argparse.ArgumentParser(
        description="Set up AWS Batch infrastructure in us-east-1"
    )
    parser.add_argument(
        "--skip-ecr",
        action="store_true",
        help="Skip ECR repository creation and image mirror instructions",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if Docker image is not found in us-east-1 ECR",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AWS Batch Setup for us-east-1")
    print("=" * 60)

    # Step 1: Create ECR repository
    if not args.skip_ecr:
        create_ecr_repository()

        # Step 2: Print mirror instructions
        print_ecr_mirror_instructions()

        # Check if image exists
        if not check_ecr_image_exists():
            print("  WARNING: Docker image not found in us-east-1 ECR")
            print("  Please run the commands above before continuing.")
            if not args.force:
                try:
                    response = input("  Continue anyway? [y/N]: ")
                    if response.lower() != "y":
                        print("  Exiting. Run again after mirroring the image.")
                        return
                except EOFError:
                    print("  Non-interactive mode. Use --force to continue anyway.")
                    return
            else:
                print("  --force flag set, continuing anyway...")

    # Step 3: Copy secret
    if not copy_secret():
        print("Error: Failed to copy secret. Exiting.")
        return

    # Step 4: Create log group
    create_log_group()

    # Step 5: Get VPC resources
    subnet_ids, security_group_ids = get_default_vpc_resources()
    if not subnet_ids or not security_group_ids:
        print("Error: Failed to get VPC resources. Exiting.")
        return

    # Step 6: Create compute environment
    compute_env_name = create_compute_environment(subnet_ids, security_group_ids)

    # Step 7: Wait for compute environment
    if not wait_for_compute_environment(compute_env_name):
        print("Error: Compute environment failed. Exiting.")
        return

    # Step 8: Create job queue
    queue_name = create_job_queue(compute_env_name)

    # Step 9: Wait for job queue
    if not wait_for_job_queue(queue_name):
        print("Error: Job queue failed. Exiting.")
        return

    # Step 10: Register job definition
    job_def = register_job_definition()

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nResources created in {TARGET_REGION}:")
    print(f"  - ECR Repository: {ECR_REPO}")
    print(f"  - Secret: {SECRET_NAME}")
    print(f"  - Log Group: {LOG_GROUP}")
    print(f"  - Compute Environment: {compute_env_name}")
    print(f"  - Job Queue: {queue_name}")
    print(f"  - Job Definition: {job_def}")
    print("\nNext steps:")
    print(f"  1. Submit a test job:")
    print(f"     python scripts/submit_whisperx_jobs.py VIDEO_ID --region {TARGET_REGION}")
    print(f"\n  2. Monitor the job:")
    print(f"     aws batch describe-jobs --jobs JOB_ID --region {TARGET_REGION}")
    print(f"     aws logs tail {LOG_GROUP} --region {TARGET_REGION} --follow")


if __name__ == "__main__":
    main()
