#!/usr/bin/env python3
"""
Launch WhisperX pipeline on a GPU EC2 instance for testing.

This script provides an alternative to AWS Batch for running WhisperX jobs
when Batch infrastructure is unavailable or for ad-hoc testing.

Usage:
    python scripts/run_whisperx_ec2.py --video-id Jiip33AMgrs
    python scripts/run_whisperx_ec2.py --video-id Jiip33AMgrs --instance-type g4dn.2xlarge
    python scripts/run_whisperx_ec2.py --video-id Jiip33AMgrs --smoke-test
"""

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

# Configuration
AWS_ACCOUNT_ID = "864981718771"
ECR_REGION = "us-east-2"  # ECR is in us-east-2
EC2_REGION = "us-east-1"  # G-instance quota is in us-east-1
S3_BUCKET = "rezora-data-pipeline-864981718771"
ECR_REPO = f"{AWS_ACCOUNT_ID}.dkr.ecr.{ECR_REGION}.amazonaws.com/whisperx-pipeline"
HF_TOKEN_SECRET_ARN = f"arn:aws:secretsmanager:{ECR_REGION}:{AWS_ACCOUNT_ID}:secret:hf-token"

# Instance configuration
DEFAULT_INSTANCE_TYPE = "g4dn.xlarge"  # 4 vCPU, 16GB RAM, 1x T4 GPU

# Amazon ECS-Optimized Amazon Linux 2 (GPU) - us-east-1
# This AMI has NVIDIA drivers, Docker, and nvidia-container-toolkit pre-installed
# AMI ID varies by region; this is for us-east-1
ECS_GPU_AMI_ID = "ami-0fb1852c38f4d357d"  # amzn2-ami-ecs-gpu-hvm-2.0.20260122-x86_64-ebs

# IAM role for the EC2 instance
INSTANCE_PROFILE_NAME = "whisperx-ec2-instance-profile"


def get_or_create_instance_profile(iam_client) -> str:
    """Get or create the IAM instance profile for WhisperX EC2 instances."""
    role_name = "whisperx-ec2-role"

    # Check if role exists
    try:
        iam_client.get_role(RoleName=role_name)
        print(f"✓ IAM role '{role_name}' exists")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            print(f"Creating IAM role '{role_name}'...")

            # Trust policy for EC2
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "ec2.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }

            iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="IAM role for WhisperX EC2 instances"
            )

            # Attach policies for ECR, S3, Secrets Manager, and SSM
            policies = [
                "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
                "arn:aws:iam::aws:policy/AmazonS3FullAccess",
                "arn:aws:iam::aws:policy/SecretsManagerReadWrite",
                "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
            ]

            for policy_arn in policies:
                iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
                print(f"  Attached: {policy_arn.split('/')[-1]}")

            print(f"✓ Created IAM role '{role_name}'")
        else:
            raise

    # Check if instance profile exists
    try:
        iam_client.get_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        print(f"✓ Instance profile '{INSTANCE_PROFILE_NAME}' exists")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            print(f"Creating instance profile '{INSTANCE_PROFILE_NAME}'...")
            iam_client.create_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
            iam_client.add_role_to_instance_profile(
                InstanceProfileName=INSTANCE_PROFILE_NAME,
                RoleName=role_name
            )
            print(f"✓ Created instance profile '{INSTANCE_PROFILE_NAME}'")
            # Wait for profile to propagate
            print("  Waiting for instance profile to propagate...")
            time.sleep(10)
        else:
            raise

    return INSTANCE_PROFILE_NAME


def get_hf_token(secrets_client) -> str:
    """Retrieve the HuggingFace token from Secrets Manager."""
    try:
        response = secrets_client.get_secret_value(SecretId=HF_TOKEN_SECRET_ARN)
        return response["SecretString"]
    except ClientError as e:
        print(f"✗ Failed to retrieve HF_TOKEN from Secrets Manager: {e}")
        sys.exit(1)


def get_default_vpc_and_subnet(ec2_client) -> tuple[str, str]:
    """Get the default VPC and a public subnet."""
    # Get default VPC
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        print("✗ No default VPC found. Please specify --subnet-id")
        sys.exit(1)

    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    # Get a public subnet (one with auto-assign public IP)
    subnets = ec2_client.describe_subnets(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "map-public-ip-on-launch", "Values": ["true"]}
        ]
    )

    if not subnets["Subnets"]:
        # Fall back to any subnet in the VPC
        subnets = ec2_client.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )

    if not subnets["Subnets"]:
        print("✗ No subnets found in default VPC")
        sys.exit(1)

    subnet_id = subnets["Subnets"][0]["SubnetId"]
    return vpc_id, subnet_id


def get_or_create_security_group(ec2_client, vpc_id: str) -> str:
    """Get or create a security group for WhisperX instances."""
    sg_name = "whisperx-ec2-sg"

    # Check if security group exists
    sgs = ec2_client.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [sg_name]},
            {"Name": "vpc-id", "Values": [vpc_id]}
        ]
    )

    if sgs["SecurityGroups"]:
        return sgs["SecurityGroups"][0]["GroupId"]

    # Create security group
    print(f"Creating security group '{sg_name}'...")
    response = ec2_client.create_security_group(
        GroupName=sg_name,
        Description="Security group for WhisperX EC2 instances",
        VpcId=vpc_id
    )
    sg_id = response["GroupId"]

    # Allow outbound traffic (default) - no inbound needed for our use case
    print(f"✓ Created security group '{sg_name}' ({sg_id})")
    return sg_id


def wait_for_g_instances_to_terminate(ec2_client, timeout: int = 300) -> None:
    """Wait for any G-type instances to fully terminate before launching."""
    # Find any G instances not yet terminated
    response = ec2_client.describe_instances(
        Filters=[
            {"Name": "instance-type", "Values": ["g*"]},
            {"Name": "instance-state-name", "Values": ["running", "pending", "stopping", "stopped", "shutting-down"]}
        ]
    )

    instance_ids = []
    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            instance_ids.append(instance["InstanceId"])

    if not instance_ids:
        return

    print(f"  Waiting for {len(instance_ids)} G instance(s) to terminate: {instance_ids}")

    # Use the EC2 waiter
    waiter = ec2_client.get_waiter("instance_terminated")
    try:
        waiter.wait(
            InstanceIds=instance_ids,
            WaiterConfig={"Delay": 10, "MaxAttempts": timeout // 10}
        )
        print("  ✓ All G instances terminated")
    except Exception as e:
        print(f"  ⚠ Timeout waiting for instances: {e}")
        print("  Proceeding anyway...")


def generate_user_data(
    video_id: str,
    run_id: str,
    hf_token: str,
    image_tag: str = "latest",
    smoke_test: bool = False,
    audio_key: str | None = None
) -> str:
    """Generate the EC2 user data script."""

    ecr_image = f"{ECR_REPO}:{image_tag}"

    # Command for the container
    if smoke_test:
        container_cmd = "--smoke-test"
    else:
        container_cmd = f"--video-id {video_id} --run-id {run_id}"
        if audio_key:
            container_cmd += f" --audio-key '{audio_key}'"

    # User data script
    # The ECS GPU AMI has Docker and nvidia-container-toolkit pre-installed, but NOT AWS CLI
    script = f"""#!/bin/bash
set -ex

# Log everything to both cloud-init and custom log
exec > >(tee /var/log/whisperx-userdata.log) 2>&1

echo "=== WhisperX EC2 Runner ==="
echo "Video ID: {video_id}"
echo "Run ID: {run_id}"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Install dependencies (awscli + jq for JSON parsing)
echo "=== Installing Dependencies ==="
sudo yum install -y awscli jq
aws --version
jq --version

# Verify GPU is available
echo "=== Checking GPU ==="
nvidia-smi || echo "WARNING: nvidia-smi failed"

# Start Docker if not running
echo "=== Starting Docker ==="
sudo systemctl start docker || true
sudo systemctl enable docker || true

# Log into ECR (cross-region pull from us-east-2)
echo "=== Logging into ECR ==="
aws ecr get-login-password --region {ECR_REGION} | docker login --username AWS --password-stdin {AWS_ACCOUNT_ID}.dkr.ecr.{ECR_REGION}.amazonaws.com

# Pull the image
echo "=== Pulling Docker Image ==="
docker pull {ecr_image}

# Run the container with host networking (boto3 inside can access IMDS directly)
echo "=== Running WhisperX Pipeline ==="
docker run --rm --gpus all --net=host \
    -e S3_BUCKET={S3_BUCKET} \
    -e HF_TOKEN="{hf_token}" \
    -e AWS_DEFAULT_REGION={ECR_REGION} \
    {ecr_image} {container_cmd}

EXIT_CODE=$?
echo "=== Pipeline Complete (exit code: $EXIT_CODE) ==="

# Upload logs to S3 before terminating
echo "=== Uploading Logs ==="
aws s3 cp /var/log/whisperx-userdata.log s3://{S3_BUCKET}/runs/{video_id}/{run_id}/ec2-userdata.log --region {ECR_REGION} || true

# Self-terminate to avoid cost leaks (use IMDSv2)
echo "=== Self-Terminating Instance ==="
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region {EC2_REGION}
"""

    return base64.b64encode(script.encode()).decode()


def launch_instance(
    ec2_client,
    video_id: str,
    run_id: str,
    hf_token: str,
    instance_type: str,
    subnet_id: str,
    security_group_id: str,
    instance_profile: str,
    image_tag: str = "latest",
    smoke_test: bool = False,
    audio_key: str | None = None
) -> str:
    """Launch the EC2 instance."""

    user_data = generate_user_data(
        video_id=video_id,
        run_id=run_id,
        hf_token=hf_token,
        image_tag=image_tag,
        smoke_test=smoke_test,
        audio_key=audio_key
    )

    # Tags for the instance
    tags = [
        {"Key": "Name", "Value": f"whisperx-{video_id}-{run_id[:8]}"},
        {"Key": "Project", "Value": "rezora-data-pipeline"},
        {"Key": "Purpose", "Value": "whisperx-transcription"},
        {"Key": "VideoId", "Value": video_id},
        {"Key": "RunId", "Value": run_id},
    ]

    print(f"Launching {instance_type} instance...")

    response = ec2_client.run_instances(
        ImageId=ECS_GPU_AMI_ID,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        SubnetId=subnet_id,
        SecurityGroupIds=[security_group_id],
        IamInstanceProfile={"Name": instance_profile},
        UserData=user_data,
        TagSpecifications=[
            {"ResourceType": "instance", "Tags": tags}
        ],
        # Spot instance for cost savings (optional)
        # InstanceMarketOptions={
        #     "MarketType": "spot",
        #     "SpotOptions": {"SpotInstanceType": "one-time"}
        # },
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": 100,  # 100GB for model cache + audio
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True
                }
            }
        ]
    )

    instance_id = response["Instances"][0]["InstanceId"]
    return instance_id


def wait_for_instance(ec2_client, instance_id: str, timeout: int = 600) -> bool:
    """Wait for instance to be running."""
    print(f"Waiting for instance {instance_id} to be running...")

    waiter = ec2_client.get_waiter("instance_running")
    try:
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={"Delay": 10, "MaxAttempts": timeout // 10}
        )
        print(f"✓ Instance {instance_id} is running")
        return True
    except Exception as e:
        print(f"✗ Instance failed to start: {e}")
        return False


def monitor_instance(ec2_client, instance_id: str, timeout: int = 7200) -> int:
    """Monitor instance until it terminates (self-terminates on completion)."""
    print(f"\nMonitoring instance {instance_id}...")
    print("Instance will self-terminate when pipeline completes.")
    print("Press Ctrl+C to stop monitoring (instance will continue running).\n")

    start_time = time.time()
    last_state = None

    try:
        while time.time() - start_time < timeout:
            response = ec2_client.describe_instances(InstanceIds=[instance_id])

            if not response["Reservations"]:
                print("Instance no longer exists (terminated)")
                return 0

            instance = response["Reservations"][0]["Instances"][0]
            state = instance["State"]["Name"]

            if state != last_state:
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed:4d}s] Instance state: {state}")
                last_state = state

            if state in ["terminated", "shutting-down"]:
                print("\n✓ Instance has terminated (pipeline complete)")
                return 0

            if state == "stopped":
                print("\n⚠ Instance stopped unexpectedly")
                return 1

            time.sleep(30)

        print(f"\n⚠ Timeout after {timeout}s - instance still running")
        return 1

    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Instance {instance_id} is still running.")
        print("To terminate manually:")
        print(f"  aws ec2 terminate-instances --instance-ids {instance_id} --region {EC2_REGION}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Launch WhisperX pipeline on a GPU EC2 instance"
    )
    parser.add_argument(
        "--video-id",
        required=True,
        help="YouTube video ID to process"
    )
    parser.add_argument(
        "--run-id",
        help="Run ID (auto-generated if not provided)"
    )
    parser.add_argument(
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help=f"EC2 instance type (default: {DEFAULT_INSTANCE_TYPE})"
    )
    parser.add_argument(
        "--image-tag",
        default="latest",
        help="Docker image tag (default: latest)"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test instead of full pipeline"
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Don't wait for instance to complete"
    )
    parser.add_argument(
        "--subnet-id",
        help="Specific subnet ID (default: auto-detect from default VPC)"
    )
    parser.add_argument(
        "--audio-key",
        help="S3 key for audio file (default: audio/{video_id}.mp3)"
    )

    args = parser.parse_args()

    # Generate run ID if not provided
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("WhisperX EC2 Runner")
    print("=" * 60)
    print(f"Video ID:      {args.video_id}")
    print(f"Run ID:        {run_id}")
    print(f"Instance Type: {args.instance_type}")
    print(f"Image Tag:     {args.image_tag}")
    print(f"EC2 Region:    {EC2_REGION}")
    print(f"ECR Region:    {ECR_REGION}")
    print(f"Smoke Test:    {args.smoke_test}")
    if args.audio_key:
        print(f"Audio Key:     {args.audio_key}")
    print("=" * 60)

    # Initialize clients
    iam_client = boto3.client("iam")
    ec2_client = boto3.client("ec2", region_name=EC2_REGION)
    secrets_client = boto3.client("secretsmanager", region_name=ECR_REGION)

    # Setup IAM
    print("\n[1/5] Setting up IAM...")
    instance_profile = get_or_create_instance_profile(iam_client)

    # Get HF token
    print("\n[2/5] Retrieving HuggingFace token...")
    hf_token = get_hf_token(secrets_client)
    print("✓ Retrieved HF_TOKEN from Secrets Manager")

    # Get VPC and subnet
    print("\n[3/5] Setting up networking...")
    if args.subnet_id:
        subnet_id = args.subnet_id
        # Get VPC from subnet
        subnet_info = ec2_client.describe_subnets(SubnetIds=[subnet_id])
        vpc_id = subnet_info["Subnets"][0]["VpcId"]
    else:
        vpc_id, subnet_id = get_default_vpc_and_subnet(ec2_client)

    print(f"✓ Using VPC: {vpc_id}")
    print(f"✓ Using Subnet: {subnet_id}")

    security_group_id = get_or_create_security_group(ec2_client, vpc_id)
    print(f"✓ Using Security Group: {security_group_id}")

    # Wait for any existing G instances to terminate (quota protection)
    print("\n[4/6] Checking GPU quota...")
    wait_for_g_instances_to_terminate(ec2_client)
    print("✓ GPU quota available")

    # Launch instance
    print("\n[5/6] Launching EC2 instance...")
    instance_id = launch_instance(
        ec2_client=ec2_client,
        video_id=args.video_id,
        run_id=run_id,
        hf_token=hf_token,
        instance_type=args.instance_type,
        subnet_id=subnet_id,
        security_group_id=security_group_id,
        instance_profile=instance_profile,
        image_tag=args.image_tag,
        smoke_test=args.smoke_test,
        audio_key=args.audio_key
    )
    print(f"✓ Launched instance: {instance_id}")

    # Wait for instance to start
    if not wait_for_instance(ec2_client, instance_id):
        print("✗ Failed to start instance")
        sys.exit(1)

    # Get instance details
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    instance = response["Reservations"][0]["Instances"][0]
    public_ip = instance.get("PublicIpAddress", "N/A")

    print(f"\n{'=' * 60}")
    print("Instance Details")
    print("=" * 60)
    print(f"Instance ID:  {instance_id}")
    print(f"Public IP:    {public_ip}")
    print(f"Console URL:  https://{EC2_REGION}.console.aws.amazon.com/ec2/home?region={EC2_REGION}#InstanceDetails:instanceId={instance_id}")
    print(f"\nOutput will be at:")
    print(f"  s3://{S3_BUCKET}/runs/{args.video_id}/{run_id}/")
    print("=" * 60)

    # Monitor instance
    if args.no_monitor:
        print("\n--no-monitor specified. Instance is running.")
        print(f"To check status: aws ec2 describe-instances --instance-ids {instance_id} --region {EC2_REGION}")
        print(f"To terminate:    aws ec2 terminate-instances --instance-ids {instance_id} --region {EC2_REGION}")
    else:
        print("\n[6/6] Monitoring instance (Ctrl+C to stop monitoring)...")
        exit_code = monitor_instance(ec2_client, instance_id)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
