#!/usr/bin/env python3
"""
Build WhisperX Docker image on EC2 and push to ECR.

Workflow:
1. Create tarball of source code with git SHA
2. Upload tarball to S3
3. Launch t3.xlarge with user data that builds and pushes
4. Poll ECR for the image
5. Terminate instance when done (backstop)

Usage:
    python scripts/build_on_ec2.py [--instance-type t3.xlarge] [--image-tag latest]
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Configuration
AWS_REGION = "us-east-2"
AWS_ACCOUNT_ID = "864981718771"
S3_BUCKET = "rezora-data-pipeline-864981718771"
ECR_REPO = "whisperx-pipeline"

# EC2 Configuration
DEFAULT_INSTANCE_TYPE = "t3.xlarge"
AMI_ID = "ami-03ea746da1a2e36e7"  # Amazon Linux 2023 in us-east-2
ROOT_VOLUME_GB = 100
POLL_INTERVAL_SECONDS = 30
TIMEOUT_MINUTES = 90  # Large base image + pip can take a while on fresh instance


def get_git_sha() -> str:
    """Get current git short SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_git_root() -> Path:
    """Get git repository root directory."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def create_tarball(git_sha: str) -> Path:
    """Create tarball of source code, excluding unnecessary files."""
    tarball_path = Path(tempfile.gettempdir()) / f"whisperx-src-{git_sha}.tgz"
    git_root = get_git_root()

    print(f"Creating tarball at {tarball_path}...")

    # Use git archive to respect .gitignore + include only tracked files
    # Then add untracked files that are needed (none currently)
    subprocess.run(
        [
            "tar",
            "--exclude=.git",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=.venv",
            "--exclude=venv",
            "--exclude=node_modules",
            "--exclude=audio",
            "--exclude=runs",
            "--exclude=artifacts",
            "--exclude=*.mp3",
            "--exclude=*.wav",
            "--exclude=*.mp4",
            "--exclude=docs",
            "--exclude=.env",
            "--exclude=.env.*",
            "-czf",
            str(tarball_path),
            "-C",
            str(git_root),
            ".",
        ],
        check=True,
    )

    size_mb = tarball_path.stat().st_size / (1024 * 1024)
    print(f"Tarball created: {size_mb:.1f} MB")
    return tarball_path


def upload_tarball_to_s3(tarball_path: Path, git_sha: str) -> str:
    """Upload tarball to S3 and return S3 key."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3_key = f"build-src/whisperx-src-{git_sha}.tgz"

    print(f"Uploading to s3://{S3_BUCKET}/{s3_key}...")
    s3.upload_file(str(tarball_path), S3_BUCKET, s3_key)
    print("Upload complete.")

    return s3_key


def get_user_data_script(git_sha: str, alias_tag: str | None) -> str:
    """Generate EC2 user data script for Docker build.

    Always pushes :<git_sha> tag. If alias_tag is provided (e.g., 'latest'),
    also pushes that tag pointing to the same image.
    """
    # Build the push commands - always push git SHA, optionally push alias
    push_commands = f"""
# Push with git SHA tag (always unique, never conflicts)
docker tag {ECR_REPO}:{git_sha} {AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO}:{git_sha}
docker push {AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO}:{git_sha}
"""
    if alias_tag:
        push_commands += f"""
# Also push alias tag (e.g., 'latest')
docker tag {ECR_REPO}:{git_sha} {AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO}:{alias_tag}
docker push {AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO}:{alias_tag}
"""

    return f"""#!/bin/bash
# Log everything to file + upload to S3 later
exec > >(tee /var/log/docker-build.log) 2>&1

# Upload logs on ANY exit (success, failure, or crash)
trap 'aws s3 cp /var/log/docker-build.log s3://{S3_BUCKET}/build-logs/{git_sha}.log --region {AWS_REGION} || true' EXIT

set -ex

echo "=== Starting Docker Build on EC2 ==="
echo "Git SHA: {git_sha}"
echo "Alias Tag: {alias_tag or 'none'}"
date

# Install Docker (Amazon Linux 2023)
# Note: curl-minimal is pre-installed on AL2023, don't try to install curl (conflicts)
dnf install -y docker git unzip
systemctl enable --now docker
docker --version

# Install AWS CLI v2 (guaranteed to work)
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
cd /tmp && unzip -q awscliv2.zip && ./aws/install --update
aws --version

# Download source tarball
echo "=== Downloading source ==="
aws s3 cp s3://{S3_BUCKET}/build-src/whisperx-src-{git_sha}.tgz /build.tgz --region {AWS_REGION}
mkdir -p /build && cd /build && tar -xzf /build.tgz

# Login to ECR
echo "=== Logging in to ECR ==="
aws ecr get-login-password --region {AWS_REGION} | docker login --username AWS --password-stdin {AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com

# Network sanity checks (fail fast if networking is broken)
echo "=== Network sanity checks ==="
curl -I https://download.pytorch.org/whl/cu118/ --max-time 10 || echo "FAILED: pytorch.org"
curl -I https://github.com/snakers4/silero-vad --max-time 10 || echo "FAILED: github"
curl -I https://registry-1.docker.io --max-time 10 || echo "FAILED: docker hub"

# Pre-pull base image (makes build progress more visible)
echo "=== Pulling base image ==="
docker pull pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Build Docker image (--progress=plain for readable logs)
echo "=== Building Docker image ==="
cd /build
docker build --progress=plain -f docker/whisperx/Dockerfile -t {ECR_REPO}:{git_sha} .

# Tag and push
echo "=== Pushing to ECR ==="
{push_commands}

echo "=== Build complete ==="
date

# Self-terminate (trap will upload logs on exit)
echo "=== Self-terminating ==="
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region {AWS_REGION}
"""


def find_or_create_instance_profile() -> str:
    """Find existing instance profile or provide instructions."""
    iam = boto3.client("iam", region_name=AWS_REGION)
    profile_name = "ec2-docker-build-profile"
    role_name = "ec2-docker-build-role"

    # Check if profile exists
    try:
        iam.get_instance_profile(InstanceProfileName=profile_name)
        print(f"Using existing instance profile: {profile_name}")
        return profile_name
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    # Profile doesn't exist, try to create it
    print(f"Creating IAM role and instance profile: {profile_name}")

    # Create role
    trust_policy = """{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }"""

    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=trust_policy,
            Description="Role for EC2 Docker builds",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise

    # Attach required policies
    policies = [
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    ]
    for policy_arn in policies:
        try:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        except ClientError:
            pass  # Already attached

    # Add inline policy for self-termination
    terminate_policy = f"""{{
        "Version": "2012-10-17",
        "Statement": [{{
            "Effect": "Allow",
            "Action": "ec2:TerminateInstances",
            "Resource": "arn:aws:ec2:{AWS_REGION}:{AWS_ACCOUNT_ID}:instance/*",
            "Condition": {{
                "StringEquals": {{
                    "ec2:ResourceTag/Purpose": "docker-build"
                }}
            }}
        }}]
    }}"""
    try:
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="self-terminate",
            PolicyDocument=terminate_policy,
        )
    except ClientError:
        pass

    # Create instance profile and add role
    try:
        iam.create_instance_profile(InstanceProfileName=profile_name)
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise

    try:
        iam.add_role_to_instance_profile(
            InstanceProfileName=profile_name, RoleName=role_name
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "LimitExceeded":
            raise  # Role already added

    # Wait for profile to propagate
    print("Waiting for instance profile to propagate...")
    time.sleep(10)

    return profile_name


def get_default_vpc_subnet() -> str:
    """Get a public subnet in the default VPC."""
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    # Get default VPC
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        raise RuntimeError("No default VPC found in region")
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    # Get a public subnet in the default VPC
    subnets = ec2.describe_subnets(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "map-public-ip-on-launch", "Values": ["true"]},
        ]
    )
    if not subnets["Subnets"]:
        # Fall back to any subnet in default VPC
        subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])

    if not subnets["Subnets"]:
        raise RuntimeError("No subnets found in default VPC")

    return subnets["Subnets"][0]["SubnetId"]


def get_security_group() -> str:
    """Get or create security group for build instances."""
    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    sg_name = "docker-build-sg"

    # Check if it exists
    try:
        result = ec2.describe_security_groups(GroupNames=[sg_name])
        return result["SecurityGroups"][0]["GroupId"]
    except ClientError as e:
        if "InvalidGroup.NotFound" not in str(e):
            raise

    # Create it - only needs outbound (for pip, Docker Hub, ECR)
    # No inbound needed since we don't SSH in
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    result = ec2.create_security_group(
        GroupName=sg_name,
        Description="Security group for Docker build instances",
        VpcId=vpc_id,
    )
    sg_id = result["GroupId"]

    # Default allows all outbound, which is what we need
    print(f"Created security group: {sg_id}")
    return sg_id


def launch_build_instance(
    git_sha: str, alias_tag: str | None, instance_type: str
) -> str:
    """Launch EC2 instance for Docker build. Returns instance ID."""
    ec2 = boto3.client("ec2", region_name=AWS_REGION)

    profile_name = find_or_create_instance_profile()
    subnet_id = get_default_vpc_subnet()
    sg_id = get_security_group()

    user_data = get_user_data_script(git_sha, alias_tag)

    print(f"Launching {instance_type} instance...")
    print(f"  Subnet: {subnet_id}")
    print(f"  Security Group: {sg_id}")
    print(f"  Instance Profile: {profile_name}")

    response = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        SubnetId=subnet_id,
        SecurityGroupIds=[sg_id],
        IamInstanceProfile={"Name": profile_name},
        UserData=user_data,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": ROOT_VOLUME_GB,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"whisperx-build-{git_sha}"},
                    {"Key": "Purpose", "Value": "docker-build"},
                    {"Key": "GitSHA", "Value": git_sha},
                    {"Key": "AliasTag", "Value": alias_tag or ""},
                ],
            }
        ],
        # Auto-terminate if instance is idle for too long
        InstanceInitiatedShutdownBehavior="terminate",
        # Prevent t3 burstable instances from throttling during long builds
        # (standard mode drains CPU credits; unlimited charges ~$0.05/hr extra)
        CreditSpecification={"CpuCredits": "unlimited"},
    )

    instance_id = response["Instances"][0]["InstanceId"]
    print(f"Launched instance: {instance_id}")
    return instance_id


def check_ecr_for_image(image_tag: str) -> bool:
    """Check if image with given tag exists in ECR."""
    ecr = boto3.client("ecr", region_name=AWS_REGION)

    try:
        response = ecr.describe_images(
            repositoryName=ECR_REPO,
            imageIds=[{"imageTag": image_tag}],
        )
        return len(response.get("imageDetails", [])) > 0
    except ClientError as e:
        if "ImageNotFoundException" in str(e):
            return False
        raise


def get_instance_state(instance_id: str) -> str:
    """Get current state of EC2 instance."""
    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    response = ec2.describe_instances(InstanceIds=[instance_id])

    if not response["Reservations"]:
        return "terminated"

    return response["Reservations"][0]["Instances"][0]["State"]["Name"]


def terminate_instance(instance_id: str) -> None:
    """Terminate EC2 instance (backstop cleanup)."""
    ec2 = boto3.client("ec2", region_name=AWS_REGION)
    print(f"Terminating instance {instance_id}...")
    ec2.terminate_instances(InstanceIds=[instance_id])


def wait_for_build(instance_id: str, image_tag: str) -> bool:
    """Poll for build completion. Returns True if image found in ECR."""
    start_time = time.time()
    timeout_seconds = TIMEOUT_MINUTES * 60

    print(f"\nWaiting for build (timeout: {TIMEOUT_MINUTES} minutes)...")
    print(f"Polling ECR every {POLL_INTERVAL_SECONDS} seconds for tag: {image_tag}")
    print()

    while True:
        elapsed = time.time() - start_time
        elapsed_min = int(elapsed / 60)

        # Check if image appeared in ECR
        if check_ecr_for_image(image_tag):
            print(f"\n[{elapsed_min}m] Image found in ECR!")
            return True

        # Check instance state
        state = get_instance_state(instance_id)
        print(f"[{elapsed_min}m] Instance: {state}, Image: not found", end="\r")

        if state in ("terminated", "shutting-down"):
            # Instance terminated but image not found - check one more time
            time.sleep(5)
            if check_ecr_for_image(image_tag):
                print(f"\n[{elapsed_min}m] Image found in ECR!")
                return True
            print(f"\n[{elapsed_min}m] Instance terminated but image not found!")
            return False

        # Check timeout
        if elapsed > timeout_seconds:
            print(f"\n[{elapsed_min}m] Timeout reached!")
            return False

        time.sleep(POLL_INTERVAL_SECONDS)


def main():
    parser = argparse.ArgumentParser(
        description="Build WhisperX Docker image on EC2 and push to ECR"
    )
    parser.add_argument(
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help=f"EC2 instance type (default: {DEFAULT_INSTANCE_TYPE})",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help="Additional tag alias (e.g., 'latest'). Image is always tagged with git SHA.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip tarball creation/upload (use existing)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EC2 Docker Build for WhisperX")
    print("=" * 60)

    # Get git SHA - this is always the primary tag
    git_sha = get_git_sha()
    print(f"Git SHA: {git_sha}")
    print(f"Primary tag: {git_sha}")
    if args.alias:
        print(f"Alias tag: {args.alias}")
    print(f"Instance Type: {args.instance_type}")
    print()

    # Check if this exact SHA was already built
    if check_ecr_for_image(git_sha):
        print(f"Image with tag '{git_sha}' already exists in ECR.")
        print("Commit new changes to build a new image, or manually delete the existing tag.")
        sys.exit(1)

    instance_id = None
    try:
        # Create and upload tarball
        if not args.skip_upload:
            tarball_path = create_tarball(git_sha)
            upload_tarball_to_s3(tarball_path, git_sha)
            # Clean up local tarball
            tarball_path.unlink()
        else:
            print("Skipping tarball upload (using existing)")

        # Launch build instance
        instance_id = launch_build_instance(git_sha, args.alias, args.instance_type)

        # Wait for build - poll for git SHA tag (the primary, unique tag)
        success = wait_for_build(instance_id, git_sha)

        if success:
            print()
            print("=" * 60)
            print("BUILD SUCCESSFUL")
            print("=" * 60)
            print(f"Image: {AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO}:{git_sha}")
            if args.alias:
                print(f"Alias: {AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO}:{args.alias}")
            print(f"Logs: s3://{S3_BUCKET}/build-logs/{git_sha}.log")
            sys.exit(0)
        else:
            print()
            print("=" * 60)
            print("BUILD FAILED")
            print("=" * 60)
            print(f"Check logs: aws s3 cp s3://{S3_BUCKET}/build-logs/{git_sha}.log -")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        if instance_id:
            terminate_instance(instance_id)
        sys.exit(130)

    finally:
        # Backstop: ensure instance is terminated
        if instance_id:
            state = get_instance_state(instance_id)
            if state not in ("terminated", "shutting-down"):
                print(f"\nCleaning up: terminating instance {instance_id}")
                terminate_instance(instance_id)


if __name__ == "__main__":
    main()
