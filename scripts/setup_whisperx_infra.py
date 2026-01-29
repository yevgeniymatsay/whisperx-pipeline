#!/usr/bin/env python3
"""
One-time setup script for WhisperX infrastructure.

Creates:
- S3 bucket in us-east-1
- DynamoDB tables for tracking videos and calls

Usage:
    python scripts/setup_whisperx_infra.py
    python scripts/setup_whisperx_infra.py --dry-run
"""
import argparse
import sys

import boto3
from botocore.exceptions import ClientError

# Import config from pipeline
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])  # Add repo root to path
from pipeline.config import (
    S3_BUCKET,
    AWS_REGION,
    DYNAMODB_VIDEOS_TABLE,
    DYNAMODB_CALLS_TABLE,
)


def create_s3_bucket(dry_run: bool = False) -> bool:
    """Create the S3 bucket if it doesn't exist."""
    s3 = boto3.client("s3", region_name=AWS_REGION)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Creating S3 bucket: {S3_BUCKET}")

    if dry_run:
        print(f"  Would create bucket in {AWS_REGION}")
        return True

    try:
        # Check if bucket exists
        s3.head_bucket(Bucket=S3_BUCKET)
        print(f"  Bucket already exists")
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            # Bucket doesn't exist, create it
            try:
                # us-east-1 doesn't need LocationConstraint
                if AWS_REGION == "us-east-1":
                    s3.create_bucket(Bucket=S3_BUCKET)
                else:
                    s3.create_bucket(
                        Bucket=S3_BUCKET,
                        CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
                    )
                print(f"  Created bucket successfully")
                return True
            except ClientError as create_error:
                print(f"  ERROR creating bucket: {create_error}")
                return False
        else:
            print(f"  ERROR checking bucket: {e}")
            return False


def create_videos_table(dry_run: bool = False) -> bool:
    """Create the whisperx-videos DynamoDB table."""
    dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Creating DynamoDB table: {DYNAMODB_VIDEOS_TABLE}")

    table_def = {
        "TableName": DYNAMODB_VIDEOS_TABLE,
        "AttributeDefinitions": [
            {"AttributeName": "video_id", "AttributeType": "S"},
        ],
        "KeySchema": [
            {"AttributeName": "video_id", "KeyType": "HASH"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
    }

    if dry_run:
        print(f"  Table definition:")
        print(f"    Primary key: video_id (String)")
        print(f"    Billing: PAY_PER_REQUEST")
        return True

    try:
        dynamodb.describe_table(TableName=DYNAMODB_VIDEOS_TABLE)
        print(f"  Table already exists")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            try:
                dynamodb.create_table(**table_def)
                print(f"  Created table successfully")
                # Wait for table to be active
                waiter = dynamodb.get_waiter("table_exists")
                print(f"  Waiting for table to be active...")
                waiter.wait(TableName=DYNAMODB_VIDEOS_TABLE)
                print(f"  Table is now active")
                return True
            except ClientError as create_error:
                print(f"  ERROR creating table: {create_error}")
                return False
        else:
            print(f"  ERROR checking table: {e}")
            return False


def create_calls_table(dry_run: bool = False) -> bool:
    """Create the whisperx-calls DynamoDB table with GSIs."""
    dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Creating DynamoDB table: {DYNAMODB_CALLS_TABLE}")

    table_def = {
        "TableName": DYNAMODB_CALLS_TABLE,
        "AttributeDefinitions": [
            {"AttributeName": "call_id", "AttributeType": "S"},
            {"AttributeName": "video_id", "AttributeType": "S"},
            {"AttributeName": "conversation_type", "AttributeType": "S"},
        ],
        "KeySchema": [
            {"AttributeName": "call_id", "KeyType": "HASH"},
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "video-index",
                "KeySchema": [
                    {"AttributeName": "video_id", "KeyType": "HASH"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "type-index",
                "KeySchema": [
                    {"AttributeName": "conversation_type", "KeyType": "HASH"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        "BillingMode": "PAY_PER_REQUEST",
    }

    if dry_run:
        print(f"  Table definition:")
        print(f"    Primary key: call_id (String)")
        print(f"    GSI 1: video-index (video_id)")
        print(f"    GSI 2: type-index (conversation_type)")
        print(f"    Billing: PAY_PER_REQUEST")
        return True

    try:
        dynamodb.describe_table(TableName=DYNAMODB_CALLS_TABLE)
        print(f"  Table already exists")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            try:
                dynamodb.create_table(**table_def)
                print(f"  Created table successfully")
                # Wait for table to be active
                waiter = dynamodb.get_waiter("table_exists")
                print(f"  Waiting for table to be active...")
                waiter.wait(TableName=DYNAMODB_CALLS_TABLE)
                print(f"  Table is now active")
                return True
            except ClientError as create_error:
                print(f"  ERROR creating table: {create_error}")
                return False
        else:
            print(f"  ERROR checking table: {e}")
            return False


def create_folder_structure(dry_run: bool = False) -> bool:
    """Create the folder structure in S3 by uploading empty marker files."""
    s3 = boto3.client("s3", region_name=AWS_REGION)

    folders = [
        "audio/pretraining/",
        "audio/expired_listing/",
        "runs/",
        "labels/",
        "models/role_classifier/",
        "exports/pretraining/",
        "exports/expired_listing/",
    ]

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Creating folder structure in S3")

    for folder in folders:
        if dry_run:
            print(f"  Would create: {folder}")
        else:
            try:
                # S3 doesn't have real folders, but we can create zero-byte objects
                # to make them visible in the console
                s3.put_object(Bucket=S3_BUCKET, Key=folder, Body=b"")
                print(f"  Created: {folder}")
            except ClientError as e:
                print(f"  ERROR creating {folder}: {e}")
                return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Set up WhisperX infrastructure (S3 bucket + DynamoDB tables)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without actually creating",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("WhisperX Infrastructure Setup")
    print("=" * 60)
    print(f"\nTarget region: {AWS_REGION}")
    print(f"S3 bucket: {S3_BUCKET}")
    print(f"Videos table: {DYNAMODB_VIDEOS_TABLE}")
    print(f"Calls table: {DYNAMODB_CALLS_TABLE}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No resources will be created ***")

    success = True

    # Create S3 bucket
    if not create_s3_bucket(args.dry_run):
        success = False

    # Create folder structure
    if success and not create_folder_structure(args.dry_run):
        success = False

    # Create DynamoDB tables
    if not create_videos_table(args.dry_run):
        success = False

    if not create_calls_table(args.dry_run):
        success = False

    print("\n" + "=" * 60)
    if success:
        print("Setup completed successfully!")
        if not args.dry_run:
            print("\nNext steps:")
            print("  1. Run migration script to copy audio files:")
            print("     python scripts/migrate_pretraining_audio.py --type pretraining")
            print("  2. Update EC2 IAM role with DynamoDB permissions")
    else:
        print("Setup completed with errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()
