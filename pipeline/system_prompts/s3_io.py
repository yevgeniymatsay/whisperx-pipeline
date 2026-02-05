"""S3 utilities for synthetic system prompt datasets."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import boto3
from botocore.exceptions import ClientError

from ..config import AWS_REGION, S3_BUCKET


def get_s3_client(region_name: str = AWS_REGION):
    return boto3.client("s3", region_name=region_name)


def list_keys(bucket: str, prefix: str, *, suffix: str | None = None) -> List[str]:
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key:
                continue
            if suffix and not key.endswith(suffix):
                continue
            keys.append(key)
    return keys


def object_exists(bucket: str, key: str) -> bool:
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def load_json(bucket: str, key: str) -> Dict[str, Any]:
    s3 = get_s3_client()
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read())


def write_json(bucket: str, key: str, data: Dict[str, Any], *, dry_run: bool = False) -> None:
    if dry_run:
        return
    s3 = get_s3_client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json",
    )


def default_bucket() -> str:
    return S3_BUCKET

