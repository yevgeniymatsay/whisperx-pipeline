"""Configuration for speaker labeler."""
import os

S3_BUCKET = os.environ.get("S3_BUCKET", "rezora-whisperx-us-east-1-864981718771")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_PREFIX = os.environ.get("S3_PREFIX", "runs/")
