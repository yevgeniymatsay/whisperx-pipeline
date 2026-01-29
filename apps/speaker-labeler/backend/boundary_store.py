"""S3-backed storage for corrected call boundaries.

Provides durable, cross-machine persistence for human-corrected boundaries.
Each video's boundaries are stored as a separate JSON file in S3.
"""
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

from config import S3_BUCKET, AWS_REGION


# Default S3 prefix for boundary storage
DEFAULT_PREFIX = "labeling/corrected_boundaries/v1/"


class S3BoundaryStore:
    """S3-backed store for corrected call boundaries.

    Storage layout:
        s3://{bucket}/{prefix}{video_id}.json

    Each JSON file contains:
        {
            "video_id": "abc123",
            "updated_at": "2026-01-29T15:12:01Z",
            "boundaries": [
                {"call_index": 0, "start_s": 12.5, "end_s": 145.8, "corrected_at": "..."},
                ...
            ]
        }
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
        """Initialize S3 boundary store.

        Args:
            bucket: S3 bucket name. Defaults to S3_BUCKET env var.
            prefix: S3 key prefix. Defaults to labeling/corrected_boundaries/v1/
        """
        self.bucket = bucket or os.environ.get("BOUNDARIES_BUCKET", S3_BUCKET)
        self.prefix = prefix or os.environ.get("BOUNDARIES_PREFIX", DEFAULT_PREFIX)
        self._s3 = None

    @property
    def s3(self):
        """Lazy-load S3 client."""
        if self._s3 is None:
            self._s3 = boto3.client("s3", region_name=AWS_REGION)
        return self._s3

    def _key(self, video_id: str) -> str:
        """Build S3 key for a video's boundaries."""
        return f"{self.prefix}{video_id}.json"

    def save(self, video_id: str, boundaries: List[Dict[str, Any]]) -> None:
        """Save boundaries for a video to S3.

        Boundaries are sorted by start_s and assigned sequential call_index values.
        Each boundary gets a corrected_at timestamp if not already present.

        Args:
            video_id: YouTube video ID
            boundaries: List of dicts with start_s and end_s (and optionally call_index, corrected_at)
        """
        now = datetime.now(timezone.utc).isoformat()

        # Sort by start time and assign call_index
        sorted_boundaries = sorted(boundaries, key=lambda x: x["start_s"])
        for i, b in enumerate(sorted_boundaries):
            b["call_index"] = i
            if "corrected_at" not in b:
                b["corrected_at"] = now

        doc = {
            "video_id": video_id,
            "updated_at": now,
            "boundaries": sorted_boundaries,
        }

        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._key(video_id),
            Body=json.dumps(doc, indent=2),
            ContentType="application/json",
        )

    def load(self, video_id: str) -> List[Dict[str, Any]]:
        """Load boundaries for a video from S3.

        Args:
            video_id: YouTube video ID

        Returns:
            List of boundary dicts. Empty list if video has no saved boundaries.
        """
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=self._key(video_id),
            )
            doc = json.loads(response["Body"].read())
            return doc.get("boundaries", [])
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise

    def list_videos(self) -> List[str]:
        """List all video IDs with saved boundaries.

        Returns:
            List of video IDs that have corrected boundaries.
        """
        paginator = self.s3.get_paginator("list_objects_v2")
        video_ids = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Extract video_id from key: {prefix}{video_id}.json
                if key.endswith(".json"):
                    video_id = key[len(self.prefix) : -5]  # Remove prefix and .json
                    if video_id:
                        video_ids.append(video_id)

        return sorted(video_ids)

    def delete(self, video_id: str) -> bool:
        """Delete boundaries for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            True if deleted, False if video had no boundaries.
        """
        try:
            self.s3.delete_object(
                Bucket=self.bucket,
                Key=self._key(video_id),
            )
            return True
        except ClientError:
            return False

    def export_labels_json(self) -> List[Dict[str, Any]]:
        """Export all boundaries in ML training format (labels.json).

        Returns format compatible with build_call_segmenter_dataset.py:
        [
            {"video_id": "abc", "calls": [{"start": 12.3, "end": 45.6}, ...]},
            ...
        ]

        Returns:
            List of video label dicts.
        """
        result = []
        for video_id in self.list_videos():
            boundaries = self.load(video_id)
            if boundaries:
                calls = [
                    {"start": b["start_s"], "end": b["end_s"]}
                    for b in sorted(boundaries, key=lambda x: x["start_s"])
                ]
                result.append({
                    "video_id": video_id,
                    "calls": calls,
                })
        return result

    def get_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a video's boundaries (without loading full boundaries).

        Args:
            video_id: YouTube video ID

        Returns:
            Dict with video_id, updated_at, and boundary count. None if not found.
        """
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=self._key(video_id),
            )
            doc = json.loads(response["Body"].read())
            return {
                "video_id": doc.get("video_id"),
                "updated_at": doc.get("updated_at"),
                "boundary_count": len(doc.get("boundaries", [])),
            }
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise


# Global instance for convenience
_store: Optional[S3BoundaryStore] = None


def get_boundary_store() -> S3BoundaryStore:
    """Get the global S3 boundary store instance."""
    global _store
    if _store is None:
        _store = S3BoundaryStore()
    return _store
