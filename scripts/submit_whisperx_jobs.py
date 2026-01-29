# scripts/submit_whisperx_jobs.py
"""Submit WhisperX transcription jobs to AWS Batch."""
import argparse
import boto3
from datetime import datetime

DEFAULT_REGION = "us-east-2"


def submit_job(
    video_id: str,
    queue: str = "whisperx-transcription-queue",
    region: str = DEFAULT_REGION,
) -> str:
    """Submit a single transcription job.

    Args:
        video_id: YouTube video ID to process
        queue: AWS Batch job queue name
        region: AWS region (us-east-1 or us-east-2)

    Returns:
        Job ID from AWS Batch
    """
    batch = boto3.client("batch", region_name=region)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    response = batch.submit_job(
        jobName=f"whisperx-{video_id}-{run_id}",
        jobQueue=queue,
        jobDefinition="whisperx-pipeline",
        parameters={"video_id": video_id, "run_id": run_id},
    )

    return response["jobId"]


def main():
    parser = argparse.ArgumentParser(
        description="Submit WhisperX transcription jobs to AWS Batch"
    )
    parser.add_argument("video_ids", nargs="+", help="Video IDs to process")
    parser.add_argument("--queue", default="whisperx-transcription-queue")
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        choices=["us-east-1", "us-east-2"],
        help=f"AWS region to submit jobs to (default: {DEFAULT_REGION})",
    )
    args = parser.parse_args()

    print(f"Submitting to {args.region}...")
    for video_id in args.video_ids:
        job_id = submit_job(video_id, args.queue, args.region)
        print(f"  {video_id}: {job_id}")


if __name__ == "__main__":
    main()
