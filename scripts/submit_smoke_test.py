# scripts/submit_smoke_test.py
"""Submit smoke test job to AWS Batch."""
import boto3
from datetime import datetime


def submit_smoke_test(queue: str = "whisperx-transcription-queue") -> str:
    """Submit smoke test job."""
    batch = boto3.client("batch", region_name="us-east-2")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    response = batch.submit_job(
        jobName=f"whisperx-smoke-test-{run_id}",
        jobQueue=queue,
        jobDefinition="whisperx-pipeline",
        containerOverrides={
            "command": ["--smoke-test"]
        }
    )

    job_id = response["jobId"]
    print(f"Submitted smoke test job: {job_id}")
    print(f"\nMonitor with:")
    print(f"  aws batch describe-jobs --jobs {job_id} --region us-east-2")
    print(f"\nView logs in CloudWatch:")
    print(f"  /aws/batch/whisperx-pipeline")
    return job_id


if __name__ == "__main__":
    submit_smoke_test()
