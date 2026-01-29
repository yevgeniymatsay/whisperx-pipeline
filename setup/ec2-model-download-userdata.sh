#!/bin/bash
# EC2 User Data script - downloads HuggingFace model and syncs to S3
# This runs automatically on instance launch

exec > >(tee /var/log/model-download.log) 2>&1
echo "=== Model Download Started: $(date) ==="

# Configuration
MODEL_ID="nvidia/Qwen3-Nemotron-32B-GenRM-Principle"
S3_BUCKET="rezora-data-pipeline-864981718771"
S3_PREFIX="models/qwen3-nemotron-32b"
AWS_REGION="us-east-2"
SECRET_NAME="hf-token"
MODEL_DIR="/home/ec2-user/models/qwen3-nemotron-32b"

# Install dependencies
echo "Installing dependencies..."
dnf install -y python3.11 python3.11-pip git
python3.11 -m pip install --upgrade pip
python3.11 -m pip install huggingface_hub[hf_transfer] boto3

# Enable fast transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Get HF token from Secrets Manager
echo "Retrieving HuggingFace token..."
HF_TOKEN=$(aws secretsmanager get-secret-value \
    --secret-id "$SECRET_NAME" \
    --region "$AWS_REGION" \
    --query 'SecretString' \
    --output text)

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Failed to retrieve HF token"
    aws s3 cp /var/log/model-download.log "s3://${S3_BUCKET}/${S3_PREFIX}/download-FAILED.log"
    exit 1
fi
echo "HF token retrieved successfully"

# Create model directory
mkdir -p "$MODEL_DIR"
cd /home/ec2-user

# Download model
echo "Downloading model: $MODEL_ID"
echo "This will take 20-40 minutes..."
HF_TOKEN="$HF_TOKEN" huggingface-cli download "$MODEL_ID" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

if [ $? -ne 0 ]; then
    echo "ERROR: Model download failed"
    aws s3 cp /var/log/model-download.log "s3://${S3_BUCKET}/${S3_PREFIX}/download-FAILED.log"
    exit 1
fi

# Verify download
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "ERROR: config.json not found after download"
    aws s3 cp /var/log/model-download.log "s3://${S3_BUCKET}/${S3_PREFIX}/download-FAILED.log"
    exit 1
fi

MODEL_SIZE=$(du -sh "$MODEL_DIR" | awk '{print $1}')
echo "Model downloaded successfully. Size: $MODEL_SIZE"

# Sync to S3
echo "Syncing to S3: s3://${S3_BUCKET}/${S3_PREFIX}/"
aws s3 sync "$MODEL_DIR" "s3://${S3_BUCKET}/${S3_PREFIX}/" --region "$AWS_REGION"

if [ $? -ne 0 ]; then
    echo "ERROR: S3 sync failed"
    aws s3 cp /var/log/model-download.log "s3://${S3_BUCKET}/${S3_PREFIX}/download-FAILED.log"
    exit 1
fi

# Upload success log
echo "=== Model Download Completed: $(date) ==="
aws s3 cp /var/log/model-download.log "s3://${S3_BUCKET}/${S3_PREFIX}/download-SUCCESS.log"

# Create completion marker
echo "{\"status\": \"complete\", \"model\": \"$MODEL_ID\", \"size\": \"$MODEL_SIZE\", \"timestamp\": \"$(date -Iseconds)\"}" > /tmp/download-complete.json
aws s3 cp /tmp/download-complete.json "s3://${S3_BUCKET}/${S3_PREFIX}/download-complete.json"

echo "All done! Model is now in S3."
echo "You can terminate this instance."
