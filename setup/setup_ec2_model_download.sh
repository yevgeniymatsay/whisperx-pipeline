#!/bin/bash
# Setup script for downloading HuggingFace model to EC2
# Model: nvidia/Qwen3-Nemotron-32B-GenRM-Principle
#
# Prerequisites:
#   - EC2 instance with IAM role attached (ec2-model-download-policy.json)
#   - EBS volume with at least 200GB free space
#   - Python 3.10+ installed

set -euo pipefail

# Configuration
MODEL_ID="nvidia/Qwen3-Nemotron-32B-GenRM-Principle"
S3_BUCKET="rezora-data-pipeline-864981718771"
S3_PREFIX="models/qwen3-nemotron-32b"
AWS_REGION="us-east-2"
SECRET_NAME="hf-token"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check disk space
check_disk_space() {
    local required_gb=150
    local available_gb=$(df -BG ~ | awk 'NR==2 {print $4}' | sed 's/G//')

    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Insufficient disk space. Need ${required_gb}GB, have ${available_gb}GB"
        exit 1
    fi
    log_info "Disk space OK: ${available_gb}GB available"
}

# Install dependencies
install_deps() {
    log_info "Installing Python dependencies..."
    pip install --upgrade pip
    pip install huggingface_hub[hf_transfer] boto3

    # Enable fast transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1
    log_info "hf_transfer enabled for fast downloads"
}

# Get HF token from Secrets Manager
get_hf_token() {
    log_info "Retrieving HuggingFace token from Secrets Manager..."
    HF_TOKEN=$(aws secretsmanager get-secret-value \
        --secret-id "$SECRET_NAME" \
        --region "$AWS_REGION" \
        --query 'SecretString' \
        --output text)

    if [ -z "$HF_TOKEN" ]; then
        log_error "Failed to retrieve HF token"
        exit 1
    fi
    log_info "HF token retrieved successfully"
}

# Download model
download_model() {
    log_info "Downloading model: $MODEL_ID"
    log_info "This will take a while (~65GB)..."

    # Use huggingface-cli for download with token
    HF_TOKEN="$HF_TOKEN" huggingface-cli download "$MODEL_ID" \
        --local-dir "./models/qwen3-nemotron-32b" \
        --local-dir-use-symlinks False

    log_info "Model download complete!"
}

# Sync to S3
sync_to_s3() {
    log_info "Syncing model to S3: s3://${S3_BUCKET}/${S3_PREFIX}/"

    aws s3 sync "./models/qwen3-nemotron-32b" \
        "s3://${S3_BUCKET}/${S3_PREFIX}/" \
        --region "$AWS_REGION"

    log_info "S3 sync complete!"
}

# Verify download
verify_download() {
    log_info "Verifying download..."

    # Check for key files
    local model_dir="./models/qwen3-nemotron-32b"

    if [ -f "${model_dir}/config.json" ]; then
        log_info "✓ config.json found"
    else
        log_error "✗ config.json missing"
        exit 1
    fi

    # Check total size
    local size=$(du -sh "$model_dir" | awk '{print $1}')
    log_info "Total model size: $size"
}

# Main
main() {
    log_info "=== HuggingFace Model Download Setup ==="
    log_info "Model: $MODEL_ID"

    check_disk_space
    install_deps
    get_hf_token
    download_model
    verify_download
    sync_to_s3

    log_info "=== Setup Complete ==="
    log_info "Model available at: ./models/qwen3-nemotron-32b"
    log_info "S3 backup at: s3://${S3_BUCKET}/${S3_PREFIX}/"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
