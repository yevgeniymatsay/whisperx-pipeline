#!/bin/bash
# vLLM startup script with OpenTelemetry tracing for Langfuse
#
# Copy this to EC2: scp infra/langfuse/start_genrm_traced.sh ubuntu@<EC2_IP>:~/
#
# Prerequisites on EC2:
#   pip install opentelemetry-distro opentelemetry-exporter-otlp
#   opentelemetry-bootstrap -a install

set -e

# Activate virtual environment
source ~/.venv/bin/activate

# Configure OpenTelemetry to send traces to Langfuse
# Note: Requires SSH tunnel from local machine running Langfuse
#   ssh -R 4318:localhost:4318 -i ~/.ssh/whisperx-key-east1.pem ubuntu@<EC2_IP>
export OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-"http://localhost:4318/v1/traces"}
export OTEL_SERVICE_NAME=${OTEL_SERVICE_NAME:-"vllm-genrm"}

# Optional: Set resource attributes for better trace organization
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=development,service.version=1.0.0"

# Model path
MODEL_PATH=${MODEL_PATH:-"$HOME/models/qwen3-nemotron-32b"}

echo "Starting vLLM with OpenTelemetry tracing..."
echo "  OTEL_EXPORTER_OTLP_ENDPOINT: $OTEL_EXPORTER_OTLP_ENDPOINT"
echo "  OTEL_SERVICE_NAME: $OTEL_SERVICE_NAME"
echo "  MODEL_PATH: $MODEL_PATH"

# Start vLLM with OpenTelemetry instrumentation
opentelemetry-instrument \
    python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000 \
    "$@"
