#!/bin/bash
# Setup script for Qwen3-Nemotron-32B GenRM inference on g5.24xlarge
# Deep Learning AMI assumed (CUDA/drivers pre-installed)

set -euo pipefail

echo "=== Setting up Qwen3-Nemotron-32B GenRM Inference ==="
echo "Instance: g5.24xlarge (4x A10G GPUs)"
echo ""

# 1. Verify GPU setup
echo "[1/5] Verifying GPU setup..."
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# 2. Create Python virtual environment
echo "[2/5] Setting up Python environment..."
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install -U pip wheel setuptools -q

# 3. Install inference packages
echo "[3/5] Installing packages..."
pip install "vllm>=0.6.0" -q
pip install "transformers>=4.45" -q
pip install boto3 hf-transfer openai -q
echo "Packages installed."

# 4. Pull model from S3
echo "[4/5] Pulling model from S3 (this may take a few minutes)..."
mkdir -p ~/models
aws s3 sync s3://rezora-data-pipeline-864981718771/models/qwen3-nemotron-32b/ \
    ~/models/qwen3-nemotron-32b/ \
    --only-show-errors

# Verify model files
if [ -f ~/models/qwen3-nemotron-32b/config.json ]; then
    MODEL_SIZE=$(du -sh ~/models/qwen3-nemotron-32b | awk '{print $1}')
    echo "Model downloaded: $MODEL_SIZE"
else
    echo "ERROR: Model download failed - config.json not found"
    exit 1
fi

# 5. Create convenience scripts
echo "[5/5] Creating convenience scripts..."

# Start server script
cat > ~/start_genrm.sh << 'EOF'
#!/bin/bash
source ~/.venv/bin/activate
echo "Starting vLLM server on port 8000..."
python -m vllm.entrypoints.openai.api_server \
    --model ~/models/qwen3-nemotron-32b \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000
EOF
chmod +x ~/start_genrm.sh

# Start server in background script
cat > ~/start_genrm_bg.sh << 'EOF'
#!/bin/bash
source ~/.venv/bin/activate
echo "Starting vLLM server in background..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model ~/models/qwen3-nemotron-32b \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000 \
    > ~/vllm.log 2>&1 &
echo "Server starting... check ~/vllm.log for status"
echo "PID: $!"
EOF
chmod +x ~/start_genrm_bg.sh

# Test script
cat > ~/test_genrm.sh << 'EOF'
#!/bin/bash
echo "Testing GenRM API..."
curl -s http://localhost:8000/v1/models | python3 -m json.tool
echo ""
echo "Sending test message..."
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-nemotron-32b", "messages": [{"role": "user", "content": "Hello, are you working?"}], "max_tokens": 50}' | python3 -m json.tool
EOF
chmod +x ~/test_genrm.sh

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the server:"
echo "  ~/start_genrm.sh        # Foreground (see logs)"
echo "  ~/start_genrm_bg.sh     # Background (logs to ~/vllm.log)"
echo ""
echo "To test:"
echo "  ~/test_genrm.sh"
echo ""
