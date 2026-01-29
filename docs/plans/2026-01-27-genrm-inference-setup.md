# GenRM Inference Setup Design

## Overview

Set up `nvidia/Qwen3-Nemotron-32B-GenRM-Principle` on EC2 g5.24xlarge for:
- Real-time API for scoring/evaluation
- Medium batch SFT data quality judging
- Future: fine-tuning with TRL SFTTrainer

## Instance Details

| Property | Value |
|----------|-------|
| Instance | Nvidia-qwen-judge |
| Type | g5.24xlarge |
| GPUs | 4x NVIDIA A10G (96GB VRAM total) |
| Region | us-east-1 |
| AMI | Deep Learning AMI |

## Architecture

```
S3 (model storage)
       │
       ▼
┌─────────────────┐
│ g5.24xlarge     │
│                 │
│  ┌───────────┐  │
│  │ vLLM      │  │  ← Tensor parallel across 4 GPUs
│  │ Server    │  │
│  └─────┬─────┘  │
│        │        │
│   Port 8000     │  ← OpenAI-compatible API
└────────┼────────┘
         │
         ▼
   Client requests
```

## Dependencies

**System (pre-installed on Deep Learning AMI):**
- NVIDIA drivers
- CUDA 12.x
- Python 3.10+

**Python packages:**
- `vllm>=0.6.0` - Inference engine with tensor parallelism
- `transformers>=4.45` - Model loading
- `boto3` - S3 access
- `hf-transfer` - Fast downloads
- `openai` - API client

## vLLM Configuration

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ~/models/qwen3-nemotron-32b \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `tensor-parallel-size` | 4 | Shard 32B model across 4 GPUs |
| `gpu-memory-utilization` | 0.90 | Leave 10% headroom for KV cache |
| `max-model-len` | 8192 | Balance context length vs memory |

## API Usage

**List models:**
```bash
curl http://localhost:8000/v1/models
```

**Score a conversation:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="qwen3-nemotron-32b",
    messages=[{"role": "user", "content": "Score this call: ..."}]
)
```

## Files

- Setup script: `scripts/setup_genrm_inference.sh`
- Model location (S3): `s3://rezora-data-pipeline-864981718771/models/qwen3-nemotron-32b/`
- Model location (EC2): `~/models/qwen3-nemotron-32b/`

## Future: Fine-tuning

When ready to fine-tune with TRL SFTTrainer:
- Use LoRA/QLoRA for memory efficiency
- Will need additional packages: `trl`, `peft`, `bitsandbytes`
- Separate design doc when data is ready
