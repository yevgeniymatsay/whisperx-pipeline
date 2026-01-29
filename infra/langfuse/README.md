# Langfuse Self-Hosted Observability

Observability stack for GenRM judge pipeline using Langfuse.

## What Langfuse Provides

- **Request tracing** - Every GenRM API call logged with full prompt/response
- **Latency monitoring** - p50, p95, p99 percentiles across all calls
- **Token tracking** - Prompt tokens, completion tokens per request
- **Error dashboard** - Stack traces, error rates, debugging
- **Analytics** - Throughput, cost estimation, model comparisons

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│ EC2 g5.24xlarge                                          │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ vLLM Server (port 8000)                            │  │
│  │ + OpenTelemetry auto-instrumentation               │  │
│  │ Qwen3-Nemotron-32B                                 │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       │                                  │
│               OTLP traces                                │
│                       ▼                                  │
└───────────────────────┼──────────────────────────────────┘
                        │
           ┌────────────▼────────────┐
           │ Langfuse (localhost)    │
           │ - langfuse-web :3000    │
           │ - PostgreSQL            │
           │ - ClickHouse (analytics)│
           │ - Redis (cache)         │
           │ - MinIO (storage)       │
           └─────────────────────────┘
```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| `langfuse-web` | 3000 | Web UI + API |
| `langfuse-worker` | - | Async processing |
| `postgres` | 5432 | Transactional DB |
| `clickhouse` | 8123, 9000 | Analytics OLAP |
| `redis` | 6379 | Cache + queue |
| `minio` | 9000, 9001 | S3-compatible storage |

## Quick Start

### 1. Configure Environment

```bash
cd infra/langfuse

# Copy example config
cp .env.example .env

# Generate secure credentials
openssl rand -base64 32  # For NEXTAUTH_SECRET, SALT
openssl rand -hex 32     # For ENCRYPTION_KEY

# Edit .env with generated values
vim .env
```

### 2. Start Langfuse

```bash
docker-compose up -d
```

### 3. Access Web UI

Open http://localhost:3000 and create an account.

### 4. Get API Keys

1. Log in to Langfuse
2. Go to Settings > API Keys
3. Create a new key pair
4. Note the Public Key and Secret Key

## vLLM OpenTelemetry Integration

### Install on EC2

```bash
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install
```

### Create Traced Startup Script

Save as `~/start_genrm_traced.sh` on EC2:

```bash
#!/bin/bash
source ~/.venv/bin/activate

export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
export OTEL_SERVICE_NAME=vllm-genrm

opentelemetry-instrument \
    python -m vllm.entrypoints.openai.api_server \
    --model ~/models/qwen3-nemotron-32b \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000
```

### What Gets Traced

- Every `/v1/chat/completions` request
- Full prompt text
- Response + completion tokens
- Latency breakdown
- Errors with stack traces

## Python SDK Integration (Optional)

For more detailed tracing from the GenRM CLI:

```bash
pip install langfuse
```

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="http://localhost:3000"
)

# Create a trace for each GenRM evaluation
trace = langfuse.trace(name="genrm-evaluate")

# Log each principle evaluation as a span
span = trace.span(
    name="evaluate-integrity",
    input={"messages": [...], "principle": "integrity"},
    output={"judgment": "Yes", "score": 1.0}
)
```

## Verification

1. Start Langfuse: `docker-compose up -d`
2. Open http://localhost:3000
3. Create account and API keys
4. Start vLLM with tracing on EC2
5. Run GenRM CLI:
   ```bash
   python -m scripts.whisperx_pipeline.genrm.cli judge \
       --s3-key runs/VIDEO/RUN/calls/CALL/spk_turns.json
   ```
6. Check Langfuse UI for traces

## Troubleshooting

### Services not starting

```bash
# Check logs
docker-compose logs langfuse-web

# Restart specific service
docker-compose restart langfuse-web
```

### Database issues

```bash
# Reset everything (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### Port conflicts

If ports 3000, 9000, or 9001 are in use:

```bash
# Check what's using port 3000
lsof -i :3000

# Modify ports in docker-compose.yml
# e.g., change "3000:3000" to "3001:3000"
```

## Cleanup

```bash
# Stop services
docker-compose down

# Stop and remove volumes (deletes all data)
docker-compose down -v
```
