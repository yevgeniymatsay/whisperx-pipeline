# docker/whisperx/Dockerfile
# Use PyTorch base image to avoid CUDA/torch version mismatches
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Unified cache directories (critical for consistent caching behavior)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV XDG_CACHE_HOME=/app/.cache
ENV PYTHONPATH=/app

# Install CUDA-specific wheels for torch audio (must match base image CUDA)
RUN pip install --no-cache-dir \
    torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

COPY docker/whisperx/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Silero VAD model during build (avoids runtime network calls in Batch)
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)"

COPY pipeline/ /app/pipeline/

# NOTE: Don't assert CUDA at build time (CI may not have GPU)
# Runtime smoke test command instead:
#   python -c "import torch; assert torch.cuda.is_available()"

ENTRYPOINT ["python", "-m", "pipeline.cli"]
