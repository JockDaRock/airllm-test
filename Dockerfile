
# Stage 1: Download and prepare the model
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as model_downloader

WORKDIR /model

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir huggingface_hub

# Download model files in chunks
RUN for i in $(seq -w 1 44); do \
    python3 <<EOF \
import sys
from huggingface_hub import hf_hub_download

try:
    hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 
                    f'model-{sys.argv[1]}-of-00044.safetensors', 
                    local_dir='/model', 
                    local_dir_use_symlinks=False)
except Exception as e:
    print(f'Error downloading chunk {sys.argv[1]}: {e}', file=sys.stderr)
    sys.exit(1)
EOF
    "$i" \
done

# Stage 2: Final image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Copy the model files from the previous stage
COPY --from=model_downloader /model /app/model

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python3", "app.py"]
