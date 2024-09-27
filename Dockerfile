# Stage 1: Download model
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as model_downloader

WORKDIR /model

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir huggingface_hub

# Download each safetensor file in a separate layer
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00001-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00002-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00003-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00004-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00005-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00006-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00007-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00008-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00009-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00010-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00011-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00012-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00013-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00014-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00015-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00016-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00017-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00018-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00019-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00020-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00021-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00022-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00023-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00024-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00025-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00026-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00027-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00028-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00029-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00030-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00031-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00032-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00033-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00034-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00035-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00036-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00037-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00038-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00039-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00040-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00041-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00042-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00043-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit', 'model-00044-of-00044.safetensors', local_dir='/model', local_dir_use_symlinks=False)"

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
COPY --from=model_downloader /model /app/model/model-00044-of-00001.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00002.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00003.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00004.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00005.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00006.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00007.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00008.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00009.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00010.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00011.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00012.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00013.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00014.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00015.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00016.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00017.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00018.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00019.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00020.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00021.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00022.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00023.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00024.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00025.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00026.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00027.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00028.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00029.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00030.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00031.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00032.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00033.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00034.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00035.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00036.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00037.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00038.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00039.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00040.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00041.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00042.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00043.safetensors
COPY --from=model_downloader /model /app/model/model-00044-of-00044.safetensors


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python3", "app.py"]