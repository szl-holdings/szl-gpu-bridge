# The pre-built burst-training image — every dependency baked in, no runtime pip.
# This is the top-company pattern for spot-style GPU fleets (Salad, spot pools):
# the container is a pure worker; jobs resume from durable checkpoints.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /work
RUN pip install --no-cache-dir \
    "transformers==4.48.0" "trl==0.13.0" "peft==0.14.0" "datasets==3.2.0" \
    "huggingface_hub>=0.27" sentencepiece protobuf
COPY salad_worker.py .
ENV HF_HOME=/work/hf-cache
# Salad sends SIGTERM before reclaiming a machine; the worker checkpoints on it.
STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "salad_worker.py"]
