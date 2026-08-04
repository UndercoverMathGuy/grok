# SEMIFINAL v2 trainer for cloud CUDA boxes. (rebuild: repo now public)
# RTX 5090 is Blackwell (sm_120): needs torch >= 2.7 built against CUDA 12.8.
# -devel (not -runtime): torch.compile's Inductor needs the C++ toolchain.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN pip install --no-cache-dir wandb safetensors numpy

WORKDIR /app
COPY train_semifinal_torch.py .

# Run outputs land in /runs — mount a volume to persist across container
# restarts (idempotent resume: runs with spectra.npz are skipped). wandb
# gets every finished run as an artifact regardless.
ENV RUNS_DIR=/runs
VOLUME /runs

# Pass -e WANDB_API_KEY=... (and optionally WANDB_PROJECT) at `docker run`.
ENTRYPOINT ["python", "-u", "train_semifinal_torch.py"]
