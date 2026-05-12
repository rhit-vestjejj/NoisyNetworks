# GPU-ready image for running NoisyNet on Atari.
# Base ships with CUDA + cuDNN + PyTorch already wired up, so the only thing
# left to install is the RL stack.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# unrar is needed by AutoROM to unpack the Atari ROMs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git unrar && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    AutoROM --accept-license

COPY . .

# Default command: NoisyNet-DQN on Breakout. Override at `docker run` time:
#   docker run --gpus all noisynet python train_atari.py --algo dueling \
#       --env-id ALE/Asteroids-v5 --out-dir runs/asteroids_dueling
CMD ["python", "train_atari.py", \
     "--algo", "dqn", \
     "--env-id", "ALE/Breakout-v5", \
     "--out-dir", "runs/breakout_dqn"]
