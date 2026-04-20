# LIBERO Evaluation Workflow — Full Setup Guide

Two virtual environments are used. They never touch each other.
- **`openpi` venv (Python 3.12)**: Model server, training, norm-stat computation
- **`libero` venv (Python 3.8)**: LIBERO simulator client only

```
openpi venv (py3.12)                    libero venv (py3.8)
─────────────────────────────────       ────────────────────────────
scripts/serve_policy.py  (server)  ←WebSocket:8000→  examples/libero/main.py
scripts/train_pytorch.py (train)
scripts/compute_norm_stats.py
```

---

## 0. Prerequisites

All commands run on **Linux** (Ubuntu 22.04 recommended). NVIDIA GPU required.

```bash
# Check CUDA driver
nvidia-smi

# Check Python versions available
python3.12 --version    # should print 3.12.x
python3.8  --version    # should print 3.8.x

# If python3.8 is missing on Ubuntu 22.04:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev

# If python3.12 is missing:
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Git LFS needed for LeRobot dataset downloads
sudo apt install -y git-lfs
git lfs install
```

---

## 1. Create the openpi venv (Python 3.12)

```bash
# Go to the repo root
cd /path/to/openpi-torch       # adjust this to your actual path

# Create venv
python3.12 -m venv openpi_venv

# Activate it
source openpi_venv/bin/activate

# Upgrade pip first — old pip causes issues with newer packages
pip install --upgrade pip setuptools wheel

# Install the project (editable, so src/ changes are picked up immediately)
pip install -e "."

# Verify: torch and transformers must be importable
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import openpi; print('openpi OK')"

# Deactivate when done
deactivate
```

---

## 2. Create the libero venv (Python 3.8)

```bash
# Must be done outside of the openpi venv — open a fresh terminal or deactivate first

# Create venv with Python 3.8
python3.8 -m venv libero_venv

# Activate it
source libero_venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ── Core LIBERO dependencies ──────────────────────────────────────────────────
# MuJoCo + robosuite (simulation engine)
pip install mujoco==3.2.3
pip install robosuite==1.4.1

# LIBERO itself — use the HuggingFace-maintained fork (assets from HF Hub)
# This avoids the old asset-bundling issue
pip install libero==0.1.1

# ── OpenPI client (lightweight, no version conflicts) ─────────────────────────
# Install from the local packages/ directory — no heavyweight dependencies
pip install /path/to/openpi-torch/packages/openpi-client/
#   ↑ This installs websocket + numpy only. It works fine on Python 3.8.

# ── Utilities used by examples/libero/main.py ─────────────────────────────────
pip install tyro imageio imageio-ffmpeg numpy tqdm

# Verify
python -c "from libero.libero import benchmark; print('LIBERO OK')"
python -c "from openpi_client import websocket_client_policy; print('openpi_client OK')"

deactivate
```

> **Why not install torch in the libero venv?**
> `examples/libero/main.py` does not train or run the model.
> It only queries the model server over WebSocket. No torch needed in the libero venv.

---

## 3. Prepare the LIBERO Dataset

LIBERO raw data is in RLDS format. You need to convert it to LeRobot format for training.

### 3.1 — Set up environment variables (add to your shell profile)

```bash
# These paths are referenced throughout the codebase
# (same pattern as exp_template.sh)

export OPENPI_DATA_HOME="/data/openpi_cache"        # openpi model cache
export HF_LEROBOT_HOME="/data/lerobot"              # LeRobot datasets go here
export HF_HOME="/data/huggingface_cache"            # HuggingFace model cache
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Create the directories
mkdir -p $OPENPI_DATA_HOME $HF_LEROBOT_HOME $HF_HOME $HF_DATASETS_CACHE
```

### 3.2 — Download the raw LIBERO RLDS data

```bash
source openpi_venv/bin/activate

# The raw RLDS data is hosted on HuggingFace.
# We need tensorflow-datasets to download it.
# Install ONLY in the openpi venv (only needed for this conversion step):
pip install tensorflow-datasets==4.9.7 tensorflow-cpu==2.15.0

# Download raw LIBERO RLDS files via Hugging Face datasets API
# They will be stored in $HF_DATASETS_CACHE by default
python -c "
import tensorflow_datasets as tfds
# Download all four task suites (~10 GB total)
for name in ['libero_10_no_noops', 'libero_goal_no_noops',
             'libero_object_no_noops', 'libero_spatial_no_noops']:
    print(f'Downloading {name}...')
    tfds.load(name,
              data_dir='/data/libero_rlds',   # raw RLDS data goes here
              split='train')
    print(f'Done: {name}')
"
```

### 3.3 — Convert RLDS → LeRobot format

```bash
# Still in openpi venv
# This script writes to $HF_LEROBOT_HOME/your_username/libero/
# Edit REPO_NAME at the top of the script to set your output dataset name.

# Open the script and set REPO_NAME:
# REPO_NAME = "local/libero"   ← use "local/" prefix for local-only datasets

python examples/libero/convert_libero_data_to_lerobot.py \
    --data_dir /data/libero_rlds

# Expected output (takes ~30 min):
#   Writing to /data/lerobot/local/libero/
#   Episodes: libero_10_no_noops  (1000 eps)
#   Episodes: libero_goal_no_noops (990 eps)
#   Episodes: libero_object_no_noops (990 eps)
#   Episodes: libero_spatial_no_noops (990 eps)
#   Total: ~3970 episodes

# Verify the dataset was created:
ls $HF_LEROBOT_HOME/local/libero/
# Should show: data/  meta/  videos/

deactivate
```

---

## 4. Training Config

The existing `pi05_libero` config in `src/openpi/training/config.py` (around line 1264) needs
one path edit for your local setup. No new file is needed — just update the existing entry.

### 4.1 — Edit the pi05_libero TrainConfig

Open `src/openpi/training/config.py` and find the `pi05_libero` block.
Change the following three fields:

```python
# ── BEFORE (original) ──────────────────────────────────────────────────────────
TrainConfig(
    name="pi05_libero",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",       # ← HuggingFace Hub path
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    pytorch_weight_path="/path/to/your/pytorch_weight_path",   # ← placeholder
    num_train_steps=30_000,
),

# ── AFTER (your local paths) ───────────────────────────────────────────────────
TrainConfig(
    name="pi05_libero",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    assets_base_dir="/data/openpi_assets",          # ← where norm_stats will be saved
    checkpoint_base_dir="/data/openpi_checkpoints", # ← where checkpoints will be saved
    data=LeRobotLiberoDataConfig(
        repo_id="local/libero",                     # ← your local LeRobot dataset name
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=64,                # ← reduce from 256 if GPU memory is limited (e.g. 40GB)
    num_workers=8,                # ← adjust to your CPU count
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,       # ← original uses 10_000 for batch_size=256; scale down
        peak_lr=5e-5,
        decay_steps=30_000,
        decay_lr=5e-6,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    pytorch_weight_path=None,     # ← None means load from weight_loader (pi05_base)
    pytorch_training_precision="bfloat16",
    num_train_steps=30_000,
    save_interval=5_000,
    keep_period=5_000,
    val_interval=5_000,
    wandb_enabled=False,          # ← set True if you have wandb configured
),
```

> **Batch size guidance:**
> - 4× A100 80GB: `batch_size=256` (original setting)
> - 2× A100 80GB: `batch_size=128`
> - 1× A100 80GB: `batch_size=64`
> - 1× A100 40GB: `batch_size=32`

> **About `pytorch_weight_path`:**
> - `None` → downloads `pi05_base` JAX weights from GCS and converts automatically
> - `/path/to/checkpoint/` → load a previously saved PyTorch checkpoint
>   (format: the directory that contains `model.safetensors`)

### 4.2 — Create output directories

```bash
mkdir -p /data/openpi_assets
mkdir -p /data/openpi_checkpoints
mkdir -p /data/logs
```

---

## 5. Compute Normalization Statistics

This MUST be done before training. Normalization stats are per-dataset.

```bash
source openpi_venv/bin/activate

# Set env vars (same as training)
export OPENPI_DATA_HOME="/data/openpi_cache"
export HF_LEROBOT_HOME="/data/lerobot"
export HF_HOME="/data/huggingface_cache"

# Run compute_norm_stats.py
# --max-frames limits computation to first N frames (optional; remove for full dataset)
# Saves to: /data/openpi_assets/pi05_libero/local/libero/norm_stats.json
#                                ↑ config name  ↑ asset_id (= repo_id)
python scripts/compute_norm_stats.py \
    pi05_libero \
    --max-frames 50000

# Verify the file was created:
ls /data/openpi_assets/local/libero/
# Should show: norm_stats.json

# Inspect the stats:
python -c "
import openpi.shared.normalize as norm
stats = norm.load('/data/openpi_assets/local/libero')
for k, v in stats.items():
    print(f'{k}: mean shape={v.mean.shape}, std shape={v.std.shape}')
"

deactivate
```

---

## 6. Training Shell Script

Create a training script following the exact pattern of `scripts/exp_template.sh`.

```bash
cat > scripts/exp_libero.sh << 'EOF'
#!/bin/bash

VENV=$1
source ${VENV}/bin/activate

RANK=$OMPI_COMM_WORLD_RANK

cleanup() {
    trap '' SIGINT SIGTERM
    echo "Node $RANK: Caught signal! Cleaning up..."
    pkill -TERM -P $$
    wait
    echo "Node $RANK: Cleanup complete."
    exit 0
}
trap cleanup SIGINT SIGTERM

if [ -z "$MASTER_ADDR" ]; then
    echo "Error: MASTER_ADDR is not set."
    exit 1
fi

# ── Data / cache paths (match exp_template.sh pattern) ──────────────────────
export OPENPI_DATA_HOME="/data/openpi_cache"
export HF_LEROBOT_HOME="/data/lerobot"
export HF_HOME="/data/huggingface_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
unset LEROBOT_HOME

mkdir -p /data/logs

# ── Variables to modify per experiment ──────────────────────────────────────
CONFIG_NAME=pi05_libero
EXP_NAME=260421_libero_sanity_check
MASTER_PORT=29502
# BASE_WEIGHT_PATH: leave empty to download from GCS (first run)
# after first download, set to the local cached path to skip re-download
BASE_WEIGHT_PATH=""
# DATALIST_FILE_PATH: leave empty unless you have a custom data list file
DATALIST_FILE_PATH=""
# ─────────────────────────────────────────────────────────────────────────────

echo "Node $RANK: Connecting to master at $MASTER_ADDR:$MASTER_PORT"

# Build optional arguments
EXTRA_ARGS=""
if [ -n "$BASE_WEIGHT_PATH" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --pytorch_weight_path $BASE_WEIGHT_PATH"
fi
if [ -n "$DATALIST_FILE_PATH" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --data.datalist_file_path $DATALIST_FILE_PATH"
fi

JAX_PLATFORMS="cpu,cuda" TORCH_NCCL_BLOCKING_WAIT=1 OMP_NUM_THREADS=16 torchrun \
    --nnodes=$OMPI_COMM_WORLD_SIZE \
    --nproc_per_node=gpu \
    --node-rank=$RANK \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    --rdzv_conf timeout=1000 \
    train_pytorch.py \
    "$CONFIG_NAME" \
    --exp-name $EXP_NAME \
    $EXTRA_ARGS \
    2>&1 | tee /data/logs/"${CONFIG_NAME}"_"${EXP_NAME}"_$(date +%y%m%d)_$(date +%H%M%S)_rank"${RANK}".log &

wait $!
EOF

chmod +x scripts/exp_libero.sh
```

---

## 7. Run Training

### Option A — Single Node, Single GPU (for quick testing)

```bash
source openpi_venv/bin/activate

export OPENPI_DATA_HOME="/data/openpi_cache"
export HF_LEROBOT_HOME="/data/lerobot"
export HF_HOME="/data/huggingface_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Change to scripts/ directory — train_pytorch.py uses relative imports
cd scripts/

python train_pytorch.py \
    pi05_libero \
    --exp-name 260421_libero_test_single \
    2>&1 | tee /data/logs/pi05_libero_260421_$(date +%H%M%S)_rank0.log

cd ..
deactivate
```

### Option B — Single Node, Multiple GPUs

```bash
source openpi_venv/bin/activate

export OPENPI_DATA_HOME="/data/openpi_cache"
export HF_LEROBOT_HOME="/data/lerobot"
export HF_HOME="/data/huggingface_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

cd scripts/

# Replace --nproc_per_node=4 with your actual GPU count
JAX_PLATFORMS="cpu,cuda" TORCH_NCCL_BLOCKING_WAIT=1 OMP_NUM_THREADS=16 \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_pytorch.py \
    pi05_libero \
    --exp-name 260421_libero_4gpu \
    2>&1 | tee /data/logs/pi05_libero_260421_$(date +%H%M%S)_rank0.log

cd ..
deactivate
```

### Option C — Multi-Node Training (via MPI wrapper, same as exp_template.sh)

```bash
# On the master node:
source openpi_venv/bin/activate

export VIRTUAL_ENV="$(pwd)/openpi_venv"
export OPENPI_DATA_HOME="/data/openpi_cache"
export HF_LEROBOT_HOME="/data/lerobot"
export HF_HOME="/data/huggingface_cache"

# The hostfile lists all nodes, one per line: <ip> slots=8
# Example hostfile:
#   192.168.1.10 slots=8
#   192.168.1.11 slots=8
bash scripts/train_mpi_wrapper.sh scripts/exp_libero.sh /path/to/hostfile

deactivate
```

---

## 8. Monitor Training

```bash
# ── Real-time log tail ────────────────────────────────────────────────────────
tail -f /data/logs/pi05_libero_260421_*.log

# ── Loss log (plain text, written by LogFile class in train_pytorch.py) ──────
tail -f /data/openpi_checkpoints/pi05_libero/260421_libero_test_single/loss.log
# Format: [2026-04-21 10:30:15] STEP   1000: train_loss=0.0523 ...

# ── TensorBoard ───────────────────────────────────────────────────────────────
source openpi_venv/bin/activate
tensorboard --logdir /data/openpi_checkpoints/pi05_libero/260421_libero_test_single/tensorboard/
# Then open http://localhost:6006 in your browser

# ── Check what checkpoints were saved ────────────────────────────────────────
ls -la /data/openpi_checkpoints/pi05_libero/260421_libero_test_single/
# Expected:
#   5000/          ← checkpoint at step 5000
#     model.safetensors
#     optimizer.pt
#     assets/
#       norm_stats.json
#   10000/
#   ...
#   loss.log
#   tensorboard/
#   wandb_id.txt   (only if wandb_enabled=True)
```

---

## 9. Inference — Two-Venv Approach

After training, you run two processes in **two separate terminals**.

---

### Terminal 1: Start the Policy Server (openpi venv, Python 3.12)

```bash
source /path/to/openpi-torch/openpi_venv/bin/activate

export OPENPI_DATA_HOME="/data/openpi_cache"
export HF_HOME="/data/huggingface_cache"

# ── Option A: Use a checkpoint from your training ────────────────────────────
python scripts/serve_policy.py \
    --env LIBERO \
    --policy.config pi05_libero \
    --policy.dir /data/openpi_checkpoints/pi05_libero/260421_libero_test_single/30000 \
    --port 8000

# ── Option B: Use the published pretrained pi05_libero weights from GCS ──────
# (downloads automatically on first run, cached in $OPENPI_DATA_HOME)
python scripts/serve_policy.py \
    --env LIBERO \
    --port 8000
# Note: --env LIBERO without --policy.config uses the DEFAULT_CHECKPOINT
#   which is: config="pi05_libero", dir="gs://openpi-assets/checkpoints/pi05_libero"

# ── Option C: Pretrained weights + custom prompt fallback ────────────────────
python scripts/serve_policy.py \
    --env LIBERO \
    --default-prompt "pick up the red block and place it in the basket" \
    --port 8000

# Expected server output:
#   [10:30:15] Creating server (host: your-machine, ip: 192.168.x.x)
#   [10:30:15] Server started on 0.0.0.0:8000
```

> **The server must be running before you launch the client.**

---

### Terminal 2: Run the LIBERO Client (libero venv, Python 3.8)

```bash
source /path/to/openpi-torch/libero_venv/bin/activate

export MUJOCO_GL=egl              # Use EGL for headless GPU rendering (no display needed)
export MUJOCO_EGL_DEVICE_ID=0    # GPU index for MuJoCo (usually 0)

cd /path/to/openpi-torch

# ── Evaluate on libero_spatial (10 tasks × 50 trials = 500 rollouts) ─────────
python examples/libero/main.py \
    --task-suite-name libero_spatial \
    --num-trials-per-task 50 \
    --host 0.0.0.0 \
    --port 8000 \
    --replan-steps 5 \
    --resize-size 224 \
    --video-out-path /data/libero/videos/libero_spatial \
    2>&1 | tee /data/logs/libero_spatial_eval_$(date +%y%m%d_%H%M%S).log

# ── Evaluate on libero_object ─────────────────────────────────────────────────
python examples/libero/main.py \
    --task-suite-name libero_object \
    --num-trials-per-task 50 \
    --host 0.0.0.0 \
    --port 8000 \
    --video-out-path /data/libero/videos/libero_object \
    2>&1 | tee /data/logs/libero_object_eval_$(date +%y%m%d_%H%M%S).log

# ── Evaluate on libero_goal ───────────────────────────────────────────────────
python examples/libero/main.py \
    --task-suite-name libero_goal \
    --num-trials-per-task 50 \
    --host 0.0.0.0 \
    --port 8000 \
    --video-out-path /data/libero/videos/libero_goal \
    2>&1 | tee /data/logs/libero_goal_eval_$(date +%y%m%d_%H%M%S).log

# ── Evaluate on libero_10 ─────────────────────────────────────────────────────
python examples/libero/main.py \
    --task-suite-name libero_10 \
    --num-trials-per-task 50 \
    --host 0.0.0.0 \
    --port 8000 \
    --video-out-path /data/libero/videos/libero_10 \
    2>&1 | tee /data/logs/libero_10_eval_$(date +%y%m%d_%H%M%S).log

deactivate
```

#### What the client outputs:

```
Task: pick the orange from the tray and put it in the basket
Starting episode 1...
Success: True
# episodes completed so far: 1
# successes: 1 (100.0%)
...
Current task success rate: 0.96
Current total success rate: 0.95
```

---

## 10. Evaluate Multiple Checkpoints (Script)

Create this script to evaluate all saved checkpoints automatically.

```bash
cat > scripts/run_libero_eval.sh << 'EOF'
#!/bin/bash
# Usage: bash scripts/run_libero_eval.sh <checkpoint_step> [task_suite]
# Example: bash scripts/run_libero_eval.sh 30000 libero_spatial

CHECKPOINT_STEP=${1:-30000}
TASK_SUITE=${2:-libero_spatial}
CONFIG_NAME=pi05_libero
EXP_NAME=260421_libero_test_single

CHECKPOINT_DIR="/data/openpi_checkpoints/${CONFIG_NAME}/${EXP_NAME}/${CHECKPOINT_STEP}"
LOG_DIR="/data/logs"
VIDEO_DIR="/data/libero/videos/${TASK_SUITE}_step${CHECKPOINT_STEP}"
OPENPI_REPO="/path/to/openpi-torch"   # ← change this

mkdir -p $LOG_DIR $VIDEO_DIR

echo "====================================================="
echo "  Config:     $CONFIG_NAME"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Task suite: $TASK_SUITE"
echo "====================================================="

# ── Step 1: Start server in background ────────────────────────────────────────
source ${OPENPI_REPO}/openpi_venv/bin/activate

export OPENPI_DATA_HOME="/data/openpi_cache"
export HF_HOME="/data/huggingface_cache"

python ${OPENPI_REPO}/scripts/serve_policy.py \
    --env LIBERO \
    --policy.config $CONFIG_NAME \
    --policy.dir $CHECKPOINT_DIR \
    --port 8000 \
    > ${LOG_DIR}/server_${CONFIG_NAME}_step${CHECKPOINT_STEP}.log 2>&1 &

SERVER_PID=$!
echo "Server started (PID: $SERVER_PID). Waiting 20s for model to load..."
sleep 20

# ── Step 2: Run client ─────────────────────────────────────────────────────────
deactivate
source ${OPENPI_REPO}/libero_venv/bin/activate

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

python ${OPENPI_REPO}/examples/libero/main.py \
    --task-suite-name $TASK_SUITE \
    --num-trials-per-task 50 \
    --host 0.0.0.0 \
    --port 8000 \
    --video-out-path $VIDEO_DIR \
    2>&1 | tee ${LOG_DIR}/${TASK_SUITE}_step${CHECKPOINT_STEP}_$(date +%y%m%d_%H%M%S).log

# ── Step 3: Clean up ──────────────────────────────────────────────────────────
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
deactivate
echo "Done. Results in: ${LOG_DIR}"
EOF

chmod +x scripts/run_libero_eval.sh
```

```bash
# Run it:
bash scripts/run_libero_eval.sh 30000 libero_spatial
bash scripts/run_libero_eval.sh 30000 libero_object
bash scripts/run_libero_eval.sh 30000 libero_goal
bash scripts/run_libero_eval.sh 30000 libero_10
```

---

## 11. Pass/Fail Criteria

After running evaluation, extract results from the log:

```bash
# Extract final success rates from log files
grep "Current total success rate" /data/logs/libero_spatial_*.log | tail -1
grep "Current total success rate" /data/logs/libero_object_*.log  | tail -1
grep "Current total success rate" /data/logs/libero_goal_*.log    | tail -1
grep "Current total success rate" /data/logs/libero_10_*.log      | tail -1
```

Expected results for `pi05_libero` pretrained weights:

| Suite | Expected | Pass Threshold |
|-------|----------|----------------|
| libero_spatial | ~98.8% | ≥ 95% |
| libero_object | ~98.2% | ≥ 95% |
| libero_goal | ~98.0% | ≥ 95% |
| libero_10 | ~92.4% | ≥ 85% |

If your fine-tuned model underperforms → check:
1. Norm stats computed correctly? (`ls /data/openpi_assets/local/libero/norm_stats.json`)
2. Correct checkpoint step loaded? (try 10k, 20k, 30k)
3. Loss log shows convergence? (`tail /data/openpi_checkpoints/.../loss.log`)

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `libero.benchmark not found` | Wrong venv | `source libero_venv/bin/activate` |
| `ConnectionRefusedError :8000` | Server not started | Start server first (Terminal 1) |
| `GLEW initialization failed` | No display | `export MUJOCO_GL=egl` |
| `EGL device not found` | Wrong GPU id | `export MUJOCO_EGL_DEVICE_ID=0` |
| `norm_stats.json not found` | Stats not computed | Run `compute_norm_stats.py` first |
| `model.safetensors not found` | Wrong checkpoint path | Check `--policy.dir` path exists |
| `CUDA out of memory` (server) | Model too big | Use `batch_size=32`, or `--pytorch_training_precision bfloat16` |
| `FileNotFoundError: libero/rlds` | Wrong data_dir in convert script | Check `--data_dir` matches where RLDS was downloaded |
| `KeyError: 'prompt'` during training | `prompt_from_task=False` | Make sure `DataConfig(prompt_from_task=True)` |
| Training loss not decreasing | LR too low or data issue | Check norm_stats; try `peak_lr=1e-4` for quick test |
| `numba==0.53.1` install fails | C compiler issue | `sudo apt install -y gcc python3.8-dev` |

---

## Summary: Complete Order of Operations

```
[One time setup]
1. python3.12 -m venv openpi_venv && pip install -e .
2. python3.8  -m venv libero_venv && pip install libero openpi-client tyro imageio

[Per dataset]
3. python examples/libero/convert_libero_data_to_lerobot.py --data_dir /data/libero_rlds
4. Edit src/openpi/training/config.py → update repo_id, paths in pi05_libero config
5. python scripts/compute_norm_stats.py pi05_libero

[Per experiment]
6. Edit scripts/exp_libero.sh → set CONFIG_NAME, EXP_NAME
7. source openpi_venv/bin/activate
8. bash scripts/exp_libero.sh openpi_venv   (or use direct torchrun for single node)
9. Monitor: tail -f /data/logs/pi05_libero_*.log

[Evaluation]
10. Terminal 1: source openpi_venv/bin/activate
               python scripts/serve_policy.py --env LIBERO --policy.config pi05_libero \
                 --policy.dir /data/openpi_checkpoints/pi05_libero/<exp>/<step> --port 8000

11. Terminal 2: source libero_venv/bin/activate
               export MUJOCO_GL=egl
               python examples/libero/main.py --task-suite-name libero_spatial \
                 --num-trials-per-task 50 --port 8000
```
