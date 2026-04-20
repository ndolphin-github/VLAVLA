# Model Evaluation & Benchmarking Guide + Code Improvement List

---

## Part 1: LIBERO Evaluation — Full Step-by-Step Process

### 1.0 The Python Version Problem

| Component | Python | PyTorch | CUDA |
|-----------|--------|---------|------|
| **OpenPI-Torch** (model server) | ≥ 3.11 | 2.7.1 | 12.2 |
| **LIBERO** (sim client) | 3.8.13 | 1.11.0 | 11.3 |

**These CANNOT coexist in the same environment.** The OpenPI evaluation uses a **client-server architecture** that completely decouples them:

```
┌───────────────────────────────┐          ┌─────────────────────────────────┐
│   LIBERO Client (Python 3.8)  │◄──WS────►│  OpenPI Server (Python 3.12)    │
│   examples/libero/main.py     │  :8000   │  scripts/serve_policy.py        │
│                               │          │                                 │
│  • Runs MuJoCo simulation     │          │  • Loads PI0/PI0.5 model        │
│  • Sends observations (JSON)  │          │  • Runs inference               │
│  • Executes returned actions  │          │  • Returns action chunks        │
│  • Tracks success metrics     │          │                                 │
└───────────────────────────────┘          └─────────────────────────────────┘
      Conda env: libero                         Conda env: openpi
      OR Docker container                       OR Docker container
```

---

### 1.1 RECOMMENDED: Docker Approach (Zero Version Conflicts)

This is the cleanest approach. The existing `examples/libero/compose.yml` handles everything.

#### Step 1: Prerequisites

```bash
# On the evaluation machine (Linux with NVIDIA GPU)
# Ensure Docker + NVIDIA Container Toolkit are installed

# Check GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

#### Step 2: Prepare Your Checkpoint

```bash
# If using a published checkpoint (e.g., pi05_libero):
# Docker will auto-download from gs://openpi-assets/checkpoints/pi05_libero/

# If using YOUR trained checkpoint, make it accessible:
export CHECKPOINT_DIR=/path/to/your/checkpoint  # e.g., /group-volume/openpi-ft/checkpoints/...
```

#### Step 3: Build & Launch (One Command)

```bash
cd /path/to/openpi-torch

# --- Option A: Evaluate with the default pretrained pi05_libero checkpoint ---
SERVER_ARGS="--env LIBERO" \
  docker compose -f examples/libero/compose.yml up --build

# --- Option B: Evaluate with YOUR custom checkpoint ---
SERVER_ARGS="--env LIBERO --policy.config pi05_libero --policy.dir /app/checkpoints/my_exp/50000" \
OPENPI_DATA_HOME=$CHECKPOINT_DIR \
  docker compose -f examples/libero/compose.yml up --build

# --- Option C: Evaluate a specific task suite ---
SERVER_ARGS="--env LIBERO" \
CLIENT_ARGS="--args.task-suite-name libero_10 --args.num-trials-per-task 50" \
  docker compose -f examples/libero/compose.yml up --build
```

#### Step 4: Collect Results

```bash
# Videos saved to: data/libero/videos/
# Console output shows success rate per task and overall
# Expected pi05_libero results:
#   libero_spatial: ~98.8%
#   libero_object:  ~98.2%
#   libero_goal:    ~98.0%
#   libero_10:      ~92.4%
```

---

### 1.2 ALTERNATIVE: Conda Dual-Environment Approach (No Docker)

Use this if Docker is not available or you need more control.

#### Step 1: Create the OpenPI Server Environment

```bash
# Environment 1: OpenPI-Torch (Python 3.12)
conda create -n openpi python=3.12 -y
conda activate openpi

cd /path/to/openpi-torch
pip install -e "."

# Verify
python -c "import openpi; import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
```

#### Step 2: Create the LIBERO Client Environment

```bash
# Environment 2: LIBERO (Python 3.8)
conda create -n libero python=3.8.13 -y
conda activate libero

# Install LIBERO from HuggingFace fork (maintained, newer assets)
pip install libero==0.1.1

# OR install from source (more control):
git clone https://github.com/huggingface/LIBERO.git /tmp/LIBERO
cd /tmp/LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .

# Install the openpi_client package (lightweight, no version conflicts)
cd /path/to/openpi-torch
pip install packages/openpi-client/

# Verify LIBERO
python -c "from libero.libero import benchmark; print('LIBERO OK')"
```

> **IMPORTANT**: `openpi-client` is a lightweight WebSocket client with minimal
> dependencies (numpy, websockets). It has NO conflict with Python 3.8.

#### Step 3: Launch the Server (Terminal 1)

```bash
conda activate openpi

# With default pretrained checkpoint
python scripts/serve_policy.py --env LIBERO --port 8000

# With YOUR checkpoint
python scripts/serve_policy.py \
  --policy.config pi05_libero \
  --policy.dir /path/to/checkpoints/my_exp/50000 \
  --port 8000
```

#### Step 4: Launch the Client (Terminal 2)

```bash
conda activate libero
cd /path/to/openpi-torch

# Set rendering backend
export MUJOCO_GL=egl            # Use 'egl' for headless, 'glx' for display
export MUJOCO_EGL_DEVICE_ID=0   # GPU index for rendering

# Run evaluation
python examples/libero/main.py \
  --task-suite-name libero_spatial \
  --num-trials-per-task 50 \
  --host 0.0.0.0 \
  --port 8000
```

#### Step 5: Run All Task Suites

```bash
# Evaluate all 4 standard suites sequentially
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo "=== Evaluating $SUITE ==="
  python examples/libero/main.py \
    --task-suite-name $SUITE \
    --num-trials-per-task 50 \
    --port 8000 \
    2>&1 | tee "results_${SUITE}.log"
done
```

---

### 1.3 Full Evaluation Workflow: Pre-training → LIBERO Benchmark → Robot Data

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 0: Prepare Base Model                                            │
│                                                                          │
│  Download pretrained pi05_base checkpoint:                               │
│    gs://openpi-assets/checkpoints/pi05_base/params                       │
│  This is the foundation model before any fine-tuning.                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Fine-tune on LIBERO (Sanity Check)                            │
│                                                                          │
│  1. Compute norm stats:                                                  │
│     python scripts/compute_norm_stats.py pi05_libero                     │
│                                                                          │
│  2. Train:                                                               │
│     torchrun --nproc_per_node=8 scripts/train_pytorch.py pi05_libero \   │
│       --exp-name libero_sanity_check                                     │
│                                                                          │
│  Config: batch_size=256, lr=5e-5, warmup=10k, steps=30k                  │
│  Expected: ~2-4 hours on 8×A100                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 2: Evaluate on LIBERO Benchmark                                   │
│                                                                          │
│  Run evaluation (Docker or dual-env approach from Section 1.1/1.2)       │
│                                                                          │
│  Pass criteria:                                                          │
│    libero_spatial ≥ 95%                                                  │
│    libero_object  ≥ 95%                                                  │
│    libero_goal    ≥ 95%                                                  │
│    libero_10      ≥ 85%                                                  │
│                                                                          │
│  If FAIL → Check: transforms, norm_stats, checkpoint loading             │
│  If PASS → Model + training pipeline are verified. Proceed to Phase 3    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 3: Pre-train / Fine-tune on YOUR Robot Dataset (rby1_xhand)       │
│                                                                          │
│  1. Compute norm stats for your config:                                  │
│     python scripts/compute_norm_stats.py pi05_rby1_df_pretrain           │
│                                                                          │
│  2. Train:                                                               │
│     Edit scripts/exp_template.sh with your CONFIG_NAME                   │
│     ./scripts/train_mpi_wrapper.sh scripts/exp_template.sh hostfile      │
│                                                                          │
│  3. Monitor loss curve, save checkpoints                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 1.4 Troubleshooting Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'libero'` | Wrong conda env | `conda activate libero` |
| `GLEW initialization failed` | No display / wrong GL | `export MUJOCO_GL=egl` |
| `WebSocket connection refused` | Server not running | Start server first, check port |
| `CUDA out of memory` on server | Model too large | Use `--default-prompt` to simplify; or use smaller batch |
| `numba` install fails on 3.8 | llvmlite version | `pip install llvmlite==0.36.0 numba==0.53.1` |
| `robosuite` crashes | Wrong MuJoCo version | `pip install mujoco==3.2.3 robosuite==1.4.1` |
| Actions look random | Norm stats mismatch | Recompute norm_stats with correct config |
| Success rate = 0% | Wrong image rotation | Ensure 180° rotation in client (already in main.py) |

---

## Part 2: Alternative Benchmarking Methods (Before Real Robot Testing)

### Recommended Evaluation Ladder

```
Level 1 (Easiest)    →    Level 2    →    Level 3    →    Level 4 (Hardest)
 Offline Metrics        Sim Eval        Cross-Task        Real Robot
```

### Level 1: Offline Metrics (No Simulation Needed)

**These can run directly in the OpenPI Python 3.12 environment.**

#### A. Training Loss Convergence Check
```bash
# Already logged during training - check loss.log or TensorBoard/W&B
# Look for: smooth decrease, no divergence, no plateaus
tensorboard --logdir checkpoints/<config>/<exp>/
```

#### B. Action Prediction Error (MSE on Held-Out Data)
```python
# In scripts/train_pytorch.py, validation already computes this.
# Key metric: val_loss should decrease and stabilize.
# Additional: per-dimension error analysis

# Add to your validation loop:
per_dim_error = ((pred_actions - gt_actions) ** 2).mean(dim=(0,1))  # [action_dim]
print(f"Per-dim MSE: {per_dim_error}")
# High error on specific dims → that joint/gripper is poorly learned
```

#### C. Action Trajectory Visualization
```python
# Already available in src/openpi/training/utils.py:
# visualize_validation_step() generates:
#   - Camera image collages
#   - GT vs predicted action trajectory plots (44 subplots)
# Saved to checkpoint_dir/vis/ during training
```

#### D. Norm Stats Verification
```bash
# Verify normalization is correct before training
python scripts/verify_norm_stats.py <config_name>
```

#### E. SE(3) Pose Error Analysis
```bash
# If using EE-pose (xhand), check SE(3) prediction accuracy
python scripts/evaluate_SE3_error.py  # Already in your repo
```

### Level 2: LIBERO Simulation Benchmark (Recommended)

**(See Part 1 above for full setup)**

| Suite | Tasks | What It Tests |
|-------|-------|---------------|
| `libero_spatial` | 10 | Spatial reasoning (left/right/behind) |
| `libero_object` | 10 | Object identification |
| `libero_goal` | 10 | Goal-directed behavior |
| `libero_10` | 10 | Multi-skill transfer |
| `libero_90` | 90 | Large-scale generalization |

**Why LIBERO is good**: Standard benchmark, reproducible, controlled initial states.

### Level 3: Cross-Domain Transfer Test

**Test if your model generalizes beyond training distribution:**

```bash
# Train on libero_90, evaluate on libero_10 (zero-shot transfer)
# This tests if the model learned generalizable skills

# 1. Train on libero_90
python scripts/train_pytorch.py pi05_libero --exp-name libero90_only
# (modify config to only use libero_90 data)

# 2. Evaluate on libero_10 (unseen tasks)
python examples/libero/main.py --task-suite-name libero_10 --num-trials-per-task 50
```

### Level 4: RoboCasa / RoboMimic (Advanced Sim Benchmarks)

If you want additional sim benchmarks beyond LIBERO:

| Benchmark | Python | Difficulty | What It Tests |
|-----------|--------|------------|---------------|
| **RoboCasa** | 3.9+ | Medium | Kitchen manipulation, long-horizon |
| **RoboMimic** | 3.8+ | Easy | Imitation learning baselines |
| **ManiSkill2** | 3.8+ | Hard | Diverse manipulation, soft body |
| **MetaWorld** | 3.8+ | Easy | 50 tabletop tasks, fast eval |

**Recommendation**: Stick with LIBERO for model validation — it's already integrated
in your codebase and is the standard benchmark for OpenPI models.

### Level 5: Action Smoothness & Physical Plausibility Check

Before deploying to a real robot, verify actions are physically safe:

```python
# Check action smoothness (no sudden jerks)
import numpy as np

def check_action_safety(action_sequence, max_delta=0.1, max_velocity=1.0):
    """Verify actions are smooth and within physical limits."""
    deltas = np.diff(action_sequence, axis=0)
    max_jerk = np.max(np.abs(deltas))
    max_vel = np.max(np.abs(action_sequence))
    
    print(f"Max jerk: {max_jerk:.4f} (limit: {max_delta})")
    print(f"Max velocity: {max_vel:.4f} (limit: {max_velocity})")
    
    if max_jerk > max_delta:
        print("WARNING: Actions have sudden jerks — unsafe for real robot!")
    if max_vel > max_velocity:
        print("WARNING: Action magnitude too high — may damage robot!")
    
    return max_jerk <= max_delta and max_vel <= max_velocity
```

---

## Part 3: Code Improvements List

### CRITICAL (Fix Immediately)

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `scripts/train_pytorch.py` | **EMA validation memory leak**: EMA model created on GPU without `dist.barrier()` between ranks | Add `dist.barrier()` before/after EMA cleanup |
| 2 | `scripts/train_pytorch.py` | **`find_unused_parameters=True`** wastes memory & slows DDP | Set to `False` (all params are used); enable `static_graph=True` for world_size ≥ 2 |
| 3 | `models_pytorch/gemma_pytorch.py` | **Debug prints in production**: `print("Forcing gradient checkpointing...")` in forward pass | Remove all debug prints; set grad checkpointing once at init |
| 4 | `models_pytorch/preprocessing_pytorch.py` | **Slow random crop**: uses `torch.randint` tensors for slicing (forces GPU sync) | Convert to Python ints with `.item()` |
| 5 | `training/data_loader.py` | **Conservative prefetching**: `prefetch_factor=max(2, num_workers//4)` — for 16 workers only 4 prefetch | Use `max(8, num_workers)` or make configurable |

### HIGH (Significant Performance Gains)

| # | File | Issue | Fix |
|---|------|-------|-----|
| 6 | `scripts/train_pytorch.py` | **No AMP (Automatic Mixed Precision)**: forward/backward not wrapped in `torch.autocast()` | Add `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around forward pass |
| 7 | `scripts/train_pytorch.py` | **EMA not loaded from checkpoints**: Comment says "not fully implemented" | Implement EMA state dict loading in `load_checkpoint()` |
| 8 | `models_pytorch/pi0_pytorch.py` | **Dtype mismatch risk**: state projection doesn't ensure dtype consistency | Always cast to `self.state_proj.weight.dtype` |
| 9 | `models_pytorch/gemma_pytorch.py` | **155-line monolithic function**: `compute_layer_complete()` does everything | Break into `_attention_forward()`, `_residual_block()`, `_mlp_forward()` |
| 10 | `models_pytorch/gemma_pytorch.py` | **Reactive dtype conversion**: converts attention output dtype only if mismatch | Set dtype proactively at start of compute layer |
| 11 | `models_pytorch/preprocessing_pytorch.py` | **Expensive rotation augmentation**: `grid_sample` every batch during training | Make optional in config; consider pre-rotating dataset |
| 12 | `training/data_loader.py` | **Streaming dataset DDP validation missing**: no check that shards align with world_size | Add assertion: `num_shards % world_size == 0` |

### MEDIUM (Nice-to-Have)

| # | File | Issue | Fix |
|---|------|-------|-----|
| 13 | `models_pytorch/pi0_pytorch.py` | **`torch.set_float32_matmul_precision("high")`** is hardcoded global state | Add to `TrainConfig` as `matmul_precision: str = "high"` |
| 14 | `scripts/train_pytorch.py` | **Gradient clearing**: manual loop when `optimizer.zero_grad(set_to_none=True)` suffices | Replace with single optimizer call |
| 15 | `models_pytorch/pi0_pytorch.py` | **`preserve_rng_state=False`** in gradient checkpointing hurts reproducibility | Make configurable; default to `True` |
| 16 | `models_pytorch/preprocessing_pytorch.py` | **Image resize logged every batch** | Log only on first batch using a flag |
| 17 | `training/data_loader.py` | **Collate function forces numpy conversion**: GPU tensors → numpy → back to GPU | Keep as PyTorch tensors when using PyTorch framework |
| 18 | `training/config.py` | **`keep_period` logic unclear** | Add docstring examples; consider `keep_milestones: List[int]` alternative |
| 19 | `training/config.py` | **Hardcoded paths** (`/group-volume/...`) | Use `os.environ.get("OPENPI_CHECKPOINT_DIR", default)` pattern |
| 20 | `scripts/train_pytorch.py` | **EMA always float32 on CPU** | Option to keep in bfloat16 for memory savings |

### LOW (Code Quality)

| # | File | Issue | Fix |
|---|------|-------|-----|
| 21 | `models_pytorch/gemma_pytorch.py` | `_debug_gc_printed` is a class variable (shared across instances) | Move to `__init__()` as instance variable |
| 22 | `training/data_loader.py` | `InfiniteSampler.__len__()` returns `float("inf")` | Return `sys.maxsize` (proper int) |
| 23 | `models_pytorch/preprocessing_pytorch.py` | `TODO: This is a hack` comment for image format handling | Document expected format; add assertion |
| 24 | `training/data_loader.py` | Worker init sets JAX env vars in PyTorch code | Add `if jax_available` guard |
| 25 | `models_pytorch/preprocessing_pytorch.py` | Saturation adjustment uses naive grayscale, not proper HSV | Use `torchvision.transforms.functional.adjust_saturation()` |
| 26 | `training/data_loader.py` | `FakeDataset` uses `jax.random` in PyTorch path | Use `torch.randint` / `numpy.random` when framework is PyTorch |
| 27 | `models_pytorch/pi0_pytorch.py` | Variable re-assignment anti-pattern (`observation = ... # noqa`) | Use different variable names (`obs_device`, etc.) |
| 28 | `models_pytorch/pi0_pytorch.py` | `torch.compile` mode not validated | Assert `mode in ["reduce-overhead", "default", "max-autotune"]` |

### Structural Improvements

| # | Area | Suggestion |
|---|------|------------|
| 29 | **Flash Attention** | Replace custom eager attention in `transformers_replace/modeling_gemma.py` with `torch.nn.functional.scaled_dot_product_attention()` which auto-selects Flash Attention 2 / Memory-Efficient Attention |
| 30 | **Data Pipeline** | Add `pin_memory=True` to DataLoader for faster CPU→GPU transfer |
| 31 | **Gradient Accumulation** | Add configurable gradient accumulation steps to simulate larger batch sizes on fewer GPUs |
| 32 | **Learning Rate Finder** | Add a quick LR range test before full training to find optimal peak_lr |
| 33 | **Checkpoint Format** | Current saves full model + optimizer. Add option for LoRA-only checkpoints (much smaller) |
| 34 | **Validation** | Add action trajectory MSE per-joint breakdown in validation logging |
| 35 | **Config Validation** | Add `TrainConfig.validate()` method that checks: action_dim matches policy, norm_stats exist, checkpoint path exists |

### Priority Implementation Order

```
Week 1: Issues #1-5 (Critical bugs & performance)
         + Issue #6 (AMP — biggest single speedup)

Week 2: Issues #7-12 (High-priority optimization)
         + Issue #29 (Flash Attention)
         + Issue #30 (pin_memory)

Week 3: Issues #13-20 (Medium improvements)
         + Issue #31 (Gradient accumulation)

Ongoing: Issues #21-28 (Code quality, as you touch those files)
```

---

## Summary: Recommended Evaluation Path

```
1. Verify training pipeline works
   └── Train pi05_libero for 30K steps
   └── Check loss convergence in TensorBoard

2. Run LIBERO benchmark (Docker approach)
   └── SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
   └── Expect ≥95% on spatial/object/goal, ≥85% on libero_10

3. If LIBERO passes → your model + pipeline are verified
   └── Proceed to rby1_xhand training with confidence

4. If LIBERO fails → debug before touching robot data
   └── Check: norm_stats, transforms, checkpoint loading, action space dimensions
```
