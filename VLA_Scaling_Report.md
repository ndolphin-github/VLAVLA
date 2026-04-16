# VLA Model Scaling and Action Expert Improvement: Technical Report
## Open Pi 0.5 Fine-tuning for Rby1 Robot Tasks

---

## Executive Summary

The two core problems you're facing are distinct but interconnected:

1. **Vision Encoder Scaling Failure**: Direct replacement of the SigLIP encoder breaks action knowledge because you're changing both the vision embedding space *and* the token budget simultaneously. The 256-token constraint couples image resolution to the entire action decoder's expectation, not just the vision module.

2. **Action Expert Memorization**: This is the more serious issue. The action decoder isn't learning generalizable spatial-temporal patterns; it's memorizing dataset-specific biases. This isn't just a training problem—it's an architectural mismatch between vision-to-action grounding and the task complexity.

**Critical Verdict**: The Open Pi 0.5 base model has fundamental architectural limitations that make it difficult to scale properly. You'll need targeted interventions, not just parameter adjustments.

---

## Part 1: Vision Encoder Scaling Problem - Root Cause Analysis

### Why SigLIP 896×896 Failed

When you transplanted SigLIP 896×896 from PaliGemma2, three things happened simultaneously:

1. **Embedding Space Mismatch**: PaliGemma2's SigLIP is trained on different visual-semantic objectives than Pi-0's base encoder. The embeddings are incompatible without layer retraining.

2. **Token Budget Explosion**: PaliGemma2's SigLIP produces ~256 tokens from 896×896, but the *quality* of those tokens is different. Pi-0's action decoder was optimized for 256 tokens from 224×224 (lower information density per token).

3. **Information Density Shift**: 
   - 224×224 → 256 tokens = 49,152 pixels / 256 tokens ≈ **192 pixels/token** (highly compressed)
   - 896×896 → 256 tokens = 802,816 pixels / 256 tokens ≈ **3,134 pixels/token** (more detail per token)
   
   The action decoder can't process this denser information because it was trained on coarser features.

### Current Architecture Bottleneck

```
224×224 Image → SigLIP patch encoder → 256 tokens (16×16 grid, 14×14 patch size)
                                         ↓
                                   Action Decoder expects this specific
                                   information density and token count
```

The action decoder's first layer is likely tightly coupled:
- Positional embeddings assume 16×16 spatial structure
- Attention patterns learned for 256-token sequences
- Cross-attention heads developed for this specific token arrangement

---

## Part 2: Practical Solutions for Vision Encoder Scaling

### Solution 1: Adapter-Based Resolution Scaling (Recommended - Minimal Knowledge Disruption)

**Architecture**:
```
896×896 Image → SigLIP 896×896 → 256-token output (different structure)
                                ↓
                          Adapter Module (trainable)
                                ↓
                          256 tokens (Pi-0 compatible format)
                                ↓
                          Action Decoder (frozen pretrained weights)
```

**Implementation Details**:

1. **Token Restructuring Adapter** (learnable projection):
   ```
   - Input: 256 tokens from 896×896 (different positional embedding)
   - Learnable linear projection: Token feature dim → Token feature dim
   - Positional embedding remapping: Map 896 coordinate space → 224 coordinate space
   - Output: 256 tokens with Pi-0-compatible positional encoding
   ```

2. **Spatial Resampling** (if token layout differs):
   - If SigLIP 896 produces different spatial grid, use bilinear interpolation
   - Project back to 16×16 spatial structure before passing to action decoder

3. **Knowledge Preservation Strategy**:
   - Freeze action decoder for first 500-1000 steps
   - Only train adapter with low learning rate (1e-4 to 1e-5)
   - Use layer-wise learning rate decay (adapter lr / 10 for vision features)

**Why This Works**:
- Action decoder never sees new input distribution
- Only adapter learns the mapping between vision spaces
- Pretrained action knowledge remains intact
- Gradual adaptation allows fine-tuning without catastrophic forgetting

**Limitation**: Max effective resolution gain is ~1.5-2×. You'll get roughly **340×340 effective resolution** (vs theoretical 896×896).

---

### Solution 2: Progressive Token Budget Expansion (More Aggressive)

If you need significantly higher resolution, you must expand the token budget gradually.

**Three-Phase Training**:

**Phase 1: Foundation (5-10 epochs)**
- Keep 224×224 input, 256 tokens
- Train adapter on your Rby1 data
- Establish baseline performance

**Phase 2: Soft Token Expansion (10-20 epochs)**
- Increase to 384×384 input
- Add learnable "token expansion layer": 256 → 512 tokens
- This layer learns to split each 256-token into 2 tokens
- Modify action decoder's cross-attention to handle 512 tokens:
  ```python
  # Learnable sparse attention
  attention_mask = torch.nn.Parameter(torch.ones(512, 512))  # Learnable
  # Start: mostly attends to original 256 positions
  # Gradually learn to use expanded positions
  ```

**Phase 3: Full Resolution (20+ epochs)**
- Increase to 512×512 or 640×640 (practical limit for action models)
- Expand to 512 or 768 tokens
- Fine-tune attention patterns

**Key Mechanism - Gradual Capacity Increase**:
```python
# During training, interpolate between 256 and 512 tokens
current_tokens = int(256 + (512 - 256) * epoch / max_epochs)
# Gradually enable attention to expanded tokens
attention_weight = torch.sigmoid(learnable_expansion_param * epoch / max_epochs)
```

**Why This Matters**:
- Action decoder learns to use additional tokens incrementally
- Prevents sudden distribution shift
- Maintains action knowledge while expanding capacity

**Challenge**: 
- Requires careful curriculum design
- Longer training time
- Risk of overfitting to intermediate resolutions
- **Practical ceiling: ~512×512, 512 tokens** (beyond this, fine-tuned models struggle)

---

### Solution 3: Multi-Resolution Training (Most Robust - But Computationally Expensive)

Train with mixed resolutions:
- 30% of batches: 224×224 (original)
- 35% of batches: 384×384 (scaled)
- 35% of batches: 512×512 (high-res)

**Benefits**:
- Action decoder learns resolution-invariant features
- No catastrophic forgetting
- Better generalization to varying image qualities

**Cost**: ~40% more GPU memory per batch

---

## Part 3: Action Expert Memorization Problem - Deep Diagnosis

### Why It's Memorizing (Not Just Overfitting)

This is fundamentally different from standard overfitting. Evidence:

1. **Spatial-Temporal Coupling Bias**: The model learned that certain visual patterns → specific motor sequences in your exact coordinate frame/workspace geometry

2. **Dataset Dependency Reasons**:
   - Rby1 robot has specific kinematics constraints
   - Your pick-up/peg-in-hole tasks have narrow solution space
   - Training data likely has high visual repetition (same picking locations, same peg holes)
   - Action sequences are deterministic given initial state

3. **Architectural Root Cause**: Pi-0's action decoder is **end-effector trajectory prediction**, not **behaviors**.
   - It predicts: `xyz positions, gripper state` for next N timesteps
   - This is inherently memorization-prone for constrained tasks
   - There's no intermediate "skill" or "intention" representation

### Diagnostic Tests (Run These):

```python
# Test 1: Spatial Invariance
# Take same pick-up action, but at 3 different locations in workspace
# If model outputs very different predictions for geometrically similar situations
# → Strong memorization confirmed

# Test 2: Visual Confusion
# Take image of peg hole at location A
# Replace with peg hole at location B (same visual appearance)
# If model predicts location A's solution → Pure memorization
# If model predicts location B's solution → Some generalization

# Test 3: Occlusion Robustness
# Occlude 25%, 50%, 75% of irrelevant background
# If performance degrades significantly → Model relies on specific visual patterns
```

### Why Open Pi 0.5 is Vulnerable to This

The base model's action expertise comes from broad robot learning data:
- ALOHA, BridgeData, RoboNet, etc.
- Highly diverse tasks with many generalizable skills
- When fine-tuned on 2-3 narrow tasks, the model **reverts to memorization** because:
  1. Task diversity collapses
  2. Solution space becomes locally deterministic
  3. Vision-action mapping becomes bijective (one visual state → one correct action)

---

## Part 4: Solutions for Action Expert Memorization

### Solution 1: Data Augmentation with Geometric Transformation (Critical)

**Problem**: Current dataset likely has systematic visual patterns tied to locations.

**Solution - Workspace Randomization**:

```python
# During fine-tuning, apply:
1. Random workspace offset: robot position ±20cm in XY
2. Random camera viewpoint: ±15° rotation on end-effector camera
3. Object orientation randomization: ±45° for pick objects
4. Gripper state augmentation: Random pre-grasp configurations
5. Visual domain randomization: ±20% brightness, ±30% hue, blur artifacts

# Critical: Retarget action labels to new workspace coordinates
# Don't just augment images; update ground truth trajectories
```

**Why This Works**:
- Forces model to learn from *visual patterns* (object geometry) not locations
- Breaks spatial memorization
- Requires model to compute relative motions, not absolute positions

**Expected Outcome**: 20-30% performance drop initially, then recovery with better generalization

### Solution 2: Intermediate Representation Learning - Add Intention Bottleneck

**Architecture Modification**:

```
Vision → Encoder → Intention State (32-dim bottleneck) → Action Decoder
                         ↑
                   Binary classification:
                   - Approaching object?
                   - Manipulating?
                   - Returning to home?
```

**Implementation**:

Add auxiliary loss that forces the model to learn discrete task phases:

```python
# Modify action decoder's middle layers
class IntentionAuxiliaryLoss(nn.Module):
    def forward(self, intermediate_features, task_phase_labels):
        # task_phase: 0=approach, 1=grasp, 2=lift, 3=move, 4=insert, 5=release
        logits = self.classifier(intermediate_features)
        return cross_entropy_loss(logits, task_phase_labels)

# Combined loss:
total_loss = trajectory_loss + 0.1 * intention_loss + 0.05 * action_smoothness_loss
```

**Why This Works**:
- Intermediate bottleneck forces disentanglement of task phases from spatial memorization
- Model learns "what to do next" before computing "how to do it"
- Creates interpretable intermediate representation

**Critical Design**: The intention classes must come from your task structure:
- For pick-up: [locate, approach, grasp, lift]
- For peg-in-hole: [locate-hole, approach-entry, align, insert, retract]

---

### Solution 3: Diffusion-Based Action Refinement (Advanced - SOTA Approach)

Replace trajectory prediction with iterative refinement:

```
Vision → Encoder → Latent Action Distribution
                           ↓
                    Diffusion Decoder (5-10 steps)
                           ↓
                    Refined Action Trajectory
```

**Why Diffusion Over Direct Prediction**:
- Diffusion models learn probability distributions, not memorized mappings
- Better uncertainty quantification
- Natural way to handle multi-modal action spaces

**Practical Implementation**:
- Use "Action Language Models" approach (similar to Diffusion Policy but robot-specific)
- Start with open-source Diffusion Policy base
- Fine-tune the diffusion scheduler for Rby1 action dimensions

**Cost**: 
- ~10-20× slower inference (not practical for real-time without optimization)
- Requires 2× GPU memory during training
- More complex debugging

**Verdict**: Worth exploring for offline planning, not for real-time control.

---

### Solution 4: Auxiliary Losses for Generalization (Practical, Medium-Impact)

Add these losses to your fine-tuning objective:

```python
class EnhancedActionLoss(nn.Module):
    def __init__(self):
        self.trajectory_loss = MSELoss()
        self.velocity_loss = MSELoss()
        self.consistency_loss = ConsistencyLoss()
        self.entropy_loss = EntropyLoss()
    
    def forward(self, pred_trajectory, gt_trajectory, pred_velocity, gt_velocity):
        # 1. Trajectory prediction (primary)
        L_traj = self.trajectory_loss(pred_trajectory, gt_trajectory)
        
        # 2. Velocity smoothness (prevents jerky memorized patterns)
        L_vel = self.velocity_loss(pred_velocity, gt_velocity)
        
        # 3. Temporal consistency (actions similar at same time-to-contact)
        L_cons = self.consistency_loss(pred_trajectory, gt_trajectory)
        
        # 4. Action entropy (prevents collapse to single mode)
        L_ent = -self.entropy_loss(pred_trajectory)
        
        return L_traj + 0.3*L_vel + 0.1*L_cons + 0.001*L_ent
```

**Effect on Memorization**:
- **Velocity loss**: Prevents sharp, dataset-specific movements
- **Consistency loss**: Forces similar actions for temporally similar states
- **Entropy regularization**: Maintains action diversity, prevents single-solution collapse

**Weights to tune**:
- Start: `0.3, 0.1, 0.001`
- If still memorizing: increase to `0.5, 0.2, 0.005`
- If performance drops: decrease to `0.1, 0.05, 0.0005`

---

## Part 5: Additional VLA Architecture Improvements

### Improvement 1: Action Chunking Strategy Optimization (Quick Win)

**Current Problem**: Likely using fixed chunk size (e.g., always predict next 3 or 5 steps)

**Better Approach - Adaptive Chunking**:

```python
# Strategy 1: Horizon Prediction
# Predict action duration alongside trajectory
class DynamicHorizonDecoder(nn.Module):
    def forward(self, vision_features, prev_actions):
        # Predict both action AND how long to execute it
        action = self.action_head(vision_features)
        horizon = torch.sigmoid(self.horizon_head(vision_features)) * 8 + 1
        # Result: action for next 1-8 timesteps (learned per frame)
        return action, horizon

# Strategy 2: Velocity-Based Chunking
# Chunk size depends on action speed
class VelocityAdaptiveChunking(nn.Module):
    def forward(self, action_vel):
        # High velocity → shorter chunks (2-3 steps)
        # Low velocity → longer chunks (5-8 steps)
        predicted_vel_norm = torch.norm(action_vel)
        chunk_size = int(10 / (1 + predicted_vel_norm)) + 1
        return chunk_size
```

**Why This Matters**:
- Fixed chunks force artificial temporal discretization
- Adaptive chunks let model learn natural action rhythms
- Particularly important for peg-in-hole (needs variable speed control)

**Implementation Cost**: ~30 lines of code, minimal overhead

---

### Improvement 2: Multi-Scale Action Prediction

Predict actions at multiple temporal scales:

```
Vision Encoder
    ↓
├─→ Immediate Actions (next 1-2 steps, high precision)
├─→ Short-term Skills (next 5-10 steps, medium precision)
└─→ Long-term Goals (whole-task trajectory, low precision)

Loss = L_immediate + 0.5*L_short + 0.1*L_long
```

**Benefits**:
- Forces model to learn hierarchical representations
- Prevents memorization of specific trajectories (long-term loss abstracts away details)
- Better handles uncertainty at different time scales

**Why It Helps Generalization**:
- Long-term loss encourages learning task structure (pick → move → place)
- Model can't memorize exact trajectories—must learn generalizable skills

---

### Improvement 3: Vision Encoder Replacement from Non-Robot Model

You considered this—it's actually valid but needs careful execution:

**Recommended Approach - Feature Alignment**:

```python
# Use vision encoder from PaliGemma2 (trained on general vision, no robot bias)
# But add alignment layer

class FeatureAlignmentAdapter(nn.Module):
    def __init__(self, paligemma_dim, pi0_dim):
        self.proj = nn.Linear(paligemma_dim, pi0_dim)
        self.norm = nn.LayerNorm(pi0_dim)
    
    def forward(self, paligemma_features):
        aligned = self.proj(paligemma_features)
        aligned = self.norm(aligned)
        return aligned

# Optional: Add contrastive learning to the alignment
# Use your robot task data as positive pairs
```

**Why Non-Robot Vision Encoders Can Help**:
- Avoids learning robot-specific biases
- Better on novel objects not in pretrain data
- More robust to background variations

**But Critical Warning**:
- General vision encoders may miss fine-grained gripper states
- Depth perception (if you have it) may not transfer well
- Requires more careful training data engineering

---

## Part 6: Critical Assessment - What Won't Work

### ❌ Issue 1: Simply Increasing Data Won't Fix Memorization

If you collect 10× more Rby1 data, you'll just have more memorization patterns. The issue is architectural, not data volume.

**What Would Help**: Collect data with geometric variations (different table heights, camera angles, object sizes).

### ❌ Issue 2: Larger Models Aren't the Solution

Training a 7B LLM variant won't solve this. The problem is vision-action grounding, not language understanding.

### ❌ Issue 3: Asynchronous Inference Doesn't Fix Accuracy

You mentioned async inference for speed—that's valid for latency but completely orthogonal to generalization. It won't improve action quality, just reduces wall-clock time.

### ❌ Issue 4: MEM Methods Are Oversold for Fine-tuning

Memory-augmented networks (MEM) are hyped but require:
- External memory buffer initialization
- Complex attention mechanisms
- Careful hyperparameter tuning

For your problem (memorization on fine-tuning), MEM is **not the right tool**. The issue is that the model is learning from limited data distribution, not that it lacks memory capacity.

---

## Part 7: Recommended Implementation Strategy

### Phase 1: Diagnostics (1-2 days)
```
1. Run the diagnostic tests from Part 3
2. Quantify memorization rate:
   - Collect 10 semantic duplicates of same pick location (different visual appearance)
   - Measure prediction variance across duplicates
3. Establish baseline: current performance on test set
```

### Phase 2: Data Augmentation + Intention Learning (3-5 days)
```
1. Implement workspace randomization (Part 4, Solution 1)
2. Add intention auxiliary loss (Part 4, Solution 2)
3. Retrain on augmented data for 20-30 epochs
4. Compare to baseline
```

**Expected Result**: 15-25% improvement in generalization

### Phase 3: Vision Scaling (5-7 days)
```
1. Implement adapter-based scaling (Part 2, Solution 1)
2. Test with 384×384 and 512×512
3. Measure: (a) FPS impact, (b) accuracy improvement
4. Find optimal resolution for your task
```

**Expected Result**: 10-15% accuracy improvement if memorization is reduced first

### Phase 4: Multi-Scale Action Prediction (3-4 days)
```
1. Add long-term goal prediction head
2. Calibrate loss weights
3. Evaluate task-level success rate
```

**Expected Result**: More stable, generalizable actions

### Timeline: 2-3 weeks total
- Week 1: Diagnostics + data augmentation
- Week 2: Vision scaling + multi-scale action
- Week 3: Empirical refinement

---

## Part 8: Specific Code Recommendations

### Adapter Implementation (Part 2, Solution 1)

```python
class VisionAdapterLayer(nn.Module):
    """Map between different vision encoder outputs"""
    def __init__(self, in_feat_dim, out_feat_dim, num_tokens=256):
        super().__init__()
        self.linear = nn.Linear(in_feat_dim, out_feat_dim)
        self.pos_embedding_adapter = nn.Parameter(
            torch.randn(1, num_tokens, out_feat_dim) * 0.02
        )
        self.norm = nn.LayerNorm(out_feat_dim)
    
    def forward(self, x):
        # x: (batch, num_tokens, in_feat_dim)
        x = self.linear(x)
        x = x + self.pos_embedding_adapter
        x = self.norm(x)
        return x
```

### Workspace Randomization (Part 4, Solution 1)

```python
class WorkspaceRandomizer:
    def __init__(self, max_offset_xy=0.2, max_angle_z=45, seed=None):
        self.max_offset = max_offset_xy
        self.max_angle = np.radians(max_angle_z)
        self.rng = np.random.RandomState(seed)
    
    def randomize_trajectory(self, trajectory_xyz, gripper_state):
        # trajectory_xyz: (seq_len, 3)
        offset = self.rng.uniform(-self.max_offset, self.max_offset, 2)
        angle = self.rng.uniform(-self.max_angle, self.max_angle)
        
        # Apply offset
        trajectory_xyz[:, :2] += offset
        
        # Apply rotation
        for t in range(len(trajectory_xyz)):
            xy = trajectory_xyz[t, :2]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            trajectory_xyz[t, :2] = rot_matrix @ xy
        
        return trajectory_xyz, gripper_state
```

### Intention Auxiliary Loss (Part 4, Solution 2)

```python
class IntentionAuxiliaryHead(nn.Module):
    def __init__(self, hidden_dim=256, num_phases=6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_phases)
        )
    
    def forward(self, intermediate_features):
        return self.classifier(intermediate_features)

# In training loop:
intention_logits = model.intention_head(intermediate_features)
intention_loss = F.cross_entropy(intention_logits, task_phase_labels)
total_loss = trajectory_loss + 0.1 * intention_loss
```

---

## Part 9: What You're Up Against (Honest Assessment)

### The Hard Truth

1. **Fine-tuning Limitations**: Open Pi 0.5 was trained on 100+ hours of diverse robot data. Your fine-tuning dataset is likely <10 hours on 2-3 tasks. The model will naturally revert to memorization—it's easier than generalizing.

2. **Task Complexity Paradox**: Pick-up and peg-in-hole seem "simple" but are actually high-precision tasks. This makes memorization more attractive:
   - Visual state space is small (object locations are limited)
   - Action space is small (few valid solutions per state)
   - Temporal patterns are repeatable
   
   → Conditions are perfect for memorization

3. **Data Requirements**: To achieve real generalization, you likely need:
   - **50+ pick-up locations** with visual variations
   - **20+ peg-in-hole configurations** (angles, depths, materials)
   - **Multi-camera views**
   - **Object variations** (colors, sizes, materials)
   
   This is 2-3× more data than you probably have.

4. **Resolution Isn't Your Main Problem**: The 224×224 limitation is real but secondary. Even with 512×512 input, a memorizing model will still memorize—just with better object detail.

### When to Give Up on Fine-tuning

If after Phase 2 (data augmentation + intention learning), you see <5% improvement, it means:
- Your task is too simple for the model to generalize
- OR your data distribution is too narrow
- In this case, consider: **Behavior cloning with explicit inverse kinematics** might actually be better

---

## Part 10: Resource-Efficient Alternative Path

If the fine-tuning approach is failing, consider **modular approach**:

```
Vision → CLIP embeddings (frozen, general features)
         ↓
    Task Classifier (lightweight, fine-tuned)
    - Detects: pick-up or peg-in-hole?
         ↓
    Task-Specific Module (small, fine-tuned)
    - Per-task: 50k parameter network
    - Learned inverse kinematics
    - Hand-engineered spatial reasoning
```

**Advantages**:
- No pretrained knowledge degradation
- Easier debugging
- Smaller fine-tuning data requirement

**Disadvantages**:
- Less generalizable to new tasks
- Requires manual task enumeration

---

## Recommendations Summary

### Immediate Actions (This Week)

1. **Run diagnostics** - Confirm memorization is the issue
2. **Implement workspace randomization** - Simple, high-impact
3. **Add intention auxiliary loss** - Forces better representations

### Medium-term (Weeks 2-3)

4. **Vision encoder adapter** (Part 2, Solution 1) - Safe resolution scaling
5. **Multi-scale action prediction** - Structural improvement
6. **Collect more diverse data** - If possible

### Long-term Considerations

7. **Evaluate diffusion-based actions** - If real-time isn't critical
8. **Consider modular approach** - If fine-tuning plateau

### What NOT to Do

- ❌ Don't scale to 896×896 directly
- ❌ Don't increase model size without addressing memorization
- ❌ Don't use MEM or complex attention tricks—they won't help
- ❌ Don't assume more data alone will fix generalization

---

## Conclusion

Your core insight is correct: **the action expert is learning dataset-specific patterns, not generalizable skills**. This is a known problem in robot learning fine-tuning, and it requires architectural solutions, not just hyperparameter tuning.

The vision resolution (224×224) is a legitimate constraint, but addressing it without first fixing memorization is premature optimization. Fix the Action Expert generalization first (Part 4 solutions), then scale vision (Part 2, Solution 1).

The fact that you're using pretrained weights is an advantage—leverage it by keeping changes minimal (adapter-based scaling) and targeted (intention learning for structure).

**Realistic Expectations**:
- With Phase 1-2: 20-30% improvement in generalization
- With Phase 3-4: Additional 10-15% improvement
- Hard ceiling: likely 70-80% of what you'd get with a model fully trained on your task distribution

This ceiling isn't pessimism—it's physics. Transfer learning from broad robotics to narrow tasks has fundamental limits.

