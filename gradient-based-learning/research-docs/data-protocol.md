# Data & Experiment Protocol

**Research Line:** Memory-Efficient Optimization for Foundation Models
**Version:** 1.0
**Last Updated:** 2025-10-30
**Owner:** Manolo Rodeta / Vera Strata AI Research

---

## Table of Contents
1. [Overview](#1-overview)
2. [Directory Structure](#2-directory-structure)
3. [Experiment Naming Convention](#3-experiment-naming-convention)
4. [Experiment Lifecycle](#4-experiment-lifecycle)
5. [Data Collection Standards](#5-data-collection-standards)
6. [Logging & Tracking](#6-logging--tracking)
7. [Results Storage](#7-results-storage)
8. [Code Organization](#8-code-organization)
9. [Reproducibility Checklist](#9-reproducibility-checklist)
10. [Backup & Archival](#10-backup--archival)

---

## 1. Overview

### 1.1 Purpose
This document establishes standards for:
- **Organizing** experimental data and code
- **Documenting** experiments systematically
- **Ensuring** reproducibility of all results
- **Facilitating** collaboration and knowledge transfer
- **Enabling** efficient paper writing from organized results

### 1.2 Principles
- **Consistency:** Every experiment follows the same structure
- **Traceability:** Every result can be traced back to code + config
- **Completeness:** All information needed to reproduce is saved
- **Accessibility:** Easy to find past experiments and their results
- **Automation:** Minimize manual work through scripts and tools

---

## 2. Directory Structure

### 2.1 Top-Level Organization

```
gradient-based-learning/
├── research-docs/              # Research protocols, plans, papers outline
│   ├── research-protocol.md
│   ├── data-protocol.md        # This document
│   ├── papers-roadmap.md
│   └── limitations-opportunities.md
│
├── experiments/                # Experimental code (version controlled)
│   ├── 01-baselines/
│   ├── 02-quantization/
│   ├── 03-lowrank/
│   ├── 04-hybrid/
│   ├── 05-scaling/
│   └── 06-theory-validation/
│
├── efficient_optim/            # Library code (package)
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── adam8bit.py
│   │   ├── hybrid_adam.py
│   │   └── adafactor.py
│   ├── utils/
│   │   ├── profiling.py
│   │   ├── logging.py
│   │   └── visualization.py
│   └── tests/
│
├── configs/                    # Experiment configurations (YAML/Hydra)
│   ├── model/
│   │   ├── gpt2-small.yaml
│   │   ├── gpt2-medium.yaml
│   │   └── gpt2-large.yaml
│   ├── optimizer/
│   │   ├── adam.yaml
│   │   ├── adam8bit.yaml
│   │   └── hybrid.yaml
│   ├── dataset/
│   │   ├── wikitext.yaml
│   │   └── openwebtext.yaml
│   └── experiment/             # Full experiment configs
│       ├── baseline-gpt2-small.yaml
│       └── hybrid-gpt2-medium.yaml
│
├── results/                    # Experimental results (NOT in git)
│   ├── runs/                   # Individual experiment runs
│   │   └── YYYY-MM-DD_HH-MM-SS_experiment-name/
│   │       ├── config.yaml     # Full config snapshot
│   │       ├── logs/           # Training logs
│   │       ├── checkpoints/    # Model checkpoints
│   │       ├── metrics/        # Saved metrics (CSV, JSON)
│   │       ├── plots/          # Generated plots
│   │       └── summary.md      # Experiment summary
│   └── archive/                # Completed experiments
│       └── paper1-characterization/
│           ├── exp001_adam_baseline/
│           ├── exp002_adam8bit/
│           └── final_results.csv
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── exploratory/            # Quick explorations
│   ├── analysis/               # Deep dives into results
│   └── figures/                # Notebooks generating paper figures
│       ├── paper1_fig1_convergence.ipynb
│       └── paper1_fig2_memory.ipynb
│
├── papers/                     # Paper writing (LaTeX)
│   ├── paper1-characterization/
│   │   ├── main.tex
│   │   ├── figures/
│   │   ├── sections/
│   │   └── bibliography.bib
│   ├── paper2-hybrid/
│   └── paper3-theory/
│
├── scripts/                    # Utility scripts
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluation script
│   ├── profile_memory.py       # Memory profiling
│   ├── visualize_landscape.py  # Loss landscape viz
│   └── aggregate_results.py    # Aggregate multiple runs
│
├── data/                       # Datasets (NOT in git if large)
│   ├── raw/                    # Original datasets
│   ├── processed/              # Preprocessed data
│   └── README.md               # Dataset documentation
│
├── .gitignore                  # Ignore results/, data/, checkpoints
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── README.md                   # Project overview
└── LICENSE                     # Open source license
```

### 2.2 Git Versioning Strategy

**Version controlled (committed to git):**
- All code (`experiments/`, `efficient_optim/`, `scripts/`)
- Configurations (`configs/`)
- Analysis notebooks (`notebooks/`) - with cleared outputs
- Documentation (`research-docs/`, `papers/`)
- Small reference results (summary CSVs, key plots)

**NOT version controlled (in .gitignore):**
- Large datasets (`data/raw/`, `data/processed/`)
- Individual run results (`results/runs/`)
- Model checkpoints (`results/runs/*/checkpoints/`)
- Temporary files, logs

**Backed up externally:**
- All results archived after paper submission
- Important checkpoints for reproducibility
- Raw experiment logs (for potential re-analysis)

---

## 3. Experiment Naming Convention

### 3.1 Experiment ID Format

```
YYYYMMDD_HHMM_phase-category_description_variant
```

**Examples:**
```
20251030_1430_baseline_adam_gpt2small_lr1e4
20251105_0920_quantization_adam8bit_gpt2medium_dynamic
20251215_1500_hybrid_hybridadam_gpt2large_adaptive
```

**Components:**
- `YYYYMMDD_HHMM`: Timestamp (start time)
- `phase`: baseline | quantization | lowrank | hybrid | scaling | theory
- `category`: Specific aspect being tested
- `description`: Key variant or method
- `variant`: Important hyperparameter or configuration detail

### 3.2 Run Directory Naming

Each experiment creates a directory:
```
results/runs/20251030_1430_baseline_adam_gpt2small_lr1e4/
```

If multiple seeds for same experiment:
```
results/runs/20251030_1430_baseline_adam_gpt2small_lr1e4_seed42/
results/runs/20251030_1430_baseline_adam_gpt2small_lr1e4_seed123/
results/runs/20251030_1430_baseline_adam_gpt2small_lr1e4_seed456/
```

### 3.3 Tags for Organization

Use consistent tags in experiment tracking (wandb):
```yaml
tags:
  - phase: baseline | quantization | lowrank | hybrid | scaling | theory
  - model: gpt2-small | gpt2-medium | gpt2-large | llama-1b
  - optimizer: adam | adam8bit | adafactor | hybrid
  - dataset: wikitext | openwebtext | pile
  - paper: paper1 | paper2 | paper3  # Which paper this is for
  - status: running | completed | failed | archived
```

---

## 4. Experiment Lifecycle

### 4.1 Phase 1: Planning

**Before running experiment:**

1. **Define hypothesis:**
   ```markdown
   ## Hypothesis
   8-bit quantization of Adam's momentum and variance will reduce
   memory by ~50% with <1% degradation in final perplexity.
   ```

2. **Create configuration file:**
   ```yaml
   # configs/experiment/20251030_adam8bit_test.yaml
   experiment_name: "20251030_1430_baseline_adam8bit_gpt2small"
   model:
     name: gpt2
     size: small  # 124M params
   optimizer:
     type: adam8bit
     lr: 1e-4
     betas: [0.9, 0.999]
     quantization: dynamic_blockwise
   # ... rest of config
   ```

3. **Document in research notebook:**
   - Create entry in `notebooks/research-log.md`
   - Link to relevant prior experiments
   - Note what's different from previous runs

### 4.2 Phase 2: Execution

**During experiment:**

1. **Start experiment:**
   ```bash
   python scripts/train.py \
     --config configs/experiment/20251030_adam8bit_test.yaml \
     --output-dir results/runs/20251030_1430_baseline_adam8bit_gpt2small \
     --wandb-project efficient-optimization-research \
     --wandb-tags baseline,adam8bit,gpt2-small,paper1
   ```

2. **Monitor in real-time:**
   - Check wandb dashboard for:
     - Loss curve
     - GPU memory usage
     - Throughput (tokens/sec)
   - Watch for anomalies (NaN, divergence, OOM)

3. **Log continuously:**
   ```python
   # In training script
   wandb.log({
       "train/loss": loss.item(),
       "train/perplexity": perplexity,
       "memory/allocated_gb": torch.cuda.memory_allocated() / 1e9,
       "memory/reserved_gb": torch.cuda.memory_reserved() / 1e9,
       "time/tokens_per_sec": throughput,
       "optimizer/grad_norm": grad_norm,
   }, step=global_step)
   ```

### 4.3 Phase 3: Completion

**After experiment finishes:**

1. **Auto-generate summary:**
   ```python
   # scripts/train.py at end of training
   summary = {
       "experiment_id": experiment_name,
       "config": config_dict,
       "final_metrics": {
           "train_loss": final_train_loss,
           "val_perplexity": final_val_perplexity,
           "peak_memory_gb": peak_memory,
           "total_time_hours": total_time,
       },
       "status": "completed",
       "timestamp": datetime.now().isoformat(),
   }
   with open(f"{output_dir}/summary.json", "w") as f:
       json.dump(summary, f, indent=2)
   ```

2. **Create markdown summary:**
   ```markdown
   # Experiment Summary

   **ID:** 20251030_1430_baseline_adam8bit_gpt2small
   **Date:** 2025-10-30
   **Status:** ✅ Completed
   **Duration:** 4.2 hours

   ## Hypothesis
   8-bit quantization reduces memory by ~50% with <1% perplexity degradation.

   ## Results
   | Metric | Value | Baseline (Adam fp32) | Delta |
   |--------|-------|----------------------|-------|
   | Val Perplexity | 24.3 | 24.1 | +0.8% |
   | Peak Memory | 2.1 GB | 4.2 GB | -50% |
   | Tokens/sec | 4500 | 4300 | +4.7% |

   ## Conclusion
   ✅ Hypothesis confirmed. 8-bit is viable.

   ## Next Steps
   - Test 4-bit quantization
   - Try on larger model (GPT-2 medium)
   ```

3. **Save artifacts:**
   ```
   results/runs/20251030_1430_baseline_adam8bit_gpt2small/
   ├── config.yaml              # Full config used
   ├── summary.json             # Structured summary
   ├── summary.md               # Human-readable summary
   ├── logs/
   │   ├── train.log            # Full training log
   │   └── metrics.csv          # All logged metrics
   ├── checkpoints/
   │   ├── checkpoint-1000.pt
   │   ├── checkpoint-5000.pt
   │   └── final.pt             # Final model
   ├── plots/
   │   ├── loss_curve.png
   │   ├── memory_profile.png
   │   └── grad_norm.png
   └── wandb/                   # wandb local files
   ```

### 4.4 Phase 4: Analysis

**Post-experiment analysis:**

1. **Create analysis notebook:**
   ```
   notebooks/analysis/20251030_adam8bit_analysis.ipynb
   ```

2. **Standard analysis includes:**
   - Load results from multiple seeds (if applicable)
   - Compute mean ± std for key metrics
   - Compare to baseline
   - Generate publication-quality plots
   - Statistical significance tests (t-test, etc.)

3. **Update research log:**
   - Add findings to `notebooks/research-log.md`
   - Link to analysis notebook
   - Note insights and questions raised

### 4.5 Phase 5: Archival

**When experiment is used in paper:**

1. **Move to archive:**
   ```bash
   mkdir -p results/archive/paper1-characterization/
   mv results/runs/20251030_1430_baseline_adam8bit_gpt2small \
      results/archive/paper1-characterization/exp002_adam8bit/
   ```

2. **Create archive manifest:**
   ```markdown
   # results/archive/paper1-characterization/MANIFEST.md

   ## Paper 1: Characterization Study

   ### Experiments

   | ID | Experiment | Figure/Table | Status |
   |----|------------|--------------|--------|
   | exp001 | Adam baseline | Fig 1, Table 1 | ✅ Reproduced |
   | exp002 | Adam 8-bit | Fig 1, Table 1 | ✅ Reproduced |
   | exp003 | Adam 4-bit | Fig 2 | ✅ Reproduced |
   ...
   ```

3. **Backup externally:**
   - Upload to cloud storage (Google Drive, Dropbox, S3)
   - Keep until paper is published + 1 year

---

## 5. Data Collection Standards

### 5.1 Metrics to Always Log

**Every experiment MUST log:**

```python
# At every training step (or every N steps)
metrics_per_step = {
    # Loss & Performance
    "train/loss": float,
    "train/perplexity": float,
    "eval/loss": float,              # Every eval_steps
    "eval/perplexity": float,

    # Memory
    "memory/allocated_gb": float,
    "memory/reserved_gb": float,
    "memory/peak_gb": float,         # Max seen so far

    # Throughput
    "time/tokens_per_sec": float,
    "time/step_time_ms": float,

    # Optimizer
    "optimizer/lr": float,           # Current learning rate
    "optimizer/grad_norm": float,    # Gradient norm (before clipping)
    "optimizer/update_norm": float,  # Update norm (θ_t - θ_{t-1})

    # Step counter
    "step": int,
    "epoch": int,
}

# Phase-specific metrics
if phase == "quantization":
    metrics_per_step.update({
        "quant/error_m": float,      # ||m_fp32 - dequantize(m_quant)||
        "quant/error_v": float,
        "quant/bits_m": int,
        "quant/bits_v": int,
    })

if phase == "lowrank":
    metrics_per_step.update({
        "lowrank/reconstruction_error": float,
        "lowrank/rank": int,
    })

if phase == "theory":
    metrics_per_step.update({
        "theory/hessian_max_eigenvalue": float,  # Approximated
        "theory/sharpness": float,
    })
```

### 5.2 Checkpoint Strategy

**Save checkpoints at:**
- Every N steps (e.g., 1000 steps)
- Every epoch
- Best validation perplexity
- Final model

**Checkpoint contents:**
```python
checkpoint = {
    "step": global_step,
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "config": config_dict,
    "metrics": {
        "train_loss": current_train_loss,
        "val_perplexity": current_val_perplexity,
    },
    "rng_states": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    },
}
torch.save(checkpoint, f"{output_dir}/checkpoints/checkpoint-{step}.pt")
```

**Retention policy:**
- Keep last 3 checkpoints
- Keep best checkpoint
- Keep final checkpoint
- Delete intermediate checkpoints after experiment completion (unless needed for reproducibility)

### 5.3 Evaluation Protocol

**Evaluate on validation set:**
- Every `eval_steps` (e.g., 500 steps)
- At end of each epoch
- On final checkpoint

**Evaluation includes:**
```python
eval_metrics = {
    "eval/loss": float,
    "eval/perplexity": float,
    "eval/tokens_per_sec": float,
}

# For final evaluation, also compute:
final_eval_metrics = {
    "test/loss": float,
    "test/perplexity": float,

    # Downstream tasks (if applicable)
    "downstream/glue_avg": float,

    # Generalization
    "test/ood_perplexity": float,  # Out-of-distribution test set
}
```

---

## 6. Logging & Tracking

### 6.1 Weights & Biases Setup

**Initialize wandb in every experiment:**

```python
import wandb

wandb.init(
    project="efficient-optimization-research",
    name=experiment_name,
    config=config_dict,
    tags=["baseline", "adam8bit", "gpt2-small", "paper1"],
    notes="Testing 8-bit Adam with dynamic blockwise quantization",
    group="paper1-baselines",  # Group related experiments
)

# Log metrics during training
wandb.log(metrics_dict, step=global_step)

# Log artifacts at end
wandb.save(f"{output_dir}/summary.json")
wandb.save(f"{output_dir}/plots/*.png")
```

**Wandb best practices:**
- Use consistent project name
- Tag appropriately for filtering
- Group related experiments (e.g., all seeds for one config)
- Log config as dict (auto-parsed by wandb)
- Save key artifacts (config, summary, plots)

### 6.2 Local Logging

**Use Python's logging module:**

```python
import logging
from pathlib import Path

def setup_logging(output_dir: Path, experiment_name: str):
    log_file = output_dir / "logs" / "train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also print to console
        ],
    )
    logger = logging.getLogger(experiment_name)
    return logger

# Usage
logger = setup_logging(output_dir, experiment_name)
logger.info(f"Starting experiment: {experiment_name}")
logger.info(f"Config: {config}")
logger.info(f"Step {step}: loss={loss:.4f}, ppl={perplexity:.2f}")
```

### 6.3 Metrics CSV

**Save metrics to CSV for easy analysis:**

```python
import csv

metrics_file = output_dir / "logs" / "metrics.csv"

# Initialize CSV
with open(metrics_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
    writer.writeheader()

# Append metrics each step
with open(metrics_file, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
    writer.writerow(metrics_dict)
```

**Benefit:** Easy to load in pandas for analysis:
```python
import pandas as pd
df = pd.read_csv("results/runs/.../logs/metrics.csv")
df.plot(x="step", y="train/loss")
```

---

## 7. Results Storage

### 7.1 Storage Tiers

**Tier 1: Hot storage (local SSD)**
- Current experiments (in progress)
- Recent experiments (last 1 month)
- **Location:** `results/runs/`

**Tier 2: Warm storage (external HDD / NAS)**
- Completed experiments (archived)
- **Location:** `results/archive/` or external drive

**Tier 3: Cold storage (cloud)**
- Published experiments (used in papers)
- Long-term backup
- **Location:** Google Drive, Dropbox, AWS S3

### 7.2 Compression for Archival

**Before archiving, compress:**

```bash
# Compress experiment directory
tar -czf exp002_adam8bit.tar.gz \
  results/archive/paper1-characterization/exp002_adam8bit/

# Exclude large checkpoints if not needed for reproducibility
tar -czf exp002_adam8bit_no_ckpt.tar.gz \
  --exclude="checkpoints" \
  results/archive/paper1-characterization/exp002_adam8bit/
```

**Storage estimates:**
- Logs + metrics: ~10-100 MB
- Plots: ~1-10 MB
- Checkpoints: 500 MB - 5 GB (per checkpoint)
- Compressed (no checkpoints): ~50-200 MB

### 7.3 Backup Strategy

**Automated backup script:**

```bash
#!/bin/bash
# scripts/backup_results.sh

# Sync archive to cloud storage
rclone sync results/archive/ gdrive:research/gradient-based-learning/archive/

# Sync important configs and notebooks
rclone sync configs/ gdrive:research/gradient-based-learning/configs/
rclone sync notebooks/ gdrive:research/gradient-based-learning/notebooks/
```

**Run weekly:** `crontab -e`
```
0 2 * * 0 /path/to/scripts/backup_results.sh
```

---

## 8. Code Organization

### 8.1 Experiment Code Structure

**Each experiment phase has consistent structure:**

```
experiments/01-baselines/
├── README.md                    # Overview of baseline experiments
├── train_adam.py                # Train with standard Adam
├── train_sgd.py                 # Train with SGD+Momentum
├── configs/
│   ├── adam_gpt2_small.yaml
│   └── sgd_gpt2_small.yaml
└── analysis.ipynb               # Analyze baseline results
```

**Shared code in library:**

```
efficient_optim/
├── optimizers/
│   ├── __init__.py
│   ├── adam8bit.py              # Custom optimizer implementations
│   └── hybrid_adam.py
├── utils/
│   ├── profiling.py             # Memory profiling utilities
│   ├── logging.py               # Logging helpers
│   ├── visualization.py         # Plot generation
│   └── metrics.py               # Metric computation
└── tests/
    ├── test_optimizers.py       # Unit tests for optimizers
    └── test_utils.py
```

### 8.2 Configuration Management

**Use Hydra for configs:**

```python
# train.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Access config: cfg.model.name, cfg.optimizer.lr, etc.
```

**Config structure:**
```yaml
# configs/experiment/baseline_adam.yaml
defaults:
  - model: gpt2-small
  - optimizer: adam
  - dataset: wikitext

experiment:
  name: baseline_adam_gpt2small
  seed: 42
  output_dir: results/runs/${experiment.name}

training:
  max_steps: 10000
  eval_steps: 500
  save_steps: 1000
  logging_steps: 100
```

**Override at runtime:**
```bash
python train.py \
  optimizer=adam8bit \
  optimizer.lr=5e-4 \
  experiment.seed=123
```

### 8.3 Version Pinning

**requirements.txt:**
```
# Core
torch==2.1.0
transformers==4.35.0
datasets==2.14.0

# Optimization
bitsandbytes==0.41.1
scipy==1.11.3

# Experiment tracking
wandb==0.16.0
hydra-core==1.3.2

# Development
pytest==7.4.3
black==23.11.0
mypy==1.7.0

# Visualization
matplotlib==3.8.0
seaborn==0.13.0
```

**Pin after environment is stable:**
```bash
pip freeze > requirements-frozen.txt
```

---

## 9. Reproducibility Checklist

### 9.1 Before Running Experiment

- [ ] Hypothesis clearly documented
- [ ] Configuration file created and saved
- [ ] Random seeds set (Python, NumPy, PyTorch, CUDA)
- [ ] Git commit current code state
- [ ] Note git commit hash in experiment config
- [ ] Hardware documented (GPU type, CUDA version)

### 9.2 During Experiment

- [ ] All metrics logged to wandb and local CSV
- [ ] Config snapshot saved to output directory
- [ ] Regular checkpoints saved
- [ ] RNG states saved in checkpoints
- [ ] Logs written to file (not just console)

### 9.3 After Experiment

- [ ] Summary generated (JSON + Markdown)
- [ ] Key plots saved (loss curve, memory profile)
- [ ] Results compared to hypothesis
- [ ] Next steps documented
- [ ] Code pushed to git (if modified)
- [ ] Wandb run marked as finished

### 9.4 For Paper Submission

- [ ] All experiments for paper archived
- [ ] Archive manifest created
- [ ] Reproducibility script created:
  ```bash
  # reproduce_paper1.sh
  python train.py --config configs/paper1/exp001_adam.yaml
  python train.py --config configs/paper1/exp002_adam8bit.yaml
  # ...
  ```
- [ ] Environment pinned (requirements-frozen.txt)
- [ ] README with reproduction instructions
- [ ] Code and data uploaded to public repository
- [ ] External backup completed

---

## 10. Backup & Archival

### 10.1 What to Backup

**Critical (must backup):**
- Final model checkpoints (used in paper)
- Configuration files
- Summary results (JSON, CSV)
- Analysis notebooks (cleared)
- Code (via git)

**Important (should backup):**
- Full training logs
- All checkpoints (intermediate)
- Generated plots

**Optional (can regenerate):**
- Wandb cache files
- Temporary files

### 10.2 Backup Locations

**Primary: Cloud storage (Google Drive / Dropbox)**
```
Vera Strata Research/
└── gradient-based-learning/
    ├── archive/
    │   ├── paper1-characterization/
    │   ├── paper2-hybrid/
    │   └── paper3-theory/
    ├── configs/
    ├── notebooks/
    └── code-snapshots/
        └── git-commits/
```

**Secondary: External hard drive**
- Full copy of `results/archive/`
- Updated monthly

**Tertiary: University cluster storage (if available)**
- Large checkpoints that don't fit in cloud

### 10.3 Backup Schedule

| Frequency | What | Where |
|-----------|------|-------|
| Daily | Current experiment results | Local backup |
| Weekly | Completed experiments | Cloud sync |
| Monthly | Full archive | External HDD |
| On paper submission | All paper-related data | Cloud + HDD + cluster |
| On paper acceptance | Published experiments | Permanent archive (Zenodo, OSF) |

### 10.4 Disaster Recovery

**If local data lost:**
1. Restore from cloud (most recent weekly backup)
2. Re-run current experiments if needed
3. Check external HDD for older archives

**If code lost:**
1. Clone from GitHub
2. Checkout specific commit (noted in experiment config)
3. Verify environment (requirements-frozen.txt)

**If cloud account lost:**
1. Restore from external HDD
2. Re-upload to new cloud account

---

## 11. Templates

### 11.1 Experiment Summary Template

```markdown
# Experiment: [EXPERIMENT_ID]

**Date:** YYYY-MM-DD
**Status:** Running | Completed | Failed
**Duration:** X.X hours
**Paper:** Paper 1 | Paper 2 | Paper 3

---

## Hypothesis

[Clear statement of what you're testing]

---

## Configuration

**Model:** GPT-2 Small (124M)
**Dataset:** WikiText-103
**Optimizer:** Adam 8-bit (dynamic blockwise)
**Learning Rate:** 1e-4
**Batch Size:** 8
**Sequence Length:** 512

---

## Results

| Metric | Value | Baseline | Delta | Target |
|--------|-------|----------|-------|--------|
| Val Perplexity | XX.X | XX.X | +X.X% | <1% |
| Peak Memory (GB) | X.X | X.X | -XX% | -50% |
| Tokens/sec | XXXX | XXXX | +X% | ≥0% |
| Total Time (hrs) | X.X | X.X | +X% | <20% |

---

## Observations

- [Key observation 1]
- [Key observation 2]
- [Unexpected behavior]

---

## Conclusion

✅ / ❌ Hypothesis [confirmed / rejected]

[Brief interpretation of results]

---

## Next Steps

- [ ] [Follow-up experiment 1]
- [ ] [Analysis to perform]
- [ ] [Question to investigate]

---

## Files

- **Config:** `configs/experiment/[name].yaml`
- **Wandb:** https://wandb.ai/[user]/[project]/runs/[id]
- **Results:** `results/runs/[experiment_id]/`
- **Analysis:** `notebooks/analysis/[experiment_id].ipynb`
```

### 11.2 Analysis Notebook Template

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analysis: [Experiment Name]
#
# **Date:** YYYY-MM-DD
# **Experiment ID:** [ID]
# **Objective:** [What are we analyzing?]

# %% [markdown]
# ## Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
# ## Load Data

# %%
exp_dir = Path("results/runs/[experiment_id]")
metrics_df = pd.read_csv(exp_dir / "logs" / "metrics.csv")
metrics_df.head()

# %% [markdown]
# ## Analysis 1: Convergence

# %%
fig, ax = plt.subplots()
ax.plot(metrics_df['step'], metrics_df['train/loss'], label='Train')
ax.plot(metrics_df['step'], metrics_df['eval/loss'], label='Eval')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Training Convergence')
ax.legend()
plt.savefig(exp_dir / "plots" / "convergence.png", dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Analysis 2: Memory Profile

# %%
# [Memory analysis code]

# %% [markdown]
# ## Comparison with Baseline

# %%
# [Comparison code]

# %% [markdown]
# ## Conclusions
#
# - [Key finding 1]
# - [Key finding 2]
#
# ## Next Steps
#
# - [ ] [Action item]
```

---

## 12. Quality Assurance

### 12.1 Code Review Checklist

Before committing code:
- [ ] Code passes unit tests (`pytest`)
- [ ] Code formatted with `black`
- [ ] Type hints added, passes `mypy` checks
- [ ] Docstrings for all public functions
- [ ] No hardcoded paths (use Path, args, configs)
- [ ] No secrets in code (API keys, etc.)

### 12.2 Experiment Review Checklist

Before archiving experiment:
- [ ] Summary.md created and complete
- [ ] All key metrics logged
- [ ] Hypothesis and conclusion documented
- [ ] Comparison to baseline (if applicable)
- [ ] Plots saved and labeled
- [ ] Config saved
- [ ] Reproducibility verified (if critical)

### 12.3 Paper Preparation Checklist

Before paper submission:
- [ ] All figures reproducible from archived experiments
- [ ] All tables regeneratable from saved results
- [ ] Reproducibility script tested on clean environment
- [ ] Code published to GitHub (or ready to publish)
- [ ] Data/results uploaded to Zenodo or OSF (if required)
- [ ] README with instructions
- [ ] License file included

---

## Appendix A: Tools Reference

### A.1 Useful Commands

**Check GPU memory:**
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

**Monitor experiment logs:**
```bash
tail -f results/runs/[experiment_id]/logs/train.log
```

**Find all experiments with tag:**
```bash
grep -r "paper1" results/runs/*/summary.json
```

**Aggregate results:**
```bash
python scripts/aggregate_results.py \
  --pattern "results/archive/paper1-*" \
  --output paper1_results.csv
```

### A.2 Useful Python Snippets

**Load experiment results:**
```python
import json
from pathlib import Path

def load_experiment(exp_dir):
    exp_dir = Path(exp_dir)
    with open(exp_dir / "summary.json") as f:
        summary = json.load(f)
    metrics = pd.read_csv(exp_dir / "logs" / "metrics.csv")
    return summary, metrics
```

**Compare multiple experiments:**
```python
import pandas as pd

def compare_experiments(exp_dirs):
    results = []
    for exp_dir in exp_dirs:
        summary, _ = load_experiment(exp_dir)
        results.append({
            "name": summary["experiment_id"],
            "val_ppl": summary["final_metrics"]["val_perplexity"],
            "memory_gb": summary["final_metrics"]["peak_memory_gb"],
        })
    return pd.DataFrame(results)

df = compare_experiments([
    "results/archive/paper1/exp001_adam",
    "results/archive/paper1/exp002_adam8bit",
])
print(df)
```

---

**Document Control:**
- **Last Updated:** 2025-10-30
- **Next Review:** 2026-01-30
- **Owner:** Manolo Rodeta / Vera Strata AI Research
- **Status:** Active
