# RL‑Portfolio: Reconfigurable Reinforcement‑Learning Research Framework

[![license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/Python-3.11%2b-informational)](https://www.python.org/)
[![hydra](https://img.shields.io/badge/Config-Hydra%201.3‑blueviolet)](https://github.com/facebookresearch/hydra)
[![wandb](https://img.shields.io/badge/Tracking-W%26B-f79520)](https://wandb.ai/)

> **Release status**: 🚧  Alpha &ndash; initial scaffolding for Anand Patel’s MSc/PhD reinforcement‑learning project.

---

## ✨ Project Overview
This repository hosts a **research‑grade pipeline** for training, evaluating, and analysing reinforcement‑learning (RL) agents in simulated tasks.  
The key design goal is **rapid iteration on your workstation** followed by **seamless scale‑out on an HPC cluster** (GPU Slurm farm).  
Features include:

| Feature | Why it matters |
|---------|----------------|
| **Hydra‑powered hierarchical configs** | One‑line overrides, composable “pyramid” ablations, automatic config logging. |
| **Stable‑Baselines3 / JAX back‑end** | Start fast with PyTorch, switch to JAX for large‑scale vectorised training. |
| **Weights & Biases integration** | Live dashboards + offline sync; keeps every run reproducible (git SHA, GPU type, hyper‑params). |
| **Conda + Docker/Singularity** | Identical environments locally and on cluster; GPU drivers handled via NVIDIA images. |
| **Slurm templates** | Zero‑boilerplate submission: `sbatch scripts/submit_job.slurm algo=ppo total_timesteps=1e8`. |
| **CI & linting** | GitHub Actions, `pre‑commit`, `pytest` smoke tests guard code quality. |

---

## 🗺️ Directory Structure

```
rl‑project/
├── config/                  # Hydra config tree (YAML)
│   ├── algo/                # PPO, SAC, etc. inherit from base_algo.yaml
│   ├── env/                 # Sim versions, domain‑randomisation toggles
│   ├── hydra/               # Job‑logging, sweep configs
│   ├── overrides/           # Figure‑ready presets
│   └── experiment.yaml      # Root config composing the pieces
├── src/                     # Importable Python package `rl_project`
│   ├── agents/              # Custom policy nets, loss functions
│   ├── envs/                # Gymnasium‑compatible simulators / wrappers
│   ├── training/            # train.py, evaluate.py, rollout.py
│   ├── utils/               # misc helpers: seeding, logging, ablations
│   └── __init__.py
├── scripts/                 # Pure Bash helpers & Slurm templates
│   ├── local_setup.sh
│   ├── cluster_setup.sh
│   ├── submit_job.slurm
│   └── sweep.yaml
├── environment.yml          # Conda spec (GPU + CPU fallback)
├── Dockerfile               # Builds nvidia‑cuda runtime image
├── tests/                   # PyTest smoke & unit tests
├── .pre‑commit‑config.yaml  # Black, isort, flake8, nbstripout
├── .github/workflows/ci.yml # Continuous‑integration pipeline
├── README.md                # ← you are here
└── LICENSE
```

---

## 🏗️ Tech Stack & Rationale

| Layer | Tooling | Rationale |
|-------|---------|-----------|
| **Language** | Python 3.11 | Wide ecosystem, async improvements, structural pattern‑matching. |
| **RL Library** | Stable‑Baselines3 (PyTorch 2.3) • optional CleanRL‑JAX | SB3 is battle‑tested & fast to prototype; JAX unlocks TPU/GPU vectorisation later. |
| **Configs** | Hydra 1.3 + OmegaConf | Nested YAML with inheritance, job‑id interpolation, dynamic defaults. |
| **Experiment tracking** | Weights & Biases | Auto‑logs git, hardware, gradients; cluster‑safe offline mode. |
| **Package/env** | Conda (mamba) | GPU builds via `pytorch` channel; clusters already ship Conda modules. |
| **Container** | Docker → Singularity(.sif) | Immutable environment; Slurm can run `--container-image`. |
| **Scheduler** | Slurm | Standard at CHPC & most university clusters. |
| **CI/CD** | GitHub Actions | Free linux runners for lint + CPU unit tests. |
| **Code Style** | `black`, `isort`, `flake8`, type hints (`mypy`) | Enforced via pre‑commit on every commit / PR. |

---

## 🚀 Getting Started

### 1. Clone & bootstrap (local workstation)

```bash
git clone https://github.com/anand1221178/Proactive-Reinforcement-Learning-for-Robust-Quadruped-Locomotion.git
cd rl‑project
bash scripts/local_setup.sh      # installs Conda env + pre‑commit hooks
conda activate rlproj
python -m rl_project.training.train dry_run=true
```

`dry_run=true` executes a 32‑step rollout to verify the plumbing.

### 2. Cluster setup (Slurm GPU node)

```bash
module load miniconda/23.1 cuda/11.8 git
bash scripts/cluster_setup.sh     # creates identical Conda env on $HOME/.conda
sbatch scripts/submit_job.slurm algo=ppo total_timesteps=1e7
```

### 3. Container workflow (optional but recommended)

```bash
docker build -t rlproj:latest .
# On cluster login node
singularity build rlproj.sif docker-daemon://rlproj:latest
sbatch --container-image=rlproj.sif scripts/submit_job.slurm algo=sac
```

Offline W&B logs accumulate under `$WANDB_DIR` and can be synced later:

```bash
wandb sync /scratch/$USER/wandb/offline‑runs
```

---

## ⚙️ Configuration 101

Hydra composes configs from `config/` according to the **defaults list** at the top of `experiment.yaml`.

```yaml
# config/experiment.yaml
defaults:
  - algo: ppo
  - env: sim_v1
  - hydra: base
  - _self_

total_timesteps: 2_000_000
seed: 123
```

Override anything from the CLI:

```bash
python -m rl_project.training.train algo=ppo clip_range=0.1 env=sim_v2 seed=42
```

Create *pyramids* for ablations by layering multiple algos or env options:

```bash
python -m rl_project.training.train   +algo=ppo   +algo.lstm=true   +env=sim_v2   +env.domain_rand=true
```

Hydra writes the **full, resolved config** and CLI exactly as run into
```
outputs/
└── 2025‑06‑20/16‑58‑42/   # timestamp
    ├── .hydra/config.yaml
    └── train.log
```

---

## 📊 Experiment Tracking

* **wandb run**: metrics, gradients, videos, git SHA, CUDA driver, peak GPU mem.
* **Group key**: `env-name_algo-seed`, e.g. `sim_v2_ppo-42`.
* **Sweep**: define grid/random in `scripts/sweep.yaml`, run `wandb sweep && wandb agent`.

> **Privacy**: offline mode is auto‑enabled when `$WANDB_MODE=offline`  
> Synchronise later with `wandb sync`.

---

## 🧪 Ablation & Sweep Workflow

| Step | Command |
|------|---------|
| 1. Define base config | `config/algo/ppo.yaml`, `config/env/sim_v2.yaml` |
| 2. Add variant YAMLs | e.g. `config/algo/ppo_small_lr.yaml` |
| 3. Launch sweep | `wandb sweep scripts/sweep.yaml` |
| 4. Agent script | Hydra injects params → SB3 trainer |


---

## 🖥️ Running Evaluations

```bash
python -m rl_project.training.evaluate   checkpoint=outputs/2025‑06‑20/16‑58‑42/model.zip   n_episodes=100   render=false
```

Results (return, length, custom metrics) are dumped to `.csv` and logged to W&B *table*.

---

## 🛠️ Developer Guide

* **Pre‑commit**: auto‑format on commit. Run manually via `pre‑commit run --all-files`.
* **Unit tests**: `pytest -q tests/`. Smoke tests run on CPU under CI.
* **Docstrings** follow the *Google* style. Public functions: full docs; private: brief one‑liner.
* **Type hints**: mandatory for new functions (`from __future__ import annotations`).
* **Branch naming**: `feat/<topic>`, `fix/<bug>`, `exp/<sweep‑name>`.

---

## 🤝 Contributing

1. Fork → `git checkout -b feat/my-awesome-feature`
2. Commit (+ tests!) → open Pull Request.
3. Ensure CI passes (lint & tests).
4. PR will trigger GPU smoke run on cluster (self‑hosted runner).

---

## 📄 License

This work is released under the **MIT License** – see [LICENSE](LICENSE).

---

## 📚 References

* Schulman, J. *et al.* “Proximal Policy Optimization Algorithms.” 2017.
* Raffin, A. *et al.* “Stable Baselines3.” *Journal of Machine Learning Open Source Software*, 2021.

---

## 📫 Contact

Questions, issues, ideas for wild ablations?  
**Anand Patel** – <anand.patel@students.wits.ac.za>  

---

Happy training & may your gradients be ever in your favour 🙌