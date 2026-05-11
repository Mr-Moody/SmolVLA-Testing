# SmolVLA / Pi0 / ACT — Dataset + Phase-Conditioned Training Pipeline

Unified pipeline for collecting, labeling, converting, and training robot demonstration data with SmolVLA, Pi0, Pi0.5, or ACT policies — extended with **Qwen3-VL offline phase annotation**, a **runtime FSM**, and **three phase-conditioned policy variants**.

---

## Project Layout

```text
Industrial-Project/
  lerobot/                     ← sibling lerobot clone (torch + policy deps)
  SmolVLA-Testing/
    main.py                    ← unified CLI (clean / label / convert / train)
    src/
      common/
        phases.py              ← SINGLE SOURCE OF TRUTH for phase IDs + transition rules
      annotation/
        serve_qwen.py          ← QwenAnnotator — vLLM wrapper for Qwen3-VL-30B
        sampler.py             ← KeyframeSampler — gripper/F·T/velocity keyframe picker
        prompt.py              ← PromptBuilder — task-aware few-shot prompt assembly
        schema.py              ← PhaseSegment + EpisodeAnnotation (Pydantic v2)
        validator.py           ← Annotation quality checks (schema, legality, FSM x-check)
      fsm/
        runtime_fsm.py         ← RuntimeFSM — predicate-based, hysteresis-debounced
      smolvla_fork/
        configuration_smolvla.py  ← SmolVLAForkedConfig (phase_dropout_prob, use_phase_conditioning)
        modeling_smolvla.py       ← VLAFlowMatchingPhased + SmolVLAPhasedPolicy
        dataset.py                ← PhaseConditionedDataset + phase_collate_fn
        CHANGELOG.md              ← upstream version + what was changed
      inference/
        policy_runner.py       ← PolicyRunner — closed-loop episode runner
      robot/
        franka_interface.py    ← MockFrankaInterface + safety envelope (NaN/Inf/vel/force/gripper)
      scripts/                 ← Python CLI entry points (moved here from scripts/)
        annotate_dataset.py      ← Batch Qwen annotation driver
        build_gold_set.py        ← Interactive gold-set builder
        deploy_check.py          ← Pre-deployment safety checklist
        eval.py                  ← Evaluation harness (dry-run + real)
        generate_annotations.py  ← Generate + write per-episode task prompts
        report.py                ← Comparison report with Wilson CI + matplotlib charts
        serve_qwen.py            ← Thin CLI wrapper around QwenAnnotator
        train_phase.py           ← Phase-conditioned training entry point
        tune_fsm.py              ← FSM threshold tuning vs Qwen ground-truth
        writeback_annotations.py ← Merge phase + subtask columns into dataset copy
      data_cleaner.py
      data_converter.py
      dataset_utils.py
      labeler.py
      create_labels.py
      merge_datasets.py
      train_act.py
      train_pi0.py
    scripts/                   ← Shell scripts only (batch ops, GPU cluster)
      00_run_params.sh … 05_extract_from_scratch.sh
      convert_all.sh  run_training.sh  setup_scratch.sh  …
    configs/
      training/
        smolvla_baseline.yaml  smolvla_fsm.yaml  pi0_subtask.yaml
      fsm/
        default.yaml  pick_place.yaml  msd_plug.yaml
      eval/
        standard.yaml
    tests/                     ← pytest suite (81 tests, all passing)
    data/gold/                 ← hand-annotated few-shot episodes
    outputs/                   ← training checkpoints, eval results, reports
    frontend/                  ← labeler web UI
```

### Data directories

```text
raw_datasets/<name>/               ← recorded session (input to clean)
cleaned_datasets/<name>/           ← output of  main.py clean
lerobot_datasets/<name>/           ← output of  main.py convert  (LeRobot v3)
lerobot_datasets/<name>_annotated/ ← phase + subtask columns written by writeback_annotations.py
outputs/<name>_smolvla/            ← output of  main.py train --model-type smolvla
outputs/<name>_pi0/                ← output of  main.py train --model-type pi0
outputs/<name>_pi05/               ← output of  main.py train --model-type pi05
outputs/<name>_act/                ← output of  main.py train --model-type act
outputs/annotations/               ← per-episode Qwen annotation parquets
outputs/eval/<run>/results.json    ← evaluation results
outputs/reports/                   ← markdown + PNG comparison reports
data/gold/                         ← hand-annotated reference episodes (few-shot source)
```

---

## Installation

Install the `lerobot` package with the relevant extras into its own uv environment from within the `lerobot` sibling directory:

```bash
cd ../lerobot

# SmolVLA
uv pip install -e ".[smolvla]"

# Pi0 / Pi0.5 (adds transformers PaliGemma support)
uv pip install -e ".[pi0]"

# ACT / base LeRobot install
uv pip install -e "."

# Both SmolVLA and Pi0/Pi0.5 extras
uv pip install -e ".[smolvla,pi0]"
```

Install phase-conditioning extras (vLLM, Pydantic v2, etc.):

```bash
pip install -r requirements_phase.txt
```

---

## Running Commands

`SmolVLA-Testing` does not have its own `pyproject.toml`. All `main.py` commands must be run against the `lerobot` project environment:

```bash
uv --project ../lerobot run python main.py <subcommand> [args]
```

Python CLI scripts in `src/scripts/` can be run directly (they manage their own `sys.path`):

```bash
python src/scripts/<script>.py [args]
# or
python -m src.scripts.<script> [args]
```

---

## Full Base Pipeline

```bash
# 1. Clean raw recording
uv --project ../lerobot run python main.py clean <dataset_name>

# 2. Generate per-episode global task prompts
uv --project ../lerobot run python main.py annotate <dataset_name>

# 3. Label episodes in the browser UI
uv --project ../lerobot run python main.py label

# 4. Convert to LeRobotDataset v3
uv --project ../lerobot run python main.py convert <dataset_name> --primary-camera <camera_name>

# 5. Train  (SmolVLA default; pass --model-type pi0, pi05, or act to switch)
uv --project ../lerobot run python main.py train \
  --dataset-root lerobot_datasets/<dataset_name> \
  [--model-type smolvla|pi0|pi05|act]
```

---

## Phase-Conditioned Pipeline (New)

Three phase-conditioned policy variants sit on top of the base pipeline.

### Architecture Overview

```
cleaned_datasets/<name>/
  └─ KeyframeSampler ──► Qwen3-VL-30B (vLLM) ──► EpisodeAnnotation (Pydantic)
                                                         │
                                              Validator (schema + FSM x-check)
                                                         │
                                          writeback_annotations.py
                                                         │
                                  lerobot_datasets/<name>_annotated/
                                    ├─ phase  (int64 per frame, IDs 0–4)
                                    └─ subtask (str, e.g. "pick_place.contact_establish")
                                                         │
                           ┌─────────────────────────────┴─────────────────────────┐
                           │                             │                          │
                  SmolVLA-baseline              SmolVLA-FSM                  π0-subtask
                  (no phase info)        (phase embedding +              (native subtask col)
                                          runtime FSM)
```

### Five-Phase Taxonomy  (`src/common/phases.py` — single source of truth)

| ID | Name | Pick-and-Place | MSD Plugging |
|----|------|----------------|--------------|
| 0 | `free_motion` | Approach to object, transport to target (no contact) | Approach to receptacle (gripper holding connector) |
| 1 | `fine_align` | Pre-grasp positioning, visual servo onto object | Pin-to-socket alignment, sub-mm positioning |
| 2 | `contact_establish` | Gripper closure on object | First pin contact / pre-mate touch |
| 3 | `constrained_motion` | Lift while grasped, in-hand manipulation | Insertion stroke under contact |
| 4 | `verify_release` | Place + gripper open + retract | Seating verification, gripper open + retract |

Phase IDs and transition logic are defined in `src/common/phases.py`, which is the single source of truth for all phase constants.

### Policy Variants

| Variant | Config | Notes |
|---------|--------|-------|
| `smolvla_baseline` | `configs/training/smolvla_baseline.yaml` | No phase info; delegates to `main.py train --model-type smolvla` |
| `smolvla_fsm` | `configs/training/smolvla_fsm.yaml` | Forked SmolVLA in `src/smolvla_fork/` + runtime FSM phase signal |
| `pi0_subtask` | `configs/training/pi0_subtask.yaml` | π0 with `subtask` column; delegates to `main.py train --model-type pi0` |

### FSM Transition Rules  (`src/fsm/runtime_fsm.py`)

```
free_motion      → fine_align           (xy dist < threshold AND z < hover_z)
fine_align       → contact_establish    (Fz > contact_force OR gripper close cmd)
fine_align       → free_motion          (pose error > threshold for >300 ms)  ← re-approach
contact_establish → constrained_motion  (gripper stable AND Fz in mate band)
constrained_motion → verify_release     (target z reached AND Fz in seated band)
verify_release   → (terminal)           (gripper open AND retract initiated)
```

Self-loops always legal; no backward jumps except `fine_align → free_motion`.  
Thresholds: `configs/fsm/pick_place.yaml` and `configs/fsm/msd_plug.yaml`.

### SmolVLA Fork — Phase Embedding

`src/smolvla_fork/modeling_smolvla.py` adds to the action expert:

```python
self.phase_embedding = nn.Embedding(6, hidden_dim)   # index 5 = unknown token
nn.init.zeros_(self.phase_embedding.weight)           # zero-init → identity at step 0
```

During training, phase IDs are randomly replaced with the unknown token (index 5) at
`phase_dropout_prob = 0.15` probability. When `use_phase_conditioning=False`, the
forward pass is byte-identical to the upstream SmolVLA.

---

## Step-by-Step Phase Annotation Workflow

```bash
# 1. Build gold episodes interactively (few-shot source of truth → data/gold/)
python src/scripts/build_gold_set.py \
  --dataset-root lerobot_datasets/<name> \
  --episode-id 0 --task pick_place

# 2. Batch-annotate all episodes with Qwen3-VL
python src/scripts/annotate_dataset.py \
  --dataset-root lerobot_datasets/<name> \
  --task pick_place \
  --output-dir outputs/annotations

# 3. Write phase + subtask columns into annotated dataset copy
python src/scripts/writeback_annotations.py \
  --annotations-dir outputs/annotations \
  --source-dataset lerobot_datasets/<name> \
  --output-dataset lerobot_datasets/<name>_annotated

# 4. (Optional) Tune FSM thresholds against Qwen ground truth
python src/scripts/tune_fsm.py \
  --episode-parquet outputs/annotations/episode_000000.parquet \
  --task pick_place \
  --annotation-json data/gold/0.json

# 5. Train phase-conditioned variant
python src/scripts/train_phase.py \
  --config configs/training/smolvla_fsm.yaml \
  --dataset-name <name>

# 6. Smoke-test training (10 steps, mock dataset if real one missing)
python src/scripts/train_phase.py \
  --config configs/training/smolvla_fsm.yaml \
  --dataset-name <name> \
  --smoke-test
```

---

## Evaluation

### Run evaluation harness (dry-run / CI)

```bash
python src/scripts/eval.py \
  --variant smolvla_baseline \
  --task pick_place \
  --dry-run
```

### Run all three variants and generate comparison report

```bash
python src/scripts/eval.py --variant smolvla_baseline --task pick_place --dry-run
python src/scripts/eval.py --variant smolvla_fsm      --task pick_place --dry-run
python src/scripts/eval.py --variant pi0_subtask      --task pick_place --dry-run

python src/scripts/report.py \
  outputs/eval/<ts>_smolvla_baseline_pick_place/results.json \
  outputs/eval/<ts>_smolvla_fsm_pick_place/results.json \
  outputs/eval/<ts>_pi0_subtask_pick_place/results.json
```

The report writes:
- `outputs/reports/<ts>_report.md` — markdown table with **95% Wilson confidence intervals**
- `outputs/reports/<ts>_failure_chart.png` — stacked failure attribution by phase
- `outputs/reports/<ts>_corruption_chart.png` — phase-corruption recovery rates

### Eval config (`configs/eval/standard.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `n_episodes` | 20 | Normal evaluation episodes |
| `n_corruption_episodes` | 10 | Phase-corruption episodes (FSM + π0 only) |
| `seed` | 42 | Random seed |
| `pick_place_success_tolerance_m` | 0.005 | Place position tolerance |
| `msd_plug_success_force_n` | 3.0 | Minimum seating force (N) |

---

## Pre-Deployment Safety Checklist

```bash
# Mock mode (CI / offline):
python src/scripts/deploy_check.py --mock

# Real robot:
python src/scripts/deploy_check.py \
  --checkpoint outputs/my_run/checkpoint_10000 \
  --variant smolvla_fsm
```

Checks: robot reachable, F/T sensor near zero, checkpoint valid, FSM valid phase, safety envelope rejects NaN actions, W&B available, e-stop confirmed.

---

## Testing

```bash
# All tests (excludes GPU and real-robot tests):
uv --project ../lerobot run python -m pytest tests/ -v -m "not gpu"

# With coverage:
uv --project ../lerobot run python -m pytest tests/ -m "not gpu" --cov=src --cov-report=term-missing
```

Test markers:
- `@pytest.mark.gpu` — requires CUDA GPU
- `@pytest.mark.robot` — requires physical Franka robot (always skipped in CI)

Current status: **81 tests, 81 pass**.

---

## Command Reference

### `main.py clean`

Removes static (non-moving) steps and trims to camera-covered frames only.

```bash
uv --project ../lerobot run python main.py clean <dataset_name> [options]
```

Input: `raw_datasets/<dataset_name>/` → Output: `cleaned_datasets/<dataset_name>/`

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets-root` | `raw_datasets` | Root containing raw recordings |
| `--output-root` | `cleaned_datasets` | Root for cleaned output |
| `--camera-tolerance-ms` | `150` | Max robot/camera sync error (ms) |
| `--joint-motion-threshold` | `5e-4` | Max joint delta (rad) considered stationary |
| `--gripper-motion-threshold` | `2e-4` | Max gripper delta (m) considered stationary |
| `--action-translation-threshold` | `5e-6` | Min translation norm considered movement |
| `--action-rotation-threshold` | `5e-5` | Min rotation norm considered movement |
| `--max-episodes` | — | Limit number of episodes processed |
| `--generate-tasks` | — | Auto-assign task prompt + write `annotations.jsonl` |
| `--force` | — | Overwrite existing output directory |

### `main.py annotate`

Generates unique natural-language task prompts (one per episode) and writes them to `annotations.jsonl` inside the cleaned dataset. The converter picks this file up automatically via `load_annotations()` and stamps every frame with the correct task string.

```bash
uv --project ../lerobot run python main.py annotate <dataset_name>
```

Input: `cleaned_datasets/<dataset_name>/episode_events.jsonl` (used to count episodes)  
Output: `cleaned_datasets/<dataset_name>/annotations.jsonl`

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets-root` | `cleaned_datasets` | Override the cleaned datasets root |
| `--overwrite` | — | Replace an existing `annotations.jsonl` |

Prompts are drawn from a fixed vocabulary of verb/object/preposition/receptacle combinations and shuffled with seed 42, so output is reproducible. The number of prompts generated matches the episode count detected in the dataset.

### `main.py label`

Launches a local Flask server for reviewing and labeling episodes in the browser.

```bash
uv --project ../lerobot run python main.py label [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--cleaned-root` | `cleaned_datasets` | Root of cleaned datasets |
| `--raw-root` | `raw_datasets` | Root of raw videos |
| `--port` | `5000` | HTTP port |
| `--no-browser` | — | Don't auto-open the browser |

### `main.py convert`

```bash
uv --project ../lerobot run python main.py convert <dataset_name> [options]
```

Input: `cleaned_datasets/<dataset_name>/` → Output: `lerobot_datasets/<dataset_name>/`

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets-root` | `cleaned_datasets` | Root containing cleaned datasets |
| `--output-root` | `lerobot_datasets` | Root for converted output |
| `--repo-id` | `local/<dataset_name>` | LeRobot metadata repo id |
| `--primary-camera` | — | Camera mapped to `observation.images.top` |
| `--cameras` | — | Comma-separated camera names to include |
| `--camera-tolerance-ms` | `150` | Max robot/camera sync error (ms) |
| `--text-tolerance-ms` | `2000` | Max text/frame sync error (ms) |
| `--vcodec` | `h264` | Output video codec |
| `--max-episodes` | — | Limit exported episodes |
| `--max-steps-per-episode` | — | Limit steps per episode |
| `--force` | — | Overwrite existing output directory |

### `main.py train`

```bash
uv --project ../lerobot run python main.py train --dataset-root <path> [options]
```

**Common options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type` | `smolvla` | `smolvla`, `pi0`, `pi05`, or `act` |
| `--dataset-root` | *(required)* | LeRobotDataset v3 directory |
| `--output-dir` | `outputs/<dataset>_<model>` | Checkpoints + logs |
| `--batch-size` | `8` | Training batch size |
| `--steps` | `20000` | Training steps |
| `--device` | `cuda` | Training device |
| `--seed` | `1000` | Training seed |
| `--use-amp` | — | Automatic mixed precision |
| `--resume` | — | Resume from last checkpoint |

**Pi0 / Pi0.5 only:**

| Flag | Default | Description |
|------|---------|-------------|
| `--freeze-vision-encoder` | — | Freeze PaliGemma vision encoder |
| `--train-expert-only` | — | Freeze VLM; train action expert only |
| `--gradient-checkpointing` | — | Reduce VRAM |
| `--dtype` | `float32` | `float32` or `bfloat16` |

**ACT only:**

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size` | `100` | Future action steps per forward pass |
| `--n-obs-steps` | `1` | Observation steps as input |
| `--vision-backbone` | `resnet18` | `resnet18`, `resnet34`, or `resnet50` |
| `--use-vae` / `--no-vae` | enabled | ACT VAE action modeling |

**Model comparison:**

| Aspect | SmolVLA | Pi0 | ACT |
|--------|---------|-----|-----|
| Vision model | SmolVLM2-500M | PaliGemma 2B | ResNet18/34/50 |
| Approx. size | 500M params | 2B+ params | 10–25M params |
| Action output | Single step | Single step | Chunked |
| VRAM (fp32) | ~16 GB | ~20–24 GB | <4 GB |
| Best fit | Language-conditioned | Dexterous with language | Fast long-horizon IL |

---

## Utilities

### Inspect an Exported Dataset

```bash
uv --project ../lerobot run python src/smolvla_franka_setup.py \
  --mode inspect --dataset-root lerobot_datasets/example
```

### SmolVLA Inference Sanity Check

```bash
uv --project ../lerobot run python src/smolvla_franka_setup.py --mode demo
```

### Verify CUDA / Torch

```bash
uv --project ../lerobot run python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

---

## Franka Robot Notes

### Docker setup (optional)

```bash
xhost +local:docker
docker run -it --name franka_noetic --net=host --privileged \
  -v ~/catkin_ws:/home/thomas/catkin_ws \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  osrf/ros:noetic-desktop-full bash
```

### Pre-run checklist

1. Host ethernet: manual IP `172.16.0.2`
2. `ping 172.16.0.1` confirms connectivity
3. Franka Desk at `https://172.16.0.1`: FCI active (blue), brakes unlocked, external activation device pressed

```bash
source devel/setup.bash
roslaunch franka_control franka_control.launch robot_ip:=172.16.0.1 load_gripper:=true
```

### Safety envelope (`src/robot/franka_interface.py`)

The `MockFrankaInterface` (used in dry-run and tests) enforces:

| Condition | Action |
|-----------|--------|
| NaN or Inf in action | Reject, log to `outputs/safety_log.jsonl` |
| Joint velocity > 2.0 rad/s | Reject |
| EEF force > 30 N | Reject |
| Gripper close cmd in `FREE_MOTION` phase | Reject |
| Action outside workspace bounds | Reject |
