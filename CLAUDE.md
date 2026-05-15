# CLAUDE.md — SmolVLA-Testing Project Conventions

## 1. Python Invocation

All scripts in this repo run via the sibling LeRobot project:

```bash
uv --project ../lerobot run python <script.py> [args]
```

Or activate the lerobot venv directly:
```bash
source ../lerobot/.venv/bin/activate
python <script.py>
```

Never create a standalone `pyproject.toml` or venv inside this repo — it would conflict.

---

## 2. Data Pipeline (existing — do not break)

```
raw_datasets/<name>/          ← raw recordings (read-only)
  └── main.py clean <name>
cleaned_datasets/<name>/      ← filtered recordings
  └── main.py label
  └── main.py convert <name> --primary-camera ee_zed_m_left
lerobot_datasets/<name>/      ← LeRobotDataset v3
  └── scripts/annotate_dataset.py + writeback_annotations.py
lerobot_datasets/<name>_annotated/   ← with phase + subtask columns
  └── scripts/train_phase.py
outputs/<name>_<model>/       ← training outputs
```

---

## 3. File Modification Rules

All **new** code goes under `src/<subpackage>/` with matching tests under `tests/`.

---

## 4. Five-Phase Taxonomy (single source of truth: `src/common/phases.py`)

Any code that references phase IDs **must** import from `src/common/phases.py`.
Never hardcode integer phase IDs.

| ID | Name | Pick-and-Place | MSD Plugging |
|----|------|----------------|--------------|
| 0 | `free_motion` | Approach to object; transport to target zone (no contact) | Approach to receptacle (gripper holding connector, no contact) |
| 1 | `fine_align` | Pre-grasp positioning; visual servo onto object | Pin-to-socket alignment; sub-mm positioning |
| 2 | `contact_establish` | Gripper closure on object | First pin contact / pre-mate touch |
| 3 | `constrained_motion` | Lift while grasped; in-hand manipulation | Insertion stroke under contact |
| 4 | `verify_release` | Place + gripper open + retract | Seating verification; gripper open + retract |

---

## 5. Three Policy Variants

| Variant | Config | Notes |
|---------|--------|-------|
| `smolvla_baseline` | `configs/training/smolvla_baseline.yaml` | No phase info; delegates to existing `main.py train` |
| `smolvla_fsm` | `configs/training/smolvla_fsm.yaml` | Forked SmolVLA in `src/smolvla_fork/` + runtime FSM |
| `pi0_subtask` | `configs/training/pi0_subtask.yaml` | π0 with `subtask` column; delegates to `main.py train --model-type pi0` |

Training entry point: `scripts/train_phase.py --config <config> --dataset-name <name>`

---

## 6. FSM Transition Rules

Implemented in `src/fsm/runtime_fsm.py`. Legal transitions:

```
free_motion      → fine_align           (xy dist < threshold AND z < hover_z)
fine_align       → contact_establish    (Fz > contact_force OR gripper close cmd)
fine_align       → free_motion          (pose error grew > threshold for >300 ms)  ← re-approach
contact_establish → constrained_motion  (gripper stable AND Fz in mate band)
constrained_motion → verify_release     (target z reached AND Fz in seated band)
verify_release   → (terminal)           (gripper open AND retract initiated)
```

All transitions must also satisfy `is_legal_transition()` from `src/common/phases.py`.
Self-loops are always legal. No backward jumps except `fine_align → free_motion`.

Thresholds live in `configs/fsm/pick_place.yaml` and `configs/fsm/msd_plug.yaml`.

---

## 7. Annotation Quality Rules

- Qwen3-VL outputs must pass `src/annotation/validator.py` before writeback.
- Validation failures: write episode ID to `outputs/failed_episodes.txt`; do not write to annotated dataset.
- Annotated dataset lives at `lerobot_datasets/<name>_annotated/` — never overwrite the source.
- Gold annotations live in `data/gold/` and are the few-shot source of truth.

---

## 8. Testing

Run tests with:
```bash
uv --project ../lerobot run python -m pytest tests/ -v
```

Skip GPU tests on CPU-only machines:
```bash
uv --project ../lerobot run python -m pytest tests/ -v -m "not gpu"
```

Tests that require the real Franka robot are marked `@pytest.mark.robot` and always skipped in CI.

---

## 9. Code Style

- Line length: 100 characters.
- No type annotations on code you didn't write.
- No docstrings on trivial methods.
- Prefer explicit over implicit; avoid clever one-liners in safety-critical paths.
