"""FSM threshold tuning tool.

Replays a recorded episode parquet through the FSM and compares its phase trace
against Qwen ground-truth labels side-by-side.

Usage:
    python src/scripts/tune_fsm.py \\
        --episode-parquet outputs/annotations/episode_000000.parquet \\
        --task pick_place \\
        --annotation-json data/gold/0.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table

from src.fsm.runtime_fsm import RuntimeFSM, Observation
from src.common.phases import Phase, PHASE_NAMES

app = typer.Typer(help="Compare FSM phase trace against Qwen ground-truth labels.")
console = Console()

_CONFIG_DIR = Path("configs/fsm")


def _load_config(task: str) -> dict:
    cfg_file = _CONFIG_DIR / f"{task}.yaml"
    if cfg_file.exists():
        return yaml.safe_load(cfg_file.read_text())
    default_file = _CONFIG_DIR / "default.yaml"
    if default_file.exists():
        return yaml.safe_load(default_file.read_text())
    return {}


def _qwen_phase_at(annotation: dict, t: float) -> int:
    for seg in annotation.get("segments", []):
        if seg["start_t"] <= t < seg["end_t"]:
            return seg["phase_id"]
    return -1


@app.command()
def main(
    episode_parquet: Path = typer.Option(..., help="Path to episode parquet (annotations/ or chunks/)"),
    task: str = typer.Option(..., help="pick_place or msd_plug"),
    annotation_json: Path = typer.Option(None, help="Qwen gold annotation JSON for comparison"),
    config_override: Path = typer.Option(None, help="Override YAML config path"),
    max_rows: int = typer.Option(200, help="Max rows to display"),
):
    import pandas as pd

    df = pd.read_parquet(episode_parquet)
    cfg = _load_config(task)
    if config_override and Path(config_override).exists():
        override = yaml.safe_load(Path(config_override).read_text())
        cfg.update(override)

    console.print(f"Loaded episode: {len(df)} rows")
    console.print(f"Config: {json.dumps(cfg, indent=2)}")

    fsm = RuntimeFSM(task=task, config=cfg, hysteresis_steps=cfg.get("hysteresis_steps", 3))

    qwen_annotation = None
    if annotation_json and Path(annotation_json).exists():
        qwen_annotation = json.loads(Path(annotation_json).read_text())

    # Build observation from dataframe columns
    fps = cfg.get("control_rate_hz", 30.0)

    table = Table(title="FSM Phase Trace vs Qwen Ground Truth")
    table.add_column("t (s)", justify="right")
    table.add_column("FSM Phase")
    table.add_column("Qwen Phase")
    table.add_column("Match?")

    transitions = []
    prev_phase = Phase.FREE_MOTION
    rows_shown = 0

    for i, row in df.iterrows():
        t = float(row.get("timestamp", i / fps))

        # Build observation
        tcp_pose = np.zeros(7)
        tcp_vel = np.zeros(6)
        wrench = np.zeros(6)
        gripper = 0.0

        if "observation.state" in df.columns:
            state = np.array(row["observation.state"])
            tcp_pose[:min(7, len(state))] = state[:7]
            gripper = float(state[-1]) if len(state) > 7 else 0.0
        if "wrench" in df.columns:
            w = np.array(row["wrench"])
            wrench[:min(6, len(w))] = w[:6]

        obs = Observation(
            tcp_pose=tcp_pose,
            tcp_velocity=tcp_vel,
            gripper_state=gripper,
            wrench=wrench,
        )
        fsm_phase = fsm.step(obs)

        if fsm_phase != prev_phase:
            transitions.append((t, prev_phase, fsm_phase))
            prev_phase = fsm_phase

        if rows_shown < max_rows:
            qwen_phase_id = _qwen_phase_at(qwen_annotation, t) if qwen_annotation else -1
            qwen_name = PHASE_NAMES.get(Phase(qwen_phase_id), "N/A") if qwen_phase_id >= 0 else "N/A"
            match = "✓" if qwen_phase_id == int(fsm_phase) else "✗" if qwen_phase_id >= 0 else "–"
            if i % 10 == 0:  # sample every 10 rows for readability
                table.add_row(
                    f"{t:.2f}",
                    PHASE_NAMES[fsm_phase],
                    qwen_name,
                    match,
                )
                rows_shown += 1

    console.print(table)

    # Print transitions
    console.print("\n[bold]FSM Phase Transitions:[/bold]")
    for t, frm, to in transitions:
        console.print(f"  t={t:.3f}s  {PHASE_NAMES[frm]} → {PHASE_NAMES[to]}")


if __name__ == "__main__":
    app()
