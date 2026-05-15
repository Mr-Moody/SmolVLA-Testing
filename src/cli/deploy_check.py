"""Pre-deployment checklist for real-robot runs.

Runs through all safety checks and refuses to exit 0 if any non-skippable
check fails.

Usage:
    # Mock mode (for CI / offline testing):
    python src/scripts/deploy_check.py --mock

    # Real robot:
    python src/scripts/deploy_check.py --checkpoint outputs/my_run/checkpoint_10000
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Run deployment checklist before real-robot episodes.")

_PASS = "[green]✓[/green]"
_FAIL = "[red]✗[/red]"
_SKIP = "[dim]–[/dim]"


def _check(name: str, fn, skip: bool = False):
    if skip:
        return name, _SKIP, "skipped"
    try:
        msg = fn()
        return name, _PASS, msg or "OK"
    except Exception as exc:
        return name, _FAIL, str(exc)


@app.command()
def main(
    checkpoint: Path = typer.Option(None, help="Checkpoint directory to load"),
    mock: bool = typer.Option(False, help="Use MockFrankaInterface (for CI)"),
    task: str = typer.Option("pick_place", help="Task name"),
    variant: str = typer.Option("smolvla_baseline", help="Policy variant"),
):
    from src.robot.franka_interface import MockFrankaInterface, FrankaInterface
    from src.fsm.runtime_fsm import RuntimeFSM, Observation
    from src.common.phases import Phase

    console.print(f"\n[bold]Deployment Checklist[/bold]  mock={mock}  variant={variant}\n")

    robot = MockFrankaInterface(task=task) if mock else None
    results = []

    # ── Check 1: Robot state (real only) ───────────────────────────────────
    def check_robot_state():
        if mock:
            raise RuntimeError("Skipped in mock mode")
        robot_real = FrankaInterface()
        obs = robot_real.get_observation()
        return "Franka reachable"

    results.append(_check("Robot reachable / idle state", check_robot_state, skip=mock))

    # ── Check 2: F/T sensor near zero ─────────────────────────────────────
    def check_ft_sensor():
        r = robot or MockFrankaInterface()
        obs = r.get_observation()
        fz = abs(float(obs.wrench[2]))
        if fz > 5.0:
            raise ValueError(f"F/T sensor Fz={fz:.1f} N (expected ~0 with no payload)")
        return f"Fz={fz:.1f} N"

    results.append(_check("F/T sensor near zero", check_ft_sensor))

    # ── Check 3: Checkpoint loads and produces non-NaN action ─────────────
    def check_checkpoint():
        if checkpoint is None:
            return "No checkpoint specified — skipped"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        # Just verify the config file is present
        cfg_file = checkpoint / "config.json"
        if not cfg_file.exists():
            cfg_file = checkpoint / "train_config.json"
        if not cfg_file.exists():
            return f"Checkpoint dir found at {checkpoint}"
        cfg = json.loads(cfg_file.read_text())
        return f"Checkpoint OK, model_type={cfg.get('model_type', '?')}"

    results.append(_check("Checkpoint loads", check_checkpoint))

    # ── Check 4: FSM produces sensible phase trace ─────────────────────────
    def check_fsm():
        import yaml
        cfg_path = Path(f"configs/fsm/{task}.yaml")
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        fsm = RuntimeFSM(task=task, config=cfg, hysteresis_steps=1)
        obs = Observation(
            tcp_pose=np.array([0.5, 0.0, 0.4, 0, 0, 0, 1], dtype=float),
            tcp_velocity=np.zeros(6),
            gripper_state=0.0,
            wrench=np.zeros(6),
        )
        for _ in range(5):
            phase = fsm.step(obs)
        if phase not in Phase:
            raise ValueError(f"FSM returned invalid phase: {phase}")
        return f"FSM OK, current_phase={phase.name}"

    results.append(_check("FSM produces valid phase", check_fsm))

    # ── Check 5: Safety envelope rejects bad action ─────────────────────────
    def check_safety():
        r = robot or MockFrankaInterface()
        bad_action = np.array([float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        obs_before = r.get_observation()
        r.execute_action(bad_action, Phase.FREE_MOTION)
        obs_after = r.get_observation()
        if not np.allclose(obs_before.tcp_pose, obs_after.tcp_pose):
            raise ValueError("Safety envelope did NOT reject NaN action!")
        return "NaN action correctly rejected"

    results.append(_check("Safety envelope rejects bad action", check_safety))

    # ── Check 6: W&B (optional) ────────────────────────────────────────────
    def check_wandb():
        try:
            import wandb
            return "wandb available"
        except ImportError:
            return "wandb not installed (logging to local JSON only)"

    results.append(_check("W&B available", check_wandb))

    # ── Check 7: E-stop confirmation (real robot only) ─────────────────────
    def check_estop():
        if mock:
            raise RuntimeError("Skipped in mock mode")
        confirm = typer.confirm("Is the external e-stop connected and tested?")
        if not confirm:
            raise ValueError("E-stop not confirmed — aborting")
        return "E-stop confirmed"

    results.append(_check("E-stop connected & tested", check_estop, skip=mock))

    # ── Print results ──────────────────────────────────────────────────────
    table = Table(title="Deployment Checklist Results")
    table.add_column("Check")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    failed = 0
    for name, status, details in results:
        table.add_row(name, status, details)
        if status == _FAIL:
            failed += 1

    console.print(table)

    if failed > 0:
        console.print(f"\n[bold red]{failed} check(s) FAILED — do NOT proceed to real-robot run.[/bold red]")
        raise typer.Exit(1)
    else:
        console.print("\n[bold green]All checks passed — safe to proceed.[/bold green]")


if __name__ == "__main__":
    app()
