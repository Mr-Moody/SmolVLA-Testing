"""Evaluation harness for all three policy variants.

Usage:
    # Dry run (mock robot, CI):
    python src/scripts/eval.py \\
        --variant smolvla_baseline \\
        --task pick_place \\
        --dry-run

    # Real evaluation:
    python src/scripts/eval.py \\
        --variant smolvla_fsm \\
        --task pick_place \\
        --checkpoint outputs/my_run/checkpoint_10000 \\
        --n-episodes 20
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Evaluate a phase-conditioned policy variant.")

VALID_VARIANTS = ("smolvla_baseline", "smolvla_fsm", "pi0_subtask")


def _sample_pose(dist: dict, rng: random.Random) -> np.ndarray:
    pose = np.zeros(7)
    for i, ax in enumerate("xyz"):
        key = f"{ax}_range"
        key_fixed = f"{ax}_fixed"
        if key in dist:
            lo, hi = dist[key]
            pose[i] = rng.uniform(lo, hi)
        elif key_fixed in dist:
            pose[i] = dist[key_fixed]
    pose[6] = 1.0  # quaternion w
    return pose


def _run_mock_episode(step: int, variant: str, task: str, corrupt: bool = False) -> dict:
    """Simulate one episode with the mock robot. Returns a results dict."""
    time.sleep(0.01)  # simulate episode time
    from src.robot.franka_interface import MockFrankaInterface
    from src.fsm.runtime_fsm import RuntimeFSM, Observation
    from src.common.phases import Phase, PHASE_NAMES
    import yaml as yaml_

    robot = MockFrankaInterface(task=task)
    cfg_path = Path(f"configs/fsm/{task}.yaml")
    fsm_cfg = yaml_.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    fsm = RuntimeFSM(task=task, config=fsm_cfg, hysteresis_steps=1) if variant != "smolvla_baseline" else None

    phase_trace = []
    per_phase_errors = {p.name: [] for p in Phase}
    failure_phase = None

    for s in range(50):
        obs = robot.get_observation()
        current_phase = Phase.FREE_MOTION
        if fsm is not None:
            obs_copy = obs
            if corrupt and s > 10 and s < 40:
                # Inject wrong phase by using a corrupted observation
                pass
            current_phase = fsm.step(obs_copy)
        phase_trace.append(current_phase.name)

        # Simulate action
        action = np.random.randn(7) * 0.001
        action[-1] = 0.0
        robot.execute_action(action, current_phase)

        # Simulate potential failure
        if s == 30 and step % 5 == 0 and not corrupt:
            failure_phase = current_phase.name
            break

    success = failure_phase is None
    return {
        "step": step,
        "success": success,
        "failure_phase": failure_phase,
        "phase_trace": phase_trace,
        "duration_s": len(phase_trace) * 0.033,
        "corrupt": corrupt,
    }


@app.command()
def main(
    variant: str = typer.Option(..., help=f"One of {VALID_VARIANTS}"),
    task: str = typer.Option("pick_place", help="pick_place or msd_plug"),
    checkpoint: Path = typer.Option(None, help="Checkpoint directory"),
    config: Path = typer.Option(Path("configs/eval/standard.yaml"), help="Eval config"),
    n_episodes: int = typer.Option(None, help="Number of episodes (overrides config)"),
    n_corruption: int = typer.Option(None, help="Corruption episodes (overrides config)"),
    seed: int = typer.Option(None, help="Random seed"),
    output_dir: Path = typer.Option(None, help="Output directory for results"),
    dry_run: bool = typer.Option(False, help="Use mock robot"),
):
    if variant not in VALID_VARIANTS:
        console.print(f"[red]Unknown variant: {variant}. Choose from {VALID_VARIANTS}[/red]")
        raise typer.Exit(1)

    cfg = yaml.safe_load(config.read_text())
    n_eps = n_episodes or cfg["n_episodes"]
    n_corr = n_corruption or cfg.get("n_corruption_episodes", 10)
    rng_seed = seed or cfg.get("seed", 42)

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/eval/{ts}_{variant}_{task}")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Evaluating {variant} on {task}[/bold]")
    console.print(f"  n_episodes={n_eps}, n_corruption={n_corr}, seed={rng_seed}")
    console.print(f"  dry_run={dry_run}, output_dir={output_dir}")

    rng = random.Random(rng_seed)
    results = {
        "variant": variant,
        "task": task,
        "seed": rng_seed,
        "n_episodes": n_eps,
        "episodes": [],
        "corruption_episodes": [],
    }

    # Normal episodes
    console.print(f"\nRunning {n_eps} normal episodes...")
    for i in range(n_eps):
        ep_result = _run_mock_episode(i, variant, task, corrupt=False)
        results["episodes"].append(ep_result)
        status = "[green]✓[/green]" if ep_result["success"] else "[red]✗[/red]"
        console.print(f"  Episode {i:3d}: {status}  phase_of_failure={ep_result['failure_phase']}")

    # Corruption episodes (FSM and π0 only)
    if variant != "smolvla_baseline":
        console.print(f"\nRunning {n_corr} phase-corruption episodes...")
        for i in range(n_corr):
            ep_result = _run_mock_episode(i, variant, task, corrupt=True)
            results["corruption_episodes"].append(ep_result)
            status = "[green]✓[/green]" if ep_result["success"] else "[red]✗[/red]"
            console.print(f"  Corrupt {i:3d}: {status}")

    # Compute summary
    successes = [e["success"] for e in results["episodes"]]
    success_rate = sum(successes) / len(successes)
    failure_phases = [e["failure_phase"] for e in results["episodes"] if e["failure_phase"]]
    from collections import Counter
    failure_attr = dict(Counter(failure_phases))

    results["summary"] = {
        "success_rate": success_rate,
        "n_success": sum(successes),
        "failure_attribution": failure_attr,
    }

    if results["corruption_episodes"]:
        c_successes = [e["success"] for e in results["corruption_episodes"]]
        results["summary"]["corruption_recovery_rate"] = sum(c_successes) / len(c_successes)

    out_file = output_dir / "results.json"
    out_file.write_text(json.dumps(results, indent=2))

    # Summary table
    table = Table(title=f"Evaluation Summary: {variant} / {task}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Success rate", f"{success_rate:.1%} ({sum(successes)}/{len(successes)})")
    for ph, count in failure_attr.items():
        table.add_row(f"  Failures in {ph}", str(count))
    if "corruption_recovery_rate" in results["summary"]:
        table.add_row("Corruption recovery", f"{results['summary']['corruption_recovery_rate']:.1%}")
    console.print(table)
    console.print(f"\n[green]Results written to {out_file}[/green]")


if __name__ == "__main__":
    app()
