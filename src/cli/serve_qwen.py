"""CLI entrypoint: batch-annotate a directory of LeRobot episode videos.

Usage:
    python src/scripts/serve_qwen.py \\
        --dataset-root lerobot_datasets/my_dataset \\
        --task pick_place \\
        --output-dir outputs/annotations
"""
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Run Qwen3-VL inference over a LeRobot episode video directory.")


@app.command()
def main(
    dataset_root: Path = typer.Option(..., help="Path to lerobot_datasets/<name>/"),
    task: str = typer.Option(..., help="Task name: pick_place or msd_plug"),
    output_dir: Path = typer.Option(Path("outputs/annotations"), help="Where to write outputs"),
    prompt_text: str = typer.Option(
        "Describe the robot phase in JSON format.", help="Prompt to send to the model"
    ),
    max_tokens: int = typer.Option(2048, help="Max tokens per episode"),
):
    from src.annotation.serve_qwen import QwenAnnotator

    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        console.print(f"[red]Videos directory not found: {videos_dir}[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    annotator = QwenAnnotator()

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        console.print(f"[yellow]No .mp4 files found in {videos_dir}[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found {len(video_files)} video(s). Loading model...")

    for vf in video_files:
        out_file = output_dir / (vf.stem + "_qwen.txt")
        if out_file.exists():
            console.print(f"[dim]Skipping {vf.name} (output already exists)[/dim]")
            continue
        console.print(f"Annotating {vf.name}...")
        result = annotator.annotate_episode(str(vf), prompt_text, max_tokens=max_tokens)
        out_file.write_text(result)
        console.print(f"  → {out_file}")

    console.print("[green]Done.[/green]")


if __name__ == "__main__":
    app()
