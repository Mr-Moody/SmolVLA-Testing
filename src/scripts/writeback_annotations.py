"""Write phase + subtask columns back into a LeRobot dataset.

Creates a copy of the source dataset at --output-dataset with two new columns:
  - phase   (int64 per frame)
  - subtask (str per frame, e.g. "pick_place.contact_establish")

Usage:
    python src/scripts/writeback_annotations.py \\
        --annotations-dir outputs/annotations \\
        --source-dataset lerobot_datasets/my_dataset \\
        --output-dataset lerobot_datasets/my_dataset_annotated
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.common.phases import Phase, PHASE_NAMES

console = Console()
app = typer.Typer(help="Add phase + subtask columns to a LeRobot dataset copy.")


@app.command()
def main(
    annotations_dir: Path = typer.Option(
        Path("outputs/annotations"), help="Directory of episode_XXXXXX.parquet annotation files"
    ),
    source_dataset: Path = typer.Option(..., help="Source lerobot_datasets/<name>/"),
    output_dataset: Path = typer.Option(..., help="Output path for annotated dataset copy"),
):
    if not annotations_dir.exists():
        console.print(f"[red]Annotations directory not found: {annotations_dir}[/red]")
        raise typer.Exit(1)
    if not source_dataset.exists():
        console.print(f"[red]Source dataset not found: {source_dataset}[/red]")
        raise typer.Exit(1)

    annotation_files = sorted(annotations_dir.glob("episode_*.parquet"))
    if not annotation_files:
        console.print(f"[red]No annotation parquet files found in {annotations_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(annotation_files)} annotation file(s).")

    # Copy source dataset to output path
    if output_dataset.exists():
        console.print(f"[yellow]Output dataset already exists at {output_dataset}. Overwriting.[/yellow]")
        shutil.rmtree(output_dataset)
    console.print(f"Copying {source_dataset} → {output_dataset} ...")
    shutil.copytree(source_dataset, output_dataset)
    console.print("Copy done.")

    chunks_dir = output_dataset / "chunks"
    valid_count = 0
    phase_freq: dict = {}

    for ann_file in annotation_files:
        ann_df = pd.read_parquet(ann_file)
        if "phase" not in ann_df.columns or "subtask" not in ann_df.columns:
            console.print(f"[yellow]Skipping {ann_file.name}: missing phase/subtask columns[/yellow]")
            continue

        # Find corresponding chunk file
        episode_stem = ann_file.stem  # e.g. "episode_000000"
        chunk_candidates = []
        if chunks_dir.exists():
            chunk_candidates = list(chunks_dir.glob(f"{episode_stem}*.parquet"))
        if not chunk_candidates:
            # Try to write standalone annotation file alongside video
            (output_dataset / "annotations").mkdir(parents=True, exist_ok=True)
            out_path = output_dataset / "annotations" / ann_file.name
            ann_df.to_parquet(out_path, index=False)
            valid_count += 1
        else:
            for chunk_file in chunk_candidates:
                chunk_df = pd.read_parquet(chunk_file)
                n_chunk = len(chunk_df)
                n_ann = len(ann_df)

                # Align lengths (annotation may have different frame count)
                if n_ann >= n_chunk:
                    phase_col = ann_df["phase"].values[:n_chunk]
                    subtask_col = ann_df["subtask"].values[:n_chunk]
                else:
                    import numpy as np
                    # Pad with last value
                    phase_col = list(ann_df["phase"].values) + [ann_df["phase"].values[-1]] * (n_chunk - n_ann)
                    subtask_col = list(ann_df["subtask"].values) + [ann_df["subtask"].values[-1]] * (n_chunk - n_ann)

                chunk_df["phase"] = phase_col
                chunk_df["subtask"] = subtask_col
                chunk_df.to_parquet(chunk_file, index=False)
                valid_count += 1

        # Accumulate phase frequencies
        for phase_id, count in ann_df["phase"].value_counts().items():
            key = PHASE_NAMES.get(Phase(int(phase_id)), str(phase_id))
            phase_freq[key] = phase_freq.get(key, 0) + int(count)

    # Update meta/info.json to note new columns
    info_file = output_dataset / "meta" / "info.json"
    if info_file.exists():
        info = json.loads(info_file.read_text())
        info.setdefault("extra_columns", [])
        if "phase" not in info["extra_columns"]:
            info["extra_columns"].append("phase")
        if "subtask" not in info["extra_columns"]:
            info["extra_columns"].append("subtask")
        info_file.write_text(json.dumps(info, indent=2))

    # Validation: spot-check the output
    console.print(f"\n[green]{valid_count} episode(s) written.[/green]")

    # Print phase frequency table
    table = Table(title="Phase Frequency Distribution")
    table.add_column("Phase")
    table.add_column("Frames", justify="right")
    table.add_column("Fraction", justify="right")
    total_frames = sum(phase_freq.values())
    for phase_name in sorted(phase_freq):
        count = phase_freq[phase_name]
        frac = count / total_frames if total_frames > 0 else 0.0
        table.add_row(phase_name, str(count), f"{frac:.3f}")
    console.print(table)
    console.print(f"\n[bold green]Annotated dataset written to {output_dataset}[/bold green]")


if __name__ == "__main__":
    app()
