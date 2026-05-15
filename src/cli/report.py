"""Comparison report generator for all three policy variants.

Consumes results.json files from eval.py runs and produces:
  1. Markdown table with 95% Wilson confidence intervals
  2. Stacked bar chart of failure attribution per phase per variant (PNG)
  3. Phase-corruption recovery plot (PNG)
  4. Full markdown report

Usage:
    python src/scripts/report.py \\
        outputs/eval/run_baseline/results.json \\
        outputs/eval/run_fsm/results.json \\
        outputs/eval/run_pi0/results.json
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Generate comparison report from eval results.")


def _wilson_ci(successes: int, n: int, z: float = 1.96):
    """95% Wilson confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    center = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / (1 + z**2 / n)
    return max(0.0, center - margin), min(1.0, center + margin)


def _load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def _markdown_success_table(all_results: list[dict]) -> str:
    lines = [
        "## Success Rate (95% Wilson CI)\n",
        "| Variant | Task | Success Rate | 95% CI | N |",
        "|---------|------|-------------|--------|---|",
    ]
    for r in all_results:
        n = len(r["episodes"])
        s = r["summary"]["n_success"]
        lo, hi = _wilson_ci(s, n)
        lines.append(
            f"| {r['variant']} | {r['task']} | {s/n:.1%} | [{lo:.1%}, {hi:.1%}] | {n} |"
        )
    return "\n".join(lines)


def _phase_failure_chart(all_results: list[dict], output_path: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        console.print("[yellow]matplotlib not installed — skipping failure chart[/yellow]")
        return None

    from src.common.phases import Phase, PHASE_NAMES
    phases = [PHASE_NAMES[p] for p in Phase]
    variants = [r["variant"] for r in all_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(variants))
    bottom = [0.0] * len(variants)
    colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b"]

    for i, phase in enumerate(phases):
        counts = []
        for r in all_results:
            attr = r["summary"].get("failure_attribution", {})
            counts.append(attr.get(phase, 0))
        ax.bar(x, counts, bottom=bottom, label=phase, color=colors[i % len(colors)])
        bottom = [b + c for b, c in zip(bottom, counts)]

    ax.set_xticks(list(x))
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylabel("Failure count")
    ax.set_title("Failure Attribution by Phase")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _corruption_chart(all_results: list[dict], output_path: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        console.print("[yellow]matplotlib not installed — skipping corruption chart[/yellow]")
        return None

    variants_with_corruption = [
        r for r in all_results if r.get("corruption_episodes")
    ]
    if not variants_with_corruption:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    for r in variants_with_corruption:
        rate = r["summary"].get("corruption_recovery_rate", 0)
        ax.bar([r["variant"]], [rate], label=r["variant"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recovery rate")
    ax.set_title("Phase-Corruption Recovery Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _full_markdown_report(all_results: list[dict], charts: dict, title: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        f"# {title}\n",
        f"Generated: {ts}\n",
        "---\n",
        _markdown_success_table(all_results),
        "\n---\n",
    ]

    if charts.get("failure_chart"):
        parts.append(f"\n## Failure Attribution\n\n![failure chart]({charts['failure_chart'].name})\n")

    if charts.get("corruption_chart"):
        parts.append(f"\n## Phase-Corruption Recovery\n\n![corruption chart]({charts['corruption_chart'].name})\n")

    parts.append("\n---\n\n## Factual Summary\n")
    for r in all_results:
        n = len(r["episodes"])
        s = r["summary"]["n_success"]
        parts.append(f"- **{r['variant']}** ({r['task']}): {s}/{n} episodes succeeded ({s/n:.1%}).")
        attr = r["summary"].get("failure_attribution", {})
        if attr:
            top_phase = max(attr, key=attr.get)
            parts.append(f"  Most failures occurred in phase `{top_phase}` ({attr[top_phase]} episodes).")
        corr = r["summary"].get("corruption_recovery_rate")
        if corr is not None:
            parts.append(f"  Phase-corruption recovery rate: {corr:.1%}.")
        parts.append("")

    return "\n".join(parts)


@app.command()
def main(
    result_files: list[Path] = typer.Argument(..., help="Path(s) to results.json files"),
    output_dir: Path = typer.Option(Path("outputs/reports"), help="Where to write the report"),
    title: str = typer.Option("Phase-Conditioned VLA Comparison", help="Report title"),
):
    if not result_files:
        console.print("[red]No result files provided.[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    for f in result_files:
        if not f.exists():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)
        all_results.append(_load_results(f))

    console.print(f"Loaded {len(all_results)} result file(s).")

    charts = {}

    # Failure attribution chart
    fail_chart_path = output_dir / f"{ts}_failure_chart.png"
    p = _phase_failure_chart(all_results, fail_chart_path)
    if p:
        charts["failure_chart"] = p
        console.print(f"  Failure chart: {p}")

    # Corruption chart
    corr_chart_path = output_dir / f"{ts}_corruption_chart.png"
    p = _corruption_chart(all_results, corr_chart_path)
    if p:
        charts["corruption_chart"] = p
        console.print(f"  Corruption chart: {p}")

    # Markdown report
    report_path = output_dir / f"{ts}_report.md"
    report_md = _full_markdown_report(all_results, charts, title)
    report_path.write_text(report_md)

    console.print(f"\n[green]Report written to {report_path}[/green]")


if __name__ == "__main__":
    app()
