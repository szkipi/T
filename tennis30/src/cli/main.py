"""Tennis30 command-line interface."""

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="tennis30",
    help="Tennis30 - Multi-Pass Precision Tennis Tracking System",
    add_completion=False,
)

console = Console()


@app.command()
def track(
    video: Path = typer.Argument(
        ...,
        help="Input video file path",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        "./output",
        "--output",
        "-o",
        help="Output directory for results",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Custom configuration file (YAML)",
        exists=True,
    ),
    passes: int = typer.Option(
        4,
        "--passes",
        "-p",
        help="Number of refinement passes (1-4)",
        min=1,
        max=4,
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        "-d",
        help="Computation device (cuda or cpu)",
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize",
        "-v",
        help="Generate visualization video",
    ),
    export_format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Export format: csv, json, unity, unreal, or all",
    ),
):
    """Track tennis ball, players, and poses in a video.

    This runs the full Tennis30 precision tracking pipeline with
    4-pass refinement for maximum accuracy.

    Example:
        tennis30 track match.mp4 --passes 4 --visualize
    """
    rprint(f"\n[bold cyan]Tennis30 Precision Tracking[/bold cyan]")
    rprint(f"Input video: [green]{video}[/green]")
    rprint(f"Output directory: [green]{output}[/green]")
    rprint(f"Refinement passes: [yellow]{passes}[/yellow]")
    rprint(f"Device: [yellow]{device}[/yellow]\n")

    # TODO: Implement actual tracking pipeline
    rprint("[yellow]⚠ Pipeline implementation pending (Phase 2 migration)[/yellow]")
    rprint("\nPlanned steps:")
    rprint("  1. Load configuration")
    rprint("  2. Initialize models (ensemble)")
    rprint("  3. Detect court and compute homography")
    rprint("  4. Run 4-pass ball tracking")
    rprint("  5. Track players and estimate poses")
    rprint("  6. Apply physics validation")
    rprint("  7. Detect game state (rallies, serves)")
    rprint("  8. Export results\n")


@app.command()
def models(
    list_models: bool = typer.Option(
        True,
        "--list",
        "-l",
        help="List all available models",
    ),
    download: Optional[str] = typer.Option(
        None,
        "--download",
        "-d",
        help="Download specific model by name",
    ),
    download_all: bool = typer.Option(
        False,
        "--download-all",
        help="Download all available models",
    ),
):
    """Manage model weights (list, download, verify).

    Example:
        tennis30 models --list
        tennis30 models --download tracknet
        tennis30 models --download-all
    """
    from tennis30.src.models.registry import MODEL_METADATA

    if download or download_all:
        rprint("[yellow]⚠ Model download not yet implemented[/yellow]")
        rprint("Please run: python scripts/download_models.py --all\n")
        return

    # List models
    table = Table(title="Tennis30 Available Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Description", style="white")

    for name, meta in MODEL_METADATA.items():
        table.add_row(
            name,
            meta["type"],
            f"{meta['size_mb']} MB",
            meta["description"][:50] + "..." if len(meta["description"]) > 50 else meta["description"],
        )

    console.print(table)
    rprint(f"\n[bold]Total:[/bold] {len(MODEL_METADATA)} models")
    rprint("\n[dim]Use 'python scripts/download_models.py --all' to download all models[/dim]\n")


@app.command()
def config(
    show: bool = typer.Option(
        True,
        "--show",
        "-s",
        help="Show current configuration",
    ),
    config_name: str = typer.Option(
        "default",
        "--name",
        "-n",
        help="Configuration file name (without .yaml)",
    ),
    key: Optional[str] = typer.Option(
        None,
        "--get",
        "-g",
        help="Get specific config value (dot-separated path)",
    ),
):
    """View and manage configuration.

    Example:
        tennis30 config --show
        tennis30 config --get ball_tracking.ensemble.tracknet.weight
    """
    from tennis30.src.utils.config import load_config

    try:
        cfg = load_config(config_name)
    except FileNotFoundError as e:
        rprint(f"[red]Error: {e}[/red]")
        return

    if key:
        # Get specific key
        from tennis30.src.utils.config import ConfigLoader

        loader = ConfigLoader()
        loader.load(config_name)
        value = loader.get(key)

        if value is not None:
            rprint(f"[cyan]{key}[/cyan] = [green]{value}[/green]")
        else:
            rprint(f"[yellow]Key not found: {key}[/yellow]")
        return

    # Show full config
    import yaml

    rprint(f"\n[bold cyan]Configuration: {config_name}.yaml[/bold cyan]\n")
    rprint(yaml.dump(cfg, default_flow_style=False, indent=2))


@app.command()
def version():
    """Show Tennis30 version information."""
    from tennis30.src.utils.config import load_config

    cfg = load_config("default")
    version = cfg.get("project", {}).get("version", "unknown")

    rprint(f"\n[bold cyan]Tennis30[/bold cyan] version [green]{version}[/green]")
    rprint("[dim]Multi-Pass Precision Tennis Tracking System[/dim]\n")


@app.command()
def benchmark(
    video: Path = typer.Argument(
        ...,
        help="Test video file",
        exists=True,
    ),
    ground_truth: Path = typer.Argument(
        ...,
        help="Ground truth CSV file",
        exists=True,
    ),
    models: str = typer.Option(
        "all",
        "--models",
        "-m",
        help="Models to benchmark (comma-separated or 'all')",
    ),
    output: Path = typer.Option(
        "./benchmark_results",
        "--output",
        "-o",
        help="Output directory for benchmark results",
    ),
):
    """Benchmark tracking models against ground truth.

    Example:
        tennis30 benchmark test.mp4 ground_truth.csv --models tracknet,yolo_v1
    """
    rprint(f"\n[bold cyan]Tennis30 Benchmark[/bold cyan]")
    rprint(f"Video: [green]{video}[/green]")
    rprint(f"Ground truth: [green]{ground_truth}[/green]")
    rprint(f"Models: [yellow]{models}[/yellow]\n")

    rprint("[yellow]⚠ Benchmark implementation pending[/yellow]\n")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
