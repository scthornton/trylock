#!/usr/bin/env python3
"""
AEGIS Data Generation CLI

Generate attack trajectories for AEGIS training data.

Usage:
    python generate_data.py generate --count 100 --output data/output.jsonl
    python generate_data.py validate --input data/trajectories.jsonl
    python generate_data.py mutate --input data/base.jsonl --output data/mutated.jsonl
    python generate_data.py benign --count 50 --output data/benign.jsonl
"""

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.pipeline import AEGISPipeline, MockPipeline, PipelineConfig, generate_benign_hard_negatives
from generation.mutation_engine import MutationEngine, SimpleMutationEngine
from data.schema.validator import TrajectoryValidator

app = typer.Typer(
    name="aegis",
    help="AEGIS Dataset Generation Tools",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    output: Path = typer.Option(
        "data/tier1_open/attacks/generated.jsonl",
        "--output", "-o",
        help="Output file path"
    ),
    count: int = typer.Option(
        100,
        "--count", "-n",
        help="Number of trajectories to generate"
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock pipeline (no API calls)"
    ),
    mutations: int = typer.Option(
        3,
        "--mutations", "-m",
        help="Number of mutations per trajectory"
    ),
    benign_ratio: float = typer.Option(
        0.5,
        "--benign-ratio",
        help="Ratio of benign hard negatives (0-1)"
    ),
    red_model: str = typer.Option(
        "claude-sonnet-4-20250514",
        "--red-model",
        help="Model for Red Bot"
    ),
    judge_model: str = typer.Option(
        "claude-sonnet-4-20250514",
        "--judge-model",
        help="Model for Judge Bot"
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate generated trajectories"
    ),
):
    """Generate attack trajectories."""
    console.print(f"\n[bold blue]AEGIS Data Generation[/bold blue]")
    console.print(f"Output: {output}")
    console.print(f"Count: {count}")
    console.print(f"Mutations per trajectory: {mutations}")
    console.print(f"Using mock pipeline: {mock}\n")

    # Configure pipeline
    config = PipelineConfig(
        red_bot_model=red_model,
        judge_model=judge_model,
        mutations_per_trajectory=mutations,
        benign_ratio=benign_ratio,
        validate_output=validate,
    )

    # Initialize pipeline
    if mock:
        pipeline = MockPipeline(config)
        console.print("[yellow]Using mock pipeline - no API calls[/yellow]\n")
    else:
        pipeline = AEGISPipeline(config)

    # Generate trajectories
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating trajectories...", total=None)

        trajectories = pipeline.run(
            count=count,
            include_mutations=mutations > 0,
            progress=False,
        )

        progress.update(task, completed=True)

    # Get stats
    stats = pipeline.get_stats()

    # Display results
    table = Table(title="Generation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Attempted", str(stats["total_attempted"]))
    table.add_row("Total Generated", str(stats["total_generated"]))
    table.add_row("Total Validated", str(stats["total_validated"]))
    table.add_row("Total Mutations", str(stats["total_mutations"]))
    table.add_row("Final Count", str(len(trajectories)))
    table.add_row("Success Rate", f"{stats['success_rate']:.1%}")

    console.print(table)

    # Show distribution
    if stats["by_family"]:
        console.print("\n[bold]By Attack Family:[/bold]")
        for family, cnt in sorted(stats["by_family"].items()):
            console.print(f"  {family}: {cnt}")

    # Save output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + "\n")

    console.print(f"\n[green]Saved {len(trajectories)} trajectories to {output}[/green]")


@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="Input JSONL file to validate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all errors"),
):
    """Validate trajectory file against AEGIS schema."""
    console.print(f"\n[bold blue]AEGIS Validator[/bold blue]")
    console.print(f"Input: {input_file}\n")

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    validator = TrajectoryValidator()
    valid, invalid, errors = validator.validate_file(input_file)

    # Display results
    table = Table(title="Validation Results")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")

    table.add_row("Valid", str(valid))
    table.add_row("Invalid", str(invalid))
    table.add_row("Total", str(valid + invalid))

    console.print(table)

    if invalid > 0:
        console.print(f"\n[yellow]Success rate: {valid / (valid + invalid) * 100:.1f}%[/yellow]")

        if verbose:
            console.print("\n[bold red]Errors:[/bold red]")
            for line_num, errs in errors:
                console.print(f"\n[red]Line {line_num}:[/red]")
                for e in errs:
                    console.print(f"  - {e}")
        else:
            console.print(f"\n[bold red]First {min(5, len(errors))} errors:[/bold red]")
            for line_num, errs in errors[:5]:
                console.print(f"\n[red]Line {line_num}:[/red]")
                for e in errs[:3]:
                    console.print(f"  - {e}")

            if len(errors) > 5:
                console.print(f"\n... and {len(errors) - 5} more errors")
                console.print("Use --verbose to see all errors")

        raise typer.Exit(1)
    else:
        console.print("\n[green]All records valid![/green]")


@app.command()
def mutate(
    input_file: Path = typer.Argument(..., help="Input JSONL file"),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (default: input_mutated.jsonl)"
    ),
    count: int = typer.Option(
        5,
        "--count", "-n",
        help="Number of mutations per trajectory"
    ),
    mock: bool = typer.Option(
        True,
        "--mock/--api",
        help="Use simple mutations (no API calls)"
    ),
):
    """Generate mutations of existing trajectories."""
    console.print(f"\n[bold blue]AEGIS Mutation Engine[/bold blue]")
    console.print(f"Input: {input_file}")
    console.print(f"Mutations per trajectory: {count}")
    console.print(f"Using mock mutations: {mock}\n")

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Set default output
    if output is None:
        output = input_file.with_stem(input_file.stem + "_mutated")

    # Load trajectories
    trajectories = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    console.print(f"Loaded {len(trajectories)} trajectories")

    # Initialize mutation engine
    if mock:
        engine = SimpleMutationEngine()
    else:
        engine = MutationEngine()

    # Generate mutations
    all_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating mutations...", total=len(trajectories))

        for traj in trajectories:
            results = engine.generate_variants(traj, n=count, include_original=True)
            for result in results:
                if result.success:
                    all_results.append(result.trajectory)
            progress.advance(task)

    # Save output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for traj in all_results:
            f.write(json.dumps(traj) + "\n")

    console.print(f"\n[green]Generated {len(all_results)} trajectories[/green]")
    console.print(f"[green]Saved to {output}[/green]")


@app.command()
def benign(
    output: Path = typer.Option(
        "data/tier1_open/benign_hard_negatives/generated.jsonl",
        "--output", "-o",
        help="Output file path"
    ),
    count: int = typer.Option(
        100,
        "--count", "-n",
        help="Number of benign examples to generate"
    ),
):
    """Generate benign hard negative examples."""
    console.print(f"\n[bold blue]AEGIS Benign Hard Negatives[/bold blue]")
    console.print(f"Output: {output}")
    console.print(f"Count: {count}\n")

    trajectories = generate_benign_hard_negatives(count, output)

    console.print(f"[green]Generated {len(trajectories)} benign hard negatives[/green]")
    console.print(f"[green]Saved to {output}[/green]")


@app.command()
def stats(
    input_file: Path = typer.Argument(..., help="Input JSONL file"),
):
    """Show statistics for a trajectory file."""
    console.print(f"\n[bold blue]AEGIS Dataset Statistics[/bold blue]")
    console.print(f"Input: {input_file}\n")

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Load and analyze
    trajectories = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    # Compute stats
    families = {}
    types = {}
    difficulties = {}
    sources = {}
    turn_counts = []

    for traj in trajectories:
        meta = traj.get("attack_metadata", {})

        family = meta.get("family", "unknown")
        families[family] = families.get(family, 0) + 1

        atype = meta.get("type", "unknown")
        types[atype] = types.get(atype, 0) + 1

        diff = meta.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1

        source = meta.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

        turn_counts.append(len(traj.get("conversation", [])))

    # Display
    console.print(f"[bold]Total trajectories:[/bold] {len(trajectories)}")
    console.print(f"[bold]Average turns:[/bold] {sum(turn_counts) / len(turn_counts):.1f}")

    console.print("\n[bold]By Attack Family:[/bold]")
    for family, cnt in sorted(families.items(), key=lambda x: -x[1]):
        pct = cnt / len(trajectories) * 100
        console.print(f"  {family}: {cnt} ({pct:.1f}%)")

    console.print("\n[bold]By Difficulty:[/bold]")
    for diff, cnt in sorted(difficulties.items()):
        pct = cnt / len(trajectories) * 100
        console.print(f"  {diff}: {cnt} ({pct:.1f}%)")

    console.print("\n[bold]By Source:[/bold]")
    for source, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        pct = cnt / len(trajectories) * 100
        console.print(f"  {source}: {cnt} ({pct:.1f}%)")


if __name__ == "__main__":
    app()
