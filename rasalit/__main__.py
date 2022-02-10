import os
import typer
import pathlib
import subprocess
from rasalit import __version__
from rasalit.common import run_streamlit_app, app_path
from rasalit.compose import generate_compose_files


app = typer.Typer(add_completion=False)
curr_dir = pathlib.Path(os.getcwd())


@app.command()
def overview(
    port: int = typer.Option(8501, help="Port number"),
    folder: str = typer.Option("", help="Folder that contains all Rasa NLU results"),
):
    """Gives an overview of all `rasa train nlu` results."""
    if folder == "":
        typer.echo(
            "You need to set the `folder` option manually. Example;\n> rasalit overview --folder path/folder"
        )
        return
    if not pathlib.Path(folder).exists():
        raise ValueError(f"You need to pass a folder that exists, got: {folder}")
    run_streamlit_app("overview", port=port, folder=folder)


@app.command()
def nlu_cluster(
    port: int = typer.Option(8501, help="Port number"),
):
    """Cluster a text file to look for clusters of intents."""
    run_streamlit_app("nlucluster", port=port)


@app.command()
def live_nlu(
    port: int = typer.Option(8501, help="Port number"),
    model_folder: pathlib.Path = typer.Option(
        curr_dir / "models", help="Folder that contains all Rasa NLU models"
    ),
    project_folder: pathlib.Path = typer.Option(
        curr_dir, help="The Rasa project folder (for custom component paths"
    ),
):
    """Select a trained Rasa model and interact with it."""
    if not pathlib.Path(model_folder).exists():
        raise ValueError(f"You need to pass a folder that exists, got: {model_folder}")
    run_streamlit_app(
        "livenlu", port=port, model_folder=model_folder, project_folder=project_folder
    )


@app.command()
def spelling(
    port: int = typer.Option(8501, help="Port number"),
    model_folder: pathlib.Path = typer.Option(
        curr_dir / "models", help="Folder that contains all Rasa NLU models"
    ),
    project_folder: pathlib.Path = typer.Option(
        curr_dir, help="The Rasa project folder (for custom component paths)"
    ),
):
    """Check the effect of spelling on NLU predictions."""
    if not pathlib.Path(model_folder).exists():
        raise ValueError(f"You need to pass a folder that exists, got: {model_folder}")
    run_streamlit_app(
        "spelling", port=port, model_folder=model_folder, project_folder=project_folder
    )


@app.command()
def diet_explorer(
    port: int = typer.Option(8501, help=("Port number")),
):
    """Allows you to explore the DIET settings."""
    app = app_path("html/diet")
    typer.echo(typer.style(f"Starting up {app}", fg="green"))
    subprocess.run(["python", "-m", "http.server", str(port), "--directory", app])


@app.command()
def generate_compose(
    rasa_project_dir: pathlib.Path = typer.Option(
        curr_dir, help="The Rasa project folder (for custom component paths)."
    ),
    overview_folder: pathlib.Path = typer.Option(
        "gridresults",
        help="Directory containing results of cross-validation from 'overview' command.",
    ),
    include_duckling: bool = typer.Option(
        False,
        help=(
            "Include the Duckling component in the generated Docker Compose - this is "
            "only required if your configured NLP pipeline relies on Duckling."
        ),
    ),
    output_dir: pathlib.Path = typer.Option(
        "compose", help="Directory to output Docker Compose files into."
    ),
):
    """Generate Docker Compose configuration file for hosting all Streamlit apps together."""
    typer.secho(f"Generating Docker Compose files into '{output_dir}'.", fg="green")
    output_dir.mkdir(exist_ok=True)
    generate_compose_files(
        rasa_project_dir, overview_folder, include_duckling, output_dir
    )
    typer.secho(
        f'Successfully generated Docker Compose files, run with: \n\n  $ cd "{output_dir}" && docker-compose up -d',
        fg="green",
    )


@app.command()
def version():
    """Prints the current version"""
    print(f"{__version__}")


def main():
    app(prog_name="rasalit")


if __name__ == "__main__":
    app()
