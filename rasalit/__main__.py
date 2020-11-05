import typer
import pathlib
import subprocess
from rasalit import __version__
from rasalit.common import run_streamlit_app, app_path


app = typer.Typer(add_completion=False)


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
def live_nlu(
    port: int = typer.Option(8501, help="Port number"),
    folder: str = typer.Option("", help="Folder that contains all Rasa NLU models"),
):
    """Select a trained Rasa model and interact with it."""
    if folder == "":
        typer.echo(
            "You need to set the `folder` option manually. Example;\n> rasalit live-nlu --folder path/folder"
        )
        return
    if not pathlib.Path(folder).exists():
        raise ValueError(f"You need to pass a folder that exists, got: {folder}")
    run_streamlit_app("livenlu", port=port, folder=folder)


@app.command()
def attention(
    port: int = typer.Option(8501, help="Port number"),
    folder: str = typer.Option("", help="Folder that contains trained Rasa models."),
):
    """Gives advanced deep-dive into the attention mechanisms in DIET/TED."""
    if folder == "":
        typer.echo(
            "You need to set the `folder` option manually. Example;\n> rasalit attention --folder path/folder"
        )
        return
    if not pathlib.Path(folder).exists():
        raise ValueError(f"You need to pass a folder that exists, got: {folder}")
    run_streamlit_app("attention", port=port, folder=folder)


@app.command()
def cluster_text(
    port: int = typer.Option(8501, help="Port number"),
):
    """Clusters texts to help you find NLU intents."""
    run_streamlit_app("attention", port=port)


@app.command()
def diet_explorer(
    port: int = typer.Option(8501, help=("Port number")),
):
    """Allows you to explore the DIET settings."""
    app = app_path("html/diet")
    typer.echo(typer.style(f"Starting up {app}", fg="green"))
    subprocess.run(["python", "-m", "http.server", str(port), "--directory", app])


@app.command()
def version():
    """Prints the current version"""
    print(f"{__version__}")


if __name__ == "__main__":
    app()
