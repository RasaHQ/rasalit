from pathlib import Path

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Backported for Python < 3.7.
    import importlib_resources as pkg_resources

from . import resources


DEPENDS_ON_DUCKLING_YAML = """depends_on: [duckling]"""

# Leading spaces are important.
DUCKLING_SERVICE_YAML = """
  duckling:
    image: rasa/duckling
    container_name: rasalit-duckling
"""

DUCKLING_URL_ENV = '"RASA_DUCKLING_HTTP_URL=http://duckling:8000"'

NGINX_FILES = ["index.html", "rasalit.conf"]


def generate_compose_files(
    rasa_project_dir: Path,
    overview_folder: Path,
    include_duckling: bool,
    output_dir: Path,
) -> None:
    """
    Render the docker-compose.yaml file based on input configuration and copy over
    necessary nginx files to run all apps using Docker Compose.
    """
    compose_yaml = pkg_resources.read_text(resources, "docker-compose-template.yaml")
    compose_yaml_rendered = compose_yaml.format(
        rasa_project_dir=str(rasa_project_dir),
        overview_folder=str(overview_folder),
        depends_on_duckling=DEPENDS_ON_DUCKLING_YAML if include_duckling else "",
        duckling_url_env=DUCKLING_URL_ENV if include_duckling else "",
        duckling_service=DUCKLING_SERVICE_YAML if include_duckling else "",
    )
    with open(output_dir / "docker-compose.yaml", "w") as compose_fp:
        compose_fp.write(compose_yaml_rendered)

    for fname in NGINX_FILES:
        with pkg_resources.open_binary(resources, fname) as fp:
            with open(output_dir / fname, "wb") as fp_out:
                fp_out.write(fp.read())
