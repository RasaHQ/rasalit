from typer.testing import CliRunner

from rasalit.__main__ import app
from rasalit import __version__

runner = CliRunner()


def test_app():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_version():
    result = runner.invoke(app, ["version"])
    assert __version__ in result.stdout
