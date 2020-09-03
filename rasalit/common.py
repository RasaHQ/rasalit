import os
import pathlib
import subprocess
from pkg_resources import resource_filename


def app_path(py_file):
    found_path = resource_filename("rasalit", py_file)
    assert pathlib.Path(found_path).exists()
    return found_path


def run_streamlit_app(subfolder, filename="app.py", port=8501, **kwargs):
    app = app_path(os.path.join("apps", subfolder, filename))
    cmd = ["streamlit", "run", "--server.port", str(port), app, "--"]
    for key, value in kwargs.items():
        cmd += [f"--{key}", f"{value}"]
    print(" ".join(cmd))
    subprocess.run(cmd)
