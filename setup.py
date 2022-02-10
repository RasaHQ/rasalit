from setuptools import setup, find_packages
from rasalit import __version__

base_packages = [
    "streamlit>=0.57.3",
    "pyyaml>=5.3.1",
    "pandas>=1.0.3",
    "altair>=4.1.0",
    "typer>=0.3.0",
    "rasa>=2.4",
    "spacy>=2.3.2",
    "tensorflow>=2.3.1",
    "nlpaug>=1.1.2",
    # Backfill for Python < 3.7.
    "importlib-resources>=5.3.0",
]

dev_packages = ["flake8>=3.6.0", "pytest>=4.0.2", "pre-commit>=2.7.1", "black"]

setup(
    name="rasalit",
    version=__version__,
    packages=find_packages(exclude=["notebooks"]),
    install_requires=base_packages,
    entry_points={
        "console_scripts": [
            "rasalit = rasalit.__main__:main",
        ],
    },
    package_data={"rasalit": ["html/*/*.html", "data/*.*"]},
    extras_require={"dev": dev_packages},
)
