from setuptools import setup, find_packages
from rasalit import __version__

base_packages = ["streamlit>=0.57.3"]

dev_packages = ["flake8>=3.6.0", "pytest==4.0.2",]


setup(
    name="rasalit",
    version=__version__,
    packages=find_packages(exclude=['notebooks']),
    install_requires=base_packages,
    entry_points={
        'console_scripts': [
            'rasalit = rasalit.__main__:main',
        ],
    },
    package_data={"rasalit": ["html/*/*.html"]},
    extras_require={
        "dev": dev_packages
    }
)