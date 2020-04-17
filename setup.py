from setuptools import setup, find_packages

base_packages = ["streamlit>=0.57.3"]

dev_packages = ["flake8>=3.6.0", "pytest==4.0.2",]


setup(
    name="rasalit",
    version="0.1.0",
    packages=find_packages(exclude=['notebooks']),
    install_requires=base_packages,
    entry_points={
        'console_scripts': [
            'rasalit = rasalit.__main__:main',
        ],
    },
    extras_require={
        "dev": dev_packages
    }
)