from setuptools import setup, find_packages

setup(
    name="gcbh2",
    version="0.0.1",
    packages=find_packages(),
    scripts=[
        "./gcbh2/scripts/unique_structures.py",
        "./gcbh2/scripts/setup_runs.py",
        "./gcbh2/scripts/launch_runs.py",
    ],
)
