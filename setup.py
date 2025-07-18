from setuptools import setup, find_packages

setup(
    name="ehr_tokenisation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "polars",
        "meds",
    ],
)