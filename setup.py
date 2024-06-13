from setuptools import setup, find_packages
import os

VERSION = "1.0.0"
DESCRIPTION = "Open-source tool for climatological time series reconstruction and extension"


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="rascal-ties",
    version=VERSION,
    author="Alvaro Gonzalez-Cervera",
    author_email="<alvaro@intermet.es>",
    description=DESCRIPTION,
    readme="README.md",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.1",
        "dask==2024.4.1",
        "xarray==2024.3.0",
        "scipy==1.13.0",
        "tqdm==4.65.0",
        "scikit-learn==1.4.1.post1",
        "seaborn==0.13.2",
        "eofs==1.4.1",
        "matplotlib==3.5.5"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/alvaro-gc95/RASCAL",
    long_description=read('README.md')
)