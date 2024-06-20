from setuptools import setup, find_packages
import os

DISTNAME = "rascal-ties"
VERSION = "1.0.5"
DESCRIPTION = "Open-source tool for climatological time series reconstruction and extension"
PROJECT_URLS = {
    "Documentation": "https://rascalv100.readthedocs.io/en/latest/",
    "Source Code": "https://github.com/alvaro-gc95/RASCAL"
}
python_requires = "==3.10"
required_python_version = (3, 10)

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=DISTNAME,
    version=VERSION,
    author="Alvaro Gonzalez-Cervera",
    author_email="<alvaro@intermet.es>",
    description=DESCRIPTION,
    readme="README.md",
    packages=["rascal"],
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
        "matplotlib==3.5.5",
        "sphinx-rtd-theme==1.2.2",
        "sphinx==7.3.7",
        "sphinx-copybutton==0.5.2",
        "cfgrib==0.9.12.0",
        "netCDF4==1.7.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=python_requires,
    url="https://github.com/alvaro-gc95/RASCAL",
    long_description=read('README.md'),
    download_url="https://github.com/alvaro-gc95/RASCAL",
)