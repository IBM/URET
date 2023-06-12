import codecs
import os

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "numpy==1.20",
    "scipy==1.4.1",
    "scikit-learn==0.22.2",
    "tensorflow==2.10",
    "pandas==1.3",
    "setuptools",
    "tqdm",
    "simanneal",
    "notebook==6.5.4"
]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r", encoding="utf-8") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="uret",
    version=get_version("uret/__init__.py"),
    description="Toolkit for generic adversarial machine learning evaluations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kevin Eykholt",
    author_email="kheykholt@ibm.com",
    maintainer="Kevin Eykholt",
    maintainer_email="kheykholt@ibm.com",
    url="https://github.com/IBM/URET",
    license="MIT",
    install_requires=install_requires,
    extras_require={
        "all": ["lief", "pandas", "tensorflow", "keras", "h5py", "keras-rl"],
        "binary": ["lief"],
        "rl": ["tensorflow", "keras", "h5py", "keras-rl"],
        "non-framework": ["pandas"]
    },
    packages=find_packages(),
    include_package_data=True,
)
