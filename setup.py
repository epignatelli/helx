import os
from setuptools import setup, find_packages


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_PATH = "requirements.txt"


def _get_version():
    filepath = os.path.join(CURRENT_DIR, "helx", "__init__.py")
    with open(filepath) as f:
        for line in f:
            if line.startswith("__version__") and "=" in line:
                version = line[line.find("=") + 1 :].strip(" '\"\n")
                if version:
                    return version
    raise ValueError("`__version__` not defined in {}".format(filepath))


def _parse_requirements():
    filepath = os.path.join(CURRENT_DIR, REQUIREMENTS_PATH)
    with open(filepath) as f:
        return [
            line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))
        ]


setup(
    name="Helx",
    version=_get_version(),
    description="Helx is a helper library for Reinforcement Learning for JAX",
    author="Eduardo Pignatelli",
    author_email="edu.pignatelli@gmail.com",
    url="https://github.com/epignatelli/helx",
    packages=find_packages(exclude=["experiments", "test", "examples"]),
    python_requires=">=3.9",
    install_requires=_parse_requirements(),
)
