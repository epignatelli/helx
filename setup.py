from setuptools import setup, find_packages


__version__ = "0.1.1.0"


def parse_requirements(filename):
    #  TODO(ep): parse git-based repositories
    return open(filename, "r").readlines()


setup(
    name="Helx",
    version=__version__,
    description="Helx is a helper library for Reinforcement Learning for JAX",
    author="Eduardo Pignatelli",
    author_email="edu.pignatelli@gmail.com",
    url="https://github.com/epignatelli/helx",
    packages=find_packages(exclude=["experiments", "test", "examples"]),
    python_requires=">=3.9",
    install_requires=[
        "pytest",
        "bsuite",
        "gymnasium[all]>=0.26",
        "gym[all]>=0.26",
        "minigrid",
        "wandb",
        "jupyterlab",
        "black",
        "flake8",
        "pytest",
        "jax",
        "chex",
        "optax",
        "rlax",
        "flax",
        "dm_env",
        "wandb",
        "absl-py",
    ],
)
