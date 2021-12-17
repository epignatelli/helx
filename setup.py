from setuptools import setup, find_packages


__version__ = "0.0.5.0"


def parse_requirements(filename):
    #  TODO(ep): parse git-based repositories
    return open(filename, "r").readlines()


setup(
    name="Helx",
    version=__version__,
    description="Helx is a helper library for JAX/stax to implement \
                 Reinforcement Learning and Deep Learning algorithms",
    author="Eduardo Pignatelli",
    author_email="edu.pignatelli@gmail.com",
    url="https://github.com/epignatelli/helx",
    packages=find_packages(exclude="experiments"),
    python_requires=">=3.7",
    install_requires=[
        "pytest",
        "jaxlib",
        "jax",
        "bsuite",
        "gym",
        "gym-minigrid",
        "wandb",
        "jupyterlab",
        "black",
    ],
)
