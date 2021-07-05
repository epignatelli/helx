#!/usr/bin/env python

from distutils.core import setup


def parse_requirements(filename):
    #  TODO(ep): parse git-based repositories
    return open(filename, "r").readlines()


setup(
    name="Helx",
    version="0.0.5.0",
    description="Helx is a helper library for JAX/stax to implement \
                 Reinforcement Learning and Deep Learning algorithms",
    author="Eduardo Pignatelli",
    author_email="edu.pignatelli@gmail.com",
    url="https://github.com/epignatelli/helx",
    packages=["helx", "helx.nn", "helx.rl", "helx.rl.baselines", "helx.optimise"],
    install_requires=parse_requirements("requirements.txt"),
)
