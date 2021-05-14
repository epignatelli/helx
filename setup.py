#!/usr/bin/env python

from distutils.core import setup

setup(
    name="Helx",
    version="0.0.2.0",
    description="Helx is a helper library for JAX/stax",
    author="Eduardo Pignatelli",
    author_email="edu.pignatelli@gmail.com",
    url="https://github.com/epignatelli/helx",
    packages=["helx"],
    install_requires=open("requirements.txt", "r").readlines(),
)
