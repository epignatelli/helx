from .info import get_version

__version__ = get_version()


from . import (
    agents,
    environment,
    experiment,
    flags,
    image,
    logging,
    mdp,
    memory,
    networks,
    preprocess,
    random,
    spaces,
    stax,
)

# Make sure that the gym registry is populated
# with exteral environments.
import procgen
import gym_minigrid
import minigrid
