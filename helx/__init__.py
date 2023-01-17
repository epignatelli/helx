__version__ = "0.2.0"

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
import procgen, minigrid, gym_minigrid
