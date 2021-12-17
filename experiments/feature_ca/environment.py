import gym
from gym_minigrid.wrappers import ImgObsWrapper
import numpy as np
from gym import spaces
from gym_minigrid.minigrid import OBJECT_TO_IDX, Goal, Grid, Lava, MiniGridEnv


class PartialObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable grid with a symbol for the state of each cell.
    The symbol is a triple of (X, Y, IDX), where IDX is
    the id of the object in the cell. Agent direction is removed.
    """

    def __init__(self, env, agent_view_size=3):
        super().__init__(env)

        assert (
            "image" in self.observation_space.spaces
        ), "Observation does not contain an image field."

        self.unwrapped.agent_view_size = agent_view_size
        self.unwrapped.observation_space["image"].shape = (
            agent_view_size,
            agent_view_size,
            self.unwrapped.observation_space.spaces["image"].shape[-1],
        )

    def observation(self, obs):
        topX, topY, botX, botY = self.unwrapped.get_view_exts()
        obs["image"] = obs["image"][topX:botX, topY:botY]
        return obs


class SymbolicObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable grid with a symbol for the state of each cell.
    The symbol is a triple of (X, Y, IDX), where IDX is
    the id of the object in the cell. Agent direction is removed.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )

    def observation(self, obs):
        w, h = self.width, self.height
        objects = np.array(
            [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
        ).reshape(1, w, h)
        grid = np.mgrid[:w, :h]
        grid = np.concatenate([grid, objects])
        grid = np.transpose(grid, (1, 2, 0))
        obs["image"] = grid
        return obs


class EmptyMultigoal(MiniGridEnv):
    def __init__(
        self,
        size=8,
        agent_start_pos=None,
        agent_start_dir=None,
        n_goals=2,
        n_traps=1,
    ):
        self.n_goals = n_goals
        self.n_traps = n_traps
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        size += 2
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            agent_view_size=size * 2 + 1,  # init as fully observable
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the goals
        for _ in range(self.n_goals):
            self.place_obj(Goal())

        # Place the traps
        for _ in range(self.n_traps):
            self.place_obj(Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square, avoid the lava"


class EmptyMultigoalEnv5x5(EmptyMultigoal):
    def __init__(self, **kwargs):
        super().__init__(size=7, **kwargs)


class EmptyMultigoalEnv8x8(EmptyMultigoal):
    def __init__(self, **kwargs):
        super().__init__(size=9, **kwargs)


class EmptyMultigoalEnv16x16(EmptyMultigoal):
    def __init__(self, **kwargs):
        super().__init__(size=18, **kwargs)


if __name__ == "__main__":
    env = EmptyMultigoal(size=5, n_goals=1, n_traps=1)
    env = SymbolicObsWrapper(env)
    env = PartialObsWrapper(env, agent_view_size=1)
    env = ImgObsWrapper(env)
    o = env.reset()
    print(o.shape)
    env.render()
    print(o)
