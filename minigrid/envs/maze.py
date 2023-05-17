from __future__ import annotations

import random
import numpy as np


from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv


class MazeEnv(MiniGridEnv):
    """
    ## Description

    This environment is random maze, and the goal of the agent is to reach
    the green goal square, which provides a sparse reward. A small penalty
    is subtracted for the numbr of steps to reach tthe goal. 
    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Maze-v0`

    """

    def __init__(
            self,
            size=10,
            agent_start_pos=(1,1),
            agent_start_dir=0,
            max_steps: int | None = None,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 10 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        maze = Maze_Generator(width-2, height-2)
        last_pos_x = 0
        last_pos_y = 0
        
        for i in range(height-2):
            for j in range(width-2):
                if maze.grid[i][j]:
                    pass
                else:
                    self.put_obj(Wall(), i+1, j+1)
                    last_pos_x = j
                    last_pos_y = i
                    
        # Place a goal square in the bottom-right corner

        self.put_obj(Goal(), last_pos_x, last_pos_y)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"




class Maze_Generator:
    """
    A maze data structure, represented as a boolean grid where
    True = Passage
    False = Wall
    """

    def __init__(self, width, height):
        self._width = width
        self._height = height
        self.grid = np.zeros((width, height), dtype=bool)
        self.__generate()

    def __frontier(self, x, y):
        """
        Returns the frontier of cell (x, y)
              The frontier of a cell are all walls with exact distance two,
              diagonals excluded.
        :param x: x coordinate of the cell
        :param y: y coordinate of the cell
        :return: set of all frontier cells
        """
        f = set()
        if x >= 0 and x < self._width and y >= 0 and y < self._height:
            if x > 1 and not self.grid[x-2][y]:
                f.add((x-2, y))
            if x + 2 < self._width and not self.grid[x+2][y]:
                f.add((x+2, y))
            if y > 1 and not self.grid[x][y-2]:
                f.add((x, y-2))
            if y + 2 < self._height and not self.grid[x][y+2]:
                f.add((x, y+2))

        return f

    def __neighbours(self, x, y):
        """
        Returns the neighbours of cell (x, y)
                 The neighbours of a cell are all passages with exact distance two,
                 diagonals excluded.
           :param x: x coordinate of the cell
           :param y: y coordinate of the cell
           :return: set of all neighbours
           """
        n = set()
        if x >= 0 and x < self._width and y >= 0 and y < self._height:
            if x > 1 and self.grid[x-2][y]:
                n.add((x-2, y))
            if x + 2 < self._width and self.grid[x+2][y]:
                n.add((x+2, y))
            if y > 1 and self.grid[x][y-2]:
                n.add((x, y-2))
            if y + 2 < self._height and self.grid[x][y+2]:
                n.add((x, y+2))

        return n

    def __connect(self, x1, y1, x2, y2):
        """
        Connects wall (x1, x2) with passage (x2 , x2), who
        are assumed to be of distance two from each other
            Connecting a wall to a passage implies converting
            that wall and the wall between them to passages
        :param x1: x coordinate of the wall
        :param y1: y coordinate of the wall
        :param x2: x coordinate of the passage
        :param y2: y coordinate of the passage
        """
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        self.grid[x1][y1] = True
        self.grid[x][y] = True


    def __generate(self):
        """
        Generates a maze using prim's algorithm
        Pseudo code:
        1. All cells are assumed to be walls
        2. Pick cell (x, y) at random and set it to passage
        3. Get frontier fs of (x, y) and add to set s that contains all frontier cells
        4. while s is not empty:
            4a. Pick a random cell (x, y) from s and remove it from s
            4b. Get neighbours ns of (x, y)
            4c. Connect (x, y) with random neighbour (nx, ny) from ns
            4d. Add the frontier fs of (x, y) to s

        :param animate: animate the maze
        """
        s = set()
        x, y = (0,0)
        self.grid[x][y] = True
        fs = self.__frontier(x, y)
        for f in fs:
            s.add(f)
        while s:
            x, y = random.choice(tuple(s))
            s.remove((x, y))
            ns = self.__neighbours(x, y)
            if ns:
                nx, ny = random.choice(tuple(ns))
                self.__connect(x, y, nx, ny)
            fs = self.__frontier(x, y)
            for f in fs:
                s.add(f)

