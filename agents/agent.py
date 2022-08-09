from environment.state import State
import numpy as np
from typing import Tuple
from utils.utils import Utils


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name

    def act(self, state: State) -> Tuple[int, int]:
        pass

    def random_action(self, state: State) -> Tuple[int, int]:
        curr_row, curr_col = state.agent_positions[self.name]
        movement = np.random.choice(Utils.POSSIBLE_MOVEMENTS)
        delta = np.random.choice(Utils.POSSIBLE_DELTA_VALS)

        if movement == Utils.VERTICAL:
            return curr_row + delta, curr_col

        else:
            return curr_row, curr_col + delta
