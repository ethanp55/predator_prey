from agents.agent import Agent, State, Tuple
import numpy as np
from utils.a_star import AStar
from utils.utils import Utils


class GreedyPlanner(Agent):
    def __init__(self, name: str) -> None:
        Agent.__init__(self, name)

    def act(self, state: State) -> Tuple[int, int]:
        prey_row, prey_col = state.agent_positions[Utils.PREY_NAME]
        curr_row, curr_col = state.agent_positions[self.name]

        # If we are already neighbors with the prey, try to move to its current position in case it moves
        if state.neighbors(prey_row, prey_col, curr_row, curr_col):
            return prey_row, prey_col

        prey_neighboring_positions, min_dist, goal = state.neighboring_positions(prey_row, prey_col), np.inf, None

        for row, col in prey_neighboring_positions:
            dist = state.n_movements(curr_row, curr_col, row, col)

            if dist < min_dist:
                goal, min_dist = (row, col), dist

        # The goal should not be None - this is just a sanity check
        if goal is None:
            raise Exception('Goal should not be none by this point; check implementation')

        goal_row, goal_col = goal

        return AStar.find_path(curr_row, curr_col, goal_row, goal_col, state)
