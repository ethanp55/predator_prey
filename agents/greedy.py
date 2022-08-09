from agents.agent import Agent, State, Tuple
import numpy as np
from utils.utils import Utils


class Greedy(Agent):
    def __init__(self, name: str) -> None:
        Agent.__init__(self, name)

    def act(self, state: State) -> Tuple[int, int]:
        prey_row, prey_col = state.agent_positions[Utils.PREY_NAME]
        curr_row, curr_col = state.agent_positions[self.name]

        if state.neighbors(prey_row, prey_col, curr_row, curr_col):
            return prey_row, prey_col

        prey_neighboring_positions, min_dist, goal = state.neighboring_positions(prey_row, prey_col), np.inf, None

        for row, col in prey_neighboring_positions:
            dist = state.n_movements(curr_row, curr_col, row, col)

            if dist < min_dist:
                goal, min_dist = (row, col), dist

        if goal is not None:
            row, col = goal
            d_row, d_col = state.delta_row(curr_row, row) % state.height, state.delta_col(curr_col, col) % state.width
            next_row, next_col = None, None

            if d_row > d_col:
                min_dist = np.inf

                for delta in Utils.POSSIBLE_DELTA_VALS:
                    new_row, new_col = curr_row + delta, curr_col

                    if state.is_available(new_row, new_col):
                        new_row, new_col = state.adjust_vals(new_row, new_col)
                        dist = state.n_movements(new_row, new_col, row, col)

                        if dist < min_dist:
                            next_row, next_col, min_dist = new_row, new_col, dist

            elif next_row is None or next_col is None:
                min_dist = np.inf

                for delta in Utils.POSSIBLE_DELTA_VALS:
                    new_row, new_col = curr_row, curr_col + delta

                    if state.is_available(new_row, new_col):
                        new_row, new_col = state.adjust_vals(new_row, new_col)
                        dist = state.n_movements(new_row, new_col, row, col)

                        if dist < min_dist:
                            next_row, next_col, min_dist = new_row, new_col, dist

            if next_row is not None and next_col is not None:
                return next_row, next_col

        return self.random_action(state)
