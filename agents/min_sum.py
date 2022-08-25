from agents.agent import Agent, State, Tuple
import numpy as np
from utils.utils import Utils


class MinSum(Agent):
    def __init__(self, name: str) -> None:
        Agent.__init__(self, name)

    def act(self, state: State) -> Tuple[int, int]:
        prey_row, prey_col = state.agent_positions[Utils.PREY_NAME]
        curr_row, curr_col = state.agent_positions[self.name]
        prey_neighboring_positions = state.neighboring_positions(prey_row, prey_col)

        # If we are already neighbors with the prey, try to move to its current position in case it moves
        if state.neighbors(prey_row, prey_col, curr_row, curr_col):
            return prey_row, prey_col

        # Calculate the distance from each predator to each available cell neighboring the prey
        neighbor_distances = {}

        for agent_name, curr_pos in state.agent_positions.items():
            if agent_name == Utils.PREY_NAME:
                continue

            row, col = curr_pos

            if state.neighbors(prey_row, prey_col, row, col):
                continue

            distances = []

            for new_row, new_col in prey_neighboring_positions:
                dist = state.n_movements(row, col, new_row, new_col)
                distances.append((new_row, new_col, dist))

            neighbor_distances[agent_name] = distances

        # Find the assignments with the smallest distance sum - there's probably a better way to do this other than
        # brute force, but there will be at most 4 * 3 * 4 = 48 entries, so it should still be very fast
        goal, possible_assignments, min_sum = None, neighbor_distances[self.name], np.inf

        for new_row, new_col, dist in possible_assignments:
            dist_sum, assigned = dist, {(new_row, new_col)}

            for agent_name, assignments in neighbor_distances.items():
                if agent_name == self.name:
                    continue

                for row, col, other_dist in assignments:
                    if (row, col) in assigned:
                        continue

                    dist_sum += other_dist
                    assigned.add((row, col))

            if dist_sum < min_sum:
                goal = (new_row, new_col)
                min_sum = dist_sum

        # The goal should not be None - this is just a sanity check
        if goal is None:
            raise Exception('Goal should not be none by this point; check implementation')

        min_dist, new_row, new_col = np.inf, None, None

        for movement in Utils.POSSIBLE_MOVEMENTS:
            for delta in Utils.POSSIBLE_DELTA_VALS:
                if movement == Utils.HORIZONTAL:
                    next_row, next_col = curr_row + delta, curr_col

                else:
                    next_row, next_col = curr_row, curr_col + delta

                if state.is_available(next_row, next_col):
                    next_row, next_col = state.adjust_vals(next_row, next_col)
                    dist = state.n_movements(next_row, next_col, goal[0], goal[1])

                    if dist < min_dist:
                        new_row, new_col, min_dist = next_row, next_col, dist

        return new_row, new_col
