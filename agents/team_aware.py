from agents.agent import Agent, State, Tuple
from utils.a_star import AStar
from utils.utils import Utils


class TeamAware(Agent):
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
        neighbor_distances, name_ordering = {}, []

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

            # Sort by distance
            distances.sort(key=lambda x: x[-1])
            neighbor_distances[agent_name] = distances

            # Keep track of the worst shortest distance
            name_ordering.append((agent_name, distances[-1][-1]))

        # Order by the worst shortest distance
        name_ordering.sort(key=lambda x: x[-1])
        assigned, goal = set(), None

        # Make assignments - assign the goal when we reach ourselves so that we can plan a path
        for agent_name, _ in name_ordering:
            for row, col, _ in neighbor_distances[agent_name]:
                if (row, col) not in assigned:
                    assigned.add((row, col))

                    if agent_name == self.name:
                        goal = (row, col)

                    break

        # The goal should not be None - this is just a sanity check
        if goal is None:
            raise Exception('Goal should not be none by this point; check implementation')

        goal_row, goal_col = goal

        return AStar.find_path(curr_row, curr_col, goal_row, goal_col, state)
