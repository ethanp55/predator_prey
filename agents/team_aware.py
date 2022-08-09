from agents.agent import Agent, State, Tuple
import numpy as np
from utils.utils import Utils

# PLEASE NOTE: There's probably a cleaner way to implement A* along with the team aware agent in general, but this still
# is a fairly straightforward and simple implementation, so hopefully it is understandable.  I also tried to insert
# comments in key places


class TeamAware(Agent):
    def __init__(self, name: str) -> None:
        Agent.__init__(self, name)

    class PathNode:
        def __init__(self, row: int, col: int, parent=None):
            self.row, self.col, self.parent = row, col, parent

            self.g, self.h, self.f = 0, 0, 0

        def position(self) -> Tuple[int, int]:
            return self.row, self.col

        def update_values(self, g: int, h: int):
            self.g, self.h, self.f = g, h, g + h

        def __eq__(self, other):
            return self.position() == other.position()

        def __hash__(self):
            return hash(str(self))

        def __str__(self):
            return str((self.row, self.col))

    def _a_star(self, curr_row: int, curr_col: int, goal_row: int, goal_col: int, state: State) -> Tuple[int, int]:
        start_node = self.PathNode(curr_row, curr_col)
        end_node = self.PathNode(goal_row, goal_col)

        open_list = [start_node]
        closed_nodes = set()

        while len(open_list) > 0:
            # Find node with smallest F score
            curr_node, idx, min_score = None, 0, np.inf

            for i, node in enumerate(open_list):
                if node.f < min_score:
                    curr_node, idx, min_score = node, i, node.f

            # Remove node with smallest F score and add it to the closed/visited list
            open_list.pop(idx)
            closed_nodes.add(curr_node)

            # Path has been found
            if curr_node == end_node:
                path = []

                while curr_node != start_node:
                    path.append(curr_node.position())
                    curr_node = curr_node.parent

                return path[-1]

            # Otherwise, continue with the algorithm - next step is to generate the children of the current node
            available_neighbors = state.neighboring_positions(curr_node.row, curr_node.col)
            children = [self.PathNode(row, col, curr_node) for row, col in available_neighbors]

            # Visit the children and update their g, h, and f values
            for child in children:
                if child in closed_nodes:
                    continue

                new_g, new_h = curr_node.g + 1, state.n_movements(child.row, child.col, end_node.row, end_node.col)
                child.update_values(new_g, new_h)

                for node in open_list:
                    if child == node and child.g > node.g:
                        continue

                open_list.append(child)

        # We might be unable to follow a path if we're blocked in by other agents; in that case, just stay where we are
        return curr_row, curr_col

    def act(self, state: State) -> Tuple[int, int]:
        prey_row, prey_col = state.agent_positions[Utils.PREY_NAME]
        prey_neighboring_positions = state.neighboring_positions(prey_row, prey_col)
        curr_row, curr_col = state.agent_positions[self.name]

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

            # Keep track of the the worst shortest distance
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

        return self._a_star(curr_row, curr_col, goal_row, goal_col, state)
