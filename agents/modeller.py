from agents.agent import Agent, State
import numpy as np
import random
from sklearn import tree
from typing import Dict, List, Tuple
from utils.utils import Utils


class Modeller(Agent):
    def __init__(self, name: str) -> None:
        Agent.__init__(self, name)
        self.teammate_models, self.teammate_labels, self.teammate_positions, self.x = {}, {}, {}, []

    def _flatten_grid(self, grid: List[List[int]]) -> List[int]:
        flattened_list = []

        for row in grid:
            flattened_list.extend(row)

        return flattened_list

    def _update_teammate_models(self, state: State):
        flattened_grid = self._flatten_grid(state.grid)
        self.x.append(flattened_grid)

        # Create new training label by comparing each teammate's old position with their new position
        if len(self.x) > 1:
            for name, (prev_row, prev_col) in self.teammate_positions.items():
                new_row, new_col = state.agent_positions[name]
                label = None

                for movement in Utils.POSSIBLE_MOVEMENTS:
                    for delta in Utils.POSSIBLE_DELTA_VALS:
                        if label is not None:
                            continue

                        if movement == Utils.VERTICAL:
                            row, col = prev_row + delta, prev_col

                        else:
                            row, col = prev_row, prev_col + delta

                        row, col = state.adjust_vals(row, col)

                        if (row, col) == (new_row, new_col):
                            label = Utils.NONE

                            if movement == Utils.VERTICAL and delta == -1:
                                label = Utils.DOWN

                            elif movement == Utils.VERTICAL and delta == 1:
                                label = Utils.UP

                            elif movement == Utils.HORIZONTAL and delta == -1:
                                label = Utils.LEFT

                            elif movement == Utils.HORIZONTAL and delta == 1:
                                label = Utils.RIGHT

                label = Utils.NONE if label is None else label
                self.teammate_labels[name] = self.teammate_labels.get(name, []) + [label]

        # Update each teammate's position
        for name, (row, col) in state.agent_positions.items():
            if name == self.name:
                continue

            self.teammate_positions[name] = (row, col)

        # Refit the decision tree for each teammate
        if len(self.x) > 1:
            for name, model in self.teammate_models.items():
                y_train = np.array(self.teammate_labels[name])
                x_train = np.array(self.x[:len(y_train)])
                model.fit(x_train, y_train)

        else:
            for name in state.agent_positions.keys():
                if name != Utils.PREY_NAME and name != self.name:
                    self.teammate_models[name] = tree.DecisionTreeClassifier()

    def _generate_teammate_actions(self, curr_state: State) -> Dict[str, Tuple[int, int]]:
        action_map, assigned = {}, set()

        if len(self.x) > 1:
            x = np.array(self._flatten_grid(curr_state.grid)).reshape(1, -1)

            for name, model in self.teammate_models.items():
                curr_row, curr_col = curr_state.agent_positions[name]
                pred = model.predict(x)[0]

                if pred == Utils.UP:
                    new_row, new_col = curr_row + 1, curr_col

                elif pred == Utils.DOWN:
                    new_row, new_col = curr_row - 1, curr_col

                elif pred == Utils.LEFT:
                    new_row, new_col = curr_row, curr_col - 1

                elif pred == Utils.RIGHT:
                    new_row, new_col = curr_row, curr_col + 1

                else:
                    new_row, new_col = curr_row, curr_col

                new_row, new_col = curr_state.adjust_vals(new_row, new_col)

                if (new_row, new_col) not in assigned:
                    assigned.add((new_row, new_col))
                    action_map[name] = (new_row, new_col)

                else:
                    action_map[name] = (curr_row, curr_col)

        else:
            possible_actions_map = curr_state.available_actions()

            for agent_name, possible_actions in possible_actions_map.items():
                if agent_name == self.name:
                    continue

                curr_row, curr_col = curr_state.agent_positions[agent_name]
                new_row, new_col = random.choice(possible_actions)

                if (new_row, new_col) not in assigned:
                    assigned.add((new_row, new_col))
                    action_map[agent_name] = (new_row, new_col)

                else:
                    action_map[agent_name] = (curr_row, curr_col)

        return action_map

    def act(self, state: State) -> Tuple[int, int]:
        self._update_teammate_models(state)

        # If already neighboring the prey, try to move onto it
        prey_row, prey_col = state.agent_positions[Utils.PREY_NAME]
        curr_row, curr_col = state.agent_positions[self.name]

        if state.neighbors(prey_row, prey_col, curr_row, curr_col):
            return prey_row, prey_col

        teammate_actions = self._generate_teammate_actions(state)
        prey_neighboring_positions = state.neighboring_positions(prey_row, prey_col)

        # Calculate the distance from each predator to each available cell neighboring the prey
        neighbor_distances = {}

        for agent_name, pos in state.agent_positions.items():
            if agent_name == Utils.PREY_NAME:
                continue

            row, col = pos if agent_name == self.name else teammate_actions[agent_name]

            if state.neighbors(prey_row, prey_col, row, col):
                continue

            distances = []

            for new_row, new_col in prey_neighboring_positions:
                if not state.is_available(new_row, new_col):
                    continue

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

    # def act(self, state: State) -> Tuple[int, int]:
    #     self._update_teammate_models(state)
    #
    #     # If already neighboring the prey, try to move onto it
    #     prey_row, prey_col = state.agent_positions[Utils.PREY_NAME]
    #     curr_row, curr_col = state.agent_positions[self.name]
    #
    #     if state.neighbors(prey_row, prey_col, curr_row, curr_col):
    #         return prey_row, prey_col
    #
    #     teammate_actions = self._generate_teammate_actions(state)
    #     prey_neighboring_positions = state.neighboring_positions(prey_row, prey_col, filter_availability=False)
    #
    #     # Calculate the distance from each predator to each available cell neighboring the prey
    #     assigned = set()
    #
    #     for agent_name, pos in state.agent_positions.items():
    #         if agent_name == Utils.PREY_NAME or agent_name == self.name:
    #             continue
    #
    #         row, col = teammate_actions[agent_name]
    #
    #         if state.neighbors(prey_row, prey_col, row, col):
    #             assigned.add((row, col))
    #             continue
    #
    #         goal, min_dist = None, np.inf
    #
    #         for new_row, new_col in prey_neighboring_positions:
    #             if (new_row, new_col) in assigned or (new_row, new_col) in teammate_actions.values():
    #                 continue
    #
    #             dist = state.n_movements(row, col, new_row, new_col)
    #
    #             if dist < min_dist:
    #                 min_dist, goal = dist, (new_row, new_col)
    #
    #         assigned.add(goal)
    #
    #     goal = None
    #     for new_row, new_col in prey_neighboring_positions:
    #         if (new_row, new_col) not in assigned:
    #             goal = new_row, new_col
    #             break
    #
    #     # The goal should not be None - this is just another sanity check
    #     if goal is None:
    #         raise Exception('Goal should not be none by this point; check implementation')
    #
    #     min_dist, new_row, new_col = np.inf, None, None
    #
    #     for movement in Utils.POSSIBLE_MOVEMENTS:
    #         for delta in Utils.POSSIBLE_DELTA_VALS:
    #             if movement == Utils.HORIZONTAL:
    #                 next_row, next_col = curr_row + delta, curr_col
    #
    #             else:
    #                 next_row, next_col = curr_row, curr_col + delta
    #
    #             if state.is_available(next_row, next_col):
    #                 next_row, next_col = state.adjust_vals(next_row, next_col)
    #                 dist = state.n_movements(next_row, next_col, goal[0], goal[1])
    #
    #                 if dist < min_dist:
    #                     new_row, new_col, min_dist = next_row, next_col, dist
    #
    #     return new_row, new_col
