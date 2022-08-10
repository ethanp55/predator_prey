from agents.agent import Agent, State
from copy import deepcopy
import numpy as np
import random
from sklearn import tree
from typing import Dict, List, Tuple
from utils.utils import Utils


class UCTNode:
    def __init__(self, state: State, n_visits: int = 0):
        self.state = state
        self.n_visits = n_visits
        self.n_steps = 0

    def __eq__(self, other):
        return str(self.state) == str(other.state)


class UCT(Agent):
    def __init__(self, name: str, n_iterations: int, teammate_names: List[str], max_depth: int = 20) -> None:
        Agent.__init__(self, name)
        self.n_iterations = n_iterations
        self.max_depth = max_depth
        self.teammate_models, self.teammate_labels, self.teammate_positions, self.x = {}, {}, {}, []

        for name in teammate_names:
            self.teammate_models[name] = tree.DecisionTreeClassifier()

    def _flatten_grid(self, grid: List[List[int]]) -> List[int]:
        flattened_list = []

        for row in grid:
            flattened_list.extend(row)

        return flattened_list

    def update_teammate_models(self, state: State):
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

    def act(self, state: State) -> Tuple[int, int]:
        # The 'root' is the current state and depth
        root = UCTNode(state)
        node_map = {f'{state}_0': root}

        # Randomly traverse until we reach a terminal state or the maximum depth
        def random_walk(curr_state: State, n_steps: int):
            if curr_state.prey_surrounded() or n_steps >= self.max_depth:
                return n_steps

            action_map, possible_actions_map = {}, curr_state.available_actions()

            for agent_name, possible_actions in possible_actions_map.items():
                action_map[agent_name] = random.choice(possible_actions)

            next_state = deepcopy(curr_state)
            next_state.process_actions(action_map)

            return random_walk(next_state, n_steps + 1)

        # From the possible child nodes, pick the 'best' one using the scoring metric
        def choose_node(children: List[Tuple[UCTNode, int, int]], curr_visits: int, is_root: bool = False) -> \
                Tuple[UCTNode, int, int]:
            min_val, min_node = np.inf, None

            for child, r, c in children:
                if child.n_visits == 0:
                    return child, r, c

                node_avg_steps = child.n_steps / child.n_visits
                node_val = node_avg_steps if is_root else \
                    node_avg_steps + 2 * ((np.log(curr_visits) * child.n_visits) ** 0.5)

                if node_val < min_val:
                    min_val, min_node = node_val, (child, r, c)

            return min_node

        # Use the decision trees (if available) to estimate our teammates' actions; otherwise, estimate randomly
        def generate_teammate_actions(curr_state: State) -> Dict[str, Tuple[int, int]]:
            action_map = {}

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

                    new_row, new_col = state.adjust_vals(new_row, new_col)
                    action_map[name] = (new_row, new_col)

            else:
                possible_actions_map = curr_state.available_actions()

                for agent_name, possible_actions in possible_actions_map.items():
                    if agent_name == self.name:
                        continue

                    action_map[agent_name] = random.choice(possible_actions)

            return action_map

        # From the current node, generate children nodes
        def generate_children(curr_state: State, n_steps: int) -> List[Tuple[UCTNode, int, int]]:
            children = []
            teammate_actions = generate_teammate_actions(curr_state)
            possible_actions = curr_state.available_actions()[self.name]

            for r, c in possible_actions:
                action_map = deepcopy(teammate_actions)
                action_map[self.name] = r, c
                next_state = deepcopy(curr_state)
                next_state.process_actions(action_map)
                child_node = node_map.get(f'{next_state}_{n_steps}', None)

                if child_node is None:
                    child_node = UCTNode(next_state)
                    node_map[f'{next_state}_{n_steps}'] = child_node

                children.append((child_node, r, c))

            return children

        # Recursively traverse until we reach a terminal state or the maximum depth
        def rec_search(curr_node: UCTNode, n_steps: int):
            curr_state = curr_node.state

            if curr_state.prey_surrounded() or n_steps >= self.max_depth:
                return n_steps

            # If the node has not been visited yet, randomly traverse
            elif curr_node.n_visits == 0:
                curr_node.n_visits += 1

                return random_walk(curr_state, n_steps)

            curr_node.n_visits += 1

            children = generate_children(curr_state, n_steps + 1)
            node_to_explore, _, _ = choose_node(children, curr_node.n_visits)

            node_n_steps = rec_search(node_to_explore, n_steps + 1)
            curr_node.n_steps += node_n_steps

            return node_n_steps

        # Run the recursive search for n_iterations
        for n in range(self.n_iterations):
            rec_search(root, 0)

        # Using estimated node values, pick the action that leads to the 'best' node
        root_children = generate_children(root.state, 1)
        _, row, col = choose_node(root_children, root.n_visits, is_root=True)

        return row, col

