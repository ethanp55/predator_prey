import numpy as np
from typing import Dict, List, Tuple
from utils.utils import Utils


class State:
    def __init__(self, height: int, width: int, agent_names: List[str]) -> None:
        self.height, self.width, self.agent_positions, self.agent_n_steps, self.round_num = height, width, {}, {}, 0

        # Initialize the grid
        self.grid = []

        for _ in range(self.height):
            new_row = [Utils.AVAILABLE for _ in range(self.width)]
            self.grid.append(new_row)

        assert len(self.grid) == self.height and len(self.grid[0]) == self.width

        # Randomly assign starting positions for the predators and prey
        for i, agent_name in enumerate(agent_names):
            while True:
                row_index = np.random.choice(list(range(self.height)))
                col_index = np.random.choice(list(range(self.width)))

                if self.grid[row_index][col_index] == Utils.AVAILABLE:
                    self.grid[row_index][col_index] = i
                    self.agent_positions[agent_name] = (row_index, col_index)
                    break

    def __hash__(self):
        return hash(str(self.grid))

    def __str__(self):
        grid_str = ''

        for row in self.grid:
            grid_str += f'{row}\n'

        return grid_str[:-1]

    def available_actions(self) -> Dict[str, List[Tuple[int, int]]]:
        possible_actions_map = {}

        for agent_name, curr_pos in self.agent_positions.items():
            curr_row, curr_col = curr_pos

            for movement in Utils.POSSIBLE_MOVEMENTS:
                for delta in Utils.POSSIBLE_DELTA_VALS:
                    if movement == Utils.VERTICAL:
                        new_row, new_col = curr_row + delta, curr_col

                    else:
                        new_row, new_col = curr_row, curr_col + delta

                    new_row, new_col = self.adjust_vals(new_row, new_col)
                    possible_positions = possible_actions_map.get(agent_name, [])

                    if (new_row, new_col) not in possible_positions:
                        possible_positions.append((new_row, new_col))

                    possible_actions_map[agent_name] = possible_positions

        return possible_actions_map

    def adjust_vals(self, row_val: int, col_val: int) -> Tuple[int, int]:
        row, col = row_val, col_val

        if row_val < 0:
            row = self.height - 1

        elif row_val >= self.height:
            row = 0

        if col_val < 0:
            col = self.width - 1

        elif col_val >= self.width:
            col = 0

        return row, col

    def is_available(self, row_val: int, col_val: int) -> bool:
        row, col = self.adjust_vals(row_val, col_val)

        return self.grid[row][col] == Utils.AVAILABLE

    def neighboring_positions(self, curr_row: int, curr_col: int) -> List[Tuple[int, int]]:
        positions = []

        for movement in Utils.POSSIBLE_MOVEMENTS:
            for delta in Utils.POSSIBLE_DELTA_VALS:
                if movement == Utils.VERTICAL:
                    new_row, new_col = curr_row + delta, curr_col

                else:
                    new_row, new_col = curr_row, curr_col + delta

                if self.is_available(new_row, new_col):
                    row, col = self.adjust_vals(new_row, new_col)
                    positions.append((row, col))

        return positions

    def delta_row(self, curr_row: int, new_row: int) -> int:
        move_down = (self.height - curr_row) + new_row
        move_up = curr_row + (self.height - new_row)
        move_regular = abs(curr_row - new_row)

        return min([move_down, move_up, move_regular])

    def delta_col(self, curr_col: int, new_col: int) -> int:
        move_left = (self.width - curr_col) + new_col
        move_right = curr_col + (self.width - new_col)
        move_regular = abs(curr_col - new_col)

        return min([move_left, move_right, move_regular])

    def n_movements(self, curr_row: int, curr_col: int, new_row: int, new_col: int) -> int:
        n_steps = self.delta_row(curr_row, new_row) + self.delta_col(curr_col, new_col)

        return n_steps

    def neighbors(self, row1: int, col1: int, row2: int, col2: int) -> bool:
        n_movements = self.n_movements(row1, col1, row2, col2)

        return n_movements <= Utils.MAX_MOVEMENT_UNITS

    def process_actions(self, action_map: Dict[str, Tuple[int, int]]) -> None:
        for agent_name, tup in action_map.items():
            new_row, new_col = tup
            curr_row, curr_col = self.agent_positions[agent_name]
            agent_idx = self.grid[curr_row][curr_col]

            # Adjust the new row and col values so that they are on the grid (i.e. apply wrap around if we go
            # beyond the grid boundaries)
            new_row, new_col = self.adjust_vals(new_row, new_col)

            # Make sure the new row and column represent a valid movement (i.e. we can only move left, right, up, or
            # down, for a total of 1 unit of movement)
            n_movements = self.n_movements(curr_row, curr_col, new_row, new_col)

            if n_movements > Utils.MAX_MOVEMENT_UNITS:
                raise Exception(f'Cannot move from {(curr_row, curr_col)} to {(new_row, new_col)} because there are '
                                f'{n_movements} > {Utils.MAX_MOVEMENT_UNITS} movements')

            # Only move the agent if its desired new position is available
            if self.is_available(new_row, new_col):
                # Update the number of steps the agent has taken if its new position is different
                if (curr_row, curr_col) != (new_row, new_col):
                    self.agent_n_steps[agent_name] = self.agent_n_steps.get(agent_name, 0) + 1

                # The old position should now be available since the agent is moving
                self.grid[curr_row][curr_col] = Utils.AVAILABLE

                # Update the new position
                self.grid[new_row][new_col] = agent_idx
                self.agent_positions[agent_name] = (new_row, new_col)

        self.round_num += 1

    def prey_surrounded(self) -> bool:
        curr_row, curr_col = self.agent_positions[Utils.PREY_NAME]

        # Try all possible movements for the prey; if it cannot move, it is surrounded
        for movement in Utils.POSSIBLE_MOVEMENTS:
            for delta in Utils.POSSIBLE_DELTA_VALS:
                if movement == Utils.VERTICAL:
                    new_row, new_col = curr_row + delta, curr_col

                else:
                    new_row, new_col = curr_row, curr_col + delta

                if self.is_available(new_row, new_col):
                    return False

        return True

    def collective_distance(self) -> float:
        collective_distance, (prey_row, prey_col) = 0, self.agent_positions[Utils.PREY_NAME]

        for agent_name, (row, col) in self.agent_positions.items():
            if agent_name == Utils.PREY_NAME:
                continue

            dist_from_prey = self.n_movements(row, col, prey_row, prey_col)
            collective_distance += dist_from_prey

        return collective_distance
