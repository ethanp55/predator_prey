from aat.assumptions import Assumptions
from agents.agent import Agent
from agents.greedy import Greedy
from agents.team_aware import TeamAware
from environment.state import State
import numpy as np
from utils.utils import Utils


class AssumptionChecker:
    def __init__(self, n_simulations: int = 30):
        self.n_simulations = n_simulations

    def _strategy_checker(self, prev_state: State, curr_state: State, sim_agent: Agent) -> float:
        vals = []

        for agent_name in prev_state.agent_positions.keys():
            # We only want to look at the other 3 predators
            if agent_name == Utils.ALEGAATR_NAME or agent_name == Utils.PREY_NAME:
                continue

            matches = []

            for _ in range(self.n_simulations):
                new_row, new_col = sim_agent.act(prev_state)

                # Match
                if (new_row, new_col) == curr_state.agent_positions[agent_name]:
                    matches.append(1.0)

                # The agent might have been blocked, so we're not sure what their true strategy is
                elif prev_state.agent_positions[agent_name] == curr_state.agent_positions[agent_name]:
                    matches.append(0.5)

                # No match
                else:
                    matches.append(0.0)

            # Use the average value from the simulations
            vals.append(np.array(matches).mean())

        # Sanity check
        assert len(vals) == 3

        # Average across the 3 predators/teammates
        return np.array(vals).mean()

    def _collective_distance(self, state: State) -> float:
        collective_distance, (prey_row, prey_col) = 0, state.agent_positions[Utils.PREY_NAME]

        for agent_name, (row, col) in state.agent_positions.items():
            if agent_name == Utils.PREY_NAME:
                continue

            dist_from_prey = state.n_movements(row, col, prey_row, prey_col)
            collective_distance += dist_from_prey

        return collective_distance

    def _closer_checker(self, prev_state: State, curr_state: State) -> float:
        prev_collective_distance = self._collective_distance(prev_state)
        curr_collective_distance = self._collective_distance(curr_state)

        return curr_collective_distance / prev_collective_distance

    def _max_dim_checker(self, prev_state: State, curr_state: State) -> float:
        vals, (prey_row, prey_col) = [], prev_state.agent_positions[Utils.PREY_NAME]

        for agent_name, (prev_row, prev_col) in prev_state.agent_positions.items():
            if agent_name == Utils.ALEGAATR_NAME or agent_name == Utils.PREY_NAME:
                continue

            curr_row, curr_col = curr_state.agent_positions[agent_name]
            prey_neighboring_positions, min_dist, goal = \
                prev_state.neighboring_positions(prey_row, prey_col), np.inf, None

            for row, col in prey_neighboring_positions:
                dist = prev_state.n_movements(prev_row, prev_col, row, col)

                if dist < min_dist:
                    goal, min_dist = (row, col), dist

            if goal is not None:
                row, col = goal
                d_row, d_col = prev_state.delta_row(prev_row, row) % prev_state.height, \
                    prev_state.delta_col(prev_col, col) % prev_state.width

                # If they moved in the maximum dimension, there is a match
                if (prev_row != curr_row and d_row > d_col) or (prev_col != curr_col and d_col > d_row):
                    vals.append(1.0)

                # If the dimensions are the same or the agent was unable to move, we're unsure
                elif d_row == d_col or (prev_row, prev_col) == (curr_row, curr_col):
                    vals.append(0.5)

                # Otherwise, there is not a match
                else:
                    vals.append(0.0)

            # We couldn't find a neighboring position, so we're not sure what their strategy is
            else:
                vals.append(0.5)

        # Sanity check
        assert len(vals) == 3

        # Average across the 3 predators/teammates
        return np.array(vals).mean()

    def _collisions_checker(self, prev_state: State, curr_state: State) -> float:
        n_collisions, (prey_row, prey_col) = 0, curr_state.agent_positions[Utils.PREY_NAME]

        for agent_name, (prev_row, prev_col) in prev_state.agent_positions.items():
            if agent_name == Utils.PREY_NAME:
                continue

            curr_row, curr_col = curr_state.agent_positions[agent_name]

            # If the agent couldn't/didn't move and it is not currently next to the prey, there was likely a collision
            if (prev_row, prev_col) == (curr_row, curr_col) and not \
                    curr_state.neighbors(curr_row, curr_col, prey_row, prey_col):
                n_collisions += 1

        return n_collisions

    def estimate_assumptions(self, prev_state: State, curr_state: State) -> Assumptions:
        greedy_estimate = self._strategy_checker(prev_state, curr_state, Greedy('GreedySim'))
        planner_estimate = self._strategy_checker(prev_state, curr_state, TeamAware('TeamAwareSim'))
        collective_distance = self._collective_distance(curr_state)
        moving_closer_estimate = self._closer_checker(prev_state, curr_state)
        max_dim_estimate = self._max_dim_checker(prev_state, curr_state)
        collisions_estimate = self._collisions_checker(prev_state, curr_state)

        return Assumptions(greedy_estimate, planner_estimate, collective_distance, moving_closer_estimate,
                           max_dim_estimate, collisions_estimate)
