from agents.agent import Agent, State
from agents.prey import Prey
import numpy as np
from typing import List


class Pursuit:
    def __init__(self, height: int, width: int, predators: List[Agent]) -> None:
        # Make sure we can set the grid up properly
        n_predators = len(predators)

        if n_predators < 4:
            raise Exception('There have to be at least 4 predators')

        elif height * width < n_predators + 1:
            raise Exception(f'Not enough cells in the grid for the prey and {n_predators} predators')

        # Generate a list of agents (the predators and prey)
        self.agents = predators + [Prey()]

        # Initialize the state
        agent_names = [agent.name for agent in self.agents]
        self.state = State(height, width, agent_names)

    def transition(self) -> None:
        # Randomize the order in which the agents will act (including the prey)
        indices = list(range(len(self.agents)))
        np.random.shuffle(indices)
        action_map = {}

        for i in indices:
            agent = self.agents[i]
            new_row, new_col = agent.act(self.state)
            action_map[agent.name] = (new_row, new_col)

        self.state.process_actions(action_map)

    def prey_caught(self) -> bool:
        return self.state.prey_surrounded()
