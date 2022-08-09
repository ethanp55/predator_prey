from agents.agent import Agent
from environment.pursuit import Pursuit
import numpy as np
from typing import List
from utils.utils import Utils


class Runner(object):
    @staticmethod
    def run(predators: List[Agent], n_epochs: int = 1000, height: int = 5, width: int = 5) -> None:
        rounds, predator_n_steps = [], {}
        fourths = n_epochs // 4

        for epoch in range(n_epochs):
            if (epoch + 1) % fourths == 0:
                print(f'Epoch {epoch + 1} / {n_epochs}')

            pursuit = Pursuit(height, width, predators)

            while not pursuit.prey_caught():
                pursuit.transition()

            rounds.append(pursuit.state.round_num)

            for agent_name, n_steps in pursuit.state.agent_n_steps.items():
                predator_n_steps[agent_name] = predator_n_steps.get(agent_name, []) + [n_steps]

        all_n_steps = []

        for agent_name, steps_list in predator_n_steps.items():
            if agent_name != Utils.PREY_NAME:
                all_n_steps.extend(steps_list)

            avg_steps = np.array(steps_list).mean()

            print(f'{agent_name} average steps = {avg_steps}')

        print(f'Average combined number of predator steps = {np.array(all_n_steps).mean()}')
        print(f'Average number of rounds = {np.array(rounds).mean()}')
