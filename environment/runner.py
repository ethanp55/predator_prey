from agents.agent import Agent
from agents.uct import UCT
from aat.assumptions import Assumptions
from aat.checker import AssumptionChecker
from copy import deepcopy
from environment.pursuit import Pursuit
import numpy as np
from typing import List, Optional
from utils.utils import Utils


class Runner(object):
    @staticmethod
    def run(predators: List[Agent], n_epochs: int = 1000, height: int = 5, width: int = 5,
            report_assumptions: bool = False) -> Optional[List[List[Assumptions]]]:
        rounds, predator_n_steps = [], {}
        fourths = n_epochs // 4
        checker, assumptions = AssumptionChecker(), []

        for epoch in range(n_epochs):
            if fourths > 0 and (epoch + 1) % fourths == 0:
                print(f'Epoch {epoch + 1} / {n_epochs}')

            pursuit = Pursuit(height, width, predators)
            prev_state, epoch_assumptions = None, []

            while not pursuit.prey_caught():
                for predator in predators:
                    if isinstance(predator, UCT):
                        predator.update_teammate_models(pursuit.state)

                pursuit.transition()

                if report_assumptions:
                    curr_state = deepcopy(pursuit.state)

                    if prev_state is not None:
                        new_assumptions = checker.estimate_assumptions(prev_state, curr_state)
                        epoch_assumptions.append(new_assumptions)

                    prev_state = curr_state

            rounds.append(pursuit.state.round_num)
            assumptions.append(epoch_assumptions)

            for agent_name, n_steps in pursuit.state.agent_n_steps.items():
                predator_n_steps[agent_name] = predator_n_steps.get(agent_name, []) + [n_steps]

        if not report_assumptions:
            all_n_steps = []

            for agent_name, steps_list in predator_n_steps.items():
                if agent_name != Utils.PREY_NAME:
                    all_n_steps.extend(steps_list)

                avg_steps = np.array(steps_list).mean()

                print(f'{agent_name} average steps = {avg_steps}')

            print(f'Average combined number of predator steps = {np.array(all_n_steps).mean()}')
            print(f'Average number of rounds = {np.array(rounds).mean()}')

        else:
            return assumptions
