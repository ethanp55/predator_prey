from agents.agent import Agent
from agents.alegaatr import Alegaatr
from agents.uct import UCT
from aat.checker import AssumptionChecker
from copy import deepcopy
from environment.pursuit import Pursuit
import numpy as np
from typing import List
from utils.utils import Utils


class Runner(object):
    @staticmethod
    def run(predators: List[Agent], n_epochs: int = 100, height: int = 5, width: int = 5,
            report_assumptions: bool = False, suppress_output: bool = False) -> any:
        rounds, predator_n_steps = [], {}
        fourths = n_epochs // 4
        checker, assumptions = AssumptionChecker(), []

        for epoch in range(n_epochs):
            if fourths > 0 and (epoch + 1) % fourths == 0 and not suppress_output:
                print(f'Epoch {epoch + 1} / {n_epochs}')

            predators_copy = deepcopy(predators)
            pursuit = Pursuit(height, width, predators_copy)
            prev_state, epoch_assumptions, aat_agent_in_predators = deepcopy(pursuit.state), [], False

            while not pursuit.prey_caught():
                for predator in predators_copy:
                    if isinstance(predator, UCT):
                        predator.update_teammate_models(pursuit.state)

                    elif isinstance(predator, Alegaatr):
                        aat_agent_in_predators = True

                        if len(epoch_assumptions) > 0:
                            predator.update_expert(pursuit.state.round_num, epoch_assumptions[-1], pursuit.state)

                pursuit.transition()

                if report_assumptions or aat_agent_in_predators:
                    curr_state = deepcopy(pursuit.state)
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

                if not suppress_output:
                    print(f'{agent_name} average steps = {avg_steps}')

            predator_steps, num_rounds = np.array(all_n_steps).mean(), np.array(rounds).mean()

            if not suppress_output:
                print(f'Average combined number of predator steps = {predator_steps}')
                print(f'Average number of rounds = {num_rounds}')

            return predator_steps, num_rounds

        else:
            return assumptions
