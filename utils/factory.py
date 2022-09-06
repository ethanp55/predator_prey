from agents.agent import Agent
from agents.greedy import Greedy
from agents.greedy_planner import GreedyPlanner
from agents.greedy_prob import GreedyProbabilistic
from agents.modeller import Modeller
from agents.team_aware import TeamAware
from copy import deepcopy
import random
from typing import List


class Factory:
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def generate_agents(self) -> List[Agent]:
        pass


class ExpertFactory(Factory):
    def __init__(self, identifier: str = 'E') -> None:
        Factory.__init__(self, identifier)

    def generate_agents(self) -> List[Agent]:
        greedy_planner = GreedyPlanner(f'GreedyPlanner_{self.identifier}')
        team_aware = TeamAware(f'TeamAware_{self.identifier}')
        modeller = Modeller(f'Modeller_{self.identifier}')

        return [greedy_planner, team_aware, modeller]


class SpecificExpertFactory(Factory):
    def __init__(self, identifier: str, expert_to_use: type) -> None:
        Factory.__init__(self, identifier)
        self.expert_to_use = expert_to_use

    def generate_agents(self) -> List[Agent]:
        agents = [self.expert_to_use(f'Agent{i}_{self.identifier}') for i in range(3)]

        return agents


class RandomExpertFactory(Factory):
    def __init__(self, identifier: str) -> None:
        Factory.__init__(self, identifier)

    def generate_agents(self) -> List[Agent]:
        agents = []

        for i in range(3):
            chosen_class = self.select_random_class()
            agents.append(chosen_class(f'Agent{i}_{self.identifier}'))

        return agents

    @staticmethod
    def select_random_class() -> type:
        expert_classes = [Greedy, GreedyProbabilistic, TeamAware]

        return random.choice(expert_classes)


class RandomSelectionFactory(Factory):
    def __init__(self, identifier: str, agent_pool: List[type]) -> None:
        Factory.__init__(self, identifier)
        self.agent_pool = agent_pool

    def generate_agents(self) -> List[Agent]:
        agents = []

        for i in range(3):
            chosen_agent = deepcopy(random.choice(self.agent_pool))(name=f'Agent{i}_{self.identifier}')
            agents.append(chosen_agent)

        return agents



