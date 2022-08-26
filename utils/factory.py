from agents.agent import Agent
from agents.greedy import Greedy
from agents.greedy_prob import GreedyProbabilistic
from agents.team_aware import TeamAware
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
        greedy = Greedy(f'Greedy_{self.identifier}')
        greedy_prob = GreedyProbabilistic(f'GreedyProb_{self.identifier}')
        team_aware = TeamAware(f'TeamAware_{self.identifier}')

        return [greedy, greedy_prob, team_aware]


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



