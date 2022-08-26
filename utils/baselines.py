from agents.agent import Agent
from agents.greedy import Greedy
from agents.greedy_prob import GreedyProbabilistic
from agents.team_aware import TeamAware


class Baselines(object):
    @staticmethod
    def baseline(agent: Agent) -> float:
        # These numbers were obtained by running the agents in a team composed of themselves for 1000 epochs; they
        # represent the average number of steps taken to surround the prey
        if isinstance(agent, Greedy):
            return 6.032542011202987

        elif isinstance(agent, GreedyProbabilistic):
            return 27.922160243407706

        elif isinstance(agent, TeamAware):
            return 2.9083265527529156

        else:
            raise Exception(f'Unsupported agent type: {type(agent)}')
