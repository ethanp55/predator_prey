from agents.agent import Agent
from agents.greedy import Greedy
from agents.greedy_planner import GreedyPlanner
from agents.greedy_prob import GreedyProbabilistic
from agents.modeller import Modeller
from agents.team_aware import TeamAware


class Baselines(object):
    @staticmethod
    def baseline(agent: Agent) -> float:
        # These numbers were obtained by running the agents in a team composed of themselves for 1000 epochs; they
        # represent the average number of rounds taken to surround the prey
        if isinstance(agent, Greedy):
            return 13.29

        elif isinstance(agent, GreedyPlanner):
            return 5.843

        elif isinstance(agent, TeamAware):
            return 5.003

        elif isinstance(agent, Modeller):
            return 5.849

        elif isinstance(agent, GreedyProbabilistic):
            return 61.411

        else:
            raise Exception(f'Unsupported agent type: {type(agent)}')
