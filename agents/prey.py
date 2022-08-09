from agents.agent import Agent, State, Tuple
from utils.utils import Utils


class Prey(Agent):
    def __init__(self) -> None:
        Agent.__init__(self, Utils.PREY_NAME)

    def act(self, state: State) -> Tuple[int, int]:
        return self.random_action(state)

