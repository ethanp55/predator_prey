from agents.agent import Agent, State, Tuple


class Random(Agent):
    def __init__(self, name: str) -> None:
        Agent.__init__(self, name)

    def act(self, state: State) -> Tuple[int, int]:
        return self.random_action(state)

