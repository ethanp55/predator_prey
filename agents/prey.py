from agents.random import Random
from utils.utils import Utils


class Prey(Random):
    def __init__(self) -> None:
        Random.__init__(self, Utils.PREY_NAME)

