from dataclasses import dataclass
from typing import List


@dataclass
class Assumptions:
    greedy: float
    planner: float
    moving_closer: float
    prefer_max_dim: float
    collisions: float

    def generate_tuple(self, round_num: int, baseline: float) -> List[float]:
        tup = [round_num]
        tup += [self.__getattribute__(field_name) for field_name in self.__annotations__.keys()]
        tup += [baseline, baseline]

        return tup
