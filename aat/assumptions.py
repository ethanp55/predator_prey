from dataclasses import dataclass
from typing import List


@dataclass
class Assumptions:
    greedy: float
    planner: float
    collective_distance: float
    moving_closer: float
    prefer_max_dim: float
    collisions: float

    def generate_tuple(self, round_num: int, baseline: float) -> List[float]:
        tup = [round_num]
        tup += [self.__getattribute__(field_name) for field_name in self.__annotations__.keys()]
        tup += [baseline]

        return tup


def distance_function(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise Exception(f'Invalid tuples with lengths {len(x)} and {len(y)}')

    round_num_dist = 2 * abs(x[0] - y[0])
    greedy_dist = 4 * abs(x[1] - y[1])
    planner_dist = 4 * abs(x[2] - y[2])
    collective_distance_dist = 4 * abs(x[3] - y[3])
    moving_closer_dist = 4 * abs(x[4] - y[4])
    prefer_max_dim_dist = 4 * abs(x[5] - y[5])
    collisions_dist = 4 * abs(x[6] - y[6])

    return sum([round_num_dist, greedy_dist, planner_dist, collective_distance_dist, moving_closer_dist,
                prefer_max_dim_dist, collisions_dist])
