from collections import deque
from dataclasses import dataclass
import pandas as pd
from typing import List


@dataclass
class Assumptions:
    greedy: float
    planner: float
    collective_distance: float
    moving_closer: float
    prefer_max_dim: float
    collisions: float

    def get_assumption_names(self) -> List[str]:
        return list(self.__annotations__.keys())

    def generate_tuple(self, round_num: int, baseline: float) -> List[float]:
        tup = [round_num]
        tup += [self.__getattribute__(assumption_name) for assumption_name in self.get_assumption_names()]
        tup += [baseline]

        return tup


class AssumptionsCollection:
    def __init__(self, lookback: int) -> None:
        self.collections = {}
        self.lookback = lookback

    def update(self, new_assumptions: Assumptions) -> None:
        for assumption_name in new_assumptions.get_assumption_names():
            collection = self.collections.get(assumption_name, deque(maxlen=self.lookback))
            collection.append(new_assumptions.__dict__[assumption_name])
            self.collections[assumption_name] = collection

    def generate_moving_averages(self) -> List[float]:
        moving_averages = []

        for collection in self.collections.values():
            moving_average = list(pd.Series.ewm(collection, span=self.lookback).mean())[-1]
            moving_averages.append(moving_average)

        return moving_averages


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
