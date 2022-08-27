from agents.agent import Agent
from aat.assumptions import Assumptions, AssumptionsCollection, distance_function
from collections import deque
from environment.state import State
import numpy as np
import pickle
import random
from typing import List, Tuple
from utils.factory import ExpertFactory
from utils.baselines import Baselines
from utils.utils import Utils


class Alegaatr(Agent):
    def __init__(self, name: str = Utils.ALEGAATR_NAME, lmbda: float = 0.95) -> None:
        Agent.__init__(self, name)
        self.lmbda = lmbda
        self.assumptions_collection = AssumptionsCollection(Utils.ESTIMATES_LOOKBACK)
        self.expert_to_use = None
        self.prev_distance = None

        factory = ExpertFactory()
        experts_list = factory.generate_agents()
        self.experts, self.models, self.scalers, self.training_datas, self.empirical_results, \
            self.n_rounds_since_played = {}, {}, {}, {}, {}, {}

        data_dir = f'../aat/training_data'

        for expert in experts_list:
            expert_name = expert.name
            expert.name = self.name
            self.experts[expert_name] = expert
            self.models[expert_name] = pickle.load(open(f'{data_dir}/{expert_name}_trained_knn_aat.pickle', 'rb'))
            self.scalers[expert_name] = \
                pickle.load(open(f'{data_dir}/{expert_name}_trained_knn_scaler_aat.pickle', 'rb'))
            self.training_datas[expert_name] = \
                np.array(pickle.load(open(f'{data_dir}/{expert_name}_training_data.pickle', 'rb')))
            self.empirical_results[expert_name] = deque(maxlen=Utils.ESTIMATES_LOOKBACK)
            self.n_rounds_since_played[expert_name] = 0

    def _knn_prediction(self, x: List[float], expert_name: str) -> Tuple[List[float], List[float]]:
        model, scaler, training_data = \
            self.models[expert_name], self.scalers[expert_name], self.training_datas[expert_name]

        x = np.array(x).reshape(1, -1)
        x_scaled = scaler.transform(x)
        neighbor_distances, neighbor_indices = model.kneighbors(x_scaled, Utils.KNN_N_NEIGHBORS)
        corrections, distances = [], []

        for i in range(len(neighbor_indices[0])):
            neighbor_idx = neighbor_indices[0][i]
            neighbor_dist = neighbor_distances[0][i]
            corrections.append(training_data[neighbor_idx, -1])
            distances.append(neighbor_dist)

        return corrections, distances

    def update_expert(self, round_num: int, new_assumptions: Assumptions, state: State) -> None:
        self.assumptions_collection.update(new_assumptions)
        new_distance = state.collective_distance()
        percentage_decrease = 1 - (new_distance / self.prev_distance)
        self.empirical_results[self.expert_to_use.name].append(percentage_decrease)

        predictions, new_tup = {}, [round_num] + self.assumptions_collection.generate_moving_averages()

        for expert_name, expert in self.experts.items():
            corrections, distances = self._knn_prediction(new_tup, expert_name)

            total_pred, inverse_distance_sum = 0, 0

            for dist in distances:
                inverse_distance_sum += (1 / dist) if dist != 0 else (1 / 0.000001)

            for i in range(len(corrections)):
                distance_i = distances[i]
                cor = corrections[i]
                inverse_distance_i = (1 / distance_i) if distance_i != 0 else (1 / 0.000001)
                distance_weight = inverse_distance_i / inverse_distance_sum
                total_pred += (Baselines.baseline(expert) * cor * distance_weight)

            if len(self.empirical_results[expert_name]) > 0:
                self.n_rounds_since_played[expert_name] += 1 if expert_name != self.expert_to_use.name else 0
                prob = self.lmbda ** self.n_rounds_since_played[expert_name]
                use_empricial_avgs = np.random.choice([1, 0], p=[prob, 1 - prob])

            else:
                use_empricial_avgs = False

            predictions[expert_name] = total_pred if not use_empricial_avgs else \
                np.array(self.empirical_results[expert_name]).mean()

        expert_key = max(predictions, key=lambda key: predictions[key])
        best_key = expert_key
        self.n_rounds_since_played[best_key] = 0
        self.expert_to_use = self.experts[best_key]

        print(f'AlgAATer expert: {best_key}')

    def act(self, state: State) -> Tuple[int, int]:
        self.prev_distance = state.collective_distance()

        if self.expert_to_use is None:
            self.expert_to_use = random.choice(list(self.experts.values()))

        return self.expert_to_use.act(state)
