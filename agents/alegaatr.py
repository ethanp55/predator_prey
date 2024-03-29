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
        self.expert_to_use, self.expert_to_use_name = None, None
        self.prev_distance = None
        self.round_switch_number = 0

        factory = ExpertFactory()
        experts_list = factory.generate_agents()
        self.experts, self.models, self.scalers, self.training_datas, self.empirical_results, \
            self.n_rounds_since_played, self.expert_counts = {}, {}, {}, {}, {}, {}, {}

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
            self.expert_counts[expert_name] = 0

    def _knn_prediction(self, x: List[float], expert_name: str) -> Tuple[List[float], List[float]]:
        model, scaler, training_data = \
            self.models[expert_name], self.scalers[expert_name], self.training_datas[expert_name]

        x = np.array(x[:-2]).reshape(1, -1)
        x_scaled = scaler.transform(x)
        neighbor_distances, neighbor_indices = model.kneighbors(x_scaled, Utils.KNN_N_NEIGHBORS)
        corrections, distances = [], []

        for i in range(len(neighbor_indices[0])):
            neighbor_idx = neighbor_indices[0][i]
            neighbor_dist = neighbor_distances[0][i]
            corrections.append(training_data[neighbor_idx, -2])
            distances.append(neighbor_dist)

        return corrections, distances

    def update_expert(self, round_num: int, new_assumptions: Assumptions, state: State) -> None:
        self.assumptions_collection.update(new_assumptions)
        new_distance = state.agent_distance(self.name)
        n_steps_closer = max(self.prev_distance - new_distance, 0)
        self.empirical_results[self.expert_to_use_name].append(n_steps_closer)
        row, col = state.agent_positions[self.name]
        prey_row, prey_col = state.agent_positions[Utils.PREY_NAME]

        if round_num > self.round_switch_number and not state.neighbors(row, col, prey_row, prey_col):
            predictions, new_tup = {}, self.assumptions_collection.generate_moving_averages()

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
                    total_pred += (new_distance * cor * distance_weight)

                if len(self.empirical_results[expert_name]) > 0:
                    self.n_rounds_since_played[expert_name] += 1 if expert_name != self.expert_to_use_name else 0
                    prob = self.lmbda ** self.n_rounds_since_played[expert_name]
                    use_empricial_avgs = np.random.choice([1, 0], p=[prob, 1 - prob])

                else:
                    use_empricial_avgs = False

                if use_empricial_avgs:
                    avg_steps = np.array(self.empirical_results[expert_name]).mean()
                    predictions[expert_name] = (total_pred / avg_steps) if avg_steps >= 1 else total_pred

                else:
                    predictions[expert_name] = total_pred

            expert_key = min(predictions, key=lambda key: predictions[key])
            best_key = expert_key
            self.n_rounds_since_played[best_key] = 0
            self.expert_to_use, self.expert_to_use_name = self.experts[best_key], best_key
            self.round_switch_number = round_num + 1

    def act(self, state: State) -> Tuple[int, int]:
        if self.expert_to_use_name is not None:
            self.expert_counts[self.expert_to_use_name] += 1

        self.prev_distance = state.agent_distance(self.name)

        if self.expert_to_use is None:
            self.expert_to_use_name, self.expert_to_use = \
                random.choice(list(zip(list(self.experts.keys()), list(self.experts.values()))))

        return self.expert_to_use.act(state)
