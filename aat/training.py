from aat.assumptions import distance_function
from environment.runner import Runner
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from utils.baselines import Baselines
from utils.factory import ExpertFactory, SpecificExpertFactory, RandomExpertFactory
from utils.utils import Utils


# Train each expert against:
#   A team where all 4 members are itself
#   A team with itself and the other 3 members are the same instance of one of the other experts
#   A team with itself and the other 3 members are selected randomly from the expert pool
# For each training phase, train on different grid dimensions

data_dir = './training_data'
training_data = {}
training_phases = [1, 2, 3]
dimensions = [(5, 5), (10, 10)]
expert_factory = ExpertFactory()

# Iterate through the different training phases
for phase in training_phases:
    print(f'PHASE {phase}')
    experts_to_train = expert_factory.generate_agents()

    for expert in experts_to_train:
        print(expert.name)
        expert_data, baseline = [], Baselines.baseline(expert)

        for height, width in dimensions:
            for epoch in range(Utils.TRAINING_EPOCHS):
                if phase == 1:
                    factory = SpecificExpertFactory('S', type(expert))

                elif phase == 2:
                    teammates_class = RandomExpertFactory.select_random_class()
                    factory = SpecificExpertFactory('S', teammates_class)

                else:
                    factory = RandomExpertFactory('R')

                predators = [expert] + factory.generate_agents()

                nested_assumptions = Runner.run(predators, n_epochs=1, height=height, width=width,
                                                report_assumptions=True)
                assert len(nested_assumptions) == 1  # Sanity check
                assumptions = nested_assumptions[0]

                n_rounds = len(assumptions)
                tups = [assumption.generate_tuple(baseline) for i, assumption in enumerate(assumptions)]

                # Adjust the correction term in the tuples
                for tup in tups:
                    tup[-1] = n_rounds / tup[-1]

                expert_data.extend(tups)

        training_data[expert.name] = training_data.get(expert.name, []) + expert_data

# Save the training data
for expert_name, data in training_data.items():
    with open(f'{data_dir}/{expert_name}_training_data.pickle', 'wb') as f:
        pickle.dump(data, f)

# Train and save KNN models for each expert
for expert in expert_factory.generate_agents():
    expert_name = expert.name

    with open(f'{data_dir}/{expert_name}_training_data.pickle', 'rb') as f:
        training_data = np.array(pickle.load(f))

    x = training_data[:, 0:-1]

    print('X train shape: ' + str(x.shape))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = NearestNeighbors(n_neighbors=Utils.KNN_N_NEIGHBORS, metric=distance_function)
    model.fit(x_scaled)

    trained_knn_file = f'{expert_name}_trained_knn_aat.pickle'
    trained_knn_scaler_file = f'{expert_name}_trained_knn_scaler_aat.pickle'

    with open(f'{data_dir}/{trained_knn_file}', 'wb') as f:
        pickle.dump(model, f)

    with open(f'{data_dir}/{trained_knn_scaler_file}', 'wb') as f:
        pickle.dump(scaler, f)



