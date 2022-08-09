from agents.prob_dest import ProbabilisticDestinations
from environment.pursuit import Pursuit
import numpy as np


prob_dest1 = ProbabilisticDestinations('ProbDest1')
prob_dest2 = ProbabilisticDestinations('ProbDest2')
prob_dest3 = ProbabilisticDestinations('ProbDest3')
prob_dest4 = ProbabilisticDestinations('ProbDest4')

height, width = 5, 5
predators = [prob_dest1, prob_dest2, prob_dest3, prob_dest4]
n_epochs, rounds = 10000, []
fourths = n_epochs // 4

for epoch in range(n_epochs):
    if (epoch + 1) % fourths == 0:
        print(f'Epoch {epoch + 1} / {n_epochs}')

    pursuit, n_rounds = Pursuit(height, width, predators), 0

    while not pursuit.prey_caught():
        n_rounds += 1
        pursuit.transition()

    rounds.append(n_rounds)

print(f'Average number of rounds needed to catch prey: {np.array(rounds).mean()}')
