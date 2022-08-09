from agents.greedy_prob import GreedyProbabilistic
from environment.pursuit import Pursuit
import numpy as np


greedy_prob1 = GreedyProbabilistic('GreedyProb1')
greedy_prob2 = GreedyProbabilistic('GreedyProb2')
greedy_prob3 = GreedyProbabilistic('GreedyProb3')
greedy_prob4 = GreedyProbabilistic('GreedyProb4')

height, width = 5, 5
predators = [greedy_prob1, greedy_prob2, greedy_prob3, greedy_prob4]
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
