from agents.greedy import Greedy
from environment.pursuit import Pursuit
import numpy as np


greedy1 = Greedy('Greedy1')
greedy2 = Greedy('Greedy2')
greedy3 = Greedy('Greedy3')
greedy4 = Greedy('Greedy4')

height, width = 5, 5
predators = [greedy1, greedy2, greedy3, greedy4]
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
