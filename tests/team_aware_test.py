from agents.team_aware import TeamAware
from environment.pursuit import Pursuit
import numpy as np


team_aware1 = TeamAware('TeamAware1')
team_aware2 = TeamAware('TeamAware2')
team_aware3 = TeamAware('TeamAware3')
team_aware4 = TeamAware('TeamAware4')

height, width = 5, 5
predators = [team_aware1, team_aware2, team_aware3, team_aware4]
n_epochs, rounds = 1, []
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
