from agents.greedy import Greedy
from agents.uct import UCT
from environment.runner import Runner


greedy1 = Greedy('Greedy1')
greedy2 = Greedy('Greedy2')
greedy3 = Greedy('Greedy3')
uct = UCT('UCT', 100, [agent.name for agent in [greedy1, greedy2, greedy3]])

predators = [greedy1, greedy2, greedy3, uct]

Runner.run(predators, n_epochs=10)
