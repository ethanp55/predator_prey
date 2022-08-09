from agents.greedy import Greedy
from environment.runner import Runner


greedy1 = Greedy('Greedy1')
greedy2 = Greedy('Greedy2')
greedy3 = Greedy('Greedy3')
greedy4 = Greedy('Greedy4')

predators = [greedy1, greedy2, greedy3, greedy4]

Runner.run(predators)
