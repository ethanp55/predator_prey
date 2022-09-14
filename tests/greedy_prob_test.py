from agents.greedy_prob import GreedyProbabilistic
from environment.runner import Runner


greedy_prob1 = GreedyProbabilistic('GreedyProb1')
greedy_prob2 = GreedyProbabilistic('GreedyProb2')
greedy_prob3 = GreedyProbabilistic('GreedyProb3')
greedy_prob4 = GreedyProbabilistic('GreedyProb4')

predators = [greedy_prob1, greedy_prob2, greedy_prob3, greedy_prob4]

Runner.run(predators, n_epochs=1000)
