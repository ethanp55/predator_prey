from agents.greedy_planner import GreedyPlanner
from environment.runner import Runner


greedy_planner1 = GreedyPlanner('GreedyPlanner1')
greedy_planner2 = GreedyPlanner('GreedyPlanner2')
greedy_planner3 = GreedyPlanner('GreedyPlanner3')
greedy_planner4 = GreedyPlanner('GreedyPlanner4')

predators = [greedy_planner1, greedy_planner2, greedy_planner3, greedy_planner4]

Runner.run(predators, n_epochs=1000)
