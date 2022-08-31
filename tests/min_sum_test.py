from agents.min_sum import MinSum
from environment.runner import Runner


min_sum1 = MinSum('MinSum1')
min_sum2 = MinSum('MinSum2')
min_sum3 = MinSum('MinSum3')
min_sum4 = MinSum('MinSum4')

predators = [min_sum1, min_sum2, min_sum3, min_sum4]

Runner.run(predators, n_epochs=100, height=10, width=10)
