from agents.prob_dest import ProbabilisticDestinations
from environment.runner import Runner

prob_dest1 = ProbabilisticDestinations('ProbDest1')
prob_dest2 = ProbabilisticDestinations('ProbDest2')
prob_dest3 = ProbabilisticDestinations('ProbDest3')
prob_dest4 = ProbabilisticDestinations('ProbDest4')

predators = [prob_dest1, prob_dest2, prob_dest3, prob_dest4]

Runner.run(predators, n_epochs=1000)



