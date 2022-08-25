from agents.random import Random
from environment.runner import Runner


random1 = Random('Random1')
random2 = Random('Random2')
random3 = Random('Random3')
random4 = Random('Random4')

predators = [random1, random2, random3, random4]

Runner.run(predators, n_epochs=100)
