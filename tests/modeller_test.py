from agents.modeller import Modeller
from environment.runner import Runner


modeller1 = Modeller('Modeller1')
modeller2 = Modeller('Modeller2')
modeller3 = Modeller('Modeller3')
modeller4 = Modeller('Modeller4')

predators = [modeller1, modeller2, modeller3, modeller4]

Runner.run(predators, n_epochs=1000)
