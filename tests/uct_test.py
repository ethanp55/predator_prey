from agents.team_aware import TeamAware
from agents.uct import UCT
from environment.runner import Runner


team_aware1 = TeamAware('TeamAware1')
team_aware2 = TeamAware('TeamAware2')
team_aware3 = TeamAware('TeamAware3')
uct = UCT('UCT', 100, [agent.name for agent in [team_aware1, team_aware2, team_aware3]])

predators = [team_aware1, team_aware2, team_aware3, uct]

Runner.run(predators, n_epochs=10, height=5, width=5)
