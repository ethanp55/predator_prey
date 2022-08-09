from agents.team_aware import TeamAware
from environment.runner import Runner


team_aware1 = TeamAware('TeamAware1')
team_aware2 = TeamAware('TeamAware2')
team_aware3 = TeamAware('TeamAware3')
team_aware4 = TeamAware('TeamAware4')

predators = [team_aware1, team_aware2, team_aware3, team_aware4]

Runner.run(predators)
