from agents.alegaatr import Alegaatr
from agents.greedy import Greedy
from agents.greedy_planner import GreedyPlanner
from agents.greedy_prob import GreedyProbabilistic
from agents.prob_dest import ProbabilisticDestinations
from agents.min_sum import MinSum
from agents.modeller import Modeller
from agents.team_aware import TeamAware
from environment.runner import Runner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.factory import SpecificExpertFactory, RandomSelectionFactory


alegaatr = Alegaatr()
greedy = Greedy('Greedy')
greedy_planner = GreedyPlanner('GreedyPlanner')
team_aware = TeamAware('TeamAware')
min_sum = MinSum('MinSum')
modeller = Modeller('Modeller')
greedy_prob = GreedyProbabilistic('GreedyProb')
prob_dest = ProbabilisticDestinations('ProbDest')

test_agents = [alegaatr, greedy, greedy_planner, team_aware, min_sum, modeller, greedy_prob, prob_dest]
agent_names = [agent.name for agent in test_agents]
dimensions = [(5, 5), (10, 10), (15, 15)]

N_EPOCHS = 50

# Deterministic
print('DETERMINISTIC')
for height, width in dimensions:
    print((height, width))
    greedy_factory = SpecificExpertFactory('S', Greedy)
    greedy_planner_factory = SpecificExpertFactory('S', GreedyPlanner)
    team_aware_factory = SpecificExpertFactory('S', TeamAware)
    min_sum_factory = SpecificExpertFactory('S', MinSum)

    factories = [('greedy', greedy_factory), ('greedy_planner', greedy_planner_factory),
                 ('team_aware', team_aware_factory), ('min_sum', min_sum_factory)]

    for label, factory in factories:
        print(label)
        results, names = [], []

        for _ in range(N_EPOCHS):
            for agent in test_agents:
                team = factory.generate_agents()
                predators = [agent] + team

                _, num_rounds = Runner.run(predators, suppress_output=True, n_epochs=1, height=height, width=width)
                results.append(num_rounds)
                names.append(agent.name)

        df = pd.DataFrame({'Rewards': results, 'Agent': names})
        df.to_csv(f'./results/deterministic_{label}_{height}_{width}.csv')

        mean_results = []

        for name in agent_names:
            mean_results.append(df.loc[df['Agent'] == name]['Rewards'].mean())

        x_pos = np.arange(len(agent_names))
        plt.bar(x_pos, mean_results, align='center', alpha=0.5,
                color=['green', 'red', 'blue', 'orange', 'purple', 'yellow', 'cyan'])
        plt.xticks(x_pos, agent_names, fontsize=6)
        plt.xlabel('Algorithm')
        plt.ylabel('# Rounds to Surround Prey')
        plt.title(f'Team Composed of {label} Agents, {width} x {height} Grid')
        plt.savefig(f'./results/images/deterministic_{label}_{height}_{width}.png', bbox_inches='tight')
        plt.clf()

# Non-deterministic
print('NON-DETERMINISTIC')
for height, width in dimensions:
    print((height, width))
    greedy_prob_factory = SpecificExpertFactory('S', GreedyProbabilistic)
    modeller_factory = SpecificExpertFactory('S', Modeller)
    prob_dest_factory = SpecificExpertFactory('S', ProbabilisticDestinations)

    factories = [('greedy_prob', greedy_prob_factory), ('modeller', modeller_factory), ('prob_dest', prob_dest_factory)]

    for label, factory in factories:
        print(label)
        results, names = [], []

        for _ in range(N_EPOCHS):
            for agent in test_agents:
                team = factory.generate_agents()
                predators = [agent] + team

                _, num_rounds = Runner.run(predators, suppress_output=True, n_epochs=1, height=height, width=width)
                results.append(num_rounds)
                names.append(agent.name)

        df = pd.DataFrame({'Rewards': results, 'Agent': names})
        df.to_csv(f'./results/nondeterministic_{label}_{height}_{width}.csv')

        mean_results = []

        for name in agent_names:
            mean_results.append(df.loc[df['Agent'] == name]['Rewards'].mean())

        x_pos = np.arange(len(agent_names))
        plt.bar(x_pos, mean_results, align='center', alpha=0.5,
                color=['green', 'red', 'blue', 'orange', 'purple', 'yellow', 'cyan'])
        plt.xticks(x_pos, agent_names, fontsize=6)
        plt.xlabel('Algorithm')
        plt.ylabel('# Rounds to Surround Prey')
        plt.title(f'Team Composed of {label} Agents, {width} x {height} Grid')
        plt.savefig(f'./results/images/nondeterministic_{label}_{height}_{width}.png', bbox_inches='tight')
        plt.clf()

# Mixed
print('MIXED')
for height, width in dimensions:
    print((height, width))
    agent_types = [type(agent) for agent in test_agents]
    random_selection_factory = RandomSelectionFactory('RS', agent_types)
    results, names = [], []

    for _ in range(N_EPOCHS):
        for agent in test_agents:
            team = random_selection_factory.generate_agents()
            predators = [agent] + team

            _, num_rounds = Runner.run(predators, suppress_output=True, n_epochs=1, height=height, width=width)
            results.append(num_rounds)
            names.append(agent.name)

        df = pd.DataFrame({'Rewards': results, 'Agent': names})
        df.to_csv(f'./results/mixed_{height}_{width}.csv')

        mean_results = []

        for name in agent_names:
            mean_results.append(df.loc[df['Agent'] == name]['Rewards'].mean())

        x_pos = np.arange(len(agent_names))
        plt.bar(x_pos, mean_results, align='center', alpha=0.5,
                color=['green', 'red', 'blue', 'orange', 'purple', 'yellow', 'cyan'])
        plt.xticks(x_pos, agent_names, fontsize=6)
        plt.xlabel('Algorithm')
        plt.ylabel('# Rounds to Surround Prey')
        plt.title(f'Team Composed of Mixed Agents, {width} x {height} Grid')
        plt.savefig(f'./results/images/mixed_{height}_{width}.png', bbox_inches='tight')
        plt.clf()
