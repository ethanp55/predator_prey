import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


names = ['mixed']
dimensions = [(5, 5), (10, 10), (15, 15)]
font_size = 18

for name in names:
    for height, width in dimensions:
        df = pd.read_csv(f'./results/{name}_{height}_{width}.csv')
        agent_names = list(df['Agent'].unique())
        mean_results, sd_results = [], []

        for agent_name in agent_names:
            mean_results.append(df.loc[df['Agent'] == agent_name]['Rewards'].mean())
            sd_results.append(df.loc[df['Agent'] == agent_name]['Rewards'].sem())

        x_pos = np.arange(len(agent_names))
        plt.bar(x_pos, mean_results, yerr=sd_results, ecolor='black', capsize=5, align='center', alpha=0.5,
                color=['green', 'red', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'black'])
        plt.xticks(x_pos, agent_names, fontsize=30, rotation=45, ha='right')
        # plt.xlabel('Algorithm', fontsize=font_size)
        if (height, width) == (5, 5):
            plt.ylabel('Rounds to Surround Prey', fontsize=26)
        # plt.rc('ytick', labelsize=font_size)
        plt.title(f'{width} x {height} Grid', fontsize=34)
        plt.savefig(f'./results/images/{name}_{height}_{width}_errorbars.png', bbox_inches='tight')
        plt.clf()
