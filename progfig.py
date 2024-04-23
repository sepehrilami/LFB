# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:00:28 2023

@author: Nunzio
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle 
import re

# Sample data dictionary (replace this with your actual data)

llm = '7b'

data = pickle.load(open(llm+"_results.txt", "rb" ))
for entry in data:
    freq = []
    for i in data[entry]:
        x = re.search(r"\b[D]\b", i)
        if x:
            freq.append(0)
        else:
            freq.append(1)
        
    data[entry] = freq

llmdict = {'7b':'LLaMa2-7b', '70b':'LLaMa2-70b'}

cont_color_dict = {'delight': '#6393A6', 'prison':'#A66D6D', 'snowdrift':'#BF9E39', 'staghunt':'#6C8C77'}
game_color_dict = {'biz':'#296478', 'team':'#D99C5D', 'IR':'#94579E', 'friendsharing':"#00A3D9", 'environment':"#9FD95D"}

# Extract contexts and games
contexts = sorted(list(set(key.split('_')[0] for key in data.keys())))
games = sorted(list(set(key.split('_')[1] for key in data.keys())))

# Initialize arrays to store average scores and standard deviations for each context and game
averages = np.zeros((len(contexts), len(games)))
std_devs = np.zeros((len(contexts), len(games)))

# Calculate average scores and standard deviations for each context and game
for i, context in enumerate(contexts):
    for j, game in enumerate(games):
        key = f"{context}_{game}"
        if key in data:
            p = np.mean(data[key])
            averages[i, j] = p
            std_devs[i, j] = np.std(data[key])
            std_devs[i, j] = np.sqrt(p*(1-p))/np.sqrt(300)

# Calculate adjusted error values to ensure they're within [0, 1]
adj_std_devs = np.minimum(std_devs, averages)
adj_std_devs = np.minimum(1 - averages, adj_std_devs)

# Create a grouped bar chart with adjusted error bars by context
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.15
index = np.arange(len(contexts))

for j, game in enumerate(games):
    ax.bar(index + j * bar_width, averages[:, j], bar_width, color = cont_color_dict[game], label=game, yerr=adj_std_devs[:, j], capsize=5)


ax.set_xlabel('Contexts')
ax.set_ylabel('Average Rate of Cooperation')
ax.set_ylim(0, 1.05)
ax.set_title('Results by Context, '+llmdict[llm])
ax.set_xticks(index + (len(games) - 1) * bar_width / 2)
ax.set_xticklabels(contexts)
ax.legend(title='Games')

plt.tight_layout()
plt.show()

# save the figure
fig.savefig("contexts_"+llm+".png")

# Create a separate grouped bar chart with adjusted error bars by game
fig, ax = plt.subplots(figsize=(10, 6))

index = np.arange(len(games))

for i, context in enumerate(contexts):
    ax.bar(index + i * bar_width, averages[i], bar_width, label=context, color = game_color_dict[context], yerr=adj_std_devs[i], capsize=5)
    
    

ax.set_xlabel('Games')
ax.set_ylabel('Average Rate of Cooperation')
ax.set_ylim(0, 1.05)
ax.set_title('Results by Game, ' +llmdict[llm])
ax.set_xticks(index + (len(contexts) - 1) * bar_width / 2)
ax.set_xticklabels(games)
ax.legend(title='Contexts')

plt.tight_layout()
plt.show()

# save the figure
fig.savefig("games_"+llm+".png")