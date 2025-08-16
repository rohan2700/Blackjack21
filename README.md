# Reinforcement Learning - BlackJack Player

This repository provides a full implementation of reinforcement learning algorithms applied to BlackJack, with special attention to real-world casino rules and agent learning from scratch.

## Overview

A Jupyter Notebook simulates BlackJack as a complex Markov Decision Process (MDP). The environment supports multiple deck shoes, realistic card dealing, and several casino rule variations. It assesses RL algorithms under different scenarios, such as basic play, card counting, late surrender, and special rule adjustments.

The project answers:

- Can RL agents learn optimal BlackJack strategies purely from experience?
- What is the comparative performance of Monte Carlo, Q-Learning, and Double Q-Learning?
- Does card counting yield improvements for an RL-driven player?
- How do critical rule changes shift player outcomes and agent behavior?


## Features

- **Configurable BlackJack Environment:** Multi-deck support, configurable rules, realistic action space.
- **Advanced Rule Variations:**
    - Basic Strategy (standard rules)
    - Card Counting (Hi-Lo system)
    - Late Surrender
    - Player 21 Always Wins
- **Legal Action Masking:** Only permissible actions allowed at each state.
- **Performance Visualization:** Heatmaps, training curves, win/loss/reward breakdowns.


## Implemented Algorithms

- **Monte Carlo Control**: First-visit updates, episodic-based learning with ϵ-greedy exploration.
- **Q-Learning**: Bootstrapping, TD(0), decaying ϵ-greedy.
- **Double Q-Learning**: Two Q-tables to reduce overestimation bias, robust in sparse/noisy reward environments.


## State and Action Spaces

- **State:** (player value, dealer up-card, usable ace, [true count bin, if counting])
- **Action:**
    - Stand
    - Hit
    - Double Down (if allowed)
    - Split (simplified, if allowed)
    - Surrender (if enabled)

Typical state-space size:

- Basic: ~1,800 states
- Counting: ~9,000 states


## Evaluation Metrics

- Win/Loss/Push Rates
- Average Reward per Hand
- Natural BlackJack Frequency
- Surrender Usage Rate
- Action Distribution Histograms
- Training Convergence Curves


## Requirements

- Python 3.7+
- Jupyter Notebook or JupyterLab
- numpy
- matplotlib
- seaborn
- tqdm


## How to Run

1. Clone or download this repository.
2. Open `TaskP3.1.ipynb` in Jupyter Notebook.
3. Run the notebook sequentially:
    - Environments are initialized with rule sets.
    - RL agents are trained and evaluated.
    - Visualizations of learned strategies and agent performance are generated.

*Note: Training deep RL agents for millions of episodes may take several hours.*

## Directory Contents

```
├── TaskP3.1.ipynb      # Main notebook: env, training, evaluation, and visualization
├── blackjack_env.py          # BlackJack environment with RL extensions
├── TaskP3.2.pdf                   # Research report explaining theory, implementation, and results
└── README.md                 # This file
```


## Key Results

- Double Q-Learning achieved the most stable and optimal reward.
- Card counting provided slight gains but with high computational requirements.
- RL agents matched or closely tracked established BlackJack “basic strategy” charts.
- By toggling rules, agents reliably adapted play to exploit or mitigate new advantages.


## Credits

**Rohan Sanjay Patil**
Master Artificial Intelligence, Faculty of Computer Science
Technical University of Applied Sciences, Würzburg-Schweinfurt
rohansanjay.patil@study.thws.de

*For a detailed theoretical and experimental discussion, consult `TaskP3.2.pdf`.*
