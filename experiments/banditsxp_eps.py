"""Experiment: practitioner epsilon-regret for UCB and TS."""

import numpy as np
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'learners', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'environments', 'src'))

import statisticalrl_environments as srl
from statisticalrl_environments.register import make

from statisticalrl_experiments.fullExperiment import runLargeMulticoreExperiment as xp
import statisticalrl_environments.register as bW

from epsilon_regret import compute_epsilon

# Create environment
env = make('mab-bernoulli')
nA = env.action_space.n

# Extract gaps for epsilon computation
means = env.means
best = max(means)
gaps = [best - m for m in means]
print(f"Arm means: {means}")
print(f"Sub-optimality gaps: {gaps}")

# Import learners
from statisticalrl_learners.Generic.Random import Random as rd
from statisticalrl_learners.MABs.UCB import UCB as ucb
from statisticalrl_learners.MABs.TS import TS as ts
from statisticalrl_learners.MABs.Oracle import Oracle as ord

# Configure agents
agents = []
agents.append([ts, {"nbArms": nA}])
agents.append([ucb, {"nbArms": nA, "delta": lambda t: 0.05}])

# Oracle
oracle = ord(env)

# Compute epsilon* for reference
T = 1000
eps_opt, c_T = compute_epsilon(gaps, T, sigma=1.0, eta=0.5)
print(f"\nOptimal epsilon* = {eps_opt:.4f}, c_T = {c_T:.4f}")
print(f"Practitioner benchmark: mu* - eps* = {best - eps_opt:.4f}")

# Run experiment — computeCumulativeRegrets will automatically compute
# epsilon-regret when gaps are provided
xp(env, agents, oracle, timeHorizon=T, nbReplicates=32, root_folder="results_mab/")
