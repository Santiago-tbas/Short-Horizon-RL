
import statisticalrl_environments as srl
from statisticalrl_environments.register import make

from statisticalrl_environments.fulldemo import print_registered_environments,random_environment,all_environments

# List register environments:
print_registered_environments()
# random_environment()
# all_environments()

from statisticalrl_experiments.fullExperiment import runLargeMulticoreExperiment as xp

#######################
# Import registered environments
import statisticalrl_environments.register as bW

#######################

# Instantiate one environment
env = make('mab-bernoulli')
#env = bW.make('river-swim-6')
#env = bW.make('random-12')
nS = env.observation_space.n
nA = env.action_space.n

#######################
# Import some learners
from statisticalrl_learners.Generic.Random import Random as rd
from statisticalrl_learners.MABs.UCB import UCB as ucb
#from statisticalrl_learners.MABs.FTL import FTL as ftl
from statisticalrl_learners.MABs.TS import TS as ts
from statisticalrl_learners.MABs.IMED import IMED as imed
from statisticalrl_learners.MABs.utils import klBern
from statisticalrl_learners.MABs.Oracle import Oracle as ord

#######################
# List a few learners to be compared:
agents = []
agents.append([rd, {"env": env}])
agents.append([ts, {"nbArms": nA}])
agents.append([ucb, {"nbArms": nA, "delta": lambda t: 0.05}])
agents.append(([imed, {"nbArms":nA,"kullback":klBern}]))

#############################
# Compute oracle policy:
oracle = ord(env)#.build_opti(env.name, env, env.observation_space.n, env.action_space.n)

#######################
# Run a full experiment
# This function produces all results including plots, logs, etc in the folder "root_folder"
#######################
xp(env, agents, oracle, timeHorizon=1000, nbReplicates=32, root_folder="results_mab/")

#######################
# Plotting Regret directly from dump files of past runs (here until time horizon tplot=500):
#######################
# from statisticalrl_experiments.plotResults import search_dump_cumRegretfiles, plot_results_from_dump
# files = search_dump_cumRegretfiles("RiverSwim-S6-v0", root_folder="results/")
# if files:
#    plot_results_from_dump(files, tplot=500)