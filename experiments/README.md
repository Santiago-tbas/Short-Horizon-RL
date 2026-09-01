# Experiments
Experimentations with Learning Agents run on Statistial Reinforcement Learning environments.


## Installation
    pip install statisticalRL-experiments

# Usage


    from statisticalrl_experiments.fullExperiment import runLargeMulticoreExperiment as xp
    
    #######################
    # Import registered environments
    import statisticalrl_environments.register as bW
    #######################
    
    # Instantiate one environment
    env = bW.make('river-swim-6')
    nS = env.observation_space.n
    nA = env.action_space.n
    
    
    #######################
    # Import some learners
    from statisticalrl_learners.Generic.Random import Random as rd
    from statisticalrl_learners.Generic.Qlearning import Qlearning as ql
    from statisticalrl_learners.MDPs_discrete.UCRL3 import UCRL3_lazy as ucrl3
    from statisticalrl_learners.MDPs_discrete.IMED_RL import IMEDRL as imedrl
    import statisticalrl_learners.MDPs_discrete.Optimal.OptimalControl  as opt
    
    #######################
    # List a few learners to be compared:
    agents = []
    agents.append( [rd, {"env": env}])
    agents.append( [ql, {"nS":nS, "nA":nA}])
    agents.append( [ucrl3, {"nS":nS, "nA":nA, "delta":0.05}])
    agents.append(([imedrl, {"nbr_states":nS, "nbr_actions":nA}]))
        
    #############################
    # Compute oracle policy:
    oracle = opt.build_opti(env.name, env, env.observation_space.n, env.action_space.n)
    
    #######################
    # Run a full experiment
    # This function produces all results including plots, logs, etc in the folder "root_folder" 
    #######################
    xp(env, agents, oracle, timeHorizon=1000, nbReplicates=16,root_folder="results/")
    
    #######################
    # Plotting Regret directly from dump files of past runs (here until time horizon tplot=500):
    #######################    
    #from statisticalrl_experiments.plotResults import search_dump_cumRegretfiles, plot_results_from_dump
    #files = search_dump_cumRegretfiles("RiverSwim-S6-v0", root_folder="results/")
    #if files:
    #    plot_results_from_dump(files, tplot=500)
