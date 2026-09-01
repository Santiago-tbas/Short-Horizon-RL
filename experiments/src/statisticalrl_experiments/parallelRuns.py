import gymnasium
import time
import copy
from joblib import Parallel, delayed

import multiprocessing


## Parallelization
def multicoreRuns(envRegisterName, learner, nbReplicates, timeHorizon, oneRunFunction, root_folder):
    num_cores = multiprocessing.cpu_count()
    envs = []
    learners = []
    timeHorizons = []
    rootFolders= []

    for i in range(nbReplicates):
        envs.append(gymnasium.make(envRegisterName).unwrapped)
        learners.append(copy.deepcopy(learner))
        timeHorizons.append(copy.deepcopy(timeHorizon))
        rootFolders.append(root_folder)

    t0 = time.time()

    cumRewards = Parallel(n_jobs=num_cores)(delayed(oneRunFunction)(*i) for i in zip(envs,learners,timeHorizons, rootFolders))

    elapsed = time.time()-t0
    return cumRewards, elapsed / nbReplicates


