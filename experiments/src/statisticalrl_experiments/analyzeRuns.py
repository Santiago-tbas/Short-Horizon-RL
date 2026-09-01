
import pickle
import time
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)
from epsilon_regret import compute_epsilon, compute_epsilon_regret
#
# from src.statisticalrl_experiments.utils import get_project_root_dir
# ROOT= get_project_root_dir()+"/src/"
#
# ROOT = "results/"


def computeCumulativeRegrets(names, dump_cumulativerewards_, timeHorizon, envName, root_folder,
                              gaps=None, sigma=1.0, eta=0.5):
    """

    :param names: get list of algorithm names
    :param dump_cumulativerewards_: list of filenames, each getting cumulative rewards for multiple runs. Last file of the list is cum reward of Oracle.
    :param timeHorizon:
    :param envName:
    :param gaps: if provided, also compute practitioner epsilon-regret using these sub-optimality gaps
    :param sigma: noise scale for epsilon computation
    :param eta: tolerance for epsilon computation
    :return: vectors median, quantile0.25, quantile0.75, timesteps, where median[i] is median of expreimnts at time timesteps[i]
    """
    median = []
    mean = []
    quantile1 = []
    quantile2 = []
    nbAlgs = len(dump_cumulativerewards_) - 1

    # Optional: compute epsilon* for practitioner regret
    epsilon = None
    if gaps is not None:
        epsilon, c_T = compute_epsilon(gaps, timeHorizon, sigma, eta)
        print(f"[INFO] Optimal epsilon* = {epsilon:.4f}, c_T = {c_T:.4f}")

    #Downsample the times, especially in case timeHorizon is huge.
    skip = max(1, (timeHorizon // 1000))
    times = [t for t in range(0,timeHorizon,skip)]

    for j in range(nbAlgs):
        data_j = []
        data_eps_j = []  # epsilon-regret data
        for i in range(len(dump_cumulativerewards_[j])):
            file_oracle = open(dump_cumulativerewards_[-1], 'rb')
            cum_rewards_oracle = pickle.load(file_oracle)
            cum_rewards_oracle = cum_rewards_oracle[0]
            file = open(dump_cumulativerewards_[j][i], 'rb')
            cum_rewards_ij = pickle.load(file)
            data_j.append([cum_rewards_oracle[t] - cum_rewards_ij[t] for t in range(0,timeHorizon,skip)])
            # Epsilon-regret: adjust oracle baseline by epsilon*
            if epsilon is not None:
                data_eps_j.append([(cum_rewards_oracle[t] - epsilon * (t + 1)) - cum_rewards_ij[t] for t in range(0,timeHorizon,skip)])
            file_oracle.close()
            file.close()

        filename = root_folder+"cumRegret_" + envName + "_" + names[j] + "_" + str(timeHorizon) + "_" + str(
            j) + "_" + str(
            time.time())
        file = open(filename, 'wb')
        pickle.dump(data_j, file)
        file.close()

        # Save epsilon-regret separately
        if epsilon is not None:
            filename_eps = root_folder + "cumEpsRegret_" + envName + "_" + names[j] + "_" + str(timeHorizon) + "_" + str(
                j) + "_" + str(time.time())
            file_eps = open(filename_eps, 'wb')
            pickle.dump(data_eps_j, file_eps)
            file_eps.close()

        mean.append(np.mean(data_j, axis=0))
        median.append(np.quantile(data_j, 0.5, axis=0))
        quantile1.append(np.quantile(data_j, 0.25, axis=0))
        quantile2.append(np.quantile(data_j, 0.75, axis=0))

    return mean,median,quantile1,quantile2,times


def computeCumulativeGaps(names, dump_cumulativegaps_, timeHorizon, envName, root_folder):
    """

    :param names: get list of algorithm names
    :param dump_cumulativerewards_: list of filenames, each getting cumulative rewards for multiple runs.
    :param timeHorizon:
    :param envName:
    :return: vectors median, quantile0.25, quantile0.75, timesteps, where median[i] is median of expreimnts at time timesteps[i]
    """
    median = []
    mean = []
    quantile1 = []
    quantile2 = []
    nbAlgs = len(dump_cumulativegaps_)

    #Downsample the times, especially in case timeHorizon is huge.
    skip = max(1, (timeHorizon // 1000))
    times = [t for t in range(0,timeHorizon,skip)]

    for j in range(nbAlgs):
        data_j = []
        for i in range(len(dump_cumulativegaps_[j])):
            file = open(dump_cumulativegaps_[j][i], 'rb')
            cum_gaps_ij = pickle.load(file)
            data_j.append([cum_gaps_ij[t] for t in range(0,timeHorizon,skip)])
            file.close()

        filename = root_folder+"cumGap_" + envName + "_" + names[j] + "_" + str(timeHorizon) + "_" + str(
            j) + "_" + str(
            time.time())
        file = open(filename, 'wb')
        pickle.dump(data_j, file)
        file.close()

        mean.append(np.mean(data_j, axis=0))
        median.append(np.quantile(data_j, 0.5, axis=0))
        quantile1.append(np.quantile(data_j, 0.25, axis=0))
        quantile2.append(np.quantile(data_j, 0.75, axis=0))

    return mean,median,quantile1,quantile2,times