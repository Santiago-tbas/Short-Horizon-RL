import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import gymnasium

# Ensure the library can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../learners/src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../environments/src')))

import statisticalrl_environments as srl
from statisticalrl_environments.register import make
from statisticalrl_learners.MABs.UCB import UCB as ucb
from statisticalrl_learners.MABs.TS import TS as ts


def compute_epsilon(gaps, T, sigma=0.5, eta=0.5):
    """Computes the optimal precision boundary epsilon* using the sweep-line relaxation.
    
    sigma is set to 0.5 by default for Bernoulli bandits (0.5-sub-Gaussian).
    """
    c_T = np.sqrt((2 * sigma**2 * np.log(T)) / (eta * T))
    sorted_unique_gaps = np.sort(np.unique(gaps))

    epsilon = 0.0
    for i in range(len(sorted_unique_gaps) - 1):
        if sorted_unique_gaps[i+1] - sorted_unique_gaps[i] <= c_T:
            epsilon = sorted_unique_gaps[i+1]
        else:
            break

    return epsilon, c_T


def oneXpPractitioner(env_name, learner_class, learner_params, timeHorizon, epsilon):
    """Runs a single simulation replicate, tracking standard and practitioner regrets."""
    # Re-import inside worker to ensure everything is initialized correctly
    import gymnasium
    import statisticalrl_environments as srl
    from statisticalrl_environments.register import make

    # Recreate the environment for this process
    env = make(env_name)
    learner = learner_class(**learner_params)
    
    observation, info = env.reset()
    learner.reset(observation)
    
    cum_std_regret = 0.0
    cum_std_regrets = []
    
    cum_eps_regret_a = 0.0  # Option A: sum(Delta_a - epsilon)
    cum_eps_regrets_a = []
    
    cum_eps_regret_b = 0.0  # Option B: sum(max(Delta_a - epsilon, 0))
    cum_eps_regrets_b = []
    
    best_mean = max(env.means)
    
    for t in range(timeHorizon):
        state = observation
        action = learner.play(state)
        observation, reward, done, truncated, info = env.step(action)
        learner.update(state, action, reward, observation)
        
        # Extract true mean of chosen arm
        mean_reward = info.get("mean", reward)
        gap = best_mean - mean_reward
        
        # 1. Standard regret
        cum_std_regret += gap
        cum_std_regrets.append(cum_std_regret)
        
        # 2. Option A: Delta - epsilon
        cum_eps_regret_a += (gap - epsilon)
        cum_eps_regrets_a.append(cum_eps_regret_a)
        
        # 3. Option B: max(Delta - epsilon, 0)
        cum_eps_regret_b += max(gap - epsilon, 0.0)
        cum_eps_regrets_b.append(cum_eps_regret_b)
        
        if done:
            observation, info = env.reset()
            
    return {
        "std_regrets": cum_std_regrets,
        "eps_regrets_a": cum_eps_regrets_a,
        "eps_regrets_b": cum_eps_regrets_b
    }


def run_experiment(env_name='mab-bernoulli', timeHorizon=1000, nbReplicates=32, root_folder="result_pract_regret/"):
    os.makedirs(root_folder, exist_ok=True)
    
    # Initialize env once to extract characteristics
    env = make(env_name)
    nA = env.action_space.n
    means = env.means
    best_mean = max(means)
    gaps = [best_mean - m for m in means]
    
    print(f"\n[EXPERIMENT] Environment: {env.name}")
    print(f"[EXPERIMENT] Arm means: {means}")
    print(f"[EXPERIMENT] Sub-optimality gaps: {gaps}")
    
    # Standard sub-Gaussian parameter for Bernoulli is 0.5
    sigma = 0.5
    eta = 0.5
    epsilon, c_T = compute_epsilon(gaps, timeHorizon, sigma=sigma, eta=eta)
    print(f"[EXPERIMENT] Time Horizon: {timeHorizon}")
    print(f"[EXPERIMENT] Computed optimal epsilon* = {epsilon:.4f} (c_T = {c_T:.4f})")
    
    agents_config = [
        {"class": ucb, "params": {"nbArms": nA, "delta": lambda t: 0.05}, "name": "UCB"},
        {"class": ts, "params": {"nbArms": nA}, "name": "TS"}
    ]
    
    results = {}
    num_cores = multiprocessing.cpu_count()
    
    for agent in agents_config:
        name = agent["name"]
        print(f"[EXPERIMENT] Running {nbReplicates} replicates for agent {name} in parallel...")
        
        # Run parallel jobs
        replicates_data = Parallel(n_jobs=num_cores)(
            delayed(oneXpPractitioner)(env_name, agent["class"], agent["params"], timeHorizon, epsilon)
            for _ in range(nbReplicates)
        )
        
        # Aggregate
        std_list = np.array([r["std_regrets"] for r in replicates_data])
        eps_a_list = np.array([r["eps_regrets_a"] for r in replicates_data])
        eps_b_list = np.array([r["eps_regrets_b"] for r in replicates_data])
        
        results[name] = {
            "std": std_list,
            "eps_a": eps_a_list,
            "eps_b": eps_b_list
        }
        
        # Save raw pickled data
        with open(os.path.join(root_folder, f"raw_data_{name}.pkl"), "wb") as f:
            pickle.dump(results[name], f)
            
    # Save experiment configuration metadata
    metadata = {
        "env_name": env_name,
        "means": means,
        "gaps": gaps,
        "epsilon": epsilon,
        "c_T": c_T,
        "timeHorizon": timeHorizon,
        "nbReplicates": nbReplicates
    }
    with open(os.path.join(root_folder, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
        
    print("[EXPERIMENT] Simulation completed. Generating plots...")
    plot_results(results, metadata, root_folder)


def plot_results(results, metadata, root_folder):
    T = metadata["timeHorizon"]
    epsilon = metadata["epsilon"]
    c_T = metadata["c_T"]
    times = np.arange(1, T + 1)
    
    # Setup beautiful aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    
    metrics = [
        {"key": "std", "title": "Standard Pseudo-Regret", "ylabel": r"Cumulative Regret $\sum \Delta_{a_t}$", "filename": "standard_regret.png"},
        {"key": "eps_a", "title": f"Epsilon-Adjusted Regret (Option A, $\\epsilon^*={epsilon:.3f}$)", "ylabel": r"Adjusted Regret $\sum (\Delta_{a_t} - \epsilon^*)$", "filename": "adjusted_regret_option_a.png"},
        {"key": "eps_b", "title": f"Thresholded Epsilon-Regret (Option B, $\\epsilon^*={epsilon:.3f}$)", "ylabel": r"Thresholded Regret $\sum \max(\Delta_{a_t} - \epsilon^*, 0)$", "filename": "thresholded_regret_option_b.png"}
    ]
    
    colors = {"UCB": "#1f77b4", "TS": "#ff7f0e"}
    
    for metric in metrics:
        key = metric["key"]
        fig = plt.figure(figsize=(10, 6), dpi=150)
        
        for name, data_dict in results.items():
            data = data_dict[key]
            
            # Compute stats
            median = np.median(data, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            
            plt.plot(times, median, label=name, color=colors[name], linewidth=2.5)
            plt.fill_between(times, q25, q75, color=colors[name], alpha=0.15)
            
        plt.title(f"{metric['title']}\nEnvironment: {metadata['env_name']} | Horizon: {T}", fontsize=14, fontweight='bold', pad=10)
        plt.xlabel("Time steps", fontsize=12)
        plt.ylabel(metric["ylabel"], fontsize=12)
        plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="gainsboro", fontsize=11)
        
        # Save plots
        fig.tight_layout()
        fig.savefig(os.path.join(root_folder, metric["filename"]))
        plt.close(fig)
        
    print(f"[EXPERIMENT] Plots saved in: '{root_folder}'")


if __name__ == '__main__':
    # Scenario 1: Long time horizon (T=1000) where epsilon* = 0 (distinguishable)
    print("=== SCENARIO 1: distinguishable environment (T=1000) ===")
    run_experiment(timeHorizon=1000, nbReplicates=32, root_folder="result_pract_regret/distinguishable/")
    
    # Scenario 2: Short time horizon (T=100) where epsilon* = 0.1 (indistinguishable)
    print("\n=== SCENARIO 2: indistinguishable environment (T=100) ===")
    run_experiment(timeHorizon=100, nbReplicates=32, root_folder="result_pract_regret/indistinguishable/")

