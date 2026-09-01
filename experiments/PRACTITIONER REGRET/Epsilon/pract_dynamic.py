import os
import sys
import pickle
import time
import numpy as np

from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path Setup
_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_HERE, '../../../learners/src'))
sys.path.insert(0, os.path.join(_HERE, '../../../environments/src'))
sys.path.insert(0, os.path.join(_HERE, '../../..'))

# Environments and Learners
from statisticalrl_environments.MABs.StochasticBandits import GaussianBandit
from statisticalrl_learners.MABs.GaussianTS import GaussianTS
from statisticalrl_learners.MABs.GaussianUCB import GaussianUCB
from epsilon import compute_epsilon_gaussian as compute_epsilon_star


def run_single_replicate(seed, learner_class, learner_kwargs, means, variances, T, gaps, precomputed_epsilons):
    """Runs a single replicate with warm-up phase (K pulls) followed by T measurement steps with dynamic epsilon."""

    np.random.seed(seed)
    env = GaussianBandit(means, variances)
    learner = learner_class(**learner_kwargs)
    learner.reset()
    nb_arms = len(means)
    best_mean = np.max(means)
    
    # Phase 1: round-robin pull per arm, not counted in regret
    for arm in range(nb_arms):
        reward = env.arms[arm].sample()
        learner.update(arm, reward)
        
    # Phase 2: Measurement (T steps)
    std_trace = np.zeros(T)
    prac_trace = np.zeros(T)
    cum_std = 0.0
    cum_prac = 0.0
    
    for t in range(T):
        action = learner.play()
        reward = env.arms[action].sample()
        learner.update(action, reward)
        gap = best_mean - means[action]
        cum_std += gap
        
        # Total steps including Phase 1 (1-indexed)
        t_total = t + 1 + nb_arms
        epsilon_star_t = precomputed_epsilons[t_total]
        
        # Accumulate practitioner regret step-by-step using the contemporaneous epsilon*_t
        cum_prac += max(gap - epsilon_star_t, 0.0)
        std_trace[t] = cum_std
        prac_trace[t] = cum_prac
        
    return std_trace, prac_trace


def main():
    # Simulation Parameters
    means = np.array([0.80, 0.75, 0.50, 0.20, -0.20])
    variances = np.array([1.0,  1.0,  1.0,  1.0,   1.0])
    sigma = 1.0          # common std-dev
    eta = 0.5          # tolerance
    T = 10000
    N = 100
    
    best_mean = np.max(means)
    gaps = best_mean - means
    
    print("=" * 70)
    print("EXPERIMENT: DYNAMIC GAUSSIAN PRACTITIONER REGRET")
    print("=" * 70)
    print(f"Arm means              : {means}")
    print(f"Sub-optimality gaps    : {gaps}")
    print(f"Time horizon (T)       : {T}")
    print(f"Simulations (N)        : {N}")
    print(f"sigma                  : {sigma}, eta: {eta}")
    print("=" * 70)
    
    # Precompute epsilon* for all total steps from 1 to T + len(means)
    precomputed_epsilons = np.zeros(T + len(means) + 1)
    for t_val in range(1, T + len(means) + 1):
        eps, _ = compute_epsilon_star(gaps, t_val, sigma=sigma, eta=eta)
        precomputed_epsilons[t_val] = eps
        
    # Configure Agents
    agents_config = [
        {
            "class": GaussianTS,
            "kwargs": {"nbArms": len(means), "sigma": sigma},
            "name": "TS",
            "color": "#e05c00"
        },
        {
            "class": GaussianUCB,
            "kwargs": {"nbArms": len(means), "sigma": sigma, "delta": lambda t: 1.0 / t},
            "name": "UCB",
            "color": "#1a6faf"
        }
    ]
    
    root_folder = os.path.abspath(os.path.join(_HERE, "../Lenient vs practitioner regret/images_experiments"))
    os.makedirs(root_folder, exist_ok=True)


    
    log_file_path = os.path.join(root_folder, "logfile_dynamic.txt")
    log_file = open(log_file_path, "w")
    log_file.write(f"Dynamic Practitioner Regret Experiment\n")
    log_file.write(f"T={T}, N={N}, sigma={sigma}, eta={eta}\n\n")
    
    results = {}
    
    for agent in agents_config:
        name = agent["name"]
        print(f"[INFO] Simulating {name}...")
        
        t0 = time.time()
        raw_traces = Parallel(n_jobs=-1)(
            delayed(run_single_replicate)(
                seed=12345 + i,
                learner_class=agent["class"],
                learner_kwargs=agent["kwargs"],
                means=means,
                variances=variances,
                T=T,
                gaps=gaps,
                precomputed_epsilons=precomputed_epsilons
            )
            for i in range(N)
        )
        elapsed = time.time() - t0
        avg_time = elapsed / N
        
        std_regrets = np.array([trace[0] for trace in raw_traces])
        prac_regrets = np.array([trace[1] for trace in raw_traces])
        
        results[name] = {
            "std_mean": np.mean(std_regrets, axis=0),
            "std_sem": np.std(std_regrets, axis=0) / np.sqrt(N),
            "prac_mean": np.mean(prac_regrets, axis=0),
            "prac_sem": np.std(prac_regrets, axis=0) / np.sqrt(N),
            "color": agent["color"]
        }
        
        # Save raw pickle files
        pkl_path = os.path.join(root_folder, f"raw_data_{name}_dynamic.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"std": std_regrets, "prac": prac_regrets, "means": means, "gaps": gaps}, f)
            
        log_file.write(f"{name} average runtime per replicate: {avg_time:.4f}s\n")
        log_file.write(f"  Final Std Regret Mean: {results[name]['std_mean'][-1]:.2f} (SEM: {results[name]['std_sem'][-1]:.2f})\n")
        log_file.write(f"  Final Prac Regret Mean: {results[name]['prac_mean'][-1]:.2f} (SEM: {results[name]['prac_sem'][-1]:.2f})\n\n")
        
    # Save experiment metadata
    metadata = {
        "means": means.tolist(),
        "gaps": gaps.tolist(),
        "T": T,
        "N": N
    }
    with open(os.path.join(root_folder, "metadata_dynamic.pkl"), "wb") as f:
        pickle.dump(metadata, f)
        
    print("\n[INFO] Simulation finished. Creating plots...")
    
    # Downsample points for cleaner plot rendering (especially for large T)
    skip = max(1, T // 1000)
    times = np.arange(1, T + 1, skip)
    
    # 1. Merged Plot on a Single Set of Axes (Same Scale for Direct Comparison)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    assert fig is not None
    
    # Plot Standard and Practitioner Regrets on the SAME axis
    for name, res in results.items():
        color = res["color"]
        
        # Standard Regret (Solid line)
        std_mean = res["std_mean"][::skip]
        std_sem = res["std_sem"][::skip]
        ax.plot(times, std_mean, label=f"{name} (Standard)", color=color, linestyle="-", linewidth=2.2)
        ax.fill_between(times, std_mean - 1.96 * std_sem, std_mean + 1.96 * std_sem, color=color, alpha=0.12)
        
        # Practitioner Regret (Dashed line)
        prac_mean = res["prac_mean"][::skip]
        prac_sem = res["prac_sem"][::skip]
        ax.plot(times, prac_mean, label=f"{name} (Practitioner)", color=color, linestyle="--", linewidth=2.2)
        ax.fill_between(times, prac_mean - 1.96 * prac_sem, prac_mean + 1.96 * prac_sem, color=color, alpha=0.12)
        
    ax.set_xlabel("Time step $t$", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title("Gaussian Bandit: Standard vs. Practitioner Regret (Dynamic $\\varepsilon^*_t$)\nComparison at Same Scale (TS vs. Gaussian UCB)", fontsize=13, fontweight='bold')
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="gainsboro", fontsize=11)
    ax.set_xlim(1, T)
    ax.set_ylim(bottom=0)
    
    # Annotation box
    info = (
        f"Arm means: {means}\n"
        f"Gaps:      {np.round(gaps, 3)}\n"
        f"sigma={sigma}, eta={eta}, T={T}, N={N}\n"
        f"Dynamic epsilon*_t"
    )
    fig.text(0.5, -0.05, info, ha='center', va='top', fontsize=9, family='monospace',
             bbox=dict(facecolor='lightyellow', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.9))
             
    fig.tight_layout()
    
    # Save merged plot
    plot_base_path = os.path.join(root_folder, "regret_dynamic")
    fig.savefig(plot_base_path + ".png", bbox_inches='tight')
    fig.savefig(plot_base_path + ".pdf", bbox_inches='tight')
    plt.close(fig)
    
    log_file.write(f"\nPlot saved in png and pdf formats.\n")
    log_file.close()
    
    print(f"[INFO] Log file created: {log_file_path}")
    print(f"[INFO] Plot saved: {os.path.abspath(plot_base_path)}.png")
    print("[DONE] Dynamic practitioner regret experiment completed.")


if __name__ == '__main__':
    main()
