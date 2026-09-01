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


def run_single_replicate(seed, learner_class, learner_kwargs, means, variances, T, epsilon_star):
    """Runs a single replicate with warm-up phase (K pulls) followed by T measurement steps."""
    np.random.seed(seed)
    
    env = GaussianBandit(means, variances)
    learner = learner_class(**learner_kwargs)
    learner.reset()
    
    nb_arms = len(means)
    best_mean = np.max(means)
    
    # Phase 1: Warm-up (1 round-robin pull per arm, not counted in regret)
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
        cum_prac += max(gap - epsilon_star, 0.0)
        
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
    
    epsilon_star, c_T = compute_epsilon_star(gaps, T + len(means), sigma=sigma, eta=eta)
    
    print("=" * 70)
    print("EXPERIMENT: STATIC GAUSSIAN PRACTITIONER REGRET")
    print("=" * 70)
    print(f"Arm means              : {means}")
    print(f"Sub-optimality gaps    : {gaps}")
    print(f"Time horizon (T)       : {T}")
    print(f"Simulations (N)        : {N}")
    print(f"c_T                    : {c_T:.6f}")
    print(f"Optimal epsilon*       : {epsilon_star:.6f}")
    print("=" * 70)
    
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


    
    log_file_path = os.path.join(root_folder, "logfile_static.txt")
    log_file = open(log_file_path, "w")
    log_file.write(f"Static Practitioner Regret Experiment\n")
    log_file.write(f"T={T}, N={N}, epsilon*={epsilon_star:.6f}, c_T={c_T:.6f}\n\n")
    
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
                epsilon_star=epsilon_star
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
        pkl_path = os.path.join(root_folder, f"raw_data_{name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"std": std_regrets, "prac": prac_regrets, "means": means, "gaps": gaps}, f)
            
        log_file.write(f"{name} average runtime per replicate: {avg_time:.4f}s\n")
        log_file.write(f"  Final Std Regret Mean: {results[name]['std_mean'][-1]:.2f} (SEM: {results[name]['std_sem'][-1]:.2f})\n")
        log_file.write(f"  Final Prac Regret Mean: {results[name]['prac_mean'][-1]:.2f} (SEM: {results[name]['prac_sem'][-1]:.2f})\n\n")
        
    # Save experiment metadata
    metadata = {
        "means": means.tolist(),
        "gaps": gaps.tolist(),
        "epsilon_star": epsilon_star,
        "c_T": c_T,
        "T": T,
        "N": N
    }
    with open(os.path.join(root_folder, "metadata_static.pkl"), "wb") as f:
        pickle.dump(metadata, f)
        
    print("\n[INFO] Simulation finished. Creating plots...")
    
    # Downsample points for cleaner plot rendering (especially for large T)
    skip = max(1, T // 1000)
    times = np.arange(1, T + 1, skip)
    
    # 1. Plotting Standard vs Practitioner Regrets (1x2 Subplots)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    assert fig is not None
    
    # Subplot 1: Standard Pseudo-Regret
    ax1 = axes[0]
    for name, res in results.items():
        mean = res["std_mean"][::skip]
        sem = res["std_sem"][::skip]
        ax1.plot(times, mean, label=name, color=res["color"], linewidth=2.2)
        ax1.fill_between(times, mean - 1.96 * sem, mean + 1.96 * sem, color=res["color"], alpha=0.18)
        
    ax1.set_xlabel("Time step $t$", fontsize=12)
    ax1.set_ylabel("Cumulative regret", fontsize=12)
    ax1.set_title("Standard Pseudo-Regret\n" r"$R(T)=\sum_{t=1}^{T}\Delta_{a_t}$", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="gainsboro", fontsize=11)
    ax1.set_xlim(1, T)
    ax1.set_ylim(bottom=0)
    
    # Subplot 2: Practitioner Regret
    ax2 = axes[1]
    for name, res in results.items():
        mean = res["prac_mean"][::skip]
        sem = res["prac_sem"][::skip]
        ax2.plot(times, mean, label=name, color=res["color"], linewidth=2.2)
        ax2.fill_between(times, mean - 1.96 * sem, mean + 1.96 * sem, color=res["color"], alpha=0.18)
        
    ax2.set_xlabel("Time step $t$", fontsize=12)
    ax2.set_ylabel("Cumulative regret", fontsize=12)
    ax2.set_title(f"Practitioner ($\\varepsilon^*$-) Regret\n" r"$R^\dagger(T)=\sum_{t=1}^{T}\max(\Delta_{a_t}-\varepsilon^*,0)$"
                  f"\n$\\varepsilon^*={epsilon_star:.4f}$, $c_T={c_T:.4f}$", fontsize=12, fontweight='bold')
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="gainsboro", fontsize=11)
    ax2.set_xlim(1, T)
    ax2.set_ylim(bottom=0)
    
    # Annotation box
    info = (
        f"Arm means: {means}\n"
        f"Gaps:      {np.round(gaps, 3)}\n"
        f"sigma={sigma}, eta={eta}, T={T}, N={N}\n"
        f"c_T={c_T:.4f},  epsilon*={epsilon_star:.4f}"
    )
    fig.text(0.5, -0.02, info, ha='center', va='top', fontsize=9, family='monospace',
             bbox=dict(facecolor='lightyellow', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.9))

    fig.tight_layout()
    fig.suptitle("Gaussian Bandit: Standard vs. Practitioner Regret (Static)\n(Thompson Sampling & Gaussian UCB)",
                 fontsize=13, fontweight='bold', y=1.01)
    
    # Save plots in both png and pdf
    plot_base_path = os.path.join(root_folder, "regret_static")
    fig.savefig(plot_base_path + ".png", bbox_inches='tight')
    fig.savefig(plot_base_path + ".pdf", bbox_inches='tight')
    plt.close(fig)
    
    # 2. Log-Log scale Plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    assert fig2 is not None
    for ax_idx, metric in enumerate(["std", "prac"]):
        ax = axes2[ax_idx]
        for name, res in results.items():
            mean = res[f"{metric}_mean"][::skip]
            sem = res[f"{metric}_sem"][::skip]
            
            mean_safe = np.maximum(mean, 1e-6)
            ax.plot(times, mean_safe, label=name, color=res["color"], linewidth=2.2)
            ax.fill_between(times, np.maximum(mean - 1.96 * sem, 1e-6), np.maximum(mean + 1.96 * sem, 1e-6),
                            color=res["color"], alpha=0.18)
            
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel("Time step $t$ (log scale)", fontsize=12)
        ax.set_ylabel("Cumulative regret (log scale)", fontsize=12)
        label = "Standard Regret" if metric == "std" else f"Practitioner Regret (ε*={epsilon_star:.4f})"
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="gainsboro", fontsize=11)
        
    fig2.tight_layout()
    fig2.suptitle("Log-log view: Standard vs. Practitioner Regret", fontsize=13, fontweight='bold')
    plot_log_path = os.path.join(root_folder, "regret_static_loglog")
    fig2.savefig(plot_log_path + ".png", bbox_inches='tight')
    fig2.savefig(plot_log_path + ".pdf", bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Ratio plot (prac / std)
    fig3, ax3 = plt.subplots(figsize=(9, 5), dpi=150)
    assert fig3 is not None
    for name, res in results.items():
        mean_std = res["std_mean"]
        mean_prac = res["prac_mean"]
        safe_std = np.where(mean_std > 0, mean_std, np.nan)
        ratio = mean_prac / safe_std
        ax3.plot(np.arange(1, T + 1), ratio, label=name, color=res["color"], linewidth=2.2)
        
    frac_clearly_suboptimal = np.mean(gaps > epsilon_star)
    ax3.axhline(frac_clearly_suboptimal, color='gray', linestyle='--', linewidth=1.4,
                label=rf"Fraction of arms with $\Delta_a > \varepsilon^*$ = {frac_clearly_suboptimal:.2f}")
                
    ax3.set_xlabel("Time step $t$", fontsize=12)
    ax3.set_ylabel(r"$R^\dagger(t) / R(t)$", fontsize=13)
    ax3.set_title("Ratio of Practitioner Regret to Standard Regret", fontsize=12, fontweight='bold')
    ax3.legend(frameon=True, facecolor='white', fontsize=10)
    ax3.set_xlim(1, T)
    ax3.set_ylim(0, 1.05)
    fig3.tight_layout()
    
    plot_ratio_path = os.path.join(root_folder, "regret_static_ratio")
    fig3.savefig(plot_ratio_path + ".png", bbox_inches='tight')
    fig3.savefig(plot_ratio_path + ".pdf", bbox_inches='tight')
    plt.close(fig3)
    
    log_file.write(f"\nPlots saved in png and pdf formats.\n")
    log_file.close()
    
    print(f"[INFO] Log file created: {log_file_path}")
    print(f"[INFO] Plot saved: {os.path.abspath(plot_base_path)}.png")
    print("[DONE] Static practitioner regret experiment completed.")


if __name__ == '__main__':
    main()
