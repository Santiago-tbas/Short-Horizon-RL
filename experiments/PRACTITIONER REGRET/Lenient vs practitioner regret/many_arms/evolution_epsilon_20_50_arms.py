import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_HERE, '../Epsilon'))
sys.path.insert(0, os.path.join(_HERE, '../../../learners/src'))
sys.path.insert(0, os.path.join(_HERE, '../../../environments/src'))
sys.path.insert(0, os.path.join(_HERE, '../../..'))
from epsilon import compute_epsilon_gaussian as compute_epsilon_star


def main():
    T = 1000
    times = np.arange(1, T + 1)
    epsilon_static = 0.2

    np.random.seed(42)
    mu_20 = np.concatenate([
        [1.0], 
        np.random.uniform(0.85, 0.98, 10), 
        np.random.uniform(0.5, 0.8, 6), 
        np.random.uniform(0.1, 0.4, 3)
    ])
    np.random.shuffle(mu_20)

    np.random.seed(43)
    mu_50 = np.concatenate([
        [1.0], 
        np.random.uniform(0.85, 0.98, 25), 
        np.random.uniform(0.5, 0.8, 15), 
        np.random.uniform(0.1, 0.4, 9)
    ])
    np.random.shuffle(mu_50)

    scenarios = [
        {"name": "20 Arms Scenario", "mu": mu_20, "color": "#1f77b4"},
        {"name": "50 Arms Scenario", "mu": mu_50, "color": "#ff7f0e"},
    ]

    out_dir = os.path.join(_HERE, "many_arms")
    os.makedirs(out_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    for i, sc in enumerate(scenarios):
        ax = axes[i]
        mu = sc["mu"]
        gaps = np.max(mu) - mu
        
        eps_seq = []
        ct_seq = []
        for t in range(1, T + 1):
            e_t, c_t = compute_epsilon_star(gaps, t, sigma=0.5, eta=0.5)
            eps_seq.append(e_t)
            ct_seq.append(c_t)

        eps_seq = np.array(eps_seq)
        ct_seq = np.array(ct_seq)

        ax.plot(times, eps_seq, label=r"Dynamic $\varepsilon^*(t)$", color=sc["color"], lw=2.0)
        ax.plot(times, ct_seq, label=r"Confidence radius $c_t$", color="gray", linestyle="--", lw=1.5, alpha=0.8)
        ax.axhline(epsilon_static, color="black", linestyle=":", lw=1.2, label=r"Static $\varepsilon=0.2$")

        ax.set_title(f"{sc['name']}", fontsize=12, fontweight='bold')
        ax.set_xlabel(r"Time Step $t$", fontsize=11)
        ax.set_ylabel(r"Leniency / Radius", fontsize=11)
        ax.set_xlim(1, T)
        ax.legend(frameon=True, facecolor="white", edgecolor="gainsboro", fontsize=9, loc="upper right")

    fig.suptitle(r"Evolution of Dynamic Practitioner Leniency $\varepsilon^*(t)$ ($T=1000$)",
                 fontsize=14, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(out_dir, "evolution_epsilon_20_50_arms.png")
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    print("=" * 70)
    print(f"[SUCCESS] Clean Epsilon evolution plot generated at: {os.path.abspath(plot_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
