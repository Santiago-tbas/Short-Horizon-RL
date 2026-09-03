import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_HERE, '../../Epsilon'))
sys.path.insert(0, os.path.join(_HERE, '../../../../learners/src'))
sys.path.insert(0, os.path.join(_HERE, '../../../../environments/src'))
sys.path.insert(0, os.path.join(_HERE, '../../../..'))

from epsilon import compute_epsilon_gaussian as compute_epsilon_star

def main():
    T = 5000
    times = np.arange(1, T + 1)
    epsilon_static = 0.2

    mu_5 = np.array([0.9, 0.82, 0.7, 0.4, 0.2])
    gaps = np.max(mu_5) - mu_5

    out_dir = _HERE
    os.makedirs(out_dir, exist_ok=True)

    eps_seq = []
    ct_seq = []
    for t in range(1, T + 1):
        e_t, c_t = compute_epsilon_star(gaps, t, sigma=0.5, eta=0.5)
        eps_seq.append(e_t)
        ct_seq.append(c_t)

    eps_seq = np.array(eps_seq)
    ct_seq = np.array(ct_seq)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    ax.plot(times, eps_seq, label=r"Dynamic $\varepsilon^*(t)$", color="#1f77b4", lw=2.0)
    ax.plot(times, ct_seq, label=r"Confidence radius $c_t$", color="gray", linestyle="--", lw=1.5, alpha=0.8)
    ax.axhline(epsilon_static, color="black", linestyle=":", lw=1.5, label=r"Static $\varepsilon=0.2$")

    ax.set_title(
        r"Evolution of Dynamic Practitioner Leniency: $\varepsilon_t^*$ Over Time (5 Arms, $T=5000$)" + "\n" +
        r"$\mu = [0.9, 0.82, 0.7, 0.4, 0.2]$, Gaps $= [0.0, 0.08, 0.2, 0.5, 0.7]$",
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel(r"Time Step $t$", fontsize=11)
    ax.set_ylabel(r"Leniency $\varepsilon^*(t)$", fontsize=11)
    ax.set_xlim(1, T)
    ax.set_ylim(-0.01, 0.35)
    ax.legend(frameon=True, facecolor="white", edgecolor="gainsboro", fontsize=9, loc="upper right")

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "evolution_epsilon_5_arms.png")
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    print("=" * 70)
    print(f"[SUCCESS] Epsilon evolution plot (5 arms) saved at: {os.path.abspath(plot_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
