
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_HERE, '../../../learners/src'))
sys.path.insert(0, os.path.join(_HERE, '../../../environments/src'))
sys.path.insert(0, os.path.join(_HERE, '../../..'))
from epsilon import compute_epsilon_gaussian as compute_epsilon_star


def main():
    T = 5000
    times = np.arange(1, T + 1)
    epsilon_static = 0.2  # the static epsilon used for e-TS

    scenarios = [
        {
            "name": "Scenario 1",
            "label": r"Sc. 1: $[\mu_1{=}0.5,\,\mu_2{=}0.2]$, gaps $= [0.0, 0.3]$",
            "mu": np.array([0.5, 0.2]),
            "color": "#d62728",
            "linestyle": "-",
            "lw": 2.5
        },
        {
            "name": "Scenario 2",
            "label": r"Sc. 2: $[\mu_1{=}0.9,\,\mu_2{=}0.6]$, gaps $= [0.0, 0.3]$",
            "mu": np.array([0.9, 0.6]),
            "color": "#ff7f0e",
            "linestyle": "--",
            "lw": 2.0
        },
        {
            "name": "Scenario 3",
            "label": r"Sc. 3: $[\mu_1{=}0.5,\,\mu_2{=}0.45,\,\mu_3{=}0.2]$, gaps $= [0.0, 0.05, 0.3]$",
            "mu": np.array([0.5, 0.45, 0.2]),
            "color": "#1f77b4",
            "linestyle": "-",
            "lw": 2.5
        },
        {
            "name": "Scenario 4",
            "label": r"Sc. 4: $[\mu_1{=}0.9,\,\mu_2{=}0.85,\,\mu_3{=}0.6]$, gaps $= [0.0, 0.05, 0.3]$",
            "mu": np.array([0.9, 0.85, 0.6]),
            "color": "#2ca02c",
            "linestyle": "--",
            "lw": 2.0
        }
    ]

    results_dir = os.path.abspath(os.path.join(_HERE, "../Lenient vs practitioner regret/images_experiments"))
    os.makedirs(results_dir, exist_ok=True)


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

    for sc in scenarios:
        mu = sc["mu"]
        mu_star = np.max(mu)
        gaps = mu_star - mu
        # eps_t = epsilon^*(T=t): tolerance computed for horizon T=t at each step
        eps_seq = np.array([
            compute_epsilon_star(gaps, t, sigma=0.5, eta=0.5)[0]
            for t in range(1, T + 1)
        ])
        ax.plot(
            times, eps_seq,
            label=sc["label"],
            color=sc["color"],
            linestyle=sc["linestyle"],
            linewidth=sc["lw"]
        )

    # Add horizontal reference line for the static epsilon = 0.2
    ax.axhline(
        y=epsilon_static, color="black", linestyle=":",
        linewidth=1.5, label=r"Static $\varepsilon = 0.2$ (used in $\varepsilon$-TS)"
    )

    # Mark the critical transition point for Scenarios 3 & 4
    # eps_t drops from 0.05 → 0.0 when c_T = sqrt(log(T)/T) crosses 0.05
    # → T* ≈ 3250
    ax.axvline(
        x=3250, color="#1f77b4", linestyle=":",
        linewidth=1.2, alpha=0.7
    )
    ax.annotate(
        r"$T^* \approx 3250$" + "\n" + r"$c_{T^*} = 0.05 = \Delta_{\min}$",
        xy=(3250, 0.026), xytext=(3400, 0.08),
        fontsize=9, color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2)
    )

    ax.set_title(
        r"Evolution Practitioner Leniency: $\varepsilon^*(t)$ Over Time" + "\n"
        r"($\varepsilon^*(t) = $ tolerance used at step $t$; Bernoulli, $\sigma=0.5$, $\eta=0.5$)",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel(r"Time Horizon $t$", fontsize=12)
    ax.set_ylabel(r"Leniency $\varepsilon^*(t)$", fontsize=12)
    ax.set_xlim(1, T)
    ax.set_ylim(-0.01, 0.33)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(frameon=True, facecolor="white", edgecolor="gainsboro", fontsize=9.5, loc="upper right")

    # Annotate the two plateau levels
    ax.annotate(r"$\varepsilon^* = 0.05$ (Sc. 3 & 4 plateau)", xy=(1200, 0.052),
                fontsize=9, color="#1f77b4", style="italic")
    ax.annotate(r"$\varepsilon^* = 0.0$ (all scenarios converge)", xy=(3600, 0.005),
                fontsize=9, color="gray", style="italic")

    fig.tight_layout()
    plot_path = os.path.join(results_dir, "evolution_practitioner_leniency.png")
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    print("=" * 70)
    print(f"[SUCCESS] Evolution Practitioner Leniency figure generated!")
    print(f"Saved at: {os.path.abspath(plot_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
