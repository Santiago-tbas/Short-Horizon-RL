
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def compute_epsilon_gaussian(gaps, T, sigma=1.0, eta=0.5):
    """Computes optimal epsilon via 1D sweep-line."""
    c_T = np.sqrt((2 * sigma**2 * np.log(T)) / (eta * T))
    sorted_unique_gaps = np.sort(np.unique(gaps))

    epsilon = 0.0
    for i in range(len(sorted_unique_gaps) - 1):
        if sorted_unique_gaps[i+1] - sorted_unique_gaps[i] <= c_T:
            epsilon = sorted_unique_gaps[i+1]
        else:
            break

    return epsilon, c_T


if __name__ == '__main__':
    # 15 Static physical sub-optimality gaps
    bandit_gaps = np.array([
        0.0, 0.02, 0.04, 0.07, 0.10, 0.14, 0.18,
        0.24, 0.32, 0.42, 0.55, 0.70, 0.88, 1.05, 1.30
    ])
    sigma = 1.0
    eta = 0.5

    # Sweep from Left to Right: T decreases so epsilon* grows rightward
    T_values = np.logspace(np.log10(10000000), np.log10(100), num=80).astype(int)

    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=120)

    def update(frame):
        ax.clear()

        T = T_values[frame]
        eps_opt, c_T = compute_epsilon_gaussian(bandit_gaps, T, sigma, eta)

        # Base horizontal axis
        ax.axhline(0, color='black', lw=1.2, zorder=1)

        # Indistinguishability Band
        ax.axvspan(eps_opt, eps_opt + c_T, color='red', alpha=0.14, zorder=2)
        ax.axvline(eps_opt, color='darkred', linestyle='--', lw=2, zorder=3)

        # Dynamic scatter and annotation
        text_height_tiers = [0.06, 0.12, 0.18]

        for i, gap in enumerate(bandit_gaps):
            if gap <= eps_opt + 1e-9:
                color = '#2ca02c'
            elif eps_opt < gap <= eps_opt + c_T + 1e-9:
                color = '#d62728'
            else:
                color = '#1f77b4'

            ax.scatter(gap, 0, color=color, s=85, zorder=4,
                       edgecolors='white', lw=1.2)

            base_height = text_height_tiers[i % 3]
            direction = 1 if (i // 3) % 2 == 0 else -1
            y_text = base_height * direction
            va_align = 'bottom' if direction == 1 else 'top'

            ax.annotate(
                rf'$\Delta={gap}$', xy=(gap, 0), xytext=(gap, y_text),
                ha='center', va=va_align, fontsize=9, fontweight='bold',
                color=color,
                arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.4)
            )

        ax.get_yaxis().set_visible(False)
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)

        ax.set_xlim(-0.05, max(bandit_gaps) + 0.12)
        ax.set_ylim(-0.25, 0.25)

        ax.set_title(
            rf'Empirical Evolution of the Optimal Precision Boundary '
            rf'$\epsilon_{{\eta}}(T, \nu)$ across Time Horizons',
            fontsize=14, pad=12, loc='center', fontweight='bold'
        )

        # Parameter box at top right
        param_text = (
            rf'$T = {T}$ | $\sigma = {sigma}$ | $\eta = {eta}$ | '
            rf'$c_T = {c_T:.4f}$'
        )
        ax.text(
            0.99, 0.97, param_text, transform=ax.transAxes, fontsize=10,
            ha='right', va='top',
            bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='gray',
                      boxstyle='round,pad=0.3')
        )

    ani = animation.FuncAnimation(
        fig, update, frames=len(T_values), interval=40, repeat=False
    )
    assert fig is not None
    fig.subplots_adjust(top=0.82, bottom=0.08)
    plt.show()