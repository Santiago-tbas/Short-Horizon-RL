import numpy as np
import matplotlib.pyplot as plt



def compute_epsilon_gaussian(gaps, T, sigma=1.0, eta=0.5):
    """Computes optimal epsilon via 1D-line."""
    c_T = np.sqrt((2 * sigma**2 * np.log(T)) / (eta * T))
    sorted_unique_gaps = np.sort(np.unique(gaps))

    epsilon = 0.0
    for i in range(len(sorted_unique_gaps) - 1):
        if sorted_unique_gaps[i+1] - sorted_unique_gaps[i] <= c_T:
            epsilon = sorted_unique_gaps[i+1]
        else:
            break

    return epsilon, c_T


def plot_iterative_relaxation(gaps, T, sigma=1.0, eta=0.5):
    c_T = np.sqrt((2 * sigma**2 * np.log(T)) / (eta * T))
    sorted_unique_gaps = np.sort(np.unique(np.concatenate(([0.0], gaps[gaps > 0]))))
    all_gaps = np.sort(gaps)

    epsilon_history = [0.0]
    epsilon = 0.0


    for i in range(len(sorted_unique_gaps) - 1):
        if sorted_unique_gaps[i+1] - sorted_unique_gaps[i] <= c_T:
            epsilon = sorted_unique_gaps[i+1]
            epsilon_history.append(epsilon)
        else:
            break

    n_steps = len(epsilon_history)
    plt.style.use('seaborn-v0_8-white')
    fig, axes = plt.subplots(n_steps, 1, figsize=(12, 2.8 * n_steps), dpi=100)

    if n_steps == 1:
        axes = [axes]

    for step, eps in enumerate(epsilon_history):
        ax = axes[step]
        ax.axhline(0, color='black', lw=1.2, zorder=1)

        # Indistinguishability band
        ax.axvspan(eps, eps + c_T, color='red', alpha=0.12, zorder=2)
        ax.axvline(eps, color='darkred', linestyle='--', lw=1.8, zorder=3)

        # Plot and color-code gaps
        for gap in all_gaps:
            if gap <= eps + 1e-9:
                color = '#2ca02c'
            elif eps < gap <= eps + c_T + 1e-9:
                color = '#d62728'
            else:
                color = '#1f77b4'

            ax.scatter(gap, 0, color=color, s=120, zorder=4, edgecolors='white', lw=1.5)
            ax.annotate(
                rf'$\Delta={gap}$', xy=(gap, 0), xytext=(gap, 0.08),
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=color
            )

        ax.get_yaxis().set_visible(False)
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)

        ax.set_xlim(-0.1, max(gaps) + 0.3)
        ax.set_ylim(-0.1, 0.25)

        if step < n_steps - 1:
            ax.set_title(
                rf'Iteration {step + 1}: $\epsilon = {eps:.2f}$ | '
                rf'Band $(\epsilon, \epsilon + c_T]$ absorbs local gaps',
                fontsize=11, color='darkred', loc='left'
            )
        else:
            ax.set_title(
                rf'Final State: $\epsilon^* = {eps:.2f}$ | '
                r'Indistinguishability band is empty',
                fontsize=11, color='darkgreen', loc='left'
            )

    # Parameter box at the bottom of the figure
    param_text = (
        rf'$T = {T}$ | $\sigma = {sigma}$ | $\eta = {eta}$ | '
        rf'$c_T = {c_T:.4f}$'
    )

    fig.text(
        0.98, 0.91, param_text, ha='right', va='top', fontsize=10,
        bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='gray',
                  boxstyle='round,pad=0.4')
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(
        rf'Discrete Boundary Progression ($T={T}$, $c_T={c_T:.2f}$)',
        fontsize=16, y=0.98, fontweight='bold'
    )
    plt.show()


# EXECUTION & VISUALIZATION

if __name__ == '__main__':
    # --- Scenario 1: Static Optimal Boundary ---
    T_static = 2000
    sigma_static = 1.0
    eta_static = 0.5
    bandit_gaps_static = np.array([0, 0.1, 0.15, 0.4, 0.8, 1.2])

    eps_opt, Ct = compute_epsilon_gaussian(
        bandit_gaps_static, T_static, sigma_static, eta_static
    )
    sorted_gaps = np.sort(bandit_gaps_static)

    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(12, 4.0), dpi=120)
    ax.axhline(0, color='black', lw=1.2, zorder=1)

    ax.scatter(
        sorted_gaps, np.zeros_like(sorted_gaps), color='#1f77b4',
        s=100, zorder=4, label=r'Gaps ($\Delta_a$)', edgecolors='white', lw=1.5
    )

    # Alternating annotations
    for i, gap in enumerate(sorted_gaps):
        direction = 1 if i % 2 == 0 else -1
        y_text = 0.06 * direction
        va_align = 'bottom' if direction == 1 else 'top'

        ax.annotate(
            rf'$\Delta={gap}$', xy=(gap, 0), xytext=(gap, y_text),
            ha='center', va=va_align, fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.8, alpha=0.5)
        )

    # Final boundaries
    ax.axvspan(
        eps_opt, eps_opt + Ct, color='red', alpha=0.12, zorder=2,
        label=r'Final Band $(\epsilon^*, \epsilon^* + c_T]$'
    )
    ax.axvline(
        eps_opt, color='darkred', linestyle='--', lw=2, zorder=3,
        label=rf'Optimal $\epsilon^* = {eps_opt:.3f}$'
    )

    ax.get_yaxis().set_visible(False)
    for spine in ['left', 'right', 'top']:
        ax.spines[spine].set_visible(False)

    ax.set_xlim(-0.1, max(bandit_gaps_static) + 0.2)
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel(r'Sub-optimality gap $\Delta_a = \mu^* - \mu_a$', fontsize=11)
    ax.set_title(
        r'Estimation of $\epsilon_{\eta}(T,\nu)$ for Gaussian Bandits',
        fontsize=13, pad=12
    )

    # Parameter info box — all parameters explicit
    param_info = (
        rf'$T = {T_static}$ | $\sigma = {sigma_static}$ | '
        rf'$\eta = {eta_static}$ | $c_T = {Ct:.4f}$ | '
        rf'$\epsilon^* = {eps_opt:.4f}$'
    )
    ax.text(
        0.01, 0.92, param_info, transform=ax.transAxes, fontsize=10,
        ha='left', va='top',
        bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='gray',
                  boxstyle='round,pad=0.3')
    )
    ax.legend(
        loc='upper right', frameon=True, shadow=False,
        facecolor='white', edgecolor='gainsboro', fontsize=9
    )

    fig.tight_layout()
    plt.show()

    # --- Scenario 2: Step-by-Step Relaxation ---
    T_dynamic = 200
    sigma_dynamic = 1.0
    eta_dynamic = 0.5
    bandit_gaps_dynamic = np.array([0, 0.1, 0.15, 0.4, 0.8, 1.2])

    plot_iterative_relaxation(
        bandit_gaps_dynamic, T_dynamic, sigma_dynamic, eta_dynamic
    )