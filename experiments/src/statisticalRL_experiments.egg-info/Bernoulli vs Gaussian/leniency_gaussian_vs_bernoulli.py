
import os
import sys
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_HERE, '../../learners/src'))
sys.path.insert(0, os.path.join(_HERE, '../../environments/src'))
sys.path.insert(0, os.path.join(_HERE, '../..'))

from statisticalrl_learners.MABs.GaussianTS import GaussianTS


# ── Environment ───────────────────────────────────────────────────────────────

class TruncatedGaussianBandit:
    """
    Bandit with arms whose rewards are drawn from TruncNorm(mu_a, sigma^2)
    restricted to [0, 1] via rejection sampling.
    """
    def __init__(self, means, sigma2=0.05):
        self.means  = np.array(means)
        self.sigma  = np.sqrt(sigma2)
        self.nbArms = len(means)

    def sample(self, action):
        mu = self.means[action]
        for _ in range(10_000):
            val = np.random.normal(mu, self.sigma)
            if 0.0 <= val <= 1.0:
                return val
        # Fallback: clipped (very rare for sigma=0.22 and mu in [0,1])
        return float(np.clip(np.random.normal(mu, self.sigma), 0.0, 1.0))


# ── LenientGaussianTS ─────────────────────────────────────────────────────────

class LenientGaussianTS:
    """
    Lenient TS for Gaussian bandits with rewards in [0, 1].

    Posterior: Normal-Normal conjugate update.
    Near-optimality: ABSOLUTE threshold  mu_post[a] > 1 - epsilon.
    Exploration: theta_a = (1 - epsilon) * Y,  Y ~ TruncNorm in [0, 1].
    """

    def __init__(self, nbArms, epsilon=0.2, sigma2=0.05,
                 mu0=0.5, sigma0_2=0.25):
        self.nbArms    = nbArms
        self.epsilon   = epsilon
        self.sigma2    = sigma2         # known likelihood variance
        self.mu0       = mu0            # prior mean
        self.sigma0_2  = sigma0_2       # prior variance
        self.reset()

    def reset(self):
        self.nbDraws    = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.theta      = np.zeros(self.nbArms)
        self._sample_theta()

    # ── posterior parameters ──────────────────────────────────────────────

    def _posterior(self, a):
        """Return (post_mean, post_var) for arm a under Normal-Normal update."""
        N_a = self.nbDraws[a]
        S_a = self.cumRewards[a]
        post_var  = 1.0 / (1.0 / self.sigma0_2 + N_a / self.sigma2)
        post_mean = post_var * (self.mu0 / self.sigma0_2 + S_a / self.sigma2)
        return post_mean, post_var

    # ── sampling rule (core of the algorithm) ────────────────────────────

    def _sample_theta(self):
        """
        Sample theta_a for each arm according to the lenient Gaussian rule.

        ABSOLUTE near-optimality condition: mu_post[a] > 1 - epsilon
        (analogous to Bernoulli ε-TS where condition is mu_hat > 1 - epsilon).

        When NOT near-optimal:
            theta_a = scale * Y
            Y ~ TruncNorm(mu_post/scale, post_var/scale^2), Y in [0, 1]
            => theta_a has support [0, scale] where scale = 1 - epsilon.
        """
        scale = 1.0 - self.epsilon   # maximum theta for non-near-optimal arms

        for a in range(self.nbArms):
            post_mean, post_var = self._posterior(a)

            if post_mean > scale:
                # ── Near-optimal: deterministic exploitation ──────────────
                # The arm is within epsilon of the maximum possible reward (1).
                # Freezing exploration here is acceptable: any draw would be
                # near-optimal regardless, and we avoid wasting variance budget.
                self.theta[a] = post_mean

            else:
                # ── Attenuated exploration ────────────────────────────────
                # Transform: if theta_a = scale * Y and theta_a has support
                # [0, scale], then Y in [0, 1] with:
                #   E[Y] = post_mean / scale
                #   Var[Y] = post_var / scale^2
                std_scaled = np.sqrt(post_var) / scale
                mu_scaled  = post_mean / scale

                # Rejection sampling for Y in [0, 1]
                for _ in range(2000):
                    Y = np.random.normal(mu_scaled, std_scaled)
                    if 0.0 <= Y <= 1.0:
                        self.theta[a] = scale * Y
                        break
                else:
                    # Very rare fallback: clip. Happens if post_mean << 0 or >> scale.
                    self.theta[a] = scale * float(np.clip(mu_scaled, 0.0, 1.0))

    def play(self):
        return int(np.argmax(self.theta))

    def update(self, arm, reward):
        self.nbDraws[arm]    += 1
        self.cumRewards[arm] += reward
        self._sample_theta()


# ── Single replicate runner ───────────────────────────────────────────────────

def run_single_replicate(seed, agent_class, agent_kwargs, bandit, T, epsilon):
    """
    Runs one replicate and returns (std_trace, hinge_trace) of shape (T,).

    Both metrics are computed EXTERNALLY from the agent's actions.
    """
    np.random.seed(seed)
    agent = agent_class(**agent_kwargs)
    agent.reset()

    mu     = bandit.means
    mu_star = np.max(mu)
    gaps   = mu_star - mu
    gaps_eps = np.maximum(0.0, gaps - epsilon)   # per-arm hinge gap

    cum_std   = 0.0
    cum_hinge = 0.0
    std_trace   = np.zeros(T)
    hinge_trace = np.zeros(T)

    for t in range(T):
        action = agent.play()
        reward = bandit.sample(action)
        agent.update(action, reward)

        cum_std   += gaps[action]
        cum_hinge += gaps_eps[action]

        std_trace[t]   = cum_std
        hinge_trace[t] = cum_hinge

    return std_trace, hinge_trace


# ── AgentConfig ───────────────────────────────────────────────────────────────

class AgentConfig:
    def __init__(self, agent_class, name, kwargs_factory, color):
        self.agent_class    = agent_class
        self.name           = name
        self.kwargs_factory = kwargs_factory
        self.color          = color


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    epsilon = 0.2
    sigma2  = 0.05        # reward variance (sigma = 0.224)
    T       = 5000
    N       = 300         # replicates

    scenarios = [
        {'name': 'Sc1', 'title': r'Arms $[\mu_1=0.5,\;\mu_2=0.2]$',
         'mu': np.array([0.5, 0.2]), 'pos': (0, 0)},
        {'name': 'Sc2', 'title': r'Arms $[\mu_1=0.9,\;\mu_2=0.6]$',
         'mu': np.array([0.9, 0.6]), 'pos': (0, 1)},
        {'name': 'Sc3', 'title': r'Arms $[\mu_1=0.5,\;\mu_2=0.45,\;\mu_3=0.2]$',
         'mu': np.array([0.5, 0.45, 0.2]), 'pos': (1, 0)},
        {'name': 'Sc4', 'title': r'Arms $[\mu_1=0.9,\;\mu_2=0.85,\;\mu_3=0.6]$',
         'mu': np.array([0.9, 0.85, 0.6]), 'pos': (1, 1)},
    ]

    agents = [
        AgentConfig(GaussianTS,
                    'GaussianTS',
                    lambda K: {'nbArms': K, 'sigma': np.sqrt(sigma2),
                               'mu0': 0.5, 'sigma0': np.sqrt(0.25)},
                    '#1f77b4'),
        AgentConfig(LenientGaussianTS,
                    r'Lenient-GaussianTS ($\varepsilon$=0.2)',
                    lambda K: {'nbArms': K, 'epsilon': epsilon,
                               'sigma2': sigma2, 'mu0': 0.5,
                               'sigma0_2': 0.25},
                    '#d62728'),
    ]

    results_dir = os.path.join(_HERE, '../../Lenient regret')
    os.makedirs(results_dir, exist_ok=True)
    num_cores = min(multiprocessing.cpu_count(), 8)

    print('=' * 70)
    print(f'LENIENCY GAUSSIAN vs. BERNOULLI   ε={epsilon}   σ²={sigma2}   T={T}   N={N}')
    print('=' * 70)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=150)
    times = np.arange(1, T + 1)

    for sc in scenarios:
        mu  = sc['mu']
        K   = len(mu)
        row, col = sc['pos']
        ax  = axes[row, col]
        bandit = TruncatedGaussianBandit(mu, sigma2=sigma2)

        print(f"\n[{sc['name']}]  mu={mu}")

        for agent in agents:
            print(f'  > {agent.name}')
            raw = Parallel(n_jobs=num_cores)(
                delayed(run_single_replicate)(
                    seed=12345 + i,
                    agent_class=agent.agent_class,
                    agent_kwargs=agent.kwargs_factory(K),
                    bandit=bandit,
                    T=T,
                    epsilon=epsilon,
                )
                for i in range(N)
            )
            std_arr   = np.array([r[0] for r in raw])   # (N, T)
            hinge_arr = np.array([r[1] for r in raw])

            std_mean   = np.mean(std_arr,   axis=0)
            std_se     = np.std( std_arr,   axis=0) / np.sqrt(N)
            hinge_mean = np.mean(hinge_arr, axis=0)
            hinge_se   = np.std( hinge_arr, axis=0) / np.sqrt(N)

            # Standard regret (solid)
            ax.plot(times, std_mean, color=agent.color, lw=2.0,
                    linestyle='-', label=agent.name)
            ax.fill_between(times,
                            std_mean - 1.96 * std_se,
                            std_mean + 1.96 * std_se,
                            color=agent.color, alpha=0.07)

            # Hinge regret (dashed)
            ax.plot(times, hinge_mean, color=agent.color, lw=2.0,
                    linestyle='--', label=f'{agent.name} (Hinge)')
            ax.fill_between(times,
                            hinge_mean - 1.96 * hinge_se,
                            hinge_mean + 1.96 * hinge_se,
                            color=agent.color, alpha=0.07)

        ax.set_title(sc['title'], fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel(r'Time horizon $T$', fontsize=11)
        ax.set_ylabel(r'$R(T)$', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlim(1, T)
        ax.set_ylim(bottom=0)

    # ── Legend ────────────────────────────────────────────────────────────
    leg = [
        Line2D([0], [0], color='#1f77b4', lw=2.5, label='GaussianTS'),
        Line2D([0], [0], color='#d62728', lw=2.5,
               label=r'Lenient-GaussianTS ($\varepsilon$=0.2)'),
        Line2D([0], [0], color='gray', lw=2.0, linestyle='-',
               label='Standard Regret'),
        Line2D([0], [0], color='gray', lw=2.0, linestyle='--',
               label=r'Hinge Regret  ($\varepsilon=0.2$)'),
    ]
    fig.legend(
        handles=leg,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        frameon=True,
        facecolor='white',
        edgecolor='#cccccc',
        fontsize=10,
        borderpad=0.8,
        handlelength=2.2,
        columnspacing=1.5,
    )

    # ── Titles ────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.985,
        r'Gaussian Lenient Regret Evaluation',
        ha='center', va='top', fontsize=15, fontweight='bold'
    )
    fig.text(
        0.5, 0.955,
        r'Truncated Normal Rewards in $[0, 1]$  ($\sigma^2 = 0.05$,  $\varepsilon = 0.2$,  $N = 300$)',
        ha='center', va='top', fontsize=10, style='italic', color='#444444'
    )

    fig.subplots_adjust(top=0.81, hspace=0.38, wspace=0.28)

    out = os.path.join(results_dir, 'leniency_gaussian_vs_bernoulli.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[SUCCESS] Saved: {os.path.abspath(out)}')
    print('=' * 70)


if __name__ == '__main__':
    main()
