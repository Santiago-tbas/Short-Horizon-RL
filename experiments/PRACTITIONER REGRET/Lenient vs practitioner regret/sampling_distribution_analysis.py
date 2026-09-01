import os
import sys
from typing import TypedDict, Tuple, List, Dict, Any, Optional, cast
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import beta as beta_dist

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_HERE, '../../../learners/src'))
sys.path.insert(0, os.path.join(_HERE, '../../../environments/src'))
sys.path.insert(0, os.path.join(_HERE, '../../..'))

from statisticalrl_learners.MABs.TS import TS
from statisticalrl_learners.MABs.Algo1 import Algo1 as EpsilonTS
from epsilon import compute_epsilon_gaussian as _compute_eps


class ScenarioDict(TypedDict):
    name: str
    title: str
    mu: np.ndarray


def compute_eps_seq(gaps: np.ndarray, T: int, sigma: float = 0.5, eta: float = 0.5) -> List[float]:
    return [float(_compute_eps(gaps, t, sigma=sigma, eta=eta)[0]) for t in range(1, T + 1)]


class EpsilonTSDynamic(EpsilonTS):
    """epsilon_t-TS: practitioner leniency with time-varying epsilon_t."""

    def __init__(self, nbArms: int, precomputed_epsilons: List[float]) -> None:
        super().__init__(nbArms, epsilon=precomputed_epsilons[0])
        self.precomputed_epsilons = precomputed_epsilons
        self.time = 0

    def reset(self) -> None:
        super().reset()
        self.time = 0

    def update(self, arm: int, reward: float) -> None:
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1
        self.empMeans[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.time += 1
        idx = min(self.time, len(self.precomputed_epsilons) - 1)
        self.epsilon = self.precomputed_epsilons[idx]
        self._sample_theta()


def run_single_replicate(seed: int, agent_class: Any, agent_kwargs: Dict[str, Any],
                         mu: np.ndarray, T: int, checkpoints: set) -> Tuple[Dict[int, Dict[int, Tuple[float, int, float]]], np.ndarray]:
    np.random.seed(seed)
    agent = agent_class(**agent_kwargs)
    agent.reset()

    K = len(mu)
    history: Dict[int, Dict[int, Tuple[float, int, float]]] = {}

    def _snapshot() -> Dict[int, Tuple[float, int, float]]:
        out = {}
        for a in range(K):
            n_draws = int(agent.nbDraws[a])
            m_mean = float(agent.cumRewards[a] / max(1, n_draws))
            out[a] = (float(agent.cumRewards[a]), n_draws, m_mean)
        return out

    action_history = np.zeros(T, dtype=int)

    for t in range(1, T + 1):
        arm = agent.play()
        action_history[t - 1] = arm
        reward = np.random.binomial(1, mu[arm])
        agent.update(arm, reward)
        if t in checkpoints:
            history[t] = _snapshot()

    return history, action_history


def simulate_agents(agents_cfg: List[Tuple[str, Any, Dict[str, Any], str]],
                    mu: np.ndarray, T: int, checkpoints: set,
                    N_rep: int, num_cores: int) -> Dict[str, Dict[str, Any]]:
    results = {}
    for name, agent_cls, kwargs, color in agents_cfg:
        raw = Parallel(n_jobs=num_cores)(
            delayed(run_single_replicate)(
                seed=42 + i,
                agent_class=agent_cls,
                agent_kwargs=kwargs,
                mu=mu,
                T=T,
                checkpoints=checkpoints,
            )
            for i in range(N_rep)
        )
        histories = [r[0] for r in raw]
        actions   = np.array([r[1] for r in raw])
        results[name] = {'histories': histories, 'actions': actions, 'color': color}
    return results


def _eps_ts_pdf_data(S_a: float, N_a: float, mu_hat: float, epsilon: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    threshold = 1.0 - epsilon
    if mu_hat > threshold:
        return None, None, True
    alpha = S_a / (1.0 - epsilon) + 1.0
    beta  = N_a + 2.0 - alpha
    alpha = max(1e-9, alpha)
    beta  = max(1e-9, beta)
    scale = 1.0 - epsilon
    x = np.linspace(0, scale, 300)
    pdf = beta_dist.pdf(x / scale, alpha, beta) / scale
    return x, pdf, False


def _ts_pdf_data(S_a: float, N_a: float) -> Tuple[np.ndarray, np.ndarray]:
    alpha = S_a + 1.0
    beta  = N_a + 1.0
    x     = np.linspace(0, 1, 300)
    pdf   = beta_dist.pdf(x, alpha, beta)
    return x, pdf


def plot_sampling_distributions(agent_results: Dict[str, Dict[str, Any]],
                                agents_cfg: List[Tuple[str, Any, Dict[str, Any], str]],
                                mu: np.ndarray, sorted_cps: List[int],
                                eps_at_cp: Dict[int, float], results_dir: str) -> str:
    n_agents = len(agents_cfg)
    n_cps    = len(sorted_cps)
    K        = len(mu)
    arm_colors = ['#2b5c8f', '#d95f02', '#7570b3']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig_raw = plt.figure(figsize=(15, 3.2 * n_agents), dpi=150)
    fig: Figure = cast(Figure, fig_raw)
    gs  = gridspec.GridSpec(n_agents, n_cps, figure=fig, hspace=0.45, wspace=0.28)


    for row_idx, (agent_name, cls, kwargs, ag_color) in enumerate(agents_cfg):
        ag_data = agent_results[agent_name]

        for col_idx, cp in enumerate(sorted_cps):
            ax: Axes = cast(Axes, fig.add_subplot(gs[row_idx, col_idx]))
            if issubclass(cls, EpsilonTSDynamic):
                eps = eps_at_cp[cp]
            elif issubclass(cls, EpsilonTS):
                eps = float(kwargs.get('epsilon', 0.0))
            else:
                eps = 0.0
            threshold = 1.0 - eps

            if eps > 0:
                ax.axvspan(threshold, 1.0, color='#ffe0b2', alpha=0.5, zorder=0)
                ax.axvline(threshold, color='#e65100', lw=1.2, linestyle='--', zorder=1)

            for a in range(K):
                ax.axvline(mu[a], color=arm_colors[a], lw=1.2, linestyle=':', alpha=0.7, zorder=1)

                S_list, N_list, M_list = [], [], []
                for rep in ag_data['histories']:
                    s, n, m = rep[cp][a]
                    S_list.append(s); N_list.append(n); M_list.append(m)

                S_mean = float(np.mean(S_list))
                N_mean = float(np.mean(N_list))
                M_mean = float(np.mean(M_list))

                if eps > 0 and issubclass(cls, EpsilonTS):
                    x, pdf, is_det = _eps_ts_pdf_data(S_mean, N_mean, M_mean, eps)
                    if is_det:
                        ax.axvline(M_mean, color=arm_colors[a], lw=2.2, linestyle='-', zorder=3)
                    elif x is not None and pdf is not None:
                        ax.plot(x, pdf, color=arm_colors[a], lw=1.8, zorder=2)
                        ax.fill_between(x, 0, pdf, color=arm_colors[a], alpha=0.15, zorder=2)
                else:
                    x, pdf = _ts_pdf_data(S_mean, N_mean)
                    ax.plot(x, pdf, color=arm_colors[a], lw=1.8, zorder=2)
                    ax.fill_between(x, 0, pdf, color=arm_colors[a], alpha=0.15, zorder=2)

            if row_idx == 0:
                eps_str = f"eps_t={eps_at_cp[cp]:.3f}" if cp in eps_at_cp else ""
                ax.set_title(f"t = {cp}   ({eps_str})", fontsize=10, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{agent_name}\nPDF / Spike", fontsize=9, fontweight='bold')

            ax.set_xlabel(r"$\theta$", fontsize=9)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=8)

    arm_handles = [
        Line2D([0], [0], color=arm_colors[a], lw=2, linestyle='-',
               label=rf'Arm {a+1} ($\mu_{a+1}={mu[a]}$)')
        for a in range(K)
    ]
    det_handle  = Line2D([0], [0], color='gray', lw=2, linestyle='-',
                         label='Deterministic Spike')
    thresh_patch = Patch(color='#ffe0b2', alpha=0.6,
                         label=r'Near-Optimal Region ($1-\varepsilon$)')

    fig.legend(
        handles=arm_handles + [det_handle, thresh_patch],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(arm_handles) + 2,
        frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.5
    )

    out = os.path.join(results_dir, 'sampling_distribution_analysis.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(out)}")
    return out


def plot_action_frequency(agent_results: Dict[str, Dict[str, Any]],
                          agents_cfg: List[Tuple[str, Any, Dict[str, Any], str]],
                          mu: np.ndarray, T: int, results_dir: str) -> str:
    K     = len(mu)
    times = np.arange(1, T + 1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig_raw, axes_raw = plt.subplots(1, K, figsize=(5 * K, 4), dpi=150, sharey=True)
    fig: Figure = cast(Figure, fig_raw)
    axes_list: List[Axes] = [cast(Axes, ax) for ax in (axes_raw if hasattr(axes_raw, '__iter__') else [axes_raw])]

    for a, ax in enumerate(axes_list):
        for name, cls, kwargs, color in agents_cfg:
            actions = agent_results[name]['actions']
            freq = np.mean(actions == a, axis=0)
            window = 50
            freq_smooth = np.convolve(freq, np.ones(window)/window, mode='same')
            ax.plot(times, freq_smooth, color=color, lw=1.8, label=name)

        ax.set_title(rf'Arm {a+1} ($\mu_{a+1}={mu[a]}$)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time t', fontsize=10)
        if a == 0:
            ax.set_ylabel('Selection Frequency P(A_t = a)', fontsize=10)
        ax.set_xlim(1, T)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=9)

    handles = [
        Line2D([0], [0], color=c, lw=2, label=n)
        for n, cls, kw, c in agents_cfg
    ]
    fig.legend(
        handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=len(agents_cfg), frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10
    )

    out = os.path.join(results_dir, 'action_selection_frequency.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(out)}")
    return out


def plot_epsilon_curves(scenarios: List[ScenarioDict], eps_seqs: List[List[float]],
                        eps_static: float, T: int, results_dir: str) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_raw, axes_raw = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)
    fig: Figure = cast(Figure, fig_raw)
    axes: Any = axes_raw
    times = np.arange(1, T + 1)

    for ax_raw, sc, eps_seq in zip(axes, scenarios, eps_seqs):
        ax: Axes = cast(Axes, ax_raw)
        arr = np.array(eps_seq)
        ax.plot(times, arr, color='#2c7bb6', lw=2.0, label=r'$\varepsilon^*_t$')
        ax.axhline(eps_static, color='#d62728', lw=1.5, linestyle='--',
                   label=rf'$\varepsilon_{{static}}={eps_static}$')
        ax.set_title(sc['title'], fontsize=10, fontweight='bold')
        ax.set_xlabel(r'Time $t$', fontsize=10)
        ax.set_ylabel(r'$\varepsilon^*_t$', fontsize=11)
        ax.set_xlim(1, T)
        ax.set_ylim(-0.01, max(arr.max(), eps_static) * 1.25)
        ax.tick_params(labelsize=9)

    handles = [
        Line2D([0], [0], color='#2c7bb6', lw=2.0, label=r'$\varepsilon^*_t$ (dynamic)'),
        Line2D([0], [0], color='#d62728', lw=1.5, linestyle='--',
               label=rf'$\varepsilon_{{static}}={eps_static}$'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.01),
               ncol=2, frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10)

    fig.suptitle(r'Evolution of $\varepsilon^*_t$ vs. Fixed $\varepsilon_{\mathrm{static}}$',
                 fontsize=12, fontweight='bold', y=1.08)
    fig.tight_layout()
    out = os.path.join(results_dir, 'lenient_vs_practitioner_epsilon.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(out)}")


def run_gap_replicate(seed: int, mu: np.ndarray, T: int, eps_static: float, eps_seq: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    agent = TS(nbArms=len(mu))
    agent.reset()

    mu_star = np.max(mu)
    gaps = mu_star - np.array(mu)
    gaps_eps = np.maximum(0.0, gaps - eps_static)

    cum_hinge = 0.0
    cum_pract = 0.0
    hinge_trace = np.zeros(T)
    pract_trace = np.zeros(T)

    for t in range(T):
        action = agent.play()
        reward = np.random.binomial(1, mu[action])
        agent.update(action, reward)

        cum_hinge += gaps_eps[action]
        cum_pract += max(0.0, gaps[action] - eps_seq[t])

        hinge_trace[t] = cum_hinge
        pract_trace[t] = cum_pract

    return hinge_trace, pract_trace


def plot_regret_gap(scenarios: List[ScenarioDict], all_eps_seqs: List[List[float]],
                    eps_static: float, T: int, N_rep: int, num_cores: int, results_dir: str) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_raw, axes_raw = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)
    fig: Figure = cast(Figure, fig_raw)
    axes: Any = axes_raw
    times = np.arange(1, T + 1)

    for ax_raw, sc, eps_seq in zip(axes, scenarios, all_eps_seqs):
        ax: Axes = cast(Axes, ax_raw)
        mu = sc['mu']
        raw = Parallel(n_jobs=num_cores)(
            delayed(run_gap_replicate)(
                seed=1000 + i,
                mu=mu,
                T=T,
                eps_static=eps_static,
                eps_seq=eps_seq
            )
            for i in range(N_rep)
        )

        hinge_means = np.mean([r[0] for r in raw], axis=0)
        pract_means = np.mean([r[1] for r in raw], axis=0)

        gap = pract_means - hinge_means
        ax.axhline(0, color='black', lw=1.0, linestyle='-', alpha=0.4)
        ax.fill_between(times, 0, gap, where=(gap > 0), color='#d62728', alpha=0.25,
                        label=r'$R^\dagger > R^\varepsilon$ (static wins)')
        ax.fill_between(times, 0, gap, where=(gap <= 0), color='#2c7bb6', alpha=0.25,
                        label=r'$R^\dagger \leq R^\varepsilon$ (dynamic wins)')
        ax.plot(times, gap, color='black', lw=1.8)

        ax.set_title(sc['title'], fontsize=10, fontweight='bold')
        ax.set_xlabel(r'Time $t$', fontsize=10)
        ax.set_ylabel(r'$R^\dagger(t) - R^{\varepsilon}(t)$', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set_xlim(1, T)
        ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(r'Cumulative Gap: $R^\dagger(T) - R^{\varepsilon}(T)$',
                 fontsize=12, fontweight='bold', y=1.04)
    fig.tight_layout()
    out = os.path.join(results_dir, 'lenient_vs_practitioner_gap.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.abspath(out)}")


def main() -> None:
    mu             = np.array([0.9, 0.85, 0.6])
    epsilon_static = 0.2
    T              = 5000
    N_rep          = 300
    checkpoints    = {100, 1000, 4000}
    sorted_cps     = sorted(checkpoints)

    results_dir = os.path.join(_HERE, "images_experiments")
    os.makedirs(results_dir, exist_ok=True)


    gaps      = np.max(mu) - mu
    eps_seq   = compute_eps_seq(gaps, T)
    eps_at_cp = {t: eps_seq[t - 1] for t in sorted_cps}

    print('=' * 65)
    print('SAMPLING DISTRIBUTION AND DIAGNOSTIC ANALYSIS')
    print('=' * 65)

    agents_cfg = [
        ('TS', TS, {'nbArms': len(mu)}, '#d62728'),
        (r'$\varepsilon$-TS', EpsilonTS, {'nbArms': len(mu), 'epsilon': epsilon_static}, '#1f77b4'),
        (r'$\varepsilon_t$-TS', EpsilonTSDynamic, {'nbArms': len(mu), 'precomputed_epsilons': eps_seq}, '#2c3e50'),
    ]

    num_cores = min(multiprocessing.cpu_count(), 8)

    print(f'\n[1/4] Running {N_rep} replicates per agent...')
    agent_results = simulate_agents(agents_cfg, mu, T, checkpoints, N_rep, num_cores)

    print('\n[2/4] Plotting sampling distributions...')
    plot_sampling_distributions(agent_results, agents_cfg, mu, sorted_cps, eps_at_cp, results_dir)

    print('\n[3/4] Plotting action selection frequency...')
    plot_action_frequency(agent_results, agents_cfg, mu, T, results_dir)

    scenarios: List[ScenarioDict] = [
        {'name': 'Sc1', 'title': r'Arms $[\mu_1=0.5,\;\mu_2=0.2]$', 'mu': np.array([0.5, 0.2])},
        {'name': 'Sc2', 'title': r'Arms $[\mu_1=0.9,\;\mu_2=0.6]$', 'mu': np.array([0.9, 0.6])},
        {'name': 'Sc3', 'title': r'Arms $[\mu_1=0.5,\;\mu_2=0.45,\;\mu_3=0.2]$', 'mu': np.array([0.5, 0.45, 0.2])},
        {'name': 'Sc4', 'title': r'Arms $[\mu_1=0.9,\;\mu_2=0.85,\;\mu_3=0.6]$', 'mu': np.array([0.9, 0.85, 0.6])},
    ]

    all_eps_seqs = [compute_eps_seq(np.max(sc['mu']) - sc['mu'], T) for sc in scenarios]

    print('\n[4/4] Plotting diagnostic epsilon curves and regret gap...')
    plot_epsilon_curves(scenarios, all_eps_seqs, epsilon_static, T, results_dir)
    plot_regret_gap(scenarios, all_eps_seqs, epsilon_static, T, N_rep, num_cores, results_dir)

    print('\n' + '=' * 65)
    print('[DONE] All figures saved to:', os.path.abspath(results_dir))
    print('=' * 65)


if __name__ == '__main__':
    main()
