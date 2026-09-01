import os
import sys
from typing import TypedDict, Tuple, List, Dict, Any
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_HERE, '../Epsilon'))
sys.path.insert(0, os.path.join(_HERE, '../../../learners/src'))
sys.path.insert(0, os.path.join(_HERE, '../../../environments/src'))
sys.path.insert(0, os.path.join(_HERE, '../../..'))

from statisticalrl_learners.MABs.TS import TS
from statisticalrl_learners.MABs.Algo1 import Algo1
from epsilon import compute_epsilon_gaussian as compute_epsilon_star


class ScenarioConfig(TypedDict):
    name: str
    title: str
    mu: np.ndarray
    pos: Tuple[int, int]
    ylim_top: float


class DynamicEpsilonTS(Algo1):
    def __init__(self, nbArms: int, precomputed_epsilons: np.ndarray):
        super().__init__(nbArms, epsilon=precomputed_epsilons[0])
        self.precomputed_epsilons = precomputed_epsilons
        self.time = 0

    def reset(self):
        super().reset()
        self.time = 0

    def update(self, arm: int, reward: float):
        self.cumRewards[arm] += reward
        self.nbDraws[arm] += 1
        self.empMeans[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.time += 1
        idx = min(self.time, len(self.precomputed_epsilons) - 1)
        self.epsilon = self.precomputed_epsilons[idx]
        self._sample_theta()


class AgentConfig:
    def __init__(self, cls: Any, name: str, kw: Any):
        self.agent_class = cls
        self.name = name
        self.kwargs_factory = kw


def run_single_replicate(seed: int, agent_class: Any, agent_kwargs: Dict[str, Any],
                         mu: np.ndarray, T: int, eps: float,
                         eps_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    agent = agent_class(**agent_kwargs)
    agent.reset()

    gaps = np.max(mu) - mu
    g_len = np.maximum(0.0, gaps - eps)

    actions = np.empty(T, dtype=int)
    for t in range(T):
        a = agent.play()
        agent.update(a, np.random.binomial(1, mu[a]))
        actions[t] = a

    g = gaps[actions]
    return (np.cumsum(g),
            np.cumsum(g_len[actions]),
            np.cumsum(np.maximum(0.0, g - eps_seq)))


def main():
    eps  = 0.2
    T    = 5000
    N    = 5000

    scenarios: List[ScenarioConfig] = [
        {"name": "S1", "title": r"$\mu=[0.5,\,0.2]$",             "mu": np.array([0.5, 0.2]),         "pos": (0, 0), "ylim_top": 9.0},
        {"name": "S2", "title": r"$\mu=[0.9,\,0.6]$",             "mu": np.array([0.9, 0.6]),         "pos": (0, 1), "ylim_top": 6.0},
        {"name": "S3", "title": r"$\mu=[0.5,\,0.45,\,0.2]$",      "mu": np.array([0.5, 0.45, 0.2]),   "pos": (1, 0), "ylim_top": 36.0},
        {"name": "S4", "title": r"$\mu=[0.9,\,0.85,\,0.6]$",      "mu": np.array([0.9, 0.85, 0.6]),   "pos": (1, 1), "ylim_top": 105.0},
    ]

    agents = [
        AgentConfig(TS,               "TS",               lambda K, e: {"nbArms": K}),
        AgentConfig(Algo1,            "EpsilonTS",        lambda K, e: {"nbArms": K, "epsilon": eps}),
        AgentConfig(DynamicEpsilonTS, "DynamicEpsilonTS", lambda K, e: {"nbArms": K, "precomputed_epsilons": e}),
    ]

    out_dir   = os.path.join(_HERE, "images_experiments")
    os.makedirs(out_dir, exist_ok=True)
    n_jobs    = min(multiprocessing.cpu_count(), 16)

    print("=" * 70)
    print(f"PRACTITIONER LENIENCY COMPARISON  eps={eps}  T={T}  N={N}")
    print("=" * 70)

    plt.style.use('default')
    fig: Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), dpi=150)  # type: ignore[assignment]
    times = np.arange(1, T + 1)

    for sc in scenarios:
        mu     = sc["mu"]
        K      = len(mu)
        ax     = axes[sc["pos"]]
        gaps   = np.max(mu) - mu
        e_seq  = np.array([float(compute_epsilon_star(gaps, t, sigma=0.5, eta=0.5)[0]) for t in range(1, T + 1)])

        print(f"\n{sc['name']}: {sc['title']}")

        traces = {}
        for ag in agents:
            runs = Parallel(n_jobs=n_jobs)(
                delayed(run_single_replicate)(
                    seed=12345 + i,
                    agent_class=ag.agent_class,
                    agent_kwargs=ag.kwargs_factory(K, e_seq),
                    mu=mu,
                    T=T,
                    eps=eps,
                    eps_seq=e_seq,
                )
                for i in range(N)
            )
            std, hng, prt = np.array(runs).mean(axis=0)
            traces[ag.name] = (std, hng, prt)

        ts_std,  ts_hng,  ts_prt  = traces["TS"]
        eps_std, eps_hng, eps_prt = traces["EpsilonTS"]
        dyn_std, dyn_hng, dyn_prt = traces["DynamicEpsilonTS"]

        ax.plot(times, ts_hng,  color="#1f77b4", linestyle="--", linewidth=1.5)
        ax.plot(times, eps_hng, color="#1f77b4", linestyle="-",  linewidth=1.5)
        ax.plot(times, dyn_hng, color="#1f77b4", linestyle=":",  linewidth=1.5)

        ax.plot(times, ts_std,  color="#ff7f0e", linestyle="--", linewidth=1.5)
        ax.plot(times, eps_std, color="#ff7f0e", linestyle="-",  linewidth=1.5)
        ax.plot(times, dyn_std, color="#ff7f0e", linestyle=":",  linewidth=1.5)

        ax.plot(times, ts_prt,  color="#2ca02c", linestyle="--", linewidth=1.5)
        ax.plot(times, eps_prt, color="#2ca02c", linestyle="-",  linewidth=1.5)
        ax.plot(times, dyn_prt, color="#2ca02c", linestyle=":",  linewidth=1.5)

        ax.set_title(sc["title"], fontsize=12, pad=6)
        ax.set_xlabel(r"$T$", fontsize=11)
        ax.set_ylabel(r"$R_f(T)$", fontsize=11)
        ax.set_xlim(0, T)
        ax.set_ylim(bottom=0, top=sc["ylim_top"])

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.suptitle("Practitioner Regret Comparison", fontsize=15, fontweight="bold", y=0.985)

    handles = [
        Line2D([0], [0], color="#1f77b4", lw=1.5, linestyle="--", label=r"Hinge: TS"),
        Line2D([0], [0], color="#1f77b4", lw=1.5, linestyle="-",  label=r"Hinge: $\epsilon$-TS"),
        Line2D([0], [0], color="#1f77b4", lw=1.5, linestyle=":",  label=r"Hinge: $\epsilon_t$-TS"),
        Line2D([0], [0], color="#ff7f0e", lw=1.5, linestyle="--", label=r"Standard: TS"),
        Line2D([0], [0], color="#ff7f0e", lw=1.5, linestyle="-",  label=r"Standard: $\epsilon$-TS"),
        Line2D([0], [0], color="#ff7f0e", lw=1.5, linestyle=":",  label=r"Standard: $\epsilon_t$-TS"),
        Line2D([0], [0], color="#2ca02c", lw=1.5, linestyle="--", label=r"Practitioner: TS"),
        Line2D([0], [0], color="#2ca02c", lw=1.5, linestyle="-",  label=r"Practitioner: $\epsilon$-TS"),
        Line2D([0], [0], color="#2ca02c", lw=1.5, linestyle=":",  label=r"Practitioner: $\epsilon_t$-TS"),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="gray",
        fontsize=9.5,
        columnspacing=2.2,
        borderpad=0.6,
    )

    out = os.path.join(out_dir, "practitioner_leniency_comparison.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n{'=' * 70}\n[OK] {os.path.abspath(out)}\n{'=' * 70}")


if __name__ == "__main__":
    main()
