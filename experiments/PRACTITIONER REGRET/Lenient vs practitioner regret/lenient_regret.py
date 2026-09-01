import os
import sys
from typing import TypedDict, Tuple, List, Dict, Any, cast
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_HERE, '../../../learners/src'))
sys.path.insert(0, os.path.join(_HERE, '../../../environments/src'))
sys.path.insert(0, os.path.join(_HERE, '../../..'))

from statisticalrl_learners.MABs.TS import TS
from statisticalrl_learners.MABs.Algo1 import Algo1

class ScenarioDict(TypedDict):
    name: str
    title: str
    mu: np.ndarray
    pos: Tuple[int, int]
    ylim_top: float

class AgentConfig:
    def __init__(self, agent_class: Any, name: str, kwargs_factory: Any) -> None:
        self.agent_class = agent_class
        self.name = name
        self.kwargs_factory = kwargs_factory


def run_single_replicate(seed: int, agent_class: Any, agent_kwargs: Dict[str, Any],
                         mu: np.ndarray, T: int, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    agent = agent_class(**agent_kwargs)
    agent.reset()
    
    mu_star = np.max(mu)
    gaps = mu_star - np.array(mu)
    gaps_eps = np.maximum(0.0, gaps - epsilon)
    
    std_trace = np.zeros(T)
    lenient_trace = np.zeros(T)
    cum_std = 0.0
    cum_lenient = 0.0

    for t in range(T):
        action = agent.play()
        reward = np.random.binomial(1, mu[action])
        agent.update(action, reward)
        
        cum_std += gaps[action]
        cum_lenient += gaps_eps[action]
        
        std_trace[t] = cum_std
        lenient_trace[t] = cum_lenient
        
    return std_trace, lenient_trace


def main() -> None:
    epsilon: float = 0.2
    T: int = 5000
    N: int = 1000   
    
    scenarios: List[ScenarioDict] = [
        {
            "name": "Scenario 1",
            "title": r"Arms = $[\mu_1=0.5, \mu_2=0.2]$",
            "mu": np.array([0.5, 0.2]),
            "pos": (0, 0),
            "ylim_top": 13.5
        },
        {
            "name": "Scenario 2",
            "title": r"Arms = $[\mu_1=0.9, \mu_2=0.6]$",
            "mu": np.array([0.9, 0.6]),
            "pos": (0, 1),
            "ylim_top": 8.5
        },
        {
            "name": "Scenario 3",
            "title": r"Arms = $[\mu_1=0.5, \mu_2=0.45, \mu_3=0.2]$",
            "mu": np.array([0.5, 0.45, 0.2]),
            "pos": (1, 0),
            "ylim_top": 35.0
        },
        {
            "name": "Scenario 4",
            "title": r"Arms = $[\mu_1=0.9, \mu_2=0.85, \mu_3=0.6]$",
            "mu": np.array([0.9, 0.85, 0.6]),
            "pos": (1, 1),
            "ylim_top": 105.0
        }
    ]
    
    agents: List[AgentConfig] = [
        AgentConfig(TS, "TS", lambda num_arms: {"nbArms": num_arms}),
        AgentConfig(Algo1, "EpsilonTS", lambda num_arms: {"nbArms": num_arms, "epsilon": epsilon})
    ]
    
    results_dir = os.path.join(_HERE, "images_experiments")
    os.makedirs(results_dir, exist_ok=True)

    
    num_cores: int = min(multiprocessing.cpu_count(), 16)
    print("=" * 70)
    print(f"PAPER REPLICATION : epsilon = {epsilon}, T = {T}, N = {N}")
    print("=" * 70)
    
    plt.style.use('default')
    fig_raw, axes_raw = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    fig: Figure = cast(Figure, fig_raw)
    axes: Any = axes_raw
    times = np.arange(1, T + 1)
    
    for sc in scenarios:
        mu = sc["mu"]
        num_arms = len(mu)
        row, col = sc["pos"]
        ax: Axes = cast(Axes, axes[row, col])
        
        print(f"\nSimulating {sc['name']}: {sc['title']}...")
        
        agent_traces: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for agent in agents:
            raw_traces = Parallel(n_jobs=num_cores)(
                delayed(run_single_replicate)(
                    seed=12345 + i,
                    agent_class=agent.agent_class,
                    agent_kwargs=agent.kwargs_factory(num_arms),
                    mu=mu,
                    T=T,
                    epsilon=epsilon
                )
                for i in range(N)
            )
            std_mean = np.mean([trace[0] for trace in raw_traces], axis=0)
            lenient_mean = np.mean([trace[1] for trace in raw_traces], axis=0)
            agent_traces[agent.name] = (std_mean, lenient_mean)

        ts_std, ts_hinge = agent_traces["TS"]
        eps_std, eps_hinge = agent_traces["EpsilonTS"]

        ax.plot(times, ts_hinge, label=r"Hinge: TS", color="#1f77b4", linestyle="--", linewidth=1.5)
        ax.plot(times, eps_hinge, label=r"Hinge: $\epsilon$-TS", color="#1f77b4", linestyle="-", linewidth=1.5)
        ax.plot(times, ts_std, label=r"Standard: TS", color="#ff7f0e", linestyle="--", linewidth=1.5)
        ax.plot(times, eps_std, label=r"Standard: $\epsilon$-TS", color="#ff7f0e", linestyle="-", linewidth=1.5)

        ax.set_title(sc["title"], fontsize=12, pad=6)
        ax.set_xlabel(r"$T$", fontsize=11)
        ax.set_ylabel(r"$R_f(T)$", fontsize=11)
        ax.set_xlim(0, T)
        ax.set_ylim(bottom=0, top=sc["ylim_top"])
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="gray", fontsize=9.5)

    fig.suptitle("Lenient Regret", fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    plot_path = os.path.join(results_dir, "lenient_regret.png")
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    print("=" * 70)
    print(f"[SUCCESS] Figure 3 saved at: {os.path.abspath(plot_path)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
