# Short-Horizon Reinforcement Learning

A research codebase for evaluating multi-armed bandit (MAB) algorithms and reinforcement learning policies in short-horizon decision-making processes. This project focuses on comparing practitioner regret metrics, analyzing lenient vs. standard regret formulations, and evaluating dynamic exploration strategies.

---

## Technical Overview

- **Practitioner vs. Lenient Regret Analysis**: Empirical evaluation of multi-armed bandit algorithms (Thompson Sampling, UCB, IMED) comparing practitioner-focused cost metrics against conventional lenient regret definitions.
- **Dynamic $\epsilon$-Thompson Sampling**: Parameter adaptation schedules designed to optimize decision-making under constrained time horizons ($T$).
- **Multi-Arm Benchmarks**: Performance evaluations across different arm distributions, parameters, and time horizons.
- **Interactive Dashboard**: A web-based server and frontend interface for monitoring experiment runs and plotting regret curves in real time.

---

## Repository Organization

```
.
├── dashboard/               # Server and frontend interface for tracking experiment execution
├── environments/            # RL environment modules (StatisticalRL framework)
├── learners/                # MAB and RL algorithm implementations (StatisticalRL framework)
├── experiments/             # Benchmark scripts and experiment configurations
│   └── PRACTITIONER REGRET/ # Practitioner regret experiments and dynamic epsilon models
├── results_mab/             # Generated figures, plots, and logs for MAB experiments
└── results_pract/           # Output metrics, plots, and data for practitioner regret runs
```

---

## Attribution & Dependencies

This repository utilizes environment modules, algorithm definitions, and execution utilities from the open-source **[StatisticalRL](https://github.com/StatisticalRL)** framework (`environments/`, `learners/`, and core `experiments/src/` components).

All incorporated framework components are used in accordance with the MIT License (Copyright © 2025 StatisticalRL).

---

## License

This repository is distributed under the MIT License. For complete terms, see the [LICENSE](LICENSE) file.
