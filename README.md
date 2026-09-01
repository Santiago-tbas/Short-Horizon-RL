# Short-Horizon Reinforcement Learning

A research codebase for evaluating multi-armed bandit (MAB) algorithms and decision-making policies in short-horizon regimes. This project focuses on formalizing and benchmarking **practitioner regret** metrics, contrasting them with standard and lenient regret formulations, and implementing dynamic exploration-exploitation mechanisms.

---

## Technical Overview

### 1. Practitioner Regret vs. Lenient Regret Metrics
Standard cumulative regret measures the difference between optimal expected reward and the algorithm's empirical performance over horizon $T$. In contrast, **practitioner regret** introduces a tolerance parameter $\epsilon$, evaluating performance based on whether selected actions satisfy domain-specific efficiency thresholds:
- **Lenient Regret**: Penalizes choices only when the suboptimality gap exceeds a tolerance threshold $\epsilon$.
- **Practitioner Regret Evaluation**: Benchmarks algorithms across various gap configurations ($\Delta$) and arm dimensionalities ($K \in \{2, 5, 20, 50\}$).
- **Sampling Distribution Analysis**: Tracks action selection frequency across time steps to study exploration decay.

### 2. Dynamic $\epsilon$-Thompson Sampling Strategies
- **Static vs. Dynamic Schedules**: Evaluates fixed tolerance thresholds $\epsilon$ against dynamic schedules $\epsilon(t)$ that adapt over time.
- **Analytical Parameter Optimization**: Integrates numerical calculations (`epsilon.py`) to compute optimal $\epsilon^*$ parameters for Gaussian and Bernoulli reward distributions.
- **Evolution Tracking**: Analyzes how dynamic decay schedules alter empirical regret bounds in short-horizon settings ($T \le 1000$).

### 3. Evaluated Algorithms
The framework evaluates both classic and modified exploration strategies:
- **Thompson Sampling (TS)** (Bernoulli and Gaussian variants)
- **Upper Confidence Bound (UCB)** (Standard UCB1 and Bernstein variants)
- **Indexed Minimum Empirical Divergence (IMED)**
- **Dynamic $\epsilon$-Thompson Sampling (`DynamicEpsilonTS`)**
- **Random Policy & Oracle Baselines**

### 4. Interactive Visualization Dashboard
Includes a web-based server and frontend dashboard (`dashboard/`) allowing real-time monitoring and comparative plotting of experiment runs:
- **Cumulative Regret Curves**: Linear and log-log scale visualization.
- **Dynamic Parameter Sweeps**: Interactive inspection of $\epsilon(t)$ decay curves and distribution shifts across iterations.

---

## Repository Organization

```
.
├── dashboard/               # Web server (server.py) and frontend (app.js, index.html) for experiment visualization
├── environments/            # Multi-armed bandit and MDP environment definitions (StatisticalRL framework)
├── learners/                # Core bandit algorithms, interfaces, and policy classes (StatisticalRL framework)
├── experiments/             # Experiment runners and evaluation suites
│   └── PRACTITIONER REGRET/ # Core research modules:
│       ├── Epsilon/         # Analytical epsilon solvers, static/dynamic schedule evaluations, animations
│       └── Lenient vs practitioner regret/
│           ├── practitioner_leniency_comparison.py  # 2-arm practitioner vs. lenient benchmarks
│           ├── sampling_distribution_analysis.py    # Action selection and pull distribution metrics
│           └── many_arms/                           # Scaled benchmarks for K=5, K=20, and K=50 arm setups
├── results_mab/             # Output plots (PDF/PNG), log files, and raw data for standard MAB benchmarks
└── results_pract/           # Output figures, metadata, and raw pickle datasets for practitioner regret runs
```

---

## Getting Started

### Prerequisites & Dependencies
Python 3.9 or higher is required. Install core dependencies:

```bash
pip install numpy scipy matplotlib joblib
```

### Running Experiments
To execute the practitioner regret comparison benchmark:

```bash
python "experiments/PRACTITIONER REGRET/Lenient vs practitioner regret/practitioner_leniency_comparison.py"
```

To run multi-arm scaling evaluations:

```bash
python "experiments/PRACTITIONER REGRET/Lenient vs practitioner regret/many_arms/practitioner_5_arms.py"
```

### Launching the Web Dashboard
Start the local dashboard server:

```bash
python dashboard/server.py
```
Open `http://localhost:8000` in your web browser to explore interactive regret plots and experiment logs.

---

## Attribution & Dependencies

This repository builds upon environment modules, algorithm interfaces, and experiment runners from the open-source **[StatisticalRL](https://github.com/StatisticalRL)** framework (`environments/`, `learners/`, and `experiments/src/`).

All incorporated framework components are used under the terms of the MIT License (Copyright © 2025 StatisticalRL).

---

## License

This repository is distributed under the MIT License. See [LICENSE](LICENSE) for full terms and conditions.
