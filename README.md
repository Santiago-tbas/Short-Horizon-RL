# Short-Horizon Reinforcement Learning (Short-Horizon-RL)

**Author:** Edison Santiago Tutalcha Bastidas  
**Repository:** [Santiago-tbas/Short-Horizon-RL](https://github.com/Santiago-tbas/Short-Horizon-RL)

---

## 📌 Project Overview

This repository contains research code, custom experiments, analysis tools, and interactive dashboards focused on **Short-Horizon Reinforcement Learning** and **Multi-Armed Bandit (MAB)** algorithms. 

Key research topics explored in this codebase:
- **Practitioner Regret vs. Lenient Regret**: Evaluating algorithms (such as Thompson Sampling, UCB, IMED) under practitioner-oriented cost and regret metrics.
- **Dynamic $\epsilon$-Thompson Sampling**: Adaptive exploration policies for short-horizon decision processes.
- **Multi-Arm Experiments**: Benchmarking regret metrics under multi-armed setups and varying horizon lengths $T$.
- **Interactive Dashboard**: A custom web server and frontend interface for inspecting experimental runs and plotting regret curves.

---

## 📁 Repository Structure & Ownership Breakdown

### 🔹 Original Contributions (Author: Edison Santiago Tutalcha Bastidas)
- **`experiments/PRACTITIONER REGRET/`**: Custom experiment scripts, dynamic $\epsilon$ strategies, lenient regret comparisons (`practitioner_leniency_comparison.py`, `pract_dynamic.py`, `practitioner_5_arms.py`).
- **`dashboard/`**: Custom web application (`server.py`, `app.js`, `index.html`, `index.css`) for live visualization of RL metrics.
- **`results_mab/` & `results_pract/`**: Experimental outputs, generated plots (PDF/PNG), log files, and raw data objects.
- **`Notes & PDF/`**: Mathematical derivations, notes (`Short_Horizon___Notes.pdf`), and referenced literature.

### 🔸 Third-Party Framework Dependencies & Attribution
This repository incorporates modules from the open-source **[StatisticalRL](https://github.com/StatisticalRL)** suite to provide standard RL environments and base bandit learners:
- **`environments/`**: Open-source environments package derived from [`StatisticalRL/environments`](https://github.com/StatisticalRL/environments).
- **`learners/`**: Open-source algorithms package derived from [`StatisticalRL/learners`](https://github.com/StatisticalRL/learners).
- **`experiments/src/`**: Framework runner components derived from [`StatisticalRL/experiments`](https://github.com/StatisticalRL/experiments).

All third-party modules are included in compliance with the **MIT License** (Copyright © 2025 StatisticalRL).

---

## ⚖️ License & Intellectual Property Notice

This repository is distributed under the **MIT License**.

- Original work & custom experiment code: **Copyright © 2026 Edison Santiago Tutalcha Bastidas**
- Included framework components (`environments/`, `learners/`, `experiments/src/`): **Copyright © 2025 StatisticalRL**

See the full [`LICENSE`](LICENSE) file for complete licensing terms.

---

## 🚀 Getting Started

### Installation
Clone the repository and set up a Python 3.9+ virtual environment:

```bash
git clone https://github.com/Santiago-tbas/Short-Horizon-RL.git
cd Short-Horizon-RL
python3 -m venv .venv
source .venv/bin/activate
pip install -r pyproject.toml
```

### Running Practitioner Regret Experiments
```bash
python "experiments/PRACTITIONER REGRET/Lenient vs practitioner regret/practitioner_leniency_comparison.py"
```

### Launching the Dashboard
```bash
python dashboard/server.py
```
Then open `http://localhost:8000` in your browser.
