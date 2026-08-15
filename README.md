# 🚚 Smart Fleet Platform: Adaptive AI Coordination

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B.svg)](https://streamlit.io)
[![Reinforcement Learning](https://img.shields.io/badge/RL-PPO_(Stable_Baselines3)-purple.svg)]()
[![Optimization](https://img.shields.io/badge/Solver-OR--Tools-orange.svg)]()

> **An RL-based fleet coordination platform for dynamic pick-and-drop environments, designed to make real-time vehicle–shipment assignment decisions under changing operating conditions.**

## 📖 Overview

Modern fleet operations are rarely static. Vehicles must continuously serve shipments while operating under changing conditions such as traffic congestion, vehicle breakdowns, and sudden demand surges.

Traditional routing approaches such as greedy dispatch heuristics and global optimization solvers can provide strong solutions for well-defined routing problems. However, dynamic environments introduce a different challenge: decisions may need to be reconsidered as the fleet state and operating conditions change.

The **Smart Fleet Platform** addresses this problem using a **reinforcement-learning-based fleet coordinator**. The PPO policy observes the current target shipment, fleet-wide state observations, vehicle availability, capacity, and environmental conditions, and selects the vehicle to assign to the shipment.

Rather than optimizing only static route distance, the platform evaluates coordination across multiple operational objectives, including mission completion, on-time delivery, fleet utilization, disruption recovery, recovery time, and adaptation cost.

The system is benchmarked against both a **Greedy Dispatch heuristic** and an **OR-Tools CVRP optimizer** using controlled, multi-seed experiments across different fleet sizes and disruption scenarios.

**We deliver an intelligent fleet coordination platform that moves beyond static route optimization to make fast online vehicle–shipment assignment decisions and adapt the fleet under changing operating conditions, backed by rigorous multi-seed evaluation against classical baselines.**

## ✨ Key Features

* **RL Fleet Coordinator:** A PPO-based policy that makes vehicle–shipment assignment decisions from the current fleet and environment state.
* **Interactive Operations Dashboard:** A Streamlit-based operational dashboard for monitoring fleet state, vehicle positions, shipments, alerts, and simulated disruptions.
* **Classical Optimization Baselines:** Includes a Greedy Dispatch heuristic and an OR-Tools CVRP-based optimizer for comparative evaluation.
* **Dynamic Disruption Modeling:** Simulates traffic congestion, vehicle breakdowns, and demand surges to evaluate how coordination strategies respond to changing conditions.
* **Multi-Seed Evaluation:** Uses controlled offline experiments across multiple random seeds and fleet sizes to assess robustness and scalability.
* **Adaptive Performance Metrics:** Evaluates not only routing efficiency but also completion rate, on-time delivery, fleet utilization, recovery rate, recovery time, and disruption adaptation cost.

## 🚀 Installation & Quick Start

**1. Clone the repository:**
git clone [https://github.com/luaGeeko/FleetSIH.git](https://github.com/luaGeeko/FleetSIH.git)
``` bash
cd FleetSIH
```
**2. Install dependencies:**
```
pip install -r requirements.txt
```

**3. Launch the Live Dashboard:**
```bash
streamlit run app.py
```

## 🧠 System Architecture

The platform consists of four main components:

1. **Fleet Simulator**
   - Maintains vehicle, shipment, route, and environmental state.
   - Simulates vehicle movement and delivery execution.
   - Supports traffic, breakdown, and demand-surge disruptions.

2. **RL Fleet Coordinator**
   - PPO policy trained using the Gymnasium environment.
   - Observes the target shipment, fleet state, vehicle capacity,
     availability, and active traffic conditions.
   - Selects the vehicle for the next shipment assignment.

3. **Classical Baselines**
   - Greedy nearest-vehicle dispatch.
   - OR-Tools CVRP optimization.

4. **Evaluation & Dashboard**
   - Streamlit dashboard for real-time visualization.
   - Offline multi-seed benchmark suite for controlled comparison.


## Scientific Validation & Results
The platform evaluates three coordination strategies under controlled operating conditions:
Greedy Dispatch — nearest-vehicle assignment heuristic.
OR-Tools CVRP — global combinatorial routing optimization.
RL Fleet Coordinator — PPO-based online vehicle–shipment assignment.
Evaluation is performed using multiple random seeds across different fleet sizes and disruption scenarios, including traffic spikes, vehicle breakdowns, demand surges, and combined disruptions.

Rather than evaluating routing distance alone, the benchmark measures:
Mission Success — shipment completion rate.
Reliability — on-time delivery rate.
Resource Management — average fleet utilization.
Static Efficiency — total distance travelled.
Resilience — recovery rate after disruption.
Agility — number of simulation ticks required to recover.
Adaptation Cost — additional distance incurred under disruption relative to the corresponding normal operating condition.

$$
\text{Adaptation Cost} = \left( \frac{D_{\text{disrupted}} - D_{\text{normal}}}{D_{\text{normal}}} \right) \times 100
$$

The experiments are designed to examine a key trade-off: static routing efficiency versus adaptability under changing operating conditions.

Fast online decision-making: The RL coordinator is not intended to replace exact routing optimization for static CVRP instances. Instead, it is evaluated as an online coordination mechanism. The trained PPO policy produces vehicle-assignment decisions directly from the current state, avoiding the need to solve a new global routing optimization problem at every decision point.

To reproduce these results, run the offline evaluation suite
```bash
# Run the multi-seed benchmark matrix
python evaluation/benchmark.py

# Generate the performance dashboard plots
python evaluation/plot_results.py
```

### AI Coordinator Performance: AI Coordinator (V0) vs. Advanced Constraints in AI Coordinator (V1)

| Metric               |   V0 (Baseline) |   V1 (Advanced) | Change (%)   | Result              |
|:---------------------|----------------:|----------------:|:-------------|:--------------------|
| Completion (%)       |           94.88 |           95.22 | +0.4%        | ✅ Improved         |
| On-Time (%)          |           92.88 |           90.96 | -2.1%        | ⚠️ Trade-off       |
| Distance (km)        |         2017.31 |         1967.55 | -2.5%        | ✅ Improved         |
| Utilization (%)      |           84.75 |           84.88 | +0.2%        | ✅ Improved         |
| Avg Recovery (Ticks) |            4.7  |            3.77 | -19.9%       | ✅ Fixed & Improved |
| Adaptation Cost (%)  |           21.19 |           23.81 | +12.3%       | ⚠️ Trade-off       |

## What We Deliver
* **Real-Time Fleet Operations Center:** Monitor vehicles, shipments, capacity, traffic, and real-time disruptions through an interactive Streamlit dashboard.
* **Intelligent Vehicle–Shipment Assignment:** Compare traditional Greedy Dispatch heuristics and global OR-Tools CVRP optimization against our learned PPO-based RL Fleet Coordinator.
* **Resilient Logistics Under Disruption:** Stress-test systems using randomized simulation scenarios—including traffic spikes, vehicle breakdowns, and demand surges.
* **Rigorous Multi-Seed Benchmarking:** Evaluate performance not just on static distance, but across mission completion, on-time delivery, fleet utilization, disruption recovery rate, recovery time, and Adaptation Cost.
