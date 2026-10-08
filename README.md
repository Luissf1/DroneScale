# Fuzzy Adaptive Scaling of PSO-Optimized PID Controllers for Quadrotor UAVs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements a hybrid methodology for quadrotor PID control that combines:

1. **Scaling Laws** — Dimensional-analysis-based scaling of PSO-optimized PID gains across quadrotor sizes
2. **Type-1 Fuzzy Adaptive Scaling** — Real-time gain adjustment based on payload, turbulence, battery state, and error dynamics

## Key Features

-  Validated scaling laws: Kp ∝ m^0.98, Ki ∝ m^0.96, Kd ∝ m^0.99
-  Fuzzy adaptive system: 12.4% average robustness improvement
-  Computational efficiency: 75% reduction in optimization effort
-  Modular architecture: clean, documented, testable code
-  Reproducible experiments: complete experiment pipeline

## Installation

```bash
git clone https://github.com/Luissf1/DroneScale.git
cd fuzzy-adaptive-quadrotor-pid
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
