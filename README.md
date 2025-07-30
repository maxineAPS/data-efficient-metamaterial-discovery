# Data-Efficient Discovery of Hyperelastic TPMS Metamaterials

This repository contains the official implementation for the paper: **Data-Efficient Discovery of Hyperelastic TPMS Metamaterials with Extreme Energy Dissipation**.

**Paper Link:** <https://doi.org/10.1145/3721238.3730759>

## Overview

This project uses a data-efficient method driven by physical experiments to discover novel metamaterial structures with exceptional energy-dissipation capabilities.

Our method uses a batch Bayesian optimization framework to guide the discovery process:

1. A **Deep Ensemble** model is trained on physical experiment data to predict stress-strain behavior and uncertainty.

2. A penalized **Upper Confidence Bound (UCB)** acquisition function selects new candidates by balancing exploration (improving model accuracy) and exploitation (maximizing energy dissipation).

3. This iterative loop efficiently discovers high-performing structures within a limited experimental budget.

## Setup and Installation

#### 1. Clone Repository

```bash
git clone https://github.com/maxineAPS/data-efficient-metamaterial-discovery
cd data-efficient-metamaterial-discovery
```

#### 2. Create Virtual Environment & Install Dependencies

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

#### 3. Download Dataset

Download the experimental dataset from [https://github.com/maxineAPS/energy-dissipation-metamaterial-dataset.git](https://github.com/maxineAPS/energy-dissipation-metamaterial-dataset.git). Unzip and place the contents in the `data/` directory.

## Usage

#### 1. Train the Ensemble

Run `train.py` to train the deep ensemble on the experimental data in `data/`. The models will be saved to the `models/` directory.

```bash
python train.py
```

#### 2. Select Next Batch

Run `bayesian.py` to identify the next batch of promising structures to fabricate. This script uses the trained ensemble to select the top 40 candidates.

```bash
python bayesian.py
```

The selected parameters will be saved to `bayesian_selected_points.json`.

#### 3. Visualize Predictions (Optional)

Run the visualization script to check the performance of your trained ensemble against test data.

```bash
python visualize_predictions.py
