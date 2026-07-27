import numpy as np
import pandas as pd
from dmdu.prim import PRIM
from dmdu.cart import ScenarioCART

# 1. Simulate experimental data (X: Uncertainties, y: Vulnerability Flag)
np.random.seed(42)
n_experiments = 1000
X = pd.DataFrame({
    "rainfall_trend": np.random.uniform(-10, 10, n_experiments),
    "demand_growth": np.random.uniform(0.5, 3.5, n_experiments),
    "infrastructure_cost": np.random.uniform(50, 150, n_experiments)
})

# Vulnerability occurs primarily when rainfall drops AND demand is high
y = ((X["rainfall_trend"] < -3.0) & (X["demand_growth"] > 2.2)).astype(int)

# 2. Run PRIM
prim = PRIM(peel_alpha=0.05, min_support=0.1)
prim.fit(X, y)
print("--- PRIM Peeling Trajectory (Top 3) ---")
print(prim.get_peeling_trajectory().tail(3))

# 3. Run CART
cart = ScenarioCART(max_depth=3, density_threshold=0.7)
cart.fit(X, y)
print("\n--- Discovered CART Scenarios ---")
print(cart.get_scenarios()[["density", "coverage", "rule_definition"]])