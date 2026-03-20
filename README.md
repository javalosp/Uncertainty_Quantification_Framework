# Uncertainty Quantification Framework for LCA & Dynamic MFA

A Python-based uncertainty analysis method for quantifying the impact of uncertainty in Life Cycle Assessment (LCA) and Dynamic Material Flow Analysis (MFA).

This tool implements a Hybrid **Independent Random Set sampling (IRS)** and **Probability Box (P-Box)** approach to separate:
* **Aleatory Variability (Natural Fluctuation):** Irreducible physical randomness (e.g., grid intensity, temperature).
* **Epistemic Uncertainty (Ignorance):** Reducible lack of knowledge (e.g., missing data, unverified estimates).

The result is a **Probability Box (P-Box)** that shows the "Ignorance Gap," allowing analysts to calculate the precise "Ignorance Penalty" required to guarantee reliability in estimations and forecasts.

---

### Key Features

#### Life Cycle Assessment (LCA)
* **SimaPro Integration:** Directly ingests "Process Contribution" exports from SimaPro (CSV/Excel).
* **Automated Classification:** Uses a Pedigree Matrix approach to automatically classify input flows as either *Aleatory* (Stochastic) or *Epistemic* (Fuzzy Intervals).
* **Hybrid Propagation Engine:** Runs a dual-loop simulation:
    * **Inner Loop:** Interval Arithmetic for epistemic variables.
    * **Outer Loop:** Monte Carlo sampling for aleatory variables.
* **Multi-Metric Batching:** Automatically detects and processes multiple impact categories (e.g., GWP, Water Scarcity, Toxicity) in a single run.
* **Decision Support Reporting:** Generates an report showing the exact safety buffer needed for compliance across multiple impact categories.


#### Dynamic Material Flow Analysis (MFA)
* **Continuous Mass-Balance Engine:** Forecasts complex, multi-node material cycles (e.g., global aluminum or copper cycles) over time (e.g., 2024 to 2050).
* **Machine Learning Preprocessing:** Replaces subjective data scoring (as the Pedigree Matrix in LCA) with algorithmic *Data Quality Indicators* (DQIs), for determining metrics of ignorance related to missing data.
* **Multi-Model Proxy Forecasting:** Maps future flows to proxies (e.g., GDP) using an ensemble of Linear Regression, Ridge Regression, and Decision Trees. The divergence between algorithms explicitly forms the structural *Epistemic* envelope.
* **Interactive Visualizations:** Generates multiple relevant charts, including: Sankey, network topologies, and temporal charts.
---

### Project Structure

```bash
project_root/
│
├── data/
│   ├── LCI_test.xlsx                   # The Master LCI file with Pedigree Scores
│   └── contributions/
│       └── Contributions_test.xlsx     # SimaPro Export (Multi-scenario)
│
├── src/
│   ├── classify.py                     # Module 1: Data Manager & Classifier
│   ├── propagate.py                    # Module 2: Hybrid IRS Propagation Engine
│   └── report.py                       # Module 3: Visualization & Reporting
│
├── main.py                             # Main Script (Run this)
├── requirements.txt                    # Python dependencies
└── README.md
```


### Getting Started
#### Prerequisites
* Python 3.8+
* SimaPro 9.x (for generating input data for LCA)

#### Installation
##### Clone the repository:
```bash
git clone https://github.com/javalosp/Uncertainty_Quantification_Framework.git
cd Uncertainty_Quantification_Framework
```

##### Install dependencies:
```bash
    pip install -r requirements.txt
```

### How to Use
The repository is divided into specific *use cases*. Navigate to the `use_cases/` directory to run the specific pipeline you need.

#### Use Case 1: LCA (using Pedigree Matrix scores and SimaPro exports.)

This tool uses a data-driven Classification logic:

**1. Epistemic Score** = Reliability + Completeness + Technological

**2. Aleatory Score** = Temporal + Geographical

**Logic:**

* If Epistemic Score >= 8 (indicating poor data quality) AND dominates the Aleatory score -> Classify as Epistemic (Possibility distribution - Interval arithmetics).
* Otherwise -> Classify as Aleatory (Probability distribution - Lognormal).

This ensures that high-quality data is treated as stochastic variability, while low-quality estimates are properly flagged as ignorance gaps.
##### **Step 1: Prepare Your LCI Data**
Prepare a CSV/Excel file (e.g., data/LCI_test.xlsx) containing your inventory flows and place it in `data/LCA/`. This file must include Pedigree scores and a mapping Column to link to SimaPro.

**Required Columns:**

* Flow_Name: Simple name (e.g., "Electricity").

* Mean: The mean value used in the calculation.

* GSD: Geometric Standard Deviation.

* Pedigree Scores: Columns named Reliability, Completeness, Temporal, Geographical, Tech.

* Contributions name: Critical. The exact string used in the SimaPro export header (e.g., "Electricity, high voltage {CL}| market for | APOS, U").

##### **Step 2: Export from SimaPro**
Place the SimaPro export (.xlsx or csv) in `data/LCA/contributions/.`
##### **Step 3: Run the Analysis**
Run the analysis:
```bash
    python use_cases/LCA_static/run_standard_lca.py
```

The script will:

1. Read the LCI file and classify every flow as **Aleatory** or **Epistemic**.
2. Scan the contributions folder.
3. Loop through every impact category found.
4. Generate a P-Box Plot (.png) and print a text report for each scenario. Results are saved in `outputs/LCA_results/`.


#### Interpreting the Output
The tool generates two key outputs for every scenario:

1. **The P-Box Plot**
* **Green Curve (Optimistic):** The performance if your unverified data turns out to be accurate.
* **Red Curve (Conservative):** The performance if your unverified data turns out to be wrong (worst-case).
* **Gray Gap (Ignorance):** The distance between curves. This represents the uncertainty caused  by poor data quality.

2. **The Executive Report**
The report provides metrics for decision-making:

```bash
1. ALEATORY VARIABILITY (Natural Fluctuation)
   >>> SAFETY BUFFER REQUIRED:  0.2787 units
   (You must design 0.2787 units above the average to handle normal swings.)

2. EPISTEMIC UNCERTAINTY (The Ignorance Penalty)
   >>> IGNORANCE PENALTY:       0.1061 units
   (You are carrying this extra risk because of poor data.)
```

* **Safety Buffer:** The cost of *Natural Variability*. You cannot reduce this by measuring more; you must engineer around it.
* **Ignorance Penalty:** The cost of *Missing Information*. You can reduce this to zero by collecting better data (improving Pedigree scores).


#### Use Case 2: Empirical Dynamic MFA

This runs a standard dynamic material flow forecast out to 2050 using synthetic, ML-driven inputs.

Run the analysis:
```bash
    python use_cases/MFA_standard/test_empirical_dynamic_mfa.py
```

**Outputs:** Check outputs/MFA_results/MFA_standard/ for the charts, including Sankeys mapping the epistemic bounds over time.


#### Use Case 3: The Bayesian Benchmark (Wang et al.)

This reproduces a methodological comparison against a Bayesian MFA model for the Aluminum cycle. Scripts must be run sequentially.

1. **Translate data:** Converts the static Bayesian priors into dynamic time-series with artificial data gaps.
```bash
    python use_cases/MFA_benchmark/01_preprocess_data.py
```
2. **Extract topology:** Parses strict mathematical mass-balance transfer coefficients.
```bash
    python use_cases/MFA_benchmark/02_extract_topology.py
```
3. **Parallel benchmark:** Runs the P-Box engine on the sabotaged dataset to demonstrate how structural ignorance compounds compared to a Bayesian point-estimate.
```bash
    python use_cases/MFA_benchmark/03_run_benchmark.py
```
4. **The "complement" scenario:** Feeds a perfectly reconciled Bayesian posterior into the P-Box as a 2020 starting baseline, chaining the two methodologies.
```bash
    python use_cases/MFA_benchmark/04_coupled_analysis.py
```
**Outputs:** Check outputs/MFA_results/MFA_benchmark/ for the comparative outputs.

### Interpreting the Visual Outputs
**Temporal Fan Charts (fan_chart.png):** Displays the trajectory of material flows. The dark inner band represents natural aleatory noise. The widening lighter bands represent expanding epistemic ignorance as the model forecasts deeper into the future.

**Uncertainty Sankey charts (uncertainty_sankey_YYYY.html):** Interactive flow diagrams where the width of the link represents the median mass flow, and the hover-text explicitly bounds the flow within a [Min - Max] structural ignorance envelope.

**Network topology charts (network_topology_YYYY.html):** Interactive PyVis maps allowing stakeholders to audit the physical routing and source/target nodes of the high-dimensional material cycle.
