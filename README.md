# Uncertainty quatification for LCA (Hybrid IRS Method)

A Python-based uncertainty analysis method for Life Cycle Assessment (LCA) that enhances standard SimaPro results by quantifying the effect of uncertainty on environmental impacts.

This tool implements an **independent random set sampling** method to rigorously separate:
* **Aleatory Variability (Natural Fluctuation):** Irreducible physical randomness (e.g., grid intensity, temperature).
* **Epistemic Uncertainty (Ignorance):** Reducible lack of knowledge (e.g., missing data, unverified estimates).

The result is a **Probability Box (P-Box)** that shows the "Ignorance Gap," allowing analysts to calculate the precise "Ignorance Penalty" required to guarantee reliability.

---

### Key Features

* **SimaPro Integration:** Directly ingests "Process Contribution" exports from SimaPro (CSV/Excel).
* **Automated Classification:** Uses a Pedigree Matrix approach to automatically classify input flows as either *Aleatory* (Stochastic) or *Epistemic* (Fuzzy Intervals).
* **Hybrid Propagation Engine:** Runs a dual-loop simulation:
    * **Inner Loop:** Interval Arithmetic for epistemic variables.
    * **Outer Loop:** Monte Carlo sampling for aleatory variables.
* **Multi-Metric Batching:** Automatically detects and processes multiple impact categories (e.g., GWP, Water Scarcity, Toxicity) in a single run.
* **Decision Support Reporting:** Generates an "Executive Robustness Report" calculating the exact safety buffer needed for compliance.

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
* SimaPro 9.x (for generating input data)

#### Installation
##### Clone the repository:
```bash
    git clone https://github.com/javalosp/LCA_uncertainty.git]
    cd LCA_uncertainty
```

##### Install dependencies:
```bash
    pip install -r requirements.txt
```

### How to Use
#### **Step 1: Prepare Your LCI Data (The "Intelligence" File)**
Prepare a CSV/Excel file (e.g., data/LCI_test.xlsx) containing your inventory flows. Crucially, it must include Pedigree Scores and a Mapping Column to link to SimaPro.

**Required Columns:**

* Flow_Name: Simple name (e.g., "Electricity").

* Mean: The mean value used in the calculation.

* GSD: Geometric Standard Deviation.

* Pedigree Scores: Columns named Reliability, Completeness, Temporal, Geographical, Tech.

* Contributions name: Critical. The exact string used in the SimaPro export header (e.g., "Electricity, high voltage {CL}| market for | APOS, U").

#### **Step 2: Export from SimaPro**
1. In SimaPro, run your calculation.
2. Go to the Process Contribution tab.
3. Select multiple impact categories (e.g., Carbon, Water, etc.).
4. Export as Excel (.xlsx) or CSV.
5. Place this file in the data/contributions/ folder.

#### **Step 3: Run the Analysis**
Execute the main script:
```bash
    python main.py
```

The script will:

1. Read the LCI file and classify every flow as Aleatory or Epistemic.
2. Scan the contributions folder.
3. Loop through every impact category found.
4. Generate a P-Box Plot (.png) and print a text report for each scenario.


## Interpreting the Output
The tool generates two key outputs for every scenario:

1. **The P-Box Plot**
* **Green Curve (Optimistic):** The performance if your unverified data turns out to be accurate.
* **Red Curve (Conservative):** The performance if your unverified data turns out to be wrong (worst-case).
* **Gray Gap (Ignorance):** The distance between curves. This represents the uncertainty caused  by poor data quality.

2. **The Executive Report**
The console output provides strategic metrics for decision-making:

```bash
1. ALEATORY VARIABILITY (Natural Fluctuation)
   >>> SAFETY BUFFER REQUIRED:  0.2787 units
   (You must design 0.2787 units above the average to handle normal swings.)

2. EPISTEMIC UNCERTAINTY (The Ignorance Penalty)
   >>> IGNORANCE PENALTY:       0.1061 units
   (You are carrying this extra risk solely because of poor data.)
```

* **Safety Buffer:** The cost of Natural Variability. You cannot reduce this by measuring more; you must engineer around it.
* **Ignorance Penalty:** The cost of Missing Information. You can reduce this to zero by collecting better data (improving Pedigree scores).

### Methodology
This tool uses a Data-Driven Classification logic:

**1. Epistemic Score** = Reliability + Completeness + Technological

**2. Aleatory Score** = Temporal + Geographical

**Logic:**

* If Epistemic Score >= 8 (indicating poor data quality) AND dominates the Aleatory score -> Classify as Epistemic (Interval).
* Otherwise -> Classify as Aleatory (Lognormal Distribution).

This ensures that high-quality data is treated as stochastic variability, while low-quality estimates are properly flagged as ignorance gaps.