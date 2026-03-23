import numpy as np
import pandas as pd

class HybridPropagationEngine:
    """
    Module 2: Hybrid Propagation Engine (Upgraded for Fuzzy Arithmetic)
    Responsibility: Execute the Independent Random Set (IRS) simulation with Alpha-Cuts.
    
    - Aleatory: Monte Carlo (Sampled once and reused to isolate epistemic effects).
    - Epistemic: Fuzzy Interval Arithmetic (Triangular Cuts).
    """
    def __init__(self, characterised_data):
        self.data = characterised_data.copy()
        self.impact_factors = {}
        self.results = {} # Stores {alpha_level: DataFrame}

    def define_impact_model(self, default_k=0.0, specific_k=None):
        if specific_k is None: specific_k = {}
        k_map = {}
        for flow in self.data['Flow_Name']:
            factor = default_k
            if flow in specific_k:
                factor = specific_k[flow]
            else:
                for key, val in specific_k.items():
                    if str(key).lower() in str(flow).lower():
                        factor = val
                        break
            k_map[flow] = factor
        self.impact_factors = k_map

    def run_simulation(self, n_iterations=10000, seed=42, alpha_cuts=[0.0, 1.0]):
        """
        Runs the propagation for multiple levels of possibility (alpha-cuts).
        
        Args:
            alpha_cuts (list): List of floats between 0 and 1. 
                               0 = Full Support (Conservative). 
                               1 = Core/Mode (Most Plausible).
        """
        if not self.impact_factors:
            print("Warning: Impact model not defined.")
        
        np.random.seed(seed)
        
        # 1. PRE-SAMPLE ALEATORY VARIABLES
        # We sample these ONCE so that the difference between alpha-levels 
        # is purely due to Epistemic shrinkage, not Monte Carlo noise.
        aleatory_cache = {}
        for index, row in self.data.iterrows():
            if row['Type'] == 'Aleatory':
                params = row['Params_Aleatory']
                if params and not pd.isna(params['mu_ln']):
                    samples = np.random.lognormal(
                        mean=params['mu_ln'], 
                        sigma=params['sigma_ln'], 
                        size=n_iterations
                    )
                    aleatory_cache[index] = samples

        print(f"[Module 2] Running Fuzzy Simulation (Alphas: {alpha_cuts})...")
        
        # 2. RUN LOOP FOR EACH ALPHA CUT
        self.results = {}
        
        for alpha in alpha_cuts:
            total_y_min = np.zeros(n_iterations)
            total_y_max = np.zeros(n_iterations)
            
            for index, row in self.data.iterrows():
                flow_name = row['Flow_Name']
                k = self.impact_factors.get(flow_name, 0)
                if k == 0: continue

                # --- ALEATORY (Stochastic) ---
                if row['Type'] == 'Aleatory':
                    # Use cached samples (Variability is constant across alpha-cuts)
                    samples = aleatory_cache.get(index, np.zeros(n_iterations))
                    total_y_min += samples * k
                    total_y_max += samples * k
                    
                # --- EPISTEMIC (Fuzzy Intervals) ---
                elif row['Type'] == 'Epistemic':
                    params = row['Params_Epistemic']
                    # Triangular Fuzzy Number: [min, mode, max]
                    a, b, c = params['min'], params['mode'], params['max']
                    
                    # Calculate Interval at current Alpha
                    # Left Bound increases toward Mode
                    val_min_alpha = a + alpha * (b - a) 
                    # Right Bound decreases toward Mode
                    val_max_alpha = c - alpha * (c - b)
                    
                    # Interval Arithmetic with k
                    if k >= 0:
                        lower = val_min_alpha * k
                        upper = val_max_alpha * k
                    else:
                        lower = val_max_alpha * k
                        upper = val_min_alpha * k
                    
                    total_y_min += lower
                    total_y_max += upper

            # Store result for this alpha
            self.results[alpha] = pd.DataFrame({
                'Y_Min_Estimation': total_y_min,
                'Y_Max_Estimation': total_y_max
            })
            
        print("[Module 2] Simulation Complete.")
        return self.results

class DynamicPropagationEngine(HybridPropagationEngine):
    """
    Extension for Dynamic Material Flow Analysis (MFA).
    Executes the Independent Random Set (IRS) simulation over a temporal horizon.
    Calculates the continuous mass balance: Stock(t) = Stock(t-1) + Inflow(t) - Outflow(t)
    """
    def __init__(self, dynamic_characterised_data, start_year, end_year):
        super().__init__(dynamic_characterised_data) # Inherit core setup
        self.start_year = start_year
        self.end_year = end_year
        self.n_steps = end_year - start_year + 1
        self.dynamic_results = {} # Stores {alpha_level: {'Stock_Min_TS': array, 'Stock_Max_TS': array}}

    def run_dynamic_simulation(self, n_iterations=10000, seed=42, alpha_cuts=[0.0, 1.0]):
        """
        Runs the propagation over time for multiple levels of possibility (alpha-cuts).
        Generates N trajectories over T time steps.
        NEW: Caches individual flow-level arrays for Sankey topology visualization.
        """
        if not self.impact_factors:
            print("Warning: Dynamic flow model (k-factors) not defined.")
        
        np.random.seed(seed)
        
        # 1. PRE-SAMPLE ALEATORY VARIABLES (Time-Series)
        # We sample an array of shape (n_iterations, n_steps) for every aleatory variable.
        # This isolates Monte Carlo stochasticity from the epistemic ignorance expanding over time.
        aleatory_cache = {}
        for index, row in self.data.iterrows():
            if row['Type'] == 'Aleatory':
                params_ts = row['Params_Aleatory_TS']
                samples = np.zeros((n_iterations, self.n_steps))
                
                # Sample the distribution for each discrete time step t
                for t in range(self.n_steps):
                    mu = params_ts['mu_ln'][t]
                    sig = params_ts['sigma_ln'][t]
                    if not pd.isna(mu):
                        samples[:, t] = np.random.lognormal(mean=mu, sigma=sig, size=n_iterations)
                aleatory_cache[index] = samples

        print(f"[Module 2 - Dynamic] Running Temporal Mass Balance (Alphas: {alpha_cuts})...")
        
        # 2. RUN LOOP FOR EACH ALPHA CUT
        for alpha in alpha_cuts:
            # Matrices to hold the evolving stock for all iterations over all time steps
            stock_min = np.zeros((n_iterations, self.n_steps))
            stock_max = np.zeros((n_iterations, self.n_steps))
            
            # VISUALISATION INTEGRATION: Initialize flow-level caches
            # ====================================================================
            flow_trajectories_min = {row['Flow_Name']: np.zeros((n_iterations, self.n_steps)) for _, row in self.data.iterrows()}
            flow_trajectories_max = {row['Flow_Name']: np.zeros((n_iterations, self.n_steps)) for _, row in self.data.iterrows()}
            
            # 3. THE TEMPORAL LOOP (t = 0 to T)
            for t in range(self.n_steps):
                delta_min_t = np.zeros(n_iterations)
                delta_max_t = np.zeros(n_iterations)
                
                # Calculate the net flow (Inflows - Outflows) at time step t
                for index, row in self.data.iterrows():
                    flow_name = row['Flow_Name']
                    
                    # k determines directionality: positive = inflow, negative = outflow
                    k = self.impact_factors.get(flow_name, 0)
                    if k == 0: continue

                    # --- ALEATORY (Stochastic Noise) ---
                    if row['Type'] == 'Aleatory':
                        samples_t = aleatory_cache[index][:, t]
                        delta_min_t += samples_t * k
                        delta_max_t += samples_t * k
                        
                    # --- EPISTEMIC (Expanding Ignorance Bounds) ---
                    elif row['Type'] == 'Epistemic':
                        params_ts = row['Params_Epistemic_TS']
                        a = params_ts['min'][t]
                        b = params_ts['mode'][t]
                        c = params_ts['max'][t]
                        
                        # Apply Alpha-Cut shrinkage for time t
                        val_min_alpha = a + alpha * (b - a) 
                        val_max_alpha = c - alpha * (c - b)
                        
                        # Interval Arithmetic with Flow Direction (k)
                        if k >= 0: # Adding to stock
                            delta_min_t += val_min_alpha * k
                            delta_max_t += val_max_alpha * k
                        else: # Subtracting from stock (k is negative, so max value creates lowest bound)
                            delta_min_t += val_max_alpha * k
                            delta_max_t += val_min_alpha * k

                        # Cache the absolute mass flow bounds (Broadcast to N iterations)
                        flow_trajectories_min[flow_name][:, t] = val_min_alpha * abs(k)
                        flow_trajectories_max[flow_name][:, t] = val_max_alpha * abs(k)

                # 4. EXECUTE MASS BALANCE: S(t) = S(t-1) + Delta(t)
                if t == 0:
                    stock_min[:, t] = delta_min_t
                    stock_max[:, t] = delta_max_t
                else:
                    stock_min[:, t] = stock_min[:, t-1] + delta_min_t
                    stock_max[:, t] = stock_max[:, t-1] + delta_max_t

            # Store the 2D arrays (Trajectories x Time) for this Alpha cut
            # Store the 2D arrays for the total stock AND the individual flows
            self.dynamic_results[alpha] = {
                'Stock_Min_TS': stock_min,
                'Stock_Max_TS': stock_max,
                'Flows_Min_TS': flow_trajectories_min, # <--- NEW
                'Flows_Max_TS': flow_trajectories_max  # <--- NEW
            }
            
        print("[Module 2 - Dynamic] Continuous MFA Simulation Complete.")
        return self.dynamic_results

class StaticReconciliationEngine:
    """
    Enforces strict Mass Balance for Retrospective MFA Auditing.
    Algebraically solves for 'Calculated' flows during every Monte Carlo 
    and Interval Arithmetic iteration.
    """
    def __init__(self, parsed_network):
        """
        Args:
            parsed_network (dict): The exact dictionary returned by MFAAuditParser.parse_network()
        """
        self.nodes = parsed_network['nodes']
        self.edges = parsed_network['edges']
        self.calculated_params = parsed_network['calculated']
        
        # Build a fast-lookup map for the topology (who connects to whom)
        self.node_map = {node: {'in': [], 'out': []} for node in self.nodes}
        
        for edge in self.edges:
            param_id = edge['id']
            # Only track mass flows (ignore pure percentage transfer coefficients for the additive balance)
            if edge['type'].lower() == 'flow':
                if edge['target'] in self.node_map:
                    self.node_map[edge['target']]['in'].append(param_id)
                if edge['source'] in self.node_map:
                    self.node_map[edge['source']]['out'].append(param_id)

    def resolve_mass_balance(self, iteration_values):
        """
        Takes a dictionary of the current Monte Carlo / Interval samples, 
        and solves for the missing 'Calculated' parameters algebraically.
        
        Args:
            iteration_values (dict): e.g. {'Import_Bauxite': 42.5, 'Primary_Aluminum': 38.1}
        Returns:
            dict: A fully balanced mass matrix for this iteration.
        """
        resolved = iteration_values.copy()
        unresolved_ids = set(self.calculated_params.keys())
        
        # Iterative Solver Loop (The "Sudoku" Method)
        progress = True
        while unresolved_ids and progress:
            progress = False
            
            for node, flows in self.node_map.items():
                # The 'Environment' is an infinite source/sink. We do not balance it.
                if node.lower() == 'environment':
                    continue
                    
                in_flows = flows['in']
                out_flows = flows['out']
                
                # Check how many unknowns are connected to this specific node
                unknowns = [f for f in (in_flows + out_flows) if f in unresolved_ids]
                
                # We can only solve the node algebraically if exactly ONE flow is missing
                if len(unknowns) == 1:
                    target_unknown = unknowns[0]
                    
                    # Sum up all the known mass entering and leaving this node
                    sum_in = sum(resolved.get(f, 0.0) for f in in_flows if f != target_unknown)
                    sum_out = sum(resolved.get(f, 0.0) for f in out_flows if f != target_unknown)
                    
                    # Conservation of Mass: Inputs = Outputs
                    if target_unknown in in_flows:
                        # Missing Input = Known Outputs - Known Inputs
                        resolved[target_unknown] = max(sum_out - sum_in, 0.0)
                    else:
                        # Missing Output = Known Inputs - Known Outputs
                        resolved[target_unknown] = max(sum_in - sum_out, 0.0)
                        
                    # Mark as solved and trigger another pass
                    unresolved_ids.remove(target_unknown)
                    progress = True
        
        # Guardrail: If the loop finishes but flows are still unresolved, 
        # the user's published paper has a mathematically under-determined system.
        if unresolved_ids:
            raise ValueError(
                f"[!] Under-determined System: The published MFA lacks enough "
                f"measured data to solve for: {unresolved_ids}"
            )
            
        return resolved

# UNIT TEST BLOCK
# (run the module directly, i.e.: python propagate.py)
if __name__ == "__main__":
    print("\n TESTING MODULE 2: PROPAGATE ")
    
    # Create mock data to simulate the expected output from classify.py
    # We create one Aleatory flow and one Epistemic flow
    mock_data = pd.DataFrame([
        {
            'Flow_Name': 'Electricity',
            'Type': 'Aleatory',
            'Params_Aleatory': {'mu_ln': 2.3, 'sigma_ln': 0.18}, # Lognormal params for mean=10 and gsd=1.2
            'Params_Epistemic': None
        },
        {
            'Flow_Name': 'Chemical X',
            'Type': 'Epistemic',
            'Params_Aleatory': None,
            'Params_Epistemic': {'min': 1.0, 'max': 5.0} # Interval [1, 5]
        }
    ])
    
    # Initialise Engine
    engine = HybridPropagationEngine(mock_data)

    # Define mock SimaPro contributions (k)   
    mock_contributions = {
        'Electricity': 0.5, # Electricity contributes 0.5 units per unit 
        'Chemical X': 2.0 # Chemical X contributes 2.0 units per unit
    }

    engine.define_impact_model(default_k=0, specific_k=mock_contributions)

    # Run Simulation
    try:
        results = engine.run_simulation(n_iterations=1000)
        metrics = engine.get_robustness_metrics()
        
        print("\nTest Results Head:")
        print(results.head(3).to_string())
        print(f"\nEpistemic Gap (Median): {metrics['Median_Gap']:.4f}")
        
        # Validation:
        # Gap should be derived purely from Chemical X: (5.0 - 1.0) * 2.0 = 8.0
        # Check if result is close to 8.0
        if abs(metrics['Median_Gap'] - 8.0) < 0.1:
            print("VALIDATION PASSED: Gap matches theoretical calculation.")
        else:
            print("VALIDATION FAILED: Gap logic incorrect.")
            
    except Exception as e:
        print(f"\nTEST FAILED: {e}")