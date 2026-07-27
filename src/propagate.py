import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

class HybridPropagationEngine:
    """
    Module 2: Hybrid Propagation Engine (Fuzzy Arithmetic and Gaussian Copula)
        -> Execute the Independent Random Set (IRS) simulation with alpha-cuts.
    
    - Aleatory: Correlated Monte Carlo via Gaussian Copula (Sampled once).
    - Epistemic: Fuzzy Interval Arithmetic (Triangular Cuts).
    """
    def __init__(self, characterised_data, correlation_matrix=None):
        self.data = characterised_data.copy()
        self.impact_factors = {}
        self.results = {} 
        # Correlation matrix for aleatory variables (DataFrame or 2D array)
        self.correlation_matrix = correlation_matrix

    def _generate_correlated_aleatory_samples(self, n_iterations, seed):
        """
        Generates correlated Monte Carlo samples using a Gaussian Copula.
        """
        np.random.seed(seed)
        
        # Isolate Aleatory Variables
        aleatory_rows = self.data[self.data['Type'] == 'Aleatory'].copy()
        n_vars = len(aleatory_rows)
        flow_names = aleatory_rows['Flow_Name'].tolist()
        
        if n_vars == 0:
            return pd.DataFrame()

        # Build the Correlation Matrix (Sigma)
        if self.correlation_matrix is not None:
            valid_flows = [f for f in flow_names if f in self.correlation_matrix.columns]
            sigma = self.correlation_matrix.loc[valid_flows, valid_flows].values
            
            if sigma.shape != (n_vars, n_vars):
                print("   [Warning] Correlation matrix mismatch. Defaulting to independent sampling.")
                sigma = np.eye(n_vars)
        else:
            sigma = np.eye(n_vars)

        # Generate Correlated Standard Normal Samples (MVN)
        mean_zero = np.zeros(n_vars)
        Z_samples = np.random.multivariate_normal(mean_zero, sigma, size=n_iterations)

        # Map back to Lognormal Marginals
        mc_samples = {}
        for i, (_, row) in enumerate(aleatory_rows.iterrows()):
            flow = row['Flow_Name']
            params = row['Params_Aleatory']
            mu = params['mu_ln']
            sig = params['sigma_ln']
            
            x_samples = np.exp(mu + sig * Z_samples[:, i])
            mc_samples[flow] = x_samples
            
        return pd.DataFrame(mc_samples)

    def define_impact_model(self, default_k=0.0, specific_k=None):
        """
        Maps LCI flows to characterization factors.
        """
        # string-matching logic
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

    def run_simulation(self, n_iterations=5000, seed=42, alpha_cuts=[0.0, 0.5, 1.0]):
        """
        Runs the propagation for multiple levels of possibility (alpha-cuts).
        Args:
        alpha_cuts (list): List of floats between 0 and 1.
        0 = Full Support (Conservative).
        1 = Core/Mode (Most Plausible).
        """
        if not self.impact_factors:
            print("Warning: Impact model not defined.")
            
        print(f"[Module 2] Running Fuzzy Simulation (Alphas: {alpha_cuts})...")
        
        # Call the Copula function instead of independent loops
        # PRE-SAMPLE ALEATORY VARIABLES (Vectorised & Correlated)
        df_aleatory = self._generate_correlated_aleatory_samples(n_iterations, seed)
        
        # Filter epistemic variables for the next step
        epistemic_rows = self.data[self.data['Type'] == 'Epistemic']
        
        self.results = {}
        
        # RUN LOOP FOR EACH ALPHA CUT
        for alpha in alpha_cuts:
            # Initialise empty arrays for the Monte Carlo results
            y_min_total = np.zeros(n_iterations)
            y_max_total = np.zeros(n_iterations)
            
            # Vectorised addition for Aleatory variables
            for flow, k in self.impact_factors.items():
                if flow in df_aleatory.columns and k != 0:
                    # Deterministic k * Stochastic X
                    y_min_total += k * df_aleatory[flow].values
                    y_max_total += k * df_aleatory[flow].values
            
            # Interval Arithmetic for Epistemic variables
            for _, row in epistemic_rows.iterrows():
                flow = row['Flow_Name']
                k = self.impact_factors.get(flow, 0.0)
                if k == 0: continue
                
                p = row['Params_Epistemic']
                
                # Alpha-cut interval shrinking
                a_alpha = p['min'] + alpha * (p['mode'] - p['min'])
                c_alpha = p['max'] - alpha * (p['max'] - p['mode'])
                
                # Interval arithmetic multiplication
                if k >= 0:
                    y_min_total += k * a_alpha
                    y_max_total += k * c_alpha
                else:
                    y_min_total += k * c_alpha
                    y_max_total += k * a_alpha
                    
            # Ensure column names match RobustnessReporter expectations
            self.results[alpha] = pd.DataFrame({
                'Y_Min_Estimation': y_min_total,
                'Y_Max_Estimation': y_max_total
            })
            
        print("[Module 2] Simulation Complete.")
        return self.results

class DynamicPropagationEngine(HybridPropagationEngine):
    """
    Extension for Dynamic Material Flow Analysis (MFA).
    Executes the Independent Random Set (IRS) simulation over a temporal horizon.
    Calculates the continuous mass balance: Stock(t) = Stock(t-1) + Inflow(t) - Outflow(t)
    """
    def __init__(self, dynamic_characterised_data, start_year, end_year, correlation_matrix=None):
        super().__init__(dynamic_characterised_data) # Inherit core setup
        self.start_year = start_year
        self.end_year = end_year
        self.n_steps = end_year - start_year + 1
        self.dynamic_results = {} # Stores {alpha_level: {'Stock_Min_TS': array, 'Stock_Max_TS': array}}
        self.correlation_matrix = correlation_matrix

    def _generate_correlated_aleatory_samples(self, n_iterations, seed, target_year):
        """
        Generates correlated Monte Carlo samples for a specific year using a Gaussian Copula.
        """
        np.random.seed(seed + target_year) # Ensure different seeds per year, but reproducible
        
        aleatory_rows = self.data[self.data['Type'] == 'Aleatory']
        n_vars = len(aleatory_rows)
        flow_names = aleatory_rows['Flow_Name'].tolist()
        
        if n_vars == 0: return pd.DataFrame()

        # Build the Correlation Matrix (Sigma)
        if self.correlation_matrix is not None:
            valid_flows = [f for f in flow_names if f in self.correlation_matrix.columns]
            # Extract subset of matrix and convert to numpy array
            sigma = self.correlation_matrix.loc[valid_flows, valid_flows].values
            
            if sigma.shape != (n_vars, n_vars):
                sigma = np.eye(n_vars) # Fallback to independent if mismatch
        else:
            sigma = np.eye(n_vars) # Fallback to independent

        # Generate Correlated Standard Normal Samples
        mean_zero = np.zeros(n_vars)
        Z_samples = np.random.multivariate_normal(mean_zero, sigma, size=n_iterations)

        # Map back to Lognormal Marginals for the target_year
        mc_samples = {}
        for i, (_, row) in enumerate(aleatory_rows.iterrows()):
            flow = row['Flow_Name']
            # Assuming dynamic params store a dictionary of parameters per year
            # TODO: Check consistency across the code
            params = row['Params_Aleatory'].get(target_year) 
            
            if params:
                mu = params['mu_ln']
                sig = params['sigma_ln']
                x_samples = np.exp(mu + sig * Z_samples[:, i])
                mc_samples[flow] = x_samples
            else:
                mc_samples[flow] = np.zeros(n_iterations)
            
        return pd.DataFrame(mc_samples)

    def run_dynamic_simulation(self, n_iterations=10000, seed=42, alpha_cuts=[0.0, 1.0]):
        """
        Runs the propagation over time for multiple levels of possibility (alpha-cuts).
        Generates N trajectories over T time steps.
        Additionally, stores individual flow-level arrays for Sankey topology visualisation.
        """
        if not self.impact_factors:
            print("Warning: Dynamic flow model (k-factors) not defined.")
        
        np.random.seed(seed)
        
        # PRE-SAMPLE ALEATORY VARIABLES (Time-Series)
        # Sample an array of shape (n_iterations, n_steps) for every aleatory variable.
        # Isolates Monte Carlo stochasticity from the epistemic ignorance expanding over time.
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
        
        # RUN LOOP FOR EACH ALPHA CUT
        for alpha in alpha_cuts:
            # Matrices to hold the evolving stock for all iterations over all time steps
            stock_min = np.zeros((n_iterations, self.n_steps))
            stock_max = np.zeros((n_iterations, self.n_steps))
            
            # VISUALISATION INTEGRATION: Initialise flow-level caches
            flow_trajectories_min = {row['Flow_Name']: np.zeros((n_iterations, self.n_steps)) for _, row in self.data.iterrows()}
            flow_trajectories_max = {row['Flow_Name']: np.zeros((n_iterations, self.n_steps)) for _, row in self.data.iterrows()}
            
            # TEMPORAL LOOP (t = 0 to T)
            for t in range(self.n_steps):
                delta_min_t = np.zeros(n_iterations)
                delta_max_t = np.zeros(n_iterations)
                
                # Calculate the net flow (Inflows - Outflows) at time step t
                for index, row in self.data.iterrows():
                    flow_name = row['Flow_Name']
                    
                    # k determines directionality: positive = inflow, negative = outflow
                    k = self.impact_factors.get(flow_name, 0)
                    if k == 0: continue

                    # ALEATORY (Stochastic Noise)
                    if row['Type'] == 'Aleatory':
                        samples_t = aleatory_cache[index][:, t]
                        delta_min_t += samples_t * k
                        delta_max_t += samples_t * k
                        
                    # EPISTEMIC (Expanding Ignorance Bounds)
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

                        # Store the absolute mass flow bounds (Broadcast to N iterations)
                        flow_trajectories_min[flow_name][:, t] = val_min_alpha * abs(k)
                        flow_trajectories_max[flow_name][:, t] = val_max_alpha * abs(k)

                # MASS BALANCE: S(t) = S(t-1) + Delta(t)
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
                'Flows_Min_TS': flow_trajectories_min,
                'Flows_Max_TS': flow_trajectories_max 
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

    def _optimisation_fallback(self, current_state):
        """
        Uses Sequential Least Squares Quadratic Programming (SLSQP) Constrained Optimisation
        to mathematically force mass balance when raw sampled data contains physical contradictions.
        """
        # Gather all parameters involved in the network
        all_params = list(self.network_data.get('aleatory', {}).keys()) + \
                     list(self.network_data.get('epistemic', {}).keys()) + \
                     list(self.network_data.get('calculated', {}).keys())
        all_params = list(set(all_params)) # Remove duplicates
        
        # Build the Initial Guess (x0) and target reference array
        # If a param is in current_state, use its value. Otherwise, guess a small positive number (0.1)
        x0 = np.array([current_state.get(p, 0.1) for p in all_params])
        
        # Objective Function: Minimise the distance between the optimised flows and the sampled data
        def objective(x):
            penalty = 0.0
            for i, p in enumerate(all_params):
                if p in current_state:
                    # Normalise the squared error to prevent massive flows from dominating small flows
                    scale = max(current_state[p], 1.0)
                    penalty += ((x[i] - current_state[p]) / scale) ** 2
            return penalty

        # Strict Physics Constraints: Inflows - Outflows - Stock_Change = 0
        nodes = set()
        for p in all_params:
            if '_to_' in p:
                src, tgt = p.split('_to_')
                nodes.update([src, tgt])
        
        constraints = []
        for node in nodes:
            if node.lower() in ['environment', 'unknown', 'nan']: continue
            
            def make_constraint(n):
                def constraint_eq(x):
                    balance = 0.0
                    for i, p in enumerate(all_params):
                        if p.startswith(f"{n}_to_"): balance -= x[i] # Outgoing mass
                        elif p.endswith(f"_to_{n}"): balance += x[i] # Incoming mass
                        elif p == f"stock_{n}": balance -= x[i]      # Mass accumulated in stock
                    return balance
                return constraint_eq
                
            constraints.append({'type': 'eq', 'fun': make_constraint(node)})

        # Bounds: Mass cannot be negative
        bounds = [(0.0, None) for _ in all_params]

        # Run SLSQP Optimiser
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Suppress noisy scipy warnings during Monte Carlo
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 100, 'ftol': 1e-3})
            
        if res.success:
            # Map optimised results back to a dictionary state
            return {p: res.x[i] for i, p in enumerate(all_params)}
        else:
            raise ValueError(f"Optimisation Fallback Failed: {res.message}")

    def resolve_mass_balance(self, iteration_values):
        """
        Takes a dictionary of the current Monte Carlo / Interval samples, 
        and solves for the missing 'Calculated' parameters algebraically.
        
        Args:
            iteration_values (dict): e.g. {'Import_Bauxite': 42.5, 'Primary_Aluminum': 38.1}
        Returns:
            dict: balanced mass matrix for this iteration.
        """

        try:
            resolved = iteration_values.copy()
            unresolved_ids = set(self.calculated_params.keys())
            
            # Iterative Solver Loop ("Sudoku" method)
            progress = True
            while unresolved_ids and progress:
                progress = False
                
                for node, flows in self.node_map.items():
                    # The 'Environment' is an infinite source/sink. Do not balance it.
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
            # the reference data (i.e. published paper) has a mathematically under-determined system.
            if unresolved_ids:
                raise ValueError(
                    f"[!] Under-determined System: The published MFA lacks enough "
                    f"measured data to solve for: {unresolved_ids}"
                )
                
            return resolved
            pass
        
        except ValueError as e:
            # If strict algebra fails due to conflicting data, fallback to optimisation
            return self._optimisation_fallback(iteration_values)

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