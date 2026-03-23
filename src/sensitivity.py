import numpy as np
import pandas as pd

class SensitivityAnalyser:
    """
    Module 4: Global Sensitivity Analysis (GSA)
    Responsibility: Decompose the uncertainty to identify key drivers.
    
    Methods:
    - Epistemic Sensitivity: Contribution to the P-Box Width (Ignorance).
    - Aleatory Sensitivity: Contribution to the Variance (Natural Variability).
    
    Since LCI models are linear (Sum of Inputs * Factors), we use Analytical Decomposition
    which is faster and more precise than Monte Carlo Sobol indices.
    """
    def __init__(self, characterised_data, impact_factors):
        """
        Args:
            characterised_data (pd.DataFrame): Output from classify.py
            impact_factors (dict): {Flow_Name: Sensitivity_Weight_k}
        """
        self.data = characterised_data.copy()
        self.k_map = impact_factors
        self.results = None

    def run_analysis(self):
        results = []
        
        total_epistemic_width = 0.0
        total_aleatory_variance = 0.0
        
        temp_stats = []
        
        # SCALING FACTOR: 
        # We convert StdDev into a "90% Confidence Interval Width" (P95-P05).
        # For a normal distribution, P95-P05 approx equals 3.29 standard deviations.
        # This ensures we compare "Range vs Range" instead of "Range vs Sigma".
        SIGMA_TO_RANGE_FACTOR = 3.29
        
        for index, row in self.data.iterrows():
            flow = row['Flow_Name']
            k = self.k_map.get(flow, 0.0)
            
            if k == 0: continue
                
            # 1. EPISTEMIC (Width contribution)
            w_contribution = 0.0
            if row['Type'] == 'Epistemic':
                p = row['Params_Epistemic']
                w_contribution = abs(k) * (p['max'] - p['min'])
            
            # 2. ALEATORY (Dual Calculation)
            var_contribution = 0.0     # For Aleatory List (Sobol-like)
            range_contribution = 0.0   # For Overall List (Magnitude-like)
            
            if row['Type'] == 'Aleatory':
                p = row['Params_Aleatory']
                s2 = p['sigma_ln']**2
                mu = p['mu_ln']
                var_x = np.exp(2*mu + s2) * (np.exp(s2) - 1)
                
                # A. Variance Contribution (k^2 * var)
                # This restores the 75% dominance for the Aleatory-only list
                var_contribution = (k**2) * var_x
                
                # B. Range Contribution (k * sigma * 3.29)
                # This keeps the logic correct for the Overall comparison
                std_dev = abs(k) * np.sqrt(var_x)
                range_contribution = std_dev * SIGMA_TO_RANGE_FACTOR
            
            temp_stats.append({
                'Flow_Name': flow,
                'Epistemic_Width': w_contribution,
                'Aleatory_Variance': var_contribution,   # <--- USE VARIANCE HERE
                'Aleatory_Range': range_contribution     # <--- USE RANGE HERE
            })
            
            total_epistemic_width += w_contribution
            total_aleatory_variance += var_contribution

        # --- CALCULATE SCORES ---
        max_score = 0
        processed_stats = []
        
        for item in temp_stats:
            # Combined Magnitude based on RANGE (Linear)
            combined_magnitude = np.sqrt(item['Epistemic_Width']**2 + item['Aleatory_Range']**2)
            
            processed_stats.append({
                'Flow_Name': item['Flow_Name'],
                'S_Epistemic_Raw': item['Epistemic_Width'],
                'S_Aleatory_Var_Raw': item['Aleatory_Variance'], # Store Variance
                'Combined_Score': combined_magnitude
            })
            
            if combined_magnitude > max_score:
                max_score = combined_magnitude
        
        # Final Normalisation
        total_magnitude = sum(x['Combined_Score'] for x in processed_stats)
        
        final_results = []
        for item in processed_stats:
            # 1. Epistemic Index (Normalized by Total Width)
            s_epi_norm = item['S_Epistemic_Raw'] / total_epistemic_width if total_epistemic_width > 0 else 0
            
            # 2. Aleatory Index (Normalized by Total VARIANCE) -> Restores 75%
            s_ale_norm = item['S_Aleatory_Var_Raw'] / total_aleatory_variance if total_aleatory_variance > 0 else 0
            
            # 3. Combined Index (Normalized by Total Magnitude) -> Keeps 26%
            s_combined = item['Combined_Score'] / total_magnitude if total_magnitude > 0 else 0
            
            final_results.append({
                'Flow_Name': item['Flow_Name'],
                'S_Epistemic': s_epi_norm,
                'S_Aleatory': s_ale_norm,
                'S_Combined': s_combined
            })
            
        self.results = pd.DataFrame(final_results)
        return self.results

    def get_top_contributors(self, n=5):
        if self.results is None: self.run_analysis()
            
        top_epi = self.results.sort_values('S_Epistemic', ascending=False).head(n)
        top_ale = self.results.sort_values('S_Aleatory', ascending=False).head(n)
        # New: Top Combined Drivers
        top_comb = self.results.sort_values('S_Combined', ascending=False).head(n)
        
        return top_epi, top_ale, top_comb
    
class DynamicSensitivityAnalyser(SensitivityAnalyser):
    """
    Extension for Dynamic Material Flow Analysis (MFA).
    Performs analytical Global Sensitivity Analysis (GSA) at specific time slices (e.g., 2030, 2050)
    to identify how the drivers of ignorance and variability evolve over the forecasting horizon.
    """
    def __init__(self, dynamic_characterised_data, impact_factors, start_year, end_year):
        super().__init__(dynamic_characterised_data, impact_factors) # Inherit basic setup
        self.start_year = start_year
        self.end_year = end_year

    def run_time_sliced_analysis(self, target_year):
        """
        Executes the analytical variance/width decomposition for a specific year.
        """
        if target_year < self.start_year or target_year > self.end_year:
            raise ValueError(f"Target year {target_year} is outside the horizon ({self.start_year}-{self.end_year}).")
            
        t_index = target_year - self.start_year
        
        results = []
        total_epistemic_width = 0.0
        total_aleatory_variance = 0.0
        temp_stats = []
        
        SIGMA_TO_RANGE_FACTOR = 3.29 # Converts StdDev to 90% Confidence Interval
        
        for index, row in self.data.iterrows():
            flow = row['Flow_Name']
            k = self.k_map.get(flow, 0.0)
            
            if k == 0: continue
                
            # 1. EPISTEMIC (Width contribution at time t)
            w_contribution = 0.0
            if row['Type'] == 'Epistemic':
                p_ts = row['Params_Epistemic_TS']
                # Extract the specific min and max for the target year
                min_t = p_ts['min'][t_index]
                max_t = p_ts['max'][t_index]
                w_contribution = abs(k) * (max_t - min_t)
            
            # 2. ALEATORY (Variance/Range contribution at time t)
            var_contribution = 0.0     
            range_contribution = 0.0   
            
            if row['Type'] == 'Aleatory':
                p_ts = row['Params_Aleatory_TS']
                mu_t = p_ts['mu_ln'][t_index]
                sig_t = p_ts['sigma_ln'][t_index]
                s2 = sig_t**2
                
                # Variance calculation for lognormal distribution at time t
                var_x = np.exp(2*mu_t + s2) * (np.exp(s2) - 1)
                
                var_contribution = (k**2) * var_x
                std_dev = abs(k) * np.sqrt(var_x)
                range_contribution = std_dev * SIGMA_TO_RANGE_FACTOR
            
            temp_stats.append({
                'Flow_Name': flow,
                'Epistemic_Width': w_contribution,
                'Aleatory_Variance': var_contribution,   
                'Aleatory_Range': range_contribution     
            })
            
            total_epistemic_width += w_contribution
            total_aleatory_variance += var_contribution

        # --- CALCULATE NORMALIZED SCORES FOR TARGET YEAR ---
        max_score = 0
        processed_stats = []
        
        for item in temp_stats:
            combined_magnitude = np.sqrt(item['Epistemic_Width']**2 + item['Aleatory_Range']**2)
            processed_stats.append({
                'Flow_Name': item['Flow_Name'],
                'S_Epistemic_Raw': item['Epistemic_Width'],
                'S_Aleatory_Var_Raw': item['Aleatory_Variance'],
                'Combined_Score': combined_magnitude
            })
            if combined_magnitude > max_score:
                max_score = combined_magnitude
        
        total_magnitude = sum(x['Combined_Score'] for x in processed_stats)
        
        final_results = []
        for item in processed_stats:
            s_epi_norm = item['S_Epistemic_Raw'] / total_epistemic_width if total_epistemic_width > 0 else 0
            s_ale_norm = item['S_Aleatory_Var_Raw'] / total_aleatory_variance if total_aleatory_variance > 0 else 0
            s_combined = item['Combined_Score'] / total_magnitude if total_magnitude > 0 else 0
            
            final_results.append({
                'Flow_Name': item['Flow_Name'],
                'Target_Year': target_year,
                'S_Epistemic': s_epi_norm,
                'S_Aleatory': s_ale_norm,
                'S_Combined': s_combined
            })
            
        self.results = pd.DataFrame(final_results)
        return self.results

    def get_dynamic_top_contributors(self, target_year, n=5):
        """
        Retrieves the top drivers of uncertainty for a specific year.
        """
        self.run_time_sliced_analysis(target_year)
        top_epi = self.results.sort_values('S_Epistemic', ascending=False).head(n)
        top_ale = self.results.sort_values('S_Aleatory', ascending=False).head(n)
        top_comb = self.results.sort_values('S_Combined', ascending=False).head(n)
        return top_epi, top_ale, top_comb