import pandas as pd
import numpy as np
import os

class LCIDataManager:
    """
    Module 1: Read and classify data
        -> Read raw LCI data, 
           clean formatting,
           calculate Pedigree scores,
           classify uncertainty types (Epistemic vs. Aleatory).
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.classified_data = None

    def load_and_clean(self):
        """
        Loads the LCI CSV/Excel file, standardises column names,
        and removes metadata rows (e.g. header rows).
        """
        if not self.file_path:
            raise ValueError("No file_path provided. Cannot run standard load_and_clean.")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            df = pd.read_csv(self.file_path, header=0)
        except:
            df = pd.read_excel(self.file_path, header=0)

        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Handle empty first column
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Flow_Name'}, inplace=True)

        # Map specific file headers to internal logic keys
        col_map = {
            'Standard deviation': 'GSD',
            'Per 1 m3 of desalinated water': 'Mean',
            'Reliability': 'Rel',
            'Completeness': 'Comp',
            'Temporal correlation': 'Temp',
            'Geographical correlation': 'Geo',
            'Further technological correlation': 'Tech',
            'Contributions name': 'Contrib_Header' 
        }
        
        # Rename columns
        df = df.rename(columns=col_map)
        
        # Clean numeric columns
        cols_to_numeric = ['Mean', 'GSD', 'Rel', 'Comp', 'Temp', 'Geo', 'Tech']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter out section headers (rows where GSD is missing)
        # Creating a copy of the dataframe
        # All columns not in 'cols_to_numeric' (like 'Contrib_Header') should be preserved.
        self.data = df.dropna(subset=['GSD']).copy()
        
        # Fill missing scores with 1
        score_cols = ['Rel', 'Comp', 'Tech', 'Temp', 'Geo']
        self.data[score_cols] = self.data[score_cols].fillna(1)
        
        print(f"[Module 1] Data Loaded: {len(self.data)} valid flows processed.")

    def classify_uncertainty(self):
        """
        Implements classification logic:
            Epistemic score = Rel + Comp + Tech
            Aleatory score = Temp + Geo
        If Rel score > 4 , then treat as Epistemic, or
        If Epistemic score > 7 AND Epistemic score > Aleatory score, then treat as Epistemic.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_clean() first.")
            
        df = self.data.copy()

        df['Score_Epistemic'] = df['Rel'] + df['Comp'] + df['Tech']
        df['Score_Aleatory'] = df['Temp'] + df['Geo']
        
        def classify(row):
            # If Reliability is 5 (Not Verified), it is fundamentally Epistemic.
            # We classify it as such regardless of other scores.
            if row['Rel'] >= 5:
                return 'Epistemic'
            
            if (row['Score_Epistemic'] >= 8) and (row['Score_Epistemic'] > row['Score_Aleatory']):
                return 'Epistemic'
            return 'Aleatory'
        
        df['Uncertainty_Type'] = df.apply(classify, axis=1)
        
        # Ensure 'Contrib_Header' is still here (for debugging)
        if 'Contrib_Header' not in df.columns:
            print("WARNING: 'Contrib_Header' column lost during classification logic.")
        
        self.classified_data = df
        return df

    def characterise_variables(self):
        """
        Implements the mathematical characterisation:
            Aleatory -> Lognormal Parameters (mu_ln, sigma_ln)
            Epistemic -> Fuzzy Interval Bounds (Min, Mode, Max)
        Includes 'Contrib_Header' in the output for mapping later
        """
        if self.classified_data is None:
            raise ValueError("Data not classified. Call classify_uncertainty() first.")

        df = self.classified_data.copy()
        results = []

        for index, row in df.iterrows():
            mean = row['Mean']
            gsd = row['GSD']
            
            if mean <= 0 or gsd <= 0:
                continue

            # Aleatory maths
            sigma_ln = np.log(gsd)
            mu_ln = np.log(mean) - 0.5 * sigma_ln**2
            
            # Epistemic maths
            fuzzy_min = mean / (gsd**2)
            fuzzy_max = mean * (gsd**2)
            
            # Get the SimaPro mapping header
            # Using simple access to ensure we get the value
            if 'Contrib_Header' in row:
                contrib_header = row['Contrib_Header']
            else:
                None

            results.append({
                'Flow_Name': row['Flow_Name'],
                'Type': row['Uncertainty_Type'],
                'Params_Aleatory': {'mu_ln': mu_ln, 'sigma_ln': sigma_ln},
                'Params_Epistemic': {'min': fuzzy_min, 'mode': mean, 'max': fuzzy_max},
                'Raw_Mean': mean,
                'Contrib_Header': contrib_header # Critical for mapping from contributions file!
            })
            
        return pd.DataFrame(results)

class DynamicDataManager(LCIDataManager): 
    
    def __init__(self, file_path=None, start_year=2024, end_year=2050):
        # Inheritance from the parent class LCIDataManager
        # This creates self.data, self.classified_data, and pedigree dicts.
        super().__init__(file_path)
        
        # Initialise the specific attributes for this class
        self.start_year = start_year
        self.end_year = end_year
        self.n_steps = end_year - start_year + 1

    def characterise_dynamic_variables(self):
        """
        Projects base parameters over the temporal horizon and calculates 
        epistemic bounds and aleatory parameters for every time step t.
        """
        if self.classified_data is None:
            raise ValueError("Data not classified. Call classify_uncertainty() first.")

        df = self.classified_data.copy()
        results = []

        for index, row in df.iterrows():
            base_mean = row['Mean']
            gsd = row['GSD']
            
            # Fetch dynamic drivers from the dataset (Default to constant if missing)
            growth_rate = row.get('Growth_Rate', 0.0)
            growth_type = row.get('Growth_Type', 'Constant')

            if base_mean <= 0 or gsd <= 0:
                continue
            
            # Gaussian process integration
            if growth_type in ['GP_Forecast', 'Proxy_Ensemble']:
                # The arrays are already fully formed by the Gaussian Process in preprocess_dynamic.py
                mode_ts = np.array(row['GP_Mean_TS'])
                min_ts = np.array(row['GP_Min_TS'])
                max_ts = np.array(row['GP_Max_TS'])
                
                # Aleatory noise (sigma) maintains the empirical GSD ratio, applied to the moving GP mean
                sigma_ln_ts = np.full(self.n_steps, np.log(gsd))
                mu_ln_ts = np.log(mode_ts + 1e-9) - 0.5 * sigma_ln_ts**2 
                
            else:
                # Standard path: Initialise arrays to store mathematically projected parameters
                mu_ln_ts = np.zeros(self.n_steps)
                sigma_ln_ts = np.zeros(self.n_steps) 
                min_ts = np.zeros(self.n_steps)
                mode_ts = np.zeros(self.n_steps)
                max_ts = np.zeros(self.n_steps)

                for t in range(self.n_steps):
                    # Project the mean over time step t according to the growth type
                    if growth_type == 'Compound':
                        mean_t = base_mean * ((1 + growth_rate) ** t)
                    elif growth_type == 'Linear':
                        mean_t = base_mean + (growth_rate * t)
                    else: # Constant
                        mean_t = base_mean

                    # Characterise Aleatory (Lognormal) for time t
                    sigma_ln = np.log(gsd)
                    mu_ln = np.log(mean_t) - 0.5 * sigma_ln**2
                    
                    # Characterise Epistemic (Fuzzy Intervals) for time t
                    fuzzy_min = mean_t / (gsd**2)
                    fuzzy_max = mean_t * (gsd**2)
                    
                    # Store in time-series arrays
                    mu_ln_ts[t] = mu_ln
                    sigma_ln_ts[t] = sigma_ln
                    min_ts[t] = fuzzy_min
                    mode_ts[t] = mean_t
                    max_ts[t] = fuzzy_max

            # Critical for mapping to the structural flow matrix
            contrib_header = row.get('Contrib_Header', None)

            # Safely extract the topology data if it exists
            source_node = row.get('Source', 'Unknown_Source')
            target_node = row.get('Target', 'Unknown_Target')

            results.append({
                'Flow_Name': row['Flow_Name'],
                'Type': row['Uncertainty_Type'],
                'Params_Aleatory_TS': {'mu_ln': mu_ln_ts, 'sigma_ln': sigma_ln_ts},
                'Params_Epistemic_TS': {'min': min_ts, 'mode': mode_ts, 'max': max_ts},
                'Raw_Mean_Base': base_mean,
                'Contrib_Header': contrib_header,
                # Ensure these columns are carried over into dynamic_params
                'Source': source_node,
                'Target': target_node
            })
            
        return pd.DataFrame(results)

# UNIT TEST BLOCK
# (run the module directly, i.e.: python classify.py)
if __name__ == "__main__":
    # Adjust path for testing
    TEST_FILE = "data/LCI_test.xlsx - Sheet1.csv"
    if os.path.exists(TEST_FILE):
        manager = LCIDataManager(TEST_FILE)
        manager.load_and_clean()
        manager.classify_uncertainty()
        params = manager.characterise_variables()
        print("\nTest Output (First 5 rows):")
        # Check if Contrib_Header is actually populated
        print(params[['Flow_Name', 'Contrib_Header']].head())