import pandas as pd
import numpy as np
import warnings
from scipy.optimize import curve_fit
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

class EmpiricalDataProcessor:
    """
    Setup for Dynamic Preprocessing.
    Acts as the bridge between raw, messy historical datasets 
    and the structured input required by the DynamicDataManager. 
    It replaces subjective Pedigree scoring with algorithmic data evaluation.
    """
    def __init__(self, raw_historical_data, proxy_data=None, simulation_start_year=2024, simulation_end_year=2050):
        """
        Args:
            raw_historical_data (dataframe): Time-series data of historical flows.
                Expected format: Rows = Years, Columns = Flow_Names.
            proxy_data (dataframe): Exogenous variables (e.g., GDP, Population) for forecasting.
            simulation_start_year (int): The baseline year (t=0) for the dynamic simulation.
        """
        self.raw_data = raw_historical_data.copy()
        self.start_year = simulation_start_year
        self.end_year = simulation_end_year
        self.target_years = np.arange(self.start_year, self.end_year + 1)
        self.structured_results = []
        if proxy_data is not None:
            self.proxy_data = proxy_data.copy()
        else:
            None
    
    # Algorithmic Data Quatlity Indicators (DQI)
    # -------------------------------------------------------------------------
    def _calculate_empirical_dqis(self, flow_series):
        """
        Calculates DQIs algorithmically from the time-series array.
        Returns a dictionary of Pedigree-style scores and the empirical GSD.
        """
        # Base Statistics
        n_total = len(flow_series)
        n_missing = flow_series.isna().sum()
        valid_data = flow_series.dropna()
        
        # Default fallback scores if data is completely empty
        # Highest equivalent scores
        if valid_data.empty:
            return {'GSD': 1.5, 'Rel': 5, 'Comp': 5, 'Temp': 5, 'Geo': 5, 'Tech': 5}

        # Completeness Penalty
        # Define equivalent completeness scores based on missing ratio
        missing_ratio = n_missing / n_total
        if missing_ratio < 0.05:
            comp_score = 1
        elif missing_ratio < 0.15:
            comp_score = 2
        elif missing_ratio < 0.30:
            comp_score = 3
        else:
            comp_score = 5 # Triggers Epistemic classification due to severe data gaps

        # Temporal Penalty
        # Similarly to completeness penalty
        # but this time using a "continuity" criteria w.r.t recent values 
        # Assuming the flow_series index represents years (e.g., 2010, 2011...)
        last_valid_year = valid_data.index[-1]
        time_gap = self.start_year - last_valid_year
        
        if time_gap <= 1:
            temp_score = 1
        elif time_gap <= 3:
            temp_score = 2
        elif time_gap <= 5:
            temp_score = 3
        elif time_gap <= 10:
            temp_score = 4
        else:
            temp_score = 5 # Triggers Epistemic classification due to outdated baseline

        # Aleatory Baseline (Calculating Empirical GSD)
        mean_val = valid_data.mean()
        std_val = valid_data.std()
        
        if mean_val > 0 and std_val > 0:
            cv = std_val / mean_val
            # Convert CV to Lognormal Geometric Standard Deviation (GSD)
            # GSD = exp(sqrt(ln(1 + CV^2)))
            gsd = np.exp(np.sqrt(np.log(1 + cv**2)))
            
            # Reliability Penalty based on excessive noise
            if cv > 1.0:
                rel_score = 5 # Unreliable if cv > 1
            else:
                if cv > 0.5:
                    rel_score = 3
                else:
                    rel_score = 1

        else:
            gsd = 1.1 # Safe minimum variance
            rel_score = 5 # Unreliable if mean/std is 0 or negative
            
        return {
            'GSD': round(gsd, 4),
            'Rel': rel_score,
            'Comp': comp_score,
            'Temp': temp_score,
            'Geo': 1,  # Default (Assume geographically accurate unless mapped otherwise)
            'Tech': 1  # Default (Assume technologically accurate unless mapped otherwise)
        }

    # Imputation and epistemic extraction (Gaussian process)
    # -------------------------------------------------------------------------
    def _impute_with_gaussian_process(self, flow_series):
        """
        Uses Gaussian Process Regression to forecast the flow out to the end_year.
        Extracts the widening predictive error to explicitly define the Epistemic Bounds.
        """
        valid_data = flow_series.dropna()
        
        # Fallback if there is virtually no data to train on
        if len(valid_data) < 3:
            if not valid_data.empty:
                default_mean = valid_data.mean()
            else:
                default_mean = 0.0
            mu_pred = np.full(len(self.target_years), default_mean)
            # Massive arbitrary penalty bounds due to total data absence
            return mu_pred, np.zeros_like(mu_pred), mu_pred * 0.1, mu_pred * 5.0

        # Prepare X (Years) and y (Flow Values) for training
        X_train = np.array(valid_data.index).reshape(-1, 1)
        y_train = valid_data.values

        # Define the Kernel: 
        # Use a Gaussian or Radial Basis Function (RBF) kernel (for  smooth non-linear trends)
        # plus a WhiteKernel (accounts for historical measurement noise)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2)) \
                 + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        
        # Suppress convergence warnings for sparse data fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_train, y_train)

        # Predict over the entire forecasting horizon (2024 to 2050)
        X_pred = self.target_years.reshape(-1, 1)
        mu_pred, std_pred = gp.predict(X_pred, return_std=True)

        # Extract Epistemic bounds (95% Confidence Interval z-score = 1.96)
        z_score = 1.96
        min_ts = mu_pred - z_score * std_pred
        max_ts = mu_pred + z_score * std_pred

        # Physical constraint: mass flows and stocks cannot be negative
        min_ts = np.maximum(min_ts, 0.0)
        mu_pred = np.maximum(mu_pred, 0.0)
        max_ts = np.maximum(max_ts, 0.0)

        return mu_pred, std_pred, min_ts, max_ts

    # Uncertainty from proxy data
    # Define the competing mathematical models
    
    # We can use specific fit models by defining their general form
    @staticmethod
    def _model_exponential(x, a, b, c):
        return a * np.exp(b * x) + c

    @staticmethod
    def _model_logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    # Alternatively we can use model ensembles
    def _evaluate_proxy_ensembles(self, flow_series, proxy_name):
        """
        Uses an ensemble of machine learning models (Linear, Ridge, 
        and Decision Tree) to map historical flows to macroeconomic proxies.
        The spread between these algorithms forms the epistemic envelope.
        """
        # Fallback check: If proxy data is missing, revert to Gaussian Process
        if self.proxy_data is None or proxy_name not in self.proxy_data.columns:
            print(f"[Warning] Proxy '{proxy_name}' not found. Falling back to GP.")
            # Gaussian process returns 4 variables, but only 3 are expected from proxy, so we slice [:3]
            gp_mu, _, gp_min, gp_max = self._impute_with_gaussian_process(flow_series)
            return gp_mu, gp_min, gp_max

        # Align historical data to train the models
        valid_data = flow_series.dropna()
        hist_years = valid_data.index.intersection(self.proxy_data.index)
        
        if len(hist_years) < 3:
            # Not enough overlapping data to train advanced models; fallback to GP
            gp_mu, _, gp_min, gp_max = self._impute_with_gaussian_process(flow_series)
            return gp_mu, gp_min, gp_max

        # Reshape for scikit-learn
        X_train = self.proxy_data.loc[hist_years, proxy_name].values.astype(float).reshape(-1, 1)
        y_train = valid_data.loc[hist_years].values.astype(float)
        X_target = self.proxy_data.loc[self.target_years, proxy_name].values.astype(float).reshape(-1, 1)

        # Define the model ensemble (basen on Wang et al.)
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Tree': DecisionTreeRegressor(max_depth=3, random_state=42) 
        }

        predictions = []
        
        # Train all models and generate competing future forecasts
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_target)
                
                # Physical constraint: material flows cannot be negative
                pred = np.maximum(pred, 0.0)
                predictions.append(pred)
            except Exception as e:
                print(f"[Warning] {name} proxy model failed on {proxy_name}: {e}")

        if not predictions:
            # If all ML models crash, use the Gaussian process
            gp_mu, _, gp_min, gp_max = self._impute_with_gaussian_process(flow_series)
            return gp_mu, gp_min, gp_max

        # Extract ignorance for P-Box
        pred_matrix = np.vstack(predictions)
        
        # The lowest prediction across all algorithms becomes the Minimum Bound
        bound_min = np.min(pred_matrix, axis=0)
        # The highest prediction becomes the Maximum Bound
        bound_max = np.max(pred_matrix, axis=0)
        # The median of the ensemble becomes the core trajectory
        core_median = np.median(pred_matrix, axis=0)

        return core_median, bound_min, bound_max

    def _evaluate_proxy_ensembles_regressions(self, flow_series, proxy_name):
        """
        Alternative version of _evaluate_proxy_ensembles function using regressions
        Fits multiple models (Linear, Exponential, Logistic) using a proxy 
        variable (e.g., GDP) to quantify uncertainty.
        Returns the envelope (min, max) and the mean forecast.
        """
        
        if self.proxy_data is None or proxy_name not in self.proxy_data.columns:
            raise ValueError(f"Proxy data '{proxy_name}' not found.")

        # Align historical flow data with historical proxy data
        valid_data = flow_series.dropna()
        hist_years = valid_data.index
        
        # Extract X (proxy) and y (Flow) for training
        X_train = self.proxy_data.loc[hist_years, proxy_name].values.astype(float)
        y_train = valid_data.values.astype(float)
        
        # Extract X (proxy) for forecasting to 2050
        X_target = self.proxy_data.loc[self.target_years, proxy_name].values.astype(float)
        
        predictions = []

        # Model A: Linear Regression
        try:
            lin_model = LinearRegression()
            lin_model.fit(X_train.reshape(-1, 1), y_train)
            pred_lin = lin_model.predict(X_target.reshape(-1, 1))
            predictions.append(np.maximum(pred_lin, 0.0)) # Prevent negative physical flows
        except Exception:
            pass

        # Model B: Exponential Growth
        try:
            # Provide sensible initial guesses to help convergence
            p0_exp = (np.mean(y_train), 0.01, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt_exp, _ = curve_fit(self._model_exponential, X_train, y_train, p0=p0_exp, maxfev=10000)
            pred_exp = self._model_exponential(X_target, *popt_exp)
            predictions.append(np.maximum(pred_exp, 0.0))
        except Exception:
            pass

        # Model C: Logistic Saturation (S-Curve)
        try:
            # Initial guesses: L = max(y)*1.5, k = 0.1, x0 = median(x)
            p0_log = (np.max(y_train)*1.5, 0.1, np.median(X_train))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt_log, _ = curve_fit(self._model_logistic, X_train, y_train, p0=p0_log, maxfev=10000)
            pred_log = self._model_logistic(X_target, *popt_log)
            predictions.append(np.maximum(pred_log, 0.0))
        except Exception:
            pass

        if not predictions:
            # If all models fail to converge, use the Gaussian process
            return self._impute_with_gaussian_process(flow_series)[:4]

        # Stack all successful model predictions (shape: N_models x T_steps)
        predictions_matrix = np.vstack(predictions)
        
        # Extract the structural envelope (Epistemic bounds)
        mu_ts = np.mean(predictions_matrix, axis=0)
        min_ts = np.min(predictions_matrix, axis=0)
        max_ts = np.max(predictions_matrix, axis=0)

        return mu_ts, min_ts, max_ts

    # Fuzzy delay distributions 
    def _calculate_fuzzy_delay_distribution(self, min_life, mode_life, max_life, std_dev):
        """
        Generates outflow fraction arrays (scrap generation rates) for minimum, 
        mode, and maximum lifetime assumptions.
        Uses a normal distribution approximation for cohort survival/failure.
        """
        # Define the maximum age we need to track (cover 99.9% of the longest tail)
        max_age = int(max_life + 4 * std_dev)
        ages = np.arange(max_age)

        def _get_fractions(mean_L):
            """Calculates the discrete probability density (fraction leaving per year)."""
            # CDF(t) - CDF(t-1) gives the fraction of stock leaving at exactly age t
            cdf = norm.cdf(ages, loc=mean_L, scale=std_dev)
            pdf_approx = np.insert(np.diff(cdf), 0, cdf[0])
            # Normalise to ensure exactly 100% of the cohort eventually leaves
            return pdf_approx / np.sum(pdf_approx)

        # Min life -> Faster turnover -> Peaks earlier
        scrap_min = _get_fractions(min_life)
        # Mode life -> Nominal turnover
        scrap_mode = _get_fractions(mode_life)
        # Max life -> Slower turnover -> Peaks later (delayed scrap availability)
        scrap_max = _get_fractions(max_life)

        return scrap_min, scrap_mode, scrap_max

    # Main flows preprocessing pipeline
    def generate_structured_parameters(self, flow_mapping_dict=None, proxy_mapping_dict=None, 
                                       delay_mapping_dict=None, topology_mapping_dict=None):
        """
        Executes the preprocessing pipeline for every flow, allowing to include:
            topology_mapping_dict (e.g., {'Cu_Mining': {'Source': 'Environment', 'Target': 'Refining'}})
            delay_mapping_dict (e.g., {'Copper_Wire': {'min': 25, 'mode': 30, 'max': 35, 'std': 5}})
            proxy_mapping_dict (e.g., {'Copper_Demand': 'Global_GDP'})
        """
        print(f"[*] Initialising Empirical Preprocessing for {len(self.raw_data.columns)} flows...")
        
        for flow_name in self.raw_data.columns:
            series = self.raw_data[flow_name]
            empirical_scores = self._calculate_empirical_dqis(series)
            
            mu_ts, min_ts, max_ts = None, None, None
            growth_type = 'Constant'
            
            # Check if this flow is mapped to an exogenous proxy (e.g., GDP).
            if proxy_mapping_dict:
                mapped_proxy = proxy_mapping_dict.get(flow_name, None)
            else:
                None

            # Integration with proxy data
            if mapped_proxy is not None and self.proxy_data is not None:
                # if proxy data
                mu_ts, min_ts, max_ts = self._evaluate_proxy_ensembles(series, mapped_proxy)
                growth_type = 'Proxy_Ensemble'
                empirical_scores['Rel'] = 5 
                empirical_scores['Temp'] = 5
                
            elif empirical_scores['Comp'] >= 3 or empirical_scores['Temp'] >= 4 or self.end_year > self.start_year:
                # Gaussian Process (if no proxy)
                mu_ts, std_ts, min_ts, max_ts = self._impute_with_gaussian_process(series)
                growth_type = 'GP_Forecast'
                empirical_scores['Rel'] = 5 
                empirical_scores['Temp'] = 5

            # Integration with fuzzy delays
            mapped_delay = delay_mapping_dict.get(flow_name, None) if delay_mapping_dict else None
            scrap_min, scrap_mode, scrap_max = None, None, None
            
            if mapped_delay is not None:
                scrap_min, scrap_mode, scrap_max = self._calculate_fuzzy_delay_distribution(
                    min_life=mapped_delay['min'],
                    mode_life=mapped_delay['mode'],
                    max_life=mapped_delay['max'],
                    std_dev=mapped_delay['std']
                )

            #  Extract Topology for visualisation
            source_node = 'Unknown_Source'
            target_node = 'Unknown_Target'
            if topology_mapping_dict and flow_name in topology_mapping_dict:
                source_node = topology_mapping_dict[flow_name].get('Source', 'Unknown_Source')
                target_node = topology_mapping_dict[flow_name].get('Target', 'Unknown_Target')

            base_mean = series.dropna().iloc[-1] if not series.dropna().empty else 0.0
            contrib_header = flow_mapping_dict.get(flow_name, None) if flow_mapping_dict else None
            
            self.structured_results.append({
                'Flow_Name': flow_name,
                'Mean': base_mean,
                'GSD': empirical_scores['GSD'],
                'Rel': empirical_scores['Rel'],
                'Comp': empirical_scores['Comp'],
                'Temp': empirical_scores['Temp'],
                'Geo': empirical_scores['Geo'],
                'Tech': empirical_scores['Tech'],
                'Growth_Rate': 0.0, 
                'Growth_Type': growth_type,
                'GP_Mean_TS': mu_ts, 
                'GP_Min_TS': min_ts,
                'GP_Max_TS': max_ts,
                'Delay_Min_Dist': scrap_min,
                'Delay_Mode_Dist': scrap_mode,
                'Delay_Max_Dist': scrap_max,
                'Contrib_Header': contrib_header,
                # Append the new physical mapping attributes
                'Source': source_node,
                'Target': target_node
            })

        df_structured = pd.DataFrame(self.structured_results)
        print("[*] Preprocessing complete. Structured parameters ready for DynamicDataManager.")
        return df_structured