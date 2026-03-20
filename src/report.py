import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pyvis.network import Network

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import sys

class RobustnessReporter:
    """
    Module 3: Visualisation and reporting
        -> Translate the P-Box simulation results into decision-support metrics.
            Visualise the 'ignorance gap' (P-Box).
            Calculate the 'ignorance penalty' (Safety buffer required due to poor data).
            (Optional: Report compliance probability against the given limits.)
        Handles Fuzzy Results (Dictionary of DataFrames) by prioritizing:
        - Alpha=0 (The Envelope): For Safety/Conservative reporting.
        - Alpha=1 (The Core): For "Most Plausible" insight.
    """
    def __init__(self, simulation_results):
        """
        Args:
            simulation_results (dict): {alpha_level: DataFrame}
        """
        self.results_map = simulation_results

        # Handle empty input
        # This allows creating a dummy instance for structural comparison
        if not simulation_results:
            self.df_conservative = None
            self.df_core = None
            return
        
        # Determine the Conservative Baseline (Alpha=0)
        # If explicit 0.0 exists, use it. Otherwise use the lowest alpha available.
        if 0.0 in simulation_results:
            self.df_conservative = simulation_results[0.0]
        else:
            min_alpha = min(simulation_results.keys())
            self.df_conservative = simulation_results[min_alpha]
            
        # Determine the Core (Alpha=1) for comparison
        max_alpha = max(simulation_results.keys())
        self.df_core = simulation_results[max_alpha]

    def get_metrics_dictionary(self):
        """
        Calculates and returns the key robustness metrics as a dictionary.
        Used for creating the summary CSV.
        """
        if self.df_conservative is None:
            return {}
            
        # Statistics on Conservative Envelope
        df_0 = self.df_conservative
        
        # Optimistic (Min Estimation) & Conservative (Max Estimation) stats
        stats_min = df_0['Y_Min_Estimation'].describe(percentiles=[0.05, 0.95])
        stats_max = df_0['Y_Max_Estimation'].describe(percentiles=[0.05, 0.5, 0.95])
        
        # Aleatory Metrics
        var_p05 = stats_max['5%']
        var_p95 = stats_max['95%']
        median = stats_max['50%']
        swing_width = var_p95 - var_p05
        safety_buffer = var_p95 - median
        
        # Epistemic Metrics
        limit_accurate = stats_min['95%']
        limit_flawed = stats_max['95%']
        ignorance_penalty = limit_flawed - limit_accurate

        # Nomina metrics (Alpha = 1)
        # This represents the "Best Estimate" range, usually centered on the mean.
        nominal_p05 = np.nan
        nominal_p95 = np.nan
        if self.df_core is not None:
             # For Alpha=1, Min and Max Estimation collapse close to each other (or are identical)
             # We use Y_Max_Estimation as the representative distribution for the core.
             core_stats = self.df_core['Y_Max_Estimation'].describe(percentiles=[0.05, 0.95])
             nominal_p05 = core_stats['5%']
             nominal_p95 = core_stats['95%']
        
        return {
            'Natural Range (P05)': var_p05,
            'Natural Range (P95)': var_p95,
            'Swing Width': swing_width,
            'SAFETY BUFFER REQUIRED': safety_buffer,
            'Limit if Data Accurate': limit_accurate,
            'Limit if Data Flawed': limit_flawed,
            'IGNORANCE PENALTY': ignorance_penalty,
            'Nominal Range (P05)': nominal_p05,
            'Nominal Range (P95)': nominal_p95 
        }

    def generate_pbox_plot(self, target_limit=None, filename="pbox_fuzzy.png", 
                           xlabel="Impact", title="Robustness: Fuzzy P-Box"):
        
        plt.figure(figsize=(12, 7)) # Slightly wider for readability
        
        sorted_alphas = sorted(self.results_map.keys())
        
        # 1. PLOT CURVES
        for alpha in sorted_alphas:
            df = self.results_map[alpha]
            y_min_sorted = np.sort(df['Y_Min_Estimation'])
            y_max_sorted = np.sort(df['Y_Max_Estimation'])
            n = len(y_min_sorted)
            probs = np.linspace(0, 1, n)
            
            # CASE A: The Envelope (Alpha=0) - The "Safety Boundary"
            if alpha == 0.0:
                # Fill Ignorance Gap
                plt.fill_betweenx(probs, y_min_sorted, y_max_sorted, color='gray', alpha=0.35, label='Epistemic Gap (Ignorance)')
                # Plot Optimistic/Conservative Bounds
                plt.plot(y_min_sorted, probs, color='green', linestyle='--', linewidth=1.5, label='Optimistic Limit (Data Accurate)')
                plt.plot(y_max_sorted, probs, color='red', linestyle='--', linewidth=1.5, label='Conservative Limit (Data Flawed)')
                
            # CASE B: The Core (Alpha=1) - The "Plausible Center"
            elif alpha == 1.0:
                #plt.fill_betweenx(probs, y_min_sorted, y_max_sorted, color='blue', alpha=0.25, label='Most Plausible Core (alpha=1)')
                plt.plot(y_min_sorted, probs, color='blue', linestyle='-', linewidth=2, label='Most Plausible Core')
                plt.plot(y_max_sorted, probs, color='blue', linestyle='-', linewidth=2)
            
            # CASE C: Intermediate Alphas - Visual Context only
            #else:
                #plt.plot(y_min_sorted, probs, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
                #plt.plot(y_max_sorted, probs, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        # 2. ANNOTATIONS (Restored Features)
        # We calculate metrics based on the Conservative Curve (Alpha=0 Max)
        # because safety margins are always defined by the worst-case boundary.
        df_cons = self.df_conservative
        y_cons_sorted = np.sort(df_cons['Y_Max_Estimation'])
        n = len(y_cons_sorted)
        
        # Get coordinates for P05 and P95
        idx_05 = int(n * 0.05)
        idx_95 = int(n * 0.95)
        x_05 = y_cons_sorted[idx_05]
        x_95 = y_cons_sorted[idx_95]
        
        # Draw Arrow for Aleatory Variability
        # Positioned at y=0.05 to avoid cluttering the center
        plt.annotate('', xy=(x_05, 0.05), xytext=(x_95, 0.05),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        plt.text((x_05 + x_95)/2, 0.07, 'Aleatory Variability\n(Natural Fluctuation)', 
                 color='red', ha='center', fontsize=9, fontweight='bold')

        # Draw Target Limit if provided
        if target_limit:
            plt.axvline(target_limit, color='black', linestyle='-', linewidth=2, label=f'Target ({target_limit})')

        # 3. FORMATTING
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Cumulative Probability (Reliability)', fontsize=18)
        plt.title(title, fontsize=14)
        plt.legend(loc='lower right', frameon=True, fontsize=16)
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        # Save and Close
        try:
            plt.savefig(filename, dpi=300)
            print(f"[Module 3] Plot saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save plot. {e}")
        plt.close()

    def print_executive_summary(self, target_limit=None, unit_label="units"):
        """
        Restores the detailed breakdown of Risk (Aleatory vs Epistemic)
        using the Alpha=0 (Conservative) case as the primary design basis.
        """
        metrics = self.get_metrics_dictionary()
        if not metrics:
            return
        # 1. CALCULATE STATISTICS (Based on Alpha=0 Envelope)
        df_0 = self.df_conservative
        
        # Optimistic Curve (Best Case assumptions)
        stats_min = df_0['Y_Min_Estimation'].describe(percentiles=[0.05, 0.5, 0.95])
        # Conservative Curve (Worst Case assumptions)
        stats_max = df_0['Y_Max_Estimation'].describe(percentiles=[0.05, 0.5, 0.95])
        
        # Aleatory Metrics ()
        var_p05_op = stats_min['5%']  # From the Optimistic Curve
        var_p05 = stats_max['5%']  # From the Conservative Curve
        var_p95 = stats_max['95%']  # From the Conservative Curve
        median = stats_max['50%']  # From the Conservative Curve
        
        natural_swing = var_p95 - var_p05
        safety_buffer = var_p95 - median
        
        # Epistemic Metrics (Gap at P95)
        limit_optimistic = stats_min['95%']
        limit_conservative = stats_max['95%']
        ignorance_penalty = limit_conservative - limit_optimistic

        # Fuzzy Insight (Alpha=1 Gap)
        # How much does the gap shrink if we assume "Mode" values are true?
        df_1 = self.df_core
        s_min_1 = df_1['Y_Min_Estimation'].describe(percentiles=[0.95])
        s_max_1 = df_1['Y_Max_Estimation'].describe(percentiles=[0.95])
        gap_1 = s_max_1['95%'] - s_min_1['95%']
        
        # 2. PRINT REPORT
        print("\n" + "="*60)
        print(" EXECUTIVE ROBUSTNESS REPORT")
        print("="*60)
        
        # --- SECTION 1: ALEATORY ---
        print(f"1. ALEATORY VARIABILITY (Natural Fluctuation)")
        print(f"   The system naturally fluctuates due to input variability (Slope).")
        print(f"   (Calculated on the Conservative/Worst-Case envelope)")
        print(f"   - Natural Range (P05 - P95): {var_p05_op:.4f} to {var_p95:.4f} {unit_label}")
        print(f"   - Swing Width:               {natural_swing:.4f} {unit_label}")
        print(f"   >>> SAFETY BUFFER REQUIRED:  {safety_buffer:.4f} {unit_label}")
        print(f"       (You must design {safety_buffer:.4f} above the average to handle normal swings.)")
        print("-" * 60)

        # --- SECTION 2: EPISTEMIC ---
        print(f"2. EPISTEMIC UNCERTAINTY (The Ignorance Penalty)")
        print(f"   Risk due to missing data (The Gap between curves).")
        print(f"   - Limit if Data Accurate:    {limit_optimistic:.4f} {unit_label}")
        print(f"   - Limit if Data Flawed:      {limit_conservative:.4f} {unit_label}")
        print(f"   >>> IGNORANCE PENALTY:       {ignorance_penalty:.4f} {unit_label}")
        print(f"       (You are carrying this extra risk solely because of poor data.)")
        
        # --- SECTION 3: NOMINAL ---
        if not np.isnan(metrics['Nominal Range (P05)']):
            print("-" * 60)
            print(f"3. NOMINAL REFERENCE (Best Estimate / Alpha=1)")
            print(f"   (The range if data is exactly as specified in SimaPro)")
            print(f"   - Nominal Range (P05 - P95): {metrics['Nominal Range (P05)']:.4f} to {metrics['Nominal Range (P95)']:.4f} {unit_label}")
            print(f"   * Check: SimaPro Total should fall inside this range.")

        # --- SECTION 4: FUZZY INSIGHT ---
        print(f"   [Fuzzy Insight]")
        print(f"   - If inputs cluster around their mean (alpha=1),")
        print(f"     the ignorance gap shrinks to: {gap_1:.4f} {unit_label}")
        print("-" * 60)

        # --- SECTION 5: COMPLIANCE ---
        if target_limit:
            prob_success_best = np.mean(df_0['Y_Min_Estimation'] < target_limit)
            prob_success_worst = np.mean(df_0['Y_Max_Estimation'] < target_limit)
            
            print(f"3. COMPLIANCE RELIABILITY (Target < {target_limit} {unit_label})")
            print(f"   - Best Case Scenario:  {prob_success_best:.1%} probability of success")
            print(f"   - Worst Case Scenario: {prob_success_worst:.1%} probability of success")
            
        print("="*60 + "\n")

    def compare_structural_scenarios(self, comparison_data, filename="structural_comparison.png", unit_label="units"):
        """
        Structural Uncertainty Visualisation.
        Overlays the P-Boxes (Alpha=0 envelopes) of multiple scenarios to show Model Uncertainty.
        
        Args:
            comparison_data (dict): Dictionary { 'Scenario Name': results_map }
                                    where results_map is the output from propagate.py
        """
        plt.figure(figsize=(12, 7))
        
        # Color palette for different scenarios
        colors = ['blue', 'orange', 'green', 'purple', 'red']
        
        print("\n" + "="*60)
        print(" STRUCTURAL UNCERTAINTY REPORT")
        print("="*60)
        
        for i, (name, res_map) in enumerate(comparison_data.items()):
            # Select color
            c = colors[i % len(colors)]
            
            # Extract Alpha=0 (Conservative Envelope)
            # This is the standard for comparison
            if 0.0 in res_map:
                df = res_map[0.0]
            else:
                df = res_map[min(res_map.keys())]
            
            # Prepare Data
            y_min = np.sort(df['Y_Min_Estimation'])
            y_max = np.sort(df['Y_Max_Estimation'])
            probs = np.linspace(0, 1, len(y_min))
            
            # Plot the Area (Shaded)
            plt.fill_betweenx(probs, y_min, y_max, color=c, alpha=0.15)
            
            # Plot the Bounds
            # Solid line for Conservative (Worst Case), Dashed for Optimistic
            plt.plot(y_max, probs, color=c, linestyle='-', linewidth=2, label=f'{name} (Conservative)')
            plt.plot(y_min, probs, color=c, linestyle='--', linewidth=1)
            
            # Print Summary Stats for this Scenario
            mean_max = df['Y_Max_Estimation'].mean()
            print(f" Scenario: {name}")
            print(f"   - Mean Conservative Impact: {mean_max:.4f} {unit_label}")
            print(f"   - 95% Worst Case Limit:     {np.percentile(df['Y_Max_Estimation'], 95):.4f} {unit_label}")
            print("-" * 30)

        # Formatting
        plt.xlabel(f"Impact ({unit_label})", fontsize=12)
        plt.ylabel('Cumulative Probability (Reliability)', fontsize=12)
        plt.title(f"Structural Uncertainty: Scenario Comparison ({unit_label})", fontsize=14)
        plt.legend(loc='lower right', frameon=True)
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        try:
            plt.savefig(filename, dpi=300)
            print(f"[Module 3] Comparison Plot saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save plot. {e}")
        plt.close()
        print("="*60 + "\n")

class DynamicRobustnessReporter(RobustnessReporter):
    """
    Extension for Dynamic Material Flow Analysis (MFA).
    Translates the 2D time-series arrays into Temporal Fan Charts,
    visually separating Aleatory Variability from Epistemic Ignorance over time.
    """
    def __init__(self, dynamic_simulation_results, start_year, end_year):
        super().__init__(dynamic_simulation_results) # Inherit basic setup
        self.start_year = start_year
        self.end_year = end_year
        self.time_steps = np.arange(start_year, end_year + 1)

    def generate_temporal_envelope_plot(self, target_limit=None, filename="temporal_fan_chart.png", 
                                        ylabel="Copper Stock (Megatonnes)", title="Dynamic MFA: Copper Cycle Projection"):
        """
        Generates a fan chart showing the expansion of uncertainty from t=0 to t=T.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if self.df_conservative is None or self.df_core is None:
            print("Error: Incomplete dynamic simulation results.")
            return

        # 1. Extract 2D Arrays (Shape: N_iterations x T_steps)
        # Conservative Envelope (Alpha = 0.0) -> Epistemic + Aleatory
        stock_min_0 = self.df_conservative['Stock_Min_TS']
        stock_max_0 = self.df_conservative['Stock_Max_TS']
        
        # Core Envelope (Alpha = 1.0) -> Pure Aleatory
        stock_max_1 = self.df_core['Stock_Max_TS'] 

        # 2. Calculate Time-Series Percentiles (Collapse N iterations via axis=0)
        # Epistemic Bounds (Outer Fan)
        epi_lower = np.percentile(stock_min_0, 5, axis=0)
        epi_upper = np.percentile(stock_max_0, 95, axis=0)
        
        # Aleatory Bounds (Inner Fan)
        ale_lower = np.percentile(stock_max_1, 5, axis=0)
        ale_upper = np.percentile(stock_max_1, 95, axis=0)
        
        # Median Trajectory (Best Estimate)
        median_trajectory = np.median(stock_max_1, axis=0)

        # 3. Plotting the Fan Chart
        plt.figure(figsize=(12, 7))
        
        # Plot Epistemic Ignorance Gap (The grey penalty area)
        plt.fill_between(self.time_steps, epi_lower, epi_upper, color='gray', alpha=0.3, 
                         label='Epistemic Ignorance (Worst-Case Bounds)')
        
        # Plot Aleatory Variability (The blue natural noise area)
        plt.fill_between(self.time_steps, ale_lower, ale_upper, color='blue', alpha=0.4, 
                         label='Aleatory Variability (90% Confidence Interval)')
        
        # Plot Median Forecast
        plt.plot(self.time_steps, median_trajectory, color='black', linewidth=2, label='Median Forecast')

        # Optional: Add a target constraint line (e.g., maximum allowable emission budget)
        if target_limit:
            plt.axhline(target_limit, color='red', linestyle='--', linewidth=2, label=f'Climate Constraint ({target_limit})')

        # 4. Formatting
        plt.xlabel("Year", fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.xlim([self.start_year, self.end_year])
        plt.legend(loc='upper left', frameon=True, fontsize=12)
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        # Save and Close
        try:
            plt.savefig(filename, dpi=300)
            print(f"[Module 3 - Dynamic] Temporal Fan Chart saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save dynamic plot. {e}")
        plt.close()

    # STEP 3: THE COLOR-CODING ENGINE
    # ====================================================================
    def _get_epistemic_color(self, score, alpha=0.5):
        """
        Maps an S_Epistemic score (0.0 to 1.0) to a Hex color string.
        0.0 = Safe/Stable (Light Grey)
        1.0 = High Ignorance (Bright Red)
        """
        if pd.isna(score):
            return f"rgba(204, 204, 204, {alpha})" # Standard light grey

        # Use the 'Reds' colormap from matplotlib
        cmap = cm.get_cmap('Reds')
        
        # Adjust the base so a score of 0.0 isn't pure white (invisible on white backgrounds).
        # We compress the scale to [0.2, 1.0]
        adjusted_score = 0.2 + (score * 0.8) 
        
        # Matplotlib returns (r, g, b, a) as floats from 0.0 to 1.0
        r, g, b, _ = cmap(adjusted_score)
        
        # Plotly expects rgb values from 0 to 255
        return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})"

    def _extract_flow_colors(self, target_year, sensitivity_analyser=None):
        """
        Runs the time-sliced GSA for the target year and returns a dictionary 
        mapping each Flow_Name to its calculated Hex color.
        """
        color_map = {}
        
        # If no analyser is provided, default all flows to a neutral grey
        if sensitivity_analyser is None:
            print("[Warning] No Sensitivity Analyser provided. Sankey flows will be grey.")
            return color_map

        # Execute the GSA for the specific year
        try:
            df_gsa = sensitivity_analyser.run_time_sliced_analysis(target_year)
            
            for _, row in df_gsa.iterrows():
                flow_name = row['Flow_Name']
                epistemic_score = row['S_Epistemic']
                color_map[flow_name] = self._get_epistemic_color(epistemic_score)
                
        except Exception as e:
            print(f"[Warning] Failed to extract sensitivity colors: {e}")
            
        return color_map
    
    # STEP 4: CONSTRUCTING THE PLOTLY SANKEY OBJECT
    # ====================================================================
    def generate_uncertainty_sankey(self, target_year, structured_params, sensitivity_analyser=None, filename="sankey.html"):
        """
        Constructs the interactive Plotly Sankey diagram using topology and color maps.
        Extracts median values for width and extreme alpha=0 bounds for tooltips.
        """
        if target_year < self.start_year or target_year > self.end_year:
            raise ValueError(f"Target year {target_year} is out of simulation bounds.")

        # Temporal index for array slicing
        t_idx = target_year - self.start_year

        # 1. Fetch the colors based on Epistemic Ignorance (Step 3)
        flow_colors = self._extract_flow_colors(target_year, sensitivity_analyser)

        # 2. Build unique node list and index mapping (Topology)
        all_nodes = set()
        for _, row in structured_params.iterrows():
            all_nodes.add(row.get('Source', 'Unknown_Source'))
            all_nodes.add(row.get('Target', 'Unknown_Target'))
        
        node_labels = list(all_nodes)
        node_indices = {name: i for i, name in enumerate(node_labels)}

        # 3. Extract Plotly Link Data
        source_idx = []
        target_idx = []
        values = []
        link_colors = []
        customdata = []

        # We need the arrays to calculate median width and extreme bounds.
        # Alpha=0.0 gives the widest bounds (absolute ignorance).
        # Alpha=1.0 gives the core mode (used for the visual flow width).
        alpha_base = min(self.results_map.keys()) # Usually 0.0
        alpha_core = max(self.results_map.keys()) # Usually 1.0

        flows_min_ts = self.results_map[alpha_base]['Flows_Min_TS']
        flows_max_ts = self.results_map[alpha_base]['Flows_Max_TS']
        flows_core_ts = self.results_map[alpha_core]['Flows_Min_TS'] 

        for _, row in structured_params.iterrows():
            flow_name = row['Flow_Name']
            src = row.get('Source', 'Unknown_Source')
            tgt = row.get('Target', 'Unknown_Target')

            if flow_name not in flows_core_ts:
                continue 

            # Calculate Median (Width) using the core mode array
            val_core_array = flows_core_ts[flow_name][:, t_idx]
            median_val = np.median(val_core_array)

            # Skip drawing edges with near-zero flow to keep the diagram clean
            if median_val <= 1e-4:
                continue

            # Calculate Epistemic Bounds (Tooltips) using alpha=0.0 extreme arrays
            bound_min = np.min(flows_min_ts[flow_name][:, t_idx])
            bound_max = np.max(flows_max_ts[flow_name][:, t_idx])

            # Map to Plotly structures
            source_idx.append(node_indices[src])
            target_idx.append(node_indices[tgt])
            values.append(median_val)
            
            # Apply Epistemic Color (RGBA formatted for Plotly compatibility)
            # The alpha channel is now natively handled by _get_epistemic_color
            link_color = flow_colors.get(flow_name, "rgba(204, 204, 204, 0.5)")
            link_colors.append(link_color) 
            
            # Customdata fuels the interactive hover tooltip: [Name, Median, Min, Max]
            customdata.append([flow_name, median_val, bound_min, bound_max])

        # 4. Construct the Plotly Figure
        fig = go.Figure(data=[go.Sankey(
            valueformat=".2f",
            valuesuffix=" Mt",
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color="#333333" # Dark grey nodes for contrast
            ),
            link=dict(
                source=source_idx,
                target=target_idx,
                value=values,
                color=link_colors,
                customdata=customdata,
                # The customized HTML Tooltip
                hovertemplate=
                "<b>%{customdata[0]}</b><br />" +
                "Median Forecast: %{customdata[1]:.2f} Mt<br />" +
                "Worst-Case Ignorance Bounds: [%{customdata[2]:.2f} - %{customdata[3]:.2f}] Mt<extra></extra>"
            )
        )])

        fig.update_layout(
            title_text=f"Uncertainty-Aware Flow Topology ({target_year})",
            font_size=12,
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 5. Save the interactive HTML
        fig.write_html(filename)
        print(f"[Module 3 - Dynamic] Uncertainty Sankey saved to {filename}")

    # NEW: NETWORK TOPOLOGY GRAPH (PYVIS + NETWORKX)
    # ====================================================================
    def generate_network_topology(self, target_year, structured_params, sensitivity_analyser=None, filename="network_topology.html"):
        """
        Generates an interactive, physics-based network graph using pyvis.
        Ideal for auditing system boundaries and highly disaggregated model structures.
        """
        # Fetch the same sensitivity colors we used for the Sankey
        flow_colors = self._extract_flow_colors(target_year, sensitivity_analyser)

        # Initialize a directed PyVis network
        # We use a white background with dark text for a clean, academic look
        net = Network(height='800px', width='100%', directed=True, bgcolor='#ffffff', font_color='#333333')
        
        # Pyvis requires nodes to be added before edges. We'll track what we've added.
        added_nodes = set()

        for _, row in structured_params.iterrows():
            flow_name = row['Flow_Name']
            src = row.get('Source', 'Unknown_Source')
            tgt = row.get('Target', 'Unknown_Target')

            # 1. Add Nodes (Processes / Stocks / Environment)
            if src not in added_nodes:
                net.add_node(src, label=src, color='#aeb6bf', shape='box', font={'size': 16})
                added_nodes.add(src)
            if tgt not in added_nodes:
                net.add_node(tgt, label=tgt, color='#aeb6bf', shape='box', font={'size': 16})
                added_nodes.add(tgt)

            # 2. Add Edges (Flows)
            # Use the hex color from the sensitivity analyser to highlight uncertain routing
            edge_color = flow_colors.get(flow_name, '#cccccc')
            
            # Add the edge. 'title' appears on hover, 'label' appears on the line itself.
            net.add_edge(
                src, 
                tgt, 
                title=f"Flow: {flow_name}", 
                label=flow_name, 
                color=edge_color,
                arrows='to'
            )

        # 3. Apply physics layout for automatic organization
        # We use hierarchical repulsion so nodes don't overlap in complex models
        net.repulsion(node_distance=200, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
        
        # Optional: Adds a UI panel to the HTML so you can manually tweak the physics
        net.show_buttons(filter_=['physics'])

        # 4. Save to HTML
        net.write_html(filename)
        print(f"[Module 3 - Dynamic] Network Topology Graph saved to {filename}")


# --- UNIT TEST BLOCK ---
if __name__ == "__main__":
    print("\n--- TESTING MODULE 3: REPORTING ---")
    
    # 1. Generate MOCK RESULTS (Simulating Module 2 Output)
    np.random.seed(42)
    n_sim = 1000
    
    # Create two offset distributions to simulate a P-Box
    # Best case: Normal(100, 10)
    y_min = np.random.normal(100, 10, n_sim)
    # Worst case: Normal(120, 10) -> The gap is ~20 units
    y_max = np.random.normal(120, 10, n_sim)
    
    mock_results = pd.DataFrame({
        'Y_Min_Estimation': y_min,
        'Y_Max_Estimation': y_max
    })
    
    print("Mock Simulation Results Created.")
    
    # 2. Initialise Reporter
    reporter = RobustnessReporter(mock_results)

    # 3. Test Reporting Methods with DYNAMIC LABELS
    try:
        # Test 1: Plotting with custom units
        print("Generating Plot...")
        reporter.generate_pbox_plot(
            target_limit=115.0, 
            filename="test_pbox_unit_test.png",
            xlabel="Water Scarcity (m3 world-eq)",  # Testing dynamic label
            title="Test P-Box: Water Footprint"
        )
        
        # Test 2: Summary with custom units
        print("Generating Summary...")
        reporter.print_executive_summary(target_limit=115.0, unit_label="m3 eq")
        
        print("TEST SUCCESSFUL.")
    except Exception as e:
        print(f"TEST FAILED: {e}")