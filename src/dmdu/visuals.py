import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List, Dict, Union


class DMDUVisualiser:
    """
    Visual Analytics module for Scenario Discovery and Vulnerability Analytics (DMDU).
    Generates publication-grade plots for PRIM peeling trajectories, dimensional trade-off matrices,
    scenario bounding ranges, and CART feature importances.
    """

    @staticmethod
    def plot_prim_trajectory(prim_model, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plots the PRIM peeling trajectory showing the trade-off between Coverage and Density across iterations.
        Bubble size represents support (fraction of total observations inside the box).
        """
        if not hasattr(prim_model, 'get_trajectory') or not prim_model.is_fitted:
            raise RuntimeError("[DMDUVisualiser] Model must be a fitted PRIMAnalyser instance.")
            
        traj_df = prim_model.get_trajectory()
        best_step = prim_model.best_box_['step'] if prim_model.best_box_ else 0
        
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        
        sizes = traj_df['support'] * 350 + 25
        scatter = ax.scatter(
            traj_df['coverage'], traj_df['density'], s=sizes, c=traj_df['step'],
            cmap='viridis', alpha=0.75, edgecolors='k', linewidth=1.2
        )
        
        ax.plot(traj_df['coverage'], traj_df['density'], linestyle='--', color='gray', alpha=0.5)
        
        # Highlight optimal box
        best_row = traj_df[traj_df['step'] == best_step].iloc[0]
        ax.scatter(
            best_row['coverage'], best_row['density'], s=450, facecolors='none',
            edgecolors='red', linewidth=2.5, marker='*', label=f"Optimal Box (Step {best_step})"
        )
        
        # Add annotation for optimal box
        ax.annotate(
            f"Step {best_step}\nDensity: {best_row['density']*100:.1f}%\nCoverage: {best_row['coverage']*100:.1f}%",
            xy=(best_row['coverage'], best_row['density']),
            xytext=(10, -25), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='red', lw=1.5),
            fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='red', alpha=0.9)
        )
        
        ax.set_title("PRIM Peeling Trajectory: Coverage vs. Density Trade-off", fontsize=12, fontweight='bold')
        ax.set_xlabel("Coverage (Recall of Vulnerabilities)", fontsize=10)
        ax.set_ylabel("Density (Precision within Box)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Peeling Iteration Step", fontsize=10)
        
        ax.legend(loc='lower left', framealpha=0.9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close(fig)
        return fig

    @staticmethod
    def plot_box_boundaries(discovery_model, box_index: int = 0, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plots a normalized horizontal bar chart showing how the discovered scenario box limits
        constrain each parameter space compared to the full uncertainty range.
        """
        if not hasattr(discovery_model, 'boxes_') or not discovery_model.boxes_:
            raise RuntimeError("[DMDUVisualiser] No scenario boxes discovered in model.")
            
        box = discovery_model.boxes_[box_index]
        box_min = box['box_min']
        box_max = box['box_max']
        
        X_df = discovery_model.X_data
        features = discovery_model.feature_names
        
        ratios = []
        for col in features:
            f_min = X_df[col].min()
            f_max = X_df[col].max()
            b_min = box_min[col]
            b_max = box_max[col]
            ratio = (b_max - b_min) / (f_max - f_min) if f_max > f_min else 1.0
            ratios.append({
                'feature': col, 'f_min': f_min, 'f_max': f_max,
                'b_min': b_min, 'b_max': b_max, 'ratio': ratio
            })
            
        df_ratios = pd.DataFrame(ratios).sort_values(by='ratio', ascending=True).reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(features) + 2.5))
        y_pos = np.arange(len(df_ratios))
        
        for i, row in df_ratios.iterrows():
            span = row['f_max'] - row['f_min']
            norm_b_min = (row['b_min'] - row['f_min']) / span if span > 0 else 0.0
            norm_b_max = (row['b_max'] - row['f_min']) / span if span > 0 else 1.0
            
            ax.barh(i, 1.0, left=0.0, height=0.45, color='#e8e8e8', edgecolor='gray', alpha=0.7,
                    label='Full Uncertainty Range' if i == 0 else "")
            ax.barh(i, norm_b_max - norm_b_min, left=norm_b_min, height=0.55, color='#1f77b4',
                    edgecolor='darkblue', linewidth=1.5, label='Vulnerable Scenario Box' if i == 0 else "")
            
            ax.text(norm_b_min, i - 0.38, f"{row['b_min']:.2f}", va='top', ha='center', fontsize=8.5, color='darkblue', fontweight='bold')
            ax.text(norm_b_max, i - 0.38, f"{row['b_max']:.2f}", va='top', ha='center', fontsize=8.5, color='darkblue', fontweight='bold')
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_ratios['feature'], fontsize=10, fontweight='bold')
        ax.set_xlabel("Normalized Parameter Range [Min = 0.0, Max = 1.0]", fontsize=10)
        ax.set_title(
            f"Scenario Box Parameter Boundaries\n(Density: {box['density']*100:.1f}%, Coverage: {box['coverage']*100:.1f}%, Support: {box['support']*100:.1f}%)",
            fontsize=11.5, fontweight='bold'
        )
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close(fig)
        return fig

    @staticmethod
    def plot_dimensional_tradeoff_matrix(discovery_model, box_index: int = 0, top_n_features: int = 3, save_path: Optional[str] = None) -> plt.Figure:
        """
        Creates a 2D scatter trade-off matrix (pair plot) across the top driving parameters.
        Observations are colored by vulnerability status, with the bounding box overlaid as a shaded rectangle.
        """
        if not hasattr(discovery_model, 'boxes_') or not discovery_model.boxes_:
            raise RuntimeError("[DMDUVisualiser] No scenario boxes discovered in model.")
            
        box = discovery_model.boxes_[box_index]
        box_min = box['box_min']
        box_max = box['box_max']
        
        X_df = discovery_model.X_data
        y_bin = discovery_model.y_binary
        features = discovery_model.feature_names
        
        ratios = []
        for col in features:
            span = X_df[col].max() - X_df[col].min()
            ratio = (box_max[col] - box_min[col]) / span if span > 0 else 1.0
            ratios.append({'feature': col, 'ratio': ratio})
            
        df_ratios = pd.DataFrame(ratios).sort_values(by='ratio', ascending=True)
        top_features = df_ratios['feature'].head(top_n_features).tolist()
        
        if len(top_features) < 2:
            top_features = features[:min(2, len(features))]
            
        k = len(top_features)
        fig, axes = plt.subplots(k, k, figsize=(3.5 * k, 3.5 * k))
        if k == 2:
            axes = np.atleast_2d(axes)
            
        for i in range(k):
            for j in range(k):
                ax = axes[i, j]
                f_y = top_features[i]
                f_x = top_features[j]
                
                if i == j:
                    safe_vals = X_df.loc[y_bin == 0, f_x]
                    vuln_vals = X_df.loc[y_bin == 1, f_x]
                    ax.hist(safe_vals, bins=15, alpha=0.45, color='forestgreen', density=True, label='Safe' if i == 0 and j == 0 else "")
                    ax.hist(vuln_vals, bins=15, alpha=0.55, color='crimson', density=True, label='Vulnerable' if i == 0 and j == 0 else "")
                    ax.axvline(box_min[f_x], color='blue', linestyle='--', linewidth=1.8)
                    ax.axvline(box_max[f_x], color='blue', linestyle='--', linewidth=1.8)
                else:
                    ax.scatter(X_df.loc[y_bin == 0, f_x], X_df.loc[y_bin == 0, f_y],
                               c='forestgreen', alpha=0.3, s=15, label='Safe' if i == 0 and j == 1 else "")
                    ax.scatter(X_df.loc[y_bin == 1, f_x], X_df.loc[y_bin == 1, f_y],
                               c='crimson', alpha=0.65, s=22, label='Vulnerable' if i == 0 and j == 1 else "")
                    
                    rect_w = box_max[f_x] - box_min[f_x]
                    rect_h = box_max[f_y] - box_min[f_y]
                    rect = patches.Rectangle(
                        (box_min[f_x], box_min[f_y]), rect_w, rect_h,
                        linewidth=2.2, edgecolor='blue', facecolor='blue', alpha=0.15
                    )
                    ax.add_patch(rect)
                    
                if i == k - 1: ax.set_xlabel(f_x, fontsize=9.5, fontweight='bold')
                if j == 0: ax.set_ylabel(f_y, fontsize=9.5, fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.6)
                
        fig.suptitle(f"Dimensional Trade-off Matrix (Top {k} Driving Parameters)", fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close(fig)
        return fig

    @staticmethod
    def plot_cart_feature_importances(cart_model, top_n: int = 10, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plots a bar chart of Gini feature importances from CARTAnalyser to identify
        the primary parameters driving systemic vulnerability.
        """
        if not hasattr(cart_model, 'get_feature_importances') or not cart_model.is_fitted:
            raise RuntimeError("[DMDUVisualiser] Model must be a fitted CARTAnalyser instance.")
            
        imp_df = cart_model.get_feature_importances().head(top_n)
        
        fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(imp_df) + 2.0))
        y_pos = np.arange(len(imp_df))
        
        bars = ax.barh(y_pos, imp_df['Importance'], color='#2ca02c', edgecolor='darkgreen', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(imp_df['Feature'], fontsize=10, fontweight='bold')
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel("Gini Feature Importance Score", fontsize=10)
        ax.set_title("CART Feature Importances: Primary Vulnerability Drivers", fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.0, f"{width:.3f}", va='center', ha='left', fontsize=9, fontweight='bold')
            
        ax.set_xlim(0, max(imp_df['Importance'].max() * 1.15, 0.1))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close(fig)
        return fig