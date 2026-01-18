"""
Visualization module for Boolean function analysis.

This module provides comprehensive plotting and visualization tools for
Boolean functions, including influence plots, Fourier spectrum visualization,
and interactive analysis tools.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available - plotting disabled")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ..core.base import BooleanFunction
from ..analysis import SpectralAnalyzer


class BooleanFunctionVisualizer:
    """
    Comprehensive visualization toolkit for Boolean functions.
    
    Provides static and interactive plots for Boolean function analysis,
    including influences, Fourier spectra, truth tables, and more.
    """
    
    def __init__(self, function: BooleanFunction, backend: str = "matplotlib"):
        """
        Initialize visualizer.
        
        Args:
            function: Boolean function to visualize
            backend: Plotting backend ("matplotlib", "plotly")
        """
        self.function = function
        self.n_vars = function.n_vars
        if self.n_vars is None:
            raise ValueError("Function must have defined number of variables")
        
        self.backend = backend.lower()
        self.analyzer = SpectralAnalyzer(function)
        
        # Validate backend availability
        if self.backend == "matplotlib" and not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not available")
        elif self.backend == "plotly" and not HAS_PLOTLY:
            raise ImportError("Plotly not available")
    
    def plot_influences(self, 
                       figsize: Tuple[int, int] = (10, 6),
                       save_path: Optional[str] = None,
                       show: bool = True) -> Any:
        """
        Plot variable influences.
        
        Args:
            figsize: Figure size for matplotlib
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        influences = self.analyzer.influences()
        
        if self.backend == "matplotlib":
            return self._plot_influences_matplotlib(influences, figsize, save_path, show)
        elif self.backend == "plotly":
            return self._plot_influences_plotly(influences, save_path, show)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _plot_influences_matplotlib(self, influences, figsize, save_path, show):
        """Plot influences using matplotlib."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        x_pos = np.arange(len(influences))
        bars = ax.bar(x_pos, influences, alpha=0.7, 
                     color='steelblue', edgecolor='navy')
        
        # Customize plot
        ax.set_xlabel('Variable Index')
        ax.set_ylabel('Influence')
        ax.set_title(f'Variable Influences (n={self.n_vars})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'x_{i}' for i in range(len(influences))])
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, influence in zip(bars, influences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{influence:.3f}', ha='center', va='bottom')
        
        # Add total influence annotation
        total_influence = np.sum(influences)
        ax.text(0.02, 0.98, f'Total Influence: {total_influence:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def _plot_influences_plotly(self, influences, save_path, show):
        """Plot influences using plotly."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly not available")
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f'x_{i}' for i in range(len(influences))],
                y=influences,
                text=[f'{inf:.3f}' for inf in influences],
                textposition='auto',
                marker_color='steelblue'
            )
        ])
        
        fig.update_layout(
            title=f'Variable Influences (n={self.n_vars})',
            xaxis_title='Variable Index',
            yaxis_title='Influence',
            showlegend=False
        )
        
        # Add total influence annotation
        total_influence = np.sum(influences)
        fig.add_annotation(
            text=f'Total Influence: {total_influence:.3f}',
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="wheat",
            bordercolor="black",
            borderwidth=1
        )
        
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        
        return fig
    
    def plot_fourier_spectrum(self,
                             max_degree: Optional[int] = None,
                             figsize: Tuple[int, int] = (12, 8),
                             save_path: Optional[str] = None,
                             show: bool = True) -> Any:
        """
        Plot Fourier spectrum grouped by degree.
        
        Args:
            max_degree: Maximum degree to plot (None for all)
            figsize: Figure size for matplotlib
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        fourier_coeffs = self.analyzer.fourier_expansion()
        
        if self.backend == "matplotlib":
            return self._plot_fourier_matplotlib(fourier_coeffs, max_degree, figsize, save_path, show)
        elif self.backend == "plotly":
            return self._plot_fourier_plotly(fourier_coeffs, max_degree, save_path, show)
    
    def _plot_fourier_matplotlib(self, fourier_coeffs, max_degree, figsize, save_path, show):
        """Plot Fourier spectrum using matplotlib."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not available")
        
        # Group coefficients by degree
        degrees = {}
        for i, coeff in enumerate(fourier_coeffs):
            degree = bin(i).count('1')
            if max_degree is None or degree <= max_degree:
                if degree not in degrees:
                    degrees[degree] = []
                degrees[degree].append(abs(coeff))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Coefficients by degree (box plot)
        if degrees:
            degree_list = sorted(degrees.keys())
            coeff_data = [degrees[d] for d in degree_list]
            
            bp = ax1.boxplot(coeff_data, labels=[f'Deg {d}' for d in degree_list],
                           patch_artist=True)
            
            # Color boxes
            colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax1.set_xlabel('Fourier Coefficient Degree')
        ax1.set_ylabel('|Fourier Coefficient|')
        ax1.set_title('Fourier Spectrum by Degree')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spectral concentration
        if degrees:
            degree_list = sorted(degrees.keys())
            concentrations = []
            total_weight = np.sum(fourier_coeffs ** 2)
            
            cumulative_weight = 0
            for d in degree_list:
                degree_weight = np.sum([c**2 for c in degrees[d]])
                cumulative_weight += degree_weight
                concentrations.append(cumulative_weight / total_weight)
            
            ax2.plot(degree_list, concentrations, 'o-', linewidth=2, markersize=8)
            ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% threshold')
            ax2.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='99% threshold')
        
        ax2.set_xlabel('Maximum Degree')
        ax2.set_ylabel('Cumulative Spectral Weight')
        ax2.set_title('Spectral Concentration')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def _plot_fourier_plotly(self, fourier_coeffs, max_degree, save_path, show):
        """Plot Fourier spectrum using plotly."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly not available")
        
        # Group coefficients by degree
        degrees = {}
        for i, coeff in enumerate(fourier_coeffs):
            degree = bin(i).count('1')
            if max_degree is None or degree <= max_degree:
                if degree not in degrees:
                    degrees[degree] = []
                degrees[degree].append(abs(coeff))
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Fourier Spectrum by Degree', 'Spectral Concentration')
        )
        
        # Plot 1: Box plot of coefficients by degree
        if degrees:
            for degree, coeffs in degrees.items():
                fig.add_trace(
                    go.Box(y=coeffs, name=f'Deg {degree}', showlegend=False),
                    row=1, col=1
                )
        
        # Plot 2: Spectral concentration
        if degrees:
            degree_list = sorted(degrees.keys())
            concentrations = []
            total_weight = np.sum(fourier_coeffs ** 2)
            
            cumulative_weight = 0
            for d in degree_list:
                degree_weight = np.sum([c**2 for c in degrees[d]])
                cumulative_weight += degree_weight
                concentrations.append(cumulative_weight / total_weight)
            
            fig.add_trace(
                go.Scatter(x=degree_list, y=concentrations, mode='lines+markers',
                          name='Cumulative Weight', showlegend=False),
                row=1, col=2
            )
            
            # Add threshold lines
            fig.add_hline(y=0.9, line_dash="dash", line_color="red", 
                         annotation_text="90%", row=1, col=2)
            fig.add_hline(y=0.99, line_dash="dash", line_color="orange",
                         annotation_text="99%", row=1, col=2)
        
        fig.update_xaxes(title_text="Fourier Coefficient Degree", row=1, col=1)
        fig.update_yaxes(title_text="|Fourier Coefficient|", row=1, col=1)
        fig.update_xaxes(title_text="Maximum Degree", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Spectral Weight", range=[0, 1.05], row=1, col=2)
        
        fig.update_layout(title_text=f"Fourier Analysis (n={self.n_vars})")
        
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        
        return fig
    
    def plot_truth_table(self,
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None,
                        show: bool = True) -> Any:
        """
        Plot truth table as a heatmap.
        
        Args:
            figsize: Figure size for matplotlib
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        if self.n_vars > 6:
            warnings.warn("Truth table visualization not recommended for >6 variables")
        
        # Get truth table
        truth_table = self.function.get_representation('truth_table')
        size = len(truth_table)
        
        if self.backend == "matplotlib":
            return self._plot_truth_table_matplotlib(truth_table, figsize, save_path, show)
        elif self.backend == "plotly":
            return self._plot_truth_table_plotly(truth_table, save_path, show)
    
    def _plot_truth_table_matplotlib(self, truth_table, figsize, save_path, show):
        """Plot truth table using matplotlib."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not available")
        
        size = len(truth_table)
        
        # For small functions, show as table
        if self.n_vars <= 4:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create input labels
            inputs = []
            outputs = []
            for i in range(size):
                binary_str = format(i, f'0{self.n_vars}b')
                inputs.append(binary_str)
                outputs.append(int(truth_table[i]))
            
            # Create table
            table_data = []
            for i, (inp, out) in enumerate(zip(inputs, outputs)):
                row = list(inp) + [str(out)]
                table_data.append(row)
            
            headers = [f'x_{i}' for i in range(self.n_vars)] + ['f(x)']
            
            # Color cells based on output
            cell_colors = []
            for row in table_data:
                row_colors = ['lightgray'] * self.n_vars
                if row[-1] == '1':
                    row_colors.append('lightgreen')
                else:
                    row_colors.append('lightcoral')
                cell_colors.append(row_colors)
            
            # Create table without cellColors for compatibility
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center')
            
            # Apply cell colors manually if supported
            try:
                for i, row_colors in enumerate(cell_colors):
                    for j, color in enumerate(row_colors):
                        table[(i+1, j)].set_facecolor(color)
            except (AttributeError, IndexError):
                pass  # Skip coloring if not supported
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            ax.axis('off')
            ax.set_title(f'Truth Table (n={self.n_vars})')
        
        else:
            # For larger functions, show as 1D heatmap
            fig, ax = plt.subplots(figsize=figsize)
            
            # Reshape for better visualization
            if size == 64:  # 6 variables
                truth_matrix = truth_table.reshape(8, 8)
            elif size == 32:  # 5 variables
                truth_matrix = truth_table.reshape(4, 8)
            else:
                truth_matrix = truth_table.reshape(1, -1)
            
            im = ax.imshow(truth_matrix, cmap='RdYlBu_r', aspect='auto')
            ax.set_title(f'Truth Table Heatmap (n={self.n_vars})')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Function Output')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def _plot_truth_table_plotly(self, truth_table, save_path, show):
        """Plot truth table using plotly."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly not available")
        
        size = len(truth_table)
        
        if self.n_vars <= 4:
            # Create table visualization
            inputs = []
            for i in range(size):
                binary_str = format(i, f'0{self.n_vars}b')
                inputs.append(list(binary_str))
            
            inputs = np.array(inputs)
            headers = [f'x_{i}' for i in range(self.n_vars)] + ['f(x)']
            
            # Combine inputs and outputs
            table_data = np.column_stack([inputs, truth_table.astype(int)])
            
            fig = go.Figure(data=[go.Table(
                header=dict(values=headers,
                           fill_color='paleturquoise',
                           align='center'),
                cells=dict(values=[table_data[:, i] for i in range(table_data.shape[1])],
                          fill_color=[['white' if val == '0' else 'lightgreen' 
                                     for val in table_data[:, -1]] if i == len(headers)-1 
                                    else 'lightgray' for i in range(len(headers))],
                          align='center'))
            ])
            
            fig.update_layout(title=f'Truth Table (n={self.n_vars})')
        
        else:
            # Heatmap for larger functions
            if size == 64:  # 6 variables
                truth_matrix = truth_table.reshape(8, 8)
            elif size == 32:  # 5 variables  
                truth_matrix = truth_table.reshape(4, 8)
            else:
                truth_matrix = truth_table.reshape(1, -1)
            
            fig = go.Figure(data=go.Heatmap(
                z=truth_matrix,
                colorscale='RdYlBu_r',
                showscale=True
            ))
            
            fig.update_layout(title=f'Truth Table Heatmap (n={self.n_vars})')
        
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        
        return fig
    
    def plot_noise_stability_curve(self,
                                  rho_range: Optional[np.ndarray] = None,
                                  figsize: Tuple[int, int] = (10, 6),
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> Any:
        """
        Plot noise stability as a function of correlation ρ.
        
        Args:
            rho_range: Range of ρ values to plot
            figsize: Figure size for matplotlib
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        if rho_range is None:
            rho_range = np.linspace(-1, 1, 50)
        
        # Compute noise stability for each ρ
        stabilities = []
        for rho in rho_range:
            try:
                stability = self.analyzer.noise_stability(rho)
                stabilities.append(stability)
            except:
                stabilities.append(np.nan)
        
        if self.backend == "matplotlib":
            return self._plot_noise_stability_matplotlib(rho_range, stabilities, figsize, save_path, show)
        elif self.backend == "plotly":
            return self._plot_noise_stability_plotly(rho_range, stabilities, save_path, show)
    
    def _plot_noise_stability_matplotlib(self, rho_range, stabilities, figsize, save_path, show):
        """Plot noise stability using matplotlib."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(rho_range, stabilities, 'b-', linewidth=2, label='Noise Stability')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random threshold')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Correlation ρ')
        ax.set_ylabel('Noise Stability')
        ax.set_title(f'Noise Stability Curve (n={self.n_vars})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def _plot_noise_stability_plotly(self, rho_range, stabilities, save_path, show):
        """Plot noise stability using plotly."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly not available")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rho_range, y=stabilities,
            mode='lines',
            name='Noise Stability',
            line=dict(width=3)
        ))
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Random threshold")
        fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f'Noise Stability Curve (n={self.n_vars})',
            xaxis_title='Correlation ρ',
            yaxis_title='Noise Stability',
            xaxis_range=[-1, 1],
            yaxis_range=[0, 1]
        )
        
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        
        return fig
    
    def create_dashboard(self, save_path: Optional[str] = None, show: bool = True) -> Any:
        """
        Create comprehensive analysis dashboard.
        
        Args:
            save_path: Path to save the dashboard
            show: Whether to display the dashboard
            
        Returns:
            Dashboard figure
        """
        if self.backend == "matplotlib":
            return self._create_matplotlib_dashboard(save_path, show)
        elif self.backend == "plotly":
            return self._create_plotly_dashboard(save_path, show)
    
    def _create_matplotlib_dashboard(self, save_path, show):
        """Create matplotlib dashboard."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not available")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Influences plot
        ax1 = fig.add_subplot(gs[0, 0])
        influences = self.analyzer.influences()
        ax1.bar(range(len(influences)), influences, alpha=0.7, color='steelblue')
        ax1.set_title('Variable Influences')
        ax1.set_xlabel('Variable')
        ax1.set_ylabel('Influence')
        
        # 2. Fourier spectrum
        ax2 = fig.add_subplot(gs[0, 1])
        fourier_coeffs = self.analyzer.fourier_expansion()
        ax2.plot(np.abs(fourier_coeffs), 'o-', alpha=0.7)
        ax2.set_title('Fourier Spectrum')
        ax2.set_xlabel('Coefficient Index')
        ax2.set_ylabel('|Coefficient|')
        
        # 3. Summary statistics
        ax3 = fig.add_subplot(gs[0, 2])
        summary = self.analyzer.summary()
        metrics = list(summary.keys())[:6]  # Show first 6 metrics
        values = [summary[m] for m in metrics]
        
        y_pos = np.arange(len(metrics))
        bars = ax3.barh(y_pos, values, alpha=0.7, color='lightgreen')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        ax3.set_title('Summary Statistics')
        
        # 4. Noise stability (if function is small enough)
        if self.n_vars <= 6:
            ax4 = fig.add_subplot(gs[1, :])
            rho_range = np.linspace(-1, 1, 20)
            stabilities = [self.analyzer.noise_stability(rho) for rho in rho_range]
            ax4.plot(rho_range, stabilities, 'b-', linewidth=2)
            ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            ax4.set_title('Noise Stability Curve')
            ax4.set_xlabel('Correlation ρ')
            ax4.set_ylabel('Stability')
            ax4.grid(True, alpha=0.3)
        
        # 5. Truth table (if small enough)
        if self.n_vars <= 4:
            ax5 = fig.add_subplot(gs[2, :])
            truth_table = self.function.get_representation('truth_table')
            
            # Create visual truth table
            size = len(truth_table)
            x_pos = np.arange(size)
            colors = ['red' if val else 'blue' for val in truth_table]
            bars = ax5.bar(x_pos, [1]*size, color=colors, alpha=0.7)
            
            # Add binary labels
            labels = [format(i, f'0{self.n_vars}b') for i in range(size)]
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(labels, rotation=45 if size > 8 else 0)
            ax5.set_title('Truth Table')
            ax5.set_ylabel('Output')
            ax5.set_ylim(0, 1.2)
        
        fig.suptitle(f'Boolean Function Analysis Dashboard (n={self.n_vars})', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def _create_plotly_dashboard(self, save_path, show):
        """Create plotly dashboard."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly not available")
        
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Variable Influences', 'Fourier Spectrum', 'Summary Statistics', 'Truth Table'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Influences plot
        influences = self.analyzer.influences()
        fig.add_trace(
            go.Bar(
                x=[f'x_{i}' for i in range(len(influences))],
                y=influences,
                name='Influences',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Fourier spectrum
        fourier_coeffs = self.analyzer.fourier_expansion()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(fourier_coeffs))),
                y=np.abs(fourier_coeffs),
                mode='markers',
                name='Fourier Coefficients',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Summary statistics
        summary = self.analyzer.summary()
        metrics = list(summary.keys())[:6]  # Show first 6 metrics
        values = [summary[m] for m in metrics]
        
        fig.add_trace(
            go.Bar(
                x=values,
                y=[m.replace('_', ' ').title() for m in metrics],
                orientation='h',
                name='Statistics',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Truth table (if small enough)
        if self.n_vars <= 4:
            truth_table = self.function.get_representation('truth_table')
            size = len(truth_table)
            colors = ['red' if val else 'blue' for val in truth_table]
            
            fig.add_trace(
                go.Bar(
                    x=[format(i, f'0{self.n_vars}b') for i in range(size)],
                    y=[1]*size,
                    marker_color=colors,
                    name='Truth Table',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'Boolean Function Analysis Dashboard (n={self.n_vars})',
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        if show:
            fig.show()
        
        return fig


# Convenience functions
def plot_function_comparison(functions: Dict[str, BooleanFunction],
                           metric: str = "influences",
                           backend: str = "matplotlib",
                           **kwargs) -> Any:
    """
    Compare multiple Boolean functions side by side.
    
    Args:
        functions: Dictionary mapping names to BooleanFunction objects
        metric: Metric to compare ("influences", "fourier", "noise_stability")
        backend: Plotting backend
        **kwargs: Additional plotting arguments
        
    Returns:
        Comparison plot
    """
    if backend == "matplotlib" and not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib not available")
    
    if metric == "influences":
        return _compare_influences_matplotlib(functions, **kwargs)
    elif metric == "fourier":
        return _compare_fourier_matplotlib(functions, **kwargs)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _compare_influences_matplotlib(functions, figsize=(12, 6), **kwargs):
    """Compare influences using matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)
    
    width = 0.8 / len(functions)
    x = np.arange(max(f.n_vars for f in functions.values()))
    
    for i, (name, func) in enumerate(functions.items()):
        analyzer = SpectralAnalyzer(func)
        influences = analyzer.influences()
        
        # Pad with zeros if needed
        padded_influences = np.zeros(len(x))
        padded_influences[:len(influences)] = influences
        
        ax.bar(x + i*width, padded_influences, width, 
               label=name, alpha=0.7)
    
    ax.set_xlabel('Variable Index')
    ax.set_ylabel('Influence')
    ax.set_title('Influence Comparison')
    ax.set_xticks(x + width*(len(functions)-1)/2)
    ax.set_xticklabels([f'x_{i}' for i in range(len(x))])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


# Import growth visualization
try:
    from .growth_plots import (
        GrowthVisualizer,
        LTFVisualizer,
        ComplexityVisualizer,
        quick_growth_plot,
    )
    HAS_GROWTH_VIZ = True
except ImportError:
    HAS_GROWTH_VIZ = False

# Export main classes
__all__ = [
    'BooleanFunctionVisualizer',
    'plot_function_comparison',
]

if HAS_GROWTH_VIZ:
    __all__.extend([
        'GrowthVisualizer',
        'LTFVisualizer',
        'ComplexityVisualizer',
        'quick_growth_plot',
    ])
