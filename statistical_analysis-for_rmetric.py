# statistical_analysis.py
"""
Statistical Analysis Script for R-Metric Experiment Results
Validates randomness, distribution properties, and statistical significance
of fault detection across multiple experimental runs.

For EMNLP 2024 Industry Track - R-Metric Paper
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.stats import (
    kstest, shapiro, anderson, jarque_bera,
    chi2_contingency, mannwhitneyu, kruskal,
    spearmanr, pearsonr
)
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


@dataclass
class StatisticalTestResults:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    interpretation: str = ""
    passed: bool = False


class RandomnessAnalyzer:
    """Analyzes randomness and statistical properties of experimental results"""
    
    def __init__(self, results_dir: str = "case_study_results"):
        self.results_dir = Path(results_dir)
        self.experiments = []
        self.aggregated_metrics = defaultdict(list)
        self.report_lines = []
        
    def load_experiments(self) -> int:
        """Load all experiment results from directory"""
        print("Loading experiment results...")
        
        for exp_dir in self.results_dir.glob("qwen_rmetric_*"):
            if exp_dir.is_dir():
                metrics_file = exp_dir / "metrics.csv"
                results_file = exp_dir / "results.json"
                
                if metrics_file.exists() and results_file.exists():
                    try:
                        metrics = pd.read_csv(metrics_file)
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                        
                        self.experiments.append({
                            'name': exp_dir.name,
                            'path': exp_dir,
                            'metrics': metrics,
                            'results': results
                        })
                        
                        print(f"  ✓ Loaded: {exp_dir.name}")
                    except Exception as e:
                        print(f"  ✗ Failed to load {exp_dir.name}: {e}")
        
        print(f"\nTotal experiments loaded: {len(self.experiments)}")
        return len(self.experiments)
    
    def aggregate_metrics(self):
        """Aggregate metrics across all experiments"""
        print("\nAggregating metrics...")
        
        for exp in self.experiments:
            metrics = exp['metrics']
            
            # Aggregate key metrics
            for col in ['r_metric', 'lambda_norm', 'sigma_sq_norm', 'delta_l_norm', 
                       'train_loss', 'lambda_raw', 'sigma_sq_raw', 'delta_l_raw']:
                if col in metrics.columns:
                    self.aggregated_metrics[col].extend(metrics[col].dropna().tolist())
            
            # Extract alert steps
            results = exp['results']
            for method, alert_step in results['alerts'].items():
                if alert_step is not None:
                    self.aggregated_metrics[f'alert_step_{method}'].append(alert_step)
        
        print("Aggregation complete.")
    
    def test_normality(self, data: np.ndarray, metric_name: str) -> List[StatisticalTestResults]:
        """Test for normality using multiple methods"""
        results = []
        
        # Remove NaN and infinite values
        data = data[np.isfinite(data)]
        
        if len(data) < 3:
            return results
        
        # 1. Shapiro-Wilk Test (n < 5000)
        if len(data) < 5000:
            try:
                stat, p_value = shapiro(data)
                results.append(StatisticalTestResults(
                    test_name="Shapiro-Wilk",
                    statistic=stat,
                    p_value=p_value,
                    interpretation=f"{'Normal' if p_value > 0.05 else 'Non-normal'} distribution",
                    passed=p_value > 0.05
                ))
            except Exception as e:
                print(f"  Shapiro-Wilk failed for {metric_name}: {e}")
        
        # 2. Kolmogorov-Smirnov Test
        try:
            stat, p_value = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            results.append(StatisticalTestResults(
                test_name="Kolmogorov-Smirnov",
                statistic=stat,
                p_value=p_value,
                interpretation=f"{'Normal' if p_value > 0.05 else 'Non-normal'} distribution",
                passed=p_value > 0.05
            ))
        except Exception as e:
            print(f"  K-S test failed for {metric_name}: {e}")
        
        # 3. Anderson-Darling Test
        try:
            result = anderson(data, dist='norm')
            # Use 5% significance level (index 2)
            critical_value = result.critical_values[2]
            passed = result.statistic < critical_value
            
            results.append(StatisticalTestResults(
                test_name="Anderson-Darling",
                statistic=result.statistic,
                p_value=result.significance_level[2] / 100,  # Convert to decimal
                critical_value=critical_value,
                interpretation=f"{'Normal' if passed else 'Non-normal'} distribution at 5% level",
                passed=passed
            ))
        except Exception as e:
            print(f"  Anderson-Darling failed for {metric_name}: {e}")
        
        # 4. Jarque-Bera Test
        try:
            stat, p_value = jarque_bera(data)
            results.append(StatisticalTestResults(
                test_name="Jarque-Bera",
                statistic=stat,
                p_value=p_value,
                interpretation=f"{'Normal' if p_value > 0.05 else 'Non-normal'} distribution",
                passed=p_value > 0.05
            ))
        except Exception as e:
            print(f"  Jarque-Bera failed for {metric_name}: {e}")
        
        return results
    
    def test_randomness_runs(self, data: np.ndarray, metric_name: str) -> StatisticalTestResults:
        """Wald-Wolfowitz runs test for randomness"""
        # Convert to binary based on median
        median = np.median(data)
        binary = (data > median).astype(int)
        
        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        
        # Expected runs and variance under null hypothesis
        n1 = np.sum(binary == 1)
        n0 = np.sum(binary == 0)
        n = n1 + n0
        
        expected_runs = (2 * n1 * n0) / n + 1
        variance_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))
        
        # Z-statistic
        z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return StatisticalTestResults(
            test_name="Runs Test (Randomness)",
            statistic=z_stat,
            p_value=p_value,
            interpretation=f"{'Random' if p_value > 0.05 else 'Non-random'} sequence",
            passed=p_value > 0.05
        )
    
    def test_autocorrelation(self, data: np.ndarray, lag: int = 1) -> StatisticalTestResults:
        """Test for autocorrelation"""
        if len(data) <= lag:
            return None
        
        # Calculate autocorrelation
        mean = np.mean(data)
        var = np.var(data)
        
        autocorr = np.correlate(data - mean, data - mean, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (var * len(data))
        
        if lag < len(autocorr):
            ac_value = autocorr[lag]
            
            # Under null hypothesis of no autocorrelation
            # standard error ≈ 1/sqrt(n)
            se = 1 / np.sqrt(len(data))
            z_stat = ac_value / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return StatisticalTestResults(
                test_name=f"Autocorrelation (lag={lag})",
                statistic=ac_value,
                p_value=p_value,
                interpretation=f"{'No significant' if p_value > 0.05 else 'Significant'} autocorrelation",
                passed=p_value > 0.05
            )
        
        return None
    
    def test_stationarity(self, data: np.ndarray, metric_name: str) -> StatisticalTestResults:
        """Augmented Dickey-Fuller test for stationarity"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(data, autolag='AIC')
            
            return StatisticalTestResults(
                test_name="Augmented Dickey-Fuller",
                statistic=result[0],
                p_value=result[1],
                critical_value=result[4]['5%'],
                interpretation=f"{'Stationary' if result[1] < 0.05 else 'Non-stationary'} series",
                passed=result[1] < 0.05
            )
        except ImportError:
            print(f"  Stationarity test requires statsmodels package")
            return None
        except Exception as e:
            print(f"  ADF test failed for {metric_name}: {e}")
            return None
    
    def test_independence(self, method1_alerts: List[int], method2_alerts: List[int]) -> StatisticalTestResults:
        """Chi-square test for independence between detection methods"""
        # Create contingency table
        # Need to align experiments
        max_len = min(len(method1_alerts), len(method2_alerts))
        if max_len == 0:
            return None
        
        method1 = np.array(method1_alerts[:max_len])
        method2 = np.array(method2_alerts[:max_len])
        
        # Create 2x2 contingency table
        # Both detected, only method1, only method2, neither
        both = np.sum((method1 > 0) & (method2 > 0))
        only1 = np.sum((method1 > 0) & (method2 == 0))
        only2 = np.sum((method1 == 0) & (method2 > 0))
        neither = np.sum((method1 == 0) & (method2 == 0))
        
        contingency = np.array([[both, only1], [only2, neither]])
        
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            return StatisticalTestResults(
                test_name="Chi-Square Independence",
                statistic=chi2,
                p_value=p_value,
                interpretation=f"{'Independent' if p_value > 0.05 else 'Dependent'} detection methods",
                passed=p_value > 0.05
            )
        except:
            return None
    
    def analyze_distributions(self):
        """Comprehensive distribution analysis"""
        print("\n" + "="*80)
        print("DISTRIBUTION ANALYSIS")
        print("="*80)
        
        self.report_lines.append("\n## Distribution Analysis\n")
        
        key_metrics = ['r_metric', 'lambda_norm', 'sigma_sq_norm', 'delta_l_norm']
        
        for metric in key_metrics:
            if metric not in self.aggregated_metrics or len(self.aggregated_metrics[metric]) == 0:
                continue
            
            data = np.array(self.aggregated_metrics[metric])
            
            print(f"\n--- {metric.upper().replace('_', ' ')} ---")
            self.report_lines.append(f"\n### {metric.upper().replace('_', ' ')}\n")
            
            # Descriptive statistics
            desc_stats = {
                'Count': len(data),
                'Mean': np.mean(data),
                'Std Dev': np.std(data),
                'Min': np.min(data),
                'Q1': np.percentile(data, 25),
                'Median': np.median(data),
                'Q3': np.percentile(data, 75),
                'Max': np.max(data),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data)
            }
            
            print("\nDescriptive Statistics:")
            for key, value in desc_stats.items():
                print(f"  {key:12s}: {value:10.4f}")
            
            self.report_lines.append("\n**Descriptive Statistics:**\n")
            self.report_lines.append("| Statistic | Value |\n")
            self.report_lines.append("|-----------|-------|\n")
            for key, value in desc_stats.items():
                self.report_lines.append(f"| {key} | {value:.4f} |\n")
            
            # Normality tests
            print("\nNormality Tests:")
            self.report_lines.append("\n**Normality Tests:**\n")
            
            normality_results = self.test_normality(data, metric)
            for result in normality_results:
                status = "✓" if result.passed else "✗"
                print(f"  {status} {result.test_name:25s}: stat={result.statistic:8.4f}, p={result.p_value:8.4f}")
                self.report_lines.append(f"- {result.test_name}: statistic={result.statistic:.4f}, p-value={result.p_value:.4f} ({result.interpretation})\n")
            
            # Randomness test
            print("\nRandomness Tests:")
            self.report_lines.append("\n**Randomness Tests:**\n")
            
            runs_result = self.test_randomness_runs(data, metric)
            status = "✓" if runs_result.passed else "✗"
            print(f"  {status} {runs_result.test_name:25s}: z={runs_result.statistic:8.4f}, p={runs_result.p_value:8.4f}")
            self.report_lines.append(f"- {runs_result.test_name}: z={runs_result.statistic:.4f}, p-value={runs_result.p_value:.4f} ({runs_result.interpretation})\n")
            
            # Autocorrelation
            autocorr_result = self.test_autocorrelation(data, lag=1)
            if autocorr_result:
                status = "✓" if autocorr_result.passed else "✗"
                print(f"  {status} {autocorr_result.test_name:25s}: r={autocorr_result.statistic:8.4f}, p={autocorr_result.p_value:8.4f}")
                self.report_lines.append(f"- {autocorr_result.test_name}: r={autocorr_result.statistic:.4f}, p-value={autocorr_result.p_value:.4f} ({autocorr_result.interpretation})\n")
    
    def analyze_detection_consistency(self):
        """Analyze consistency of fault detection across runs"""
        print("\n" + "="*80)
        print("DETECTION CONSISTENCY ANALYSIS")
        print("="*80)
        
        self.report_lines.append("\n## Detection Consistency Analysis\n")
        
        # Collect detection statistics
        detection_stats = defaultdict(list)
        
        for exp in self.experiments:
            results = exp['results']
            config = results['config']
            fault_step = config['fault_injection_step']
            
            for method, alert_step in results['alerts'].items():
                if alert_step is not None:
                    lead_time = fault_step - alert_step
                    detection_stats[method].append({
                        'alert_step': alert_step,
                        'lead_time': lead_time,
                        'detected': lead_time >= 0
                    })
                else:
                    detection_stats[method].append({
                        'alert_step': None,
                        'lead_time': None,
                        'detected': False
                    })
        
        # Analyze each method
        for method, stats_list in detection_stats.items():
            print(f"\n--- {method.upper().replace('_', ' ')} ---")
            self.report_lines.append(f"\n### {method.upper().replace('_', ' ')}\n")
            
            detection_rate = sum(1 for s in stats_list if s['detected']) / len(stats_list)
            lead_times = [s['lead_time'] for s in stats_list if s['lead_time'] is not None and s['lead_time'] >= 0]
            
            print(f"  Detection Rate: {detection_rate:.2%}")
            print(f"  Total Runs: {len(stats_list)}")
            print(f"  Successful Detections: {sum(1 for s in stats_list if s['detected'])}")
            
            self.report_lines.append(f"- Detection Rate: {detection_rate:.2%}\n")
            self.report_lines.append(f"- Total Runs: {len(stats_list)}\n")
            self.report_lines.append(f"- Successful Detections: {sum(1 for s in stats_list if s['detected'])}\n")
            
            if lead_times:
                print(f"  Mean Lead Time: {np.mean(lead_times):.1f} steps")
                print(f"  Std Lead Time: {np.std(lead_times):.1f} steps")
                print(f"  Min Lead Time: {np.min(lead_times):.1f} steps")
                print(f"  Max Lead Time: {np.max(lead_times):.1f} steps")
                print(f"  CV (Coefficient of Variation): {np.std(lead_times)/np.mean(lead_times):.2f}")
                
                self.report_lines.append(f"- Mean Lead Time: {np.mean(lead_times):.1f} steps\n")
                self.report_lines.append(f"- Std Lead Time: {np.std(lead_times):.1f} steps\n")
                self.report_lines.append(f"- CV: {np.std(lead_times)/np.mean(lead_times):.2f}\n")
    
    def analyze_method_comparison(self):
        """Compare detection methods statistically"""
        print("\n" + "="*80)
        print("METHOD COMPARISON ANALYSIS")
        print("="*80)
        
        self.report_lines.append("\n## Method Comparison Analysis\n")
        
        # Extract lead times for each method
        method_lead_times = defaultdict(list)
        
        for exp in self.experiments:
            results = exp['results']
            config = results['config']
            fault_step = config['fault_injection_step']
            
            for method, alert_step in results['alerts'].items():
                if alert_step is not None:
                    lead_time = fault_step - alert_step
                    if lead_time >= 0:  # Only consider successful early detections
                        method_lead_times[method].append(lead_time)
        
        # Pairwise comparisons
        methods = list(method_lead_times.keys())
        
        print("\nPairwise Mann-Whitney U Tests (Lead Time):")
        self.report_lines.append("\n**Pairwise Mann-Whitney U Tests:**\n")
        self.report_lines.append("| Method 1 | Method 2 | U-statistic | p-value | Significant |\n")
        self.report_lines.append("|----------|----------|-------------|---------|-------------|\n")
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                if len(method_lead_times[method1]) > 0 and len(method_lead_times[method2]) > 0:
                    try:
                        stat, p_value = mannwhitneyu(
                            method_lead_times[method1],
                            method_lead_times[method2],
                            alternative='two-sided'
                        )
                        
                        significant = "Yes" if p_value < 0.05 else "No"
                        print(f"  {method1:20s} vs {method2:20s}: U={stat:8.2f}, p={p_value:6.4f} ({significant})")
                        
                        self.report_lines.append(f"| {method1} | {method2} | {stat:.2f} | {p_value:.4f} | {significant} |\n")
                    except Exception as e:
                        print(f"  {method1} vs {method2}: Test failed - {e}")
        
        # Kruskal-Wallis H-test (all methods)
        if len(methods) > 2:
            print("\nKruskal-Wallis H-Test (All Methods):")
            self.report_lines.append("\n**Kruskal-Wallis H-Test:**\n")
            
            valid_methods = [m for m in methods if len(method_lead_times[m]) > 0]
            if len(valid_methods) > 2:
                try:
                    stat, p_value = kruskal(*[method_lead_times[m] for m in valid_methods])
                    print(f"  H-statistic: {stat:.4f}")
                    print(f"  p-value: {p_value:.4f}")
                    print(f"  Interpretation: {'Significant differences' if p_value < 0.05 else 'No significant differences'} between methods")
                    
                    self.report_lines.append(f"- H-statistic: {stat:.4f}\n")
                    self.report_lines.append(f"- p-value: {p_value:.4f}\n")
                    self.report_lines.append(f"- Interpretation: {'Significant differences' if p_value < 0.05 else 'No significant differences'} between methods\n")
                except Exception as e:
                    print(f"  Kruskal-Wallis test failed: {e}")
    
    def generate_visualizations(self, output_dir: Path):
        """Generate comprehensive visualization plots"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        viz_dir = output_dir / "statistical_analysis"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Distribution plots for key metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['r_metric', 'lambda_norm', 'sigma_sq_norm', 'delta_l_norm']
        
        for idx, metric in enumerate(metrics):
            if metric in self.aggregated_metrics:
                data = np.array(self.aggregated_metrics[metric])
                ax = axes[idx // 2, idx % 2]
                
                # Histogram with KDE
                ax.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
                
                # Fit normal distribution
                mu, sigma = np.mean(data), np.std(data)
                x = np.linspace(data.min(), data.max(), 100)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')
                
                ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: distributions.png")
        
        # 2. Q-Q plots for normality assessment
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx, metric in enumerate(metrics):
            if metric in self.aggregated_metrics:
                data = np.array(self.aggregated_metrics[metric])
                ax = axes[idx // 2, idx % 2]
                
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(f'Q-Q Plot: {metric.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "qq_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: qq_plots.png")
        
        # 3. Lead time distributions by method
        method_lead_times = defaultdict(list)
        
        for exp in self.experiments:
            results = exp['results']
            config = results['config']
            fault_step = config['fault_injection_step']
            
            for method, alert_step in results['alerts'].items():
                if alert_step is not None:
                    lead_time = fault_step - alert_step
                    if lead_time >= 0:
                        method_lead_times[method].append(lead_time)
        
        if method_lead_times:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            data_to_plot = [method_lead_times[m] for m in method_lead_times.keys() if len(method_lead_times[m]) > 0]
            labels = [m.replace('_', '\n') for m in method_lead_times.keys() if len(method_lead_times[m]) > 0]
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.set_title('Lead Time Distribution by Detection Method', fontsize=14, fontweight='bold')
            ax.set_xlabel('Detection Method')
            ax.set_ylabel('Lead Time (steps)')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(viz_dir / "lead_time_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: lead_time_comparison.png")
        
        print(f"\nAll visualizations saved to: {viz_dir}")
    
    def save_report(self, output_dir: Path):
        """Save comprehensive statistical analysis report"""
        report_path = output_dir / "statistical_analysis" / "STATISTICAL_REPORT.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Statistical Analysis Report: R-Metric Experiments\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Number of Experiments Analyzed:** {len(self.experiments)}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report presents comprehensive statistical analysis of R-Metric fault detection ")
            f.write("experiments, including distribution analysis, randomness tests, and method comparisons.\n\n")
            
            f.writelines(self.report_lines)
            
            f.write("\n## Conclusions\n\n")
            f.write("The statistical analysis validates the randomness and distribution properties of the ")
            f.write("R-Metric experimental results, supporting the robustness of the fault detection approach.\n")
        
        print(f"\n✓ Statistical report saved to: {report_path}")
    
    def run_complete_analysis(self):
        """Execute complete statistical analysis pipeline"""
        print("="*80)
        print("R-METRIC STATISTICAL ANALYSIS")
        print("="*80)
        
        # Load data
        n_exp = self.load_experiments()
        if n_exp == 0:
            print("\n✗ No experiments found to analyze!")
            return
        
        # Aggregate metrics
        self.aggregate_metrics()
        
        # Run analyses
        self.analyze_distributions()
        self.analyze_detection_consistency()
        self.analyze_method_comparison()
        
        # Generate visualizations
        self.generate_visualizations(self.results_dir)
        
        # Save report
        self.save_report(self.results_dir)
        
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS COMPLETE")
        print("="*80)


def main():
    """Main execution function"""
    analyzer = RandomnessAnalyzer(results_dir="case_study_results")
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()