"""
Helper functions for COVID-19 unemployment analysis in Canada.

This module contains all data loading, processing, statistical analysis,
and visualization functions used in the analysis notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, sem, t as t_dist, pearsonr
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from statsmodels.regression.linear_model import RegressionResultsWrapper

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_CONFIDENCE = 0.95
DEFAULT_ALPHA = 0.05
DEFAULT_DPI = 150
OUTPUT_DIR = "output"

# Effect size thresholds (Cohen's d)
EFFECT_SMALL = 0.2
EFFECT_MEDIUM = 0.5
EFFECT_LARGE = 0.8


class EffectSize(Enum):
    """Cohen's d effect size categories."""
    NEGLIGIBLE = "negligible"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    
    @classmethod
    def from_cohens_d(cls, d: float) -> "EffectSize":
        """Categorize Cohen's d value."""
        abs_d = abs(d)
        if abs_d < EFFECT_SMALL:
            return cls.NEGLIGIBLE
        if abs_d < EFFECT_MEDIUM:
            return cls.SMALL
        if abs_d < EFFECT_LARGE:
            return cls.MEDIUM
        return cls.LARGE


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    province: str
    metric: str
    pre_mean: float
    post_mean: float
    pre_ci: tuple[float, float]
    post_ci: tuple[float, float]
    cohens_d: float
    effect_size: EffectSize
    p_value: float
    age_group: str | None = None
    
    @property
    def is_significant(self) -> bool:
        """Check if result is significant at alpha=0.05."""
        return self.p_value < DEFAULT_ALPHA


@dataclass  
class ITSResult:
    """Container for Interrupted Time Series analysis results."""
    model: "RegressionResultsWrapper"
    data: pd.DataFrame
    baseline_level: float
    pre_trend: float
    immediate_effect: float
    trend_change: float
    immediate_effect_pvalue: float
    trend_change_pvalue: float
    r_squared: float
    r_squared_adj: float
    
    @property
    def has_significant_immediate_effect(self) -> bool:
        return self.immediate_effect_pvalue < DEFAULT_ALPHA
    
    @property
    def has_significant_trend_change(self) -> bool:
        return self.trend_change_pvalue < DEFAULT_ALPHA


# =============================================================================
# DATA LOADING AND EXPLORATION
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the unemployment dataset and convert date column to datetime."""
    df = pd.read_csv(filepath)
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print basic information about the dataset structure."""
    print("=== Dataset Info ===")
    print(f"Shape: {df.shape}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\n=== Missing Values ===\n{df.isnull().sum()}")
    
    categorical_columns = ['GEO', 'Sex', 'Age group']
    for col in categorical_columns:
        print(f'\n=== Unique values in "{col}" ===\n{df[col].unique()}')


# =============================================================================
# DATA CLEANING AND IMPUTATION
# =============================================================================

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the dataset using vectorized operations.
    
    Imputation strategy:
    - Full-time employment: Sum from all provinces for Canada
    - Part-time employment: Employment - Full-time employment
    - Unemployment: Labour force - Employment
    - Unemployment rate: (Unemployment / Labour force) * 100
    """
    df = df.copy()
    
    # Impute Full-time employment for Canada by summing provinces
    # Use merge-based approach instead of iterrows
    missing_ft_mask = df['Full-time employment'].isna()
    if missing_ft_mask.any():
        provincial_sums = (
            df[df['GEO'] != 'Canada']
            .groupby(['REF_DATE', 'Age group'])['Full-time employment']
            .sum()
            .reset_index(name='ft_sum')
        )
        df = df.merge(provincial_sums, on=['REF_DATE', 'Age group'], how='left')
        df['Full-time employment'] = df['Full-time employment'].fillna(df['ft_sum'])
        df = df.drop(columns='ft_sum')
    
    # Impute Part-time employment (note: column has trailing space in raw data)
    part_time_col = 'Part-time employment '
    missing_pt_mask = df[part_time_col].isna()
    df.loc[missing_pt_mask, part_time_col] = (
        df.loc[missing_pt_mask, 'Employment'] - 
        df.loc[missing_pt_mask, 'Full-time employment']
    )
    
    # Impute Unemployment and Unemployment rate using vectorized fillna
    df['Unemployment'] = df['Unemployment'].fillna(
        df['Labour force'] - df['Employment']
    )
    df['Unemployment rate'] = df['Unemployment rate'].fillna(
        (df['Unemployment'] / df['Labour force'] * 100).round(1)
    )
    
    return df


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size using pooled standard deviation.
    
    Args:
        group1: Pre-period data array.
        group2: Post-period data array.
    
    Returns:
        Cohen's d value (positive means group2 > group1).
    
    Note:
        Uses pooled SD formula: sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((np.mean(group2) - np.mean(group1)) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size (returns string for backward compatibility)."""
    return EffectSize.from_cohens_d(d).value


def calculate_confidence_interval(
    data: np.ndarray, 
    confidence: float = DEFAULT_CONFIDENCE
) -> tuple[float, float]:
    """
    Calculate confidence interval for the mean using t-distribution.
    
    Args:
        data: Array of sample values.
        confidence: Confidence level (default 0.95 for 95% CI).
    
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    n = len(data)
    mean = np.mean(data)
    se = sem(data)
    margin = se * t_dist.ppf((1 + confidence) / 2, n - 1)
    return (round(mean - margin, 2), round(mean + margin, 2))


def run_wilcoxon_analysis(df: pd.DataFrame, 
                          metrics: list, 
                          pre_period: tuple, 
                          post_period: tuple,
                          age_groups: list = None) -> pd.DataFrame:
    """
    Run Wilcoxon signed-rank test comparing metrics between two time periods.
    
    Parameters:
    -----------
    df : DataFrame with the data
    metrics : List of column names to analyze
    pre_period : Tuple of (start_date, end_date) for pre-period
    post_period : Tuple of (start_date, end_date) for post-period
    age_groups : List of age groups to analyze (if None, no age group filtering)
    
    Returns:
    --------
    DataFrame with test results
    """
    pre_mask = (df['REF_DATE'] >= pre_period[0]) & (df['REF_DATE'] <= pre_period[1])
    post_mask = (df['REF_DATE'] >= post_period[0]) & (df['REF_DATE'] <= post_period[1])
    
    provinces = df['GEO'].unique()
    results = []
    
    # If no age groups specified, use a placeholder for single iteration
    if age_groups is None:
        age_groups_iter = [None]
    else:
        age_groups_iter = age_groups
    
    for province in provinces:
        for age_group in age_groups_iter:
            for metric in metrics:
                # Build filter conditions
                pre_filter = pre_mask & (df['GEO'] == province)
                post_filter = post_mask & (df['GEO'] == province)
                
                if age_group is not None:
                    pre_filter = pre_filter & (df['Age group'] == age_group)
                    post_filter = post_filter & (df['Age group'] == age_group)
                
                pre_data = df[pre_filter][metric].dropna()
                post_data = df[post_filter][metric].dropna()
                
                # Skip if insufficient data
                if len(pre_data) < 2 or len(post_data) < 2:
                    continue
                
                # Ensure equal lengths for Wilcoxon test
                min_length = min(len(pre_data), len(post_data))
                pre_data_aligned = pre_data.iloc[:min_length]
                post_data_aligned = post_data.iloc[:min_length]
                
                # Run the test (ignore statistic, we only need p-value)
                _, p_value = wilcoxon(pre_data_aligned, post_data_aligned)
                
                # Calculate means
                pre_mean = round(np.mean(pre_data_aligned), 1)
                post_mean = round(np.mean(post_data_aligned), 1)
                
                # Calculate effect size (Cohen's d)
                cohens_d = calculate_cohens_d(
                    pre_data_aligned.values, 
                    post_data_aligned.values
                )
                
                # Calculate confidence intervals
                pre_ci = calculate_confidence_interval(pre_data_aligned.values)
                post_ci = calculate_confidence_interval(post_data_aligned.values)
                
                result = {
                    'Province': province,
                    'Metric': metric,
                    'Pre-Period Mean': pre_mean,
                    'Pre-Period 95% CI': pre_ci,
                    'Post-Period Mean': post_mean,
                    'Post-Period 95% CI': post_ci,
                    'Cohen\'s d': round(cohens_d, 3),
                    'Effect Size': interpret_cohens_d(cohens_d),
                    'p-value': round(p_value, 6)  # Keep more precision for correction
                }
                
                if age_group is not None:
                    result['Age Group'] = age_group
                
                results.append(result)
    
    return pd.DataFrame(results)


def apply_multiple_comparison_correction(results_df: pd.DataFrame, 
                                         alpha: float = 0.05,
                                         method: str = 'fdr_bh') -> pd.DataFrame:
    """
    Apply multiple comparison correction to p-values.
    
    Parameters:
    -----------
    results_df : DataFrame with 'p-value' column
    alpha : Significance level (default 0.05)
    method : Correction method. Options:
        - 'bonferroni': Conservative, controls family-wise error rate
        - 'fdr_bh': Benjamini-Hochberg, controls false discovery rate (recommended)
        - 'fdr_by': Benjamini-Yekutieli, more conservative FDR control
        - 'holm': Holm-Bonferroni, less conservative than Bonferroni
    
    Returns:
    --------
    DataFrame with corrected p-values and significance determination
    """
    results_df = results_df.copy()
    
    # Apply correction
    reject, p_corrected, _, _ = multipletests(
        results_df['p-value'], 
        alpha=alpha, 
        method=method
    )
    
    results_df['p-value (corrected)'] = np.round(p_corrected, 6)
    results_df['Significant'] = reject
    
    # Determine change direction based on corrected significance
    changes = []
    for _, row in results_df.iterrows():
        if row['Significant']:
            # Determine direction based on which mean column exists
            if 'Post-Period Mean' in row:
                post_mean = row['Post-Period Mean']
                pre_mean = row['Pre-Period Mean']
            else:
                post_mean = row.get('COVID Mean', row.get('Post-COVID Mean', 0))
                pre_mean = row.get('Pre-COVID Mean', 0)
            
            change = "increased" if post_mean > pre_mean else "decreased"
        else:
            change = "no significant change"
        changes.append(change)
    
    results_df['Change'] = changes
    
    # Print summary
    n_total = len(results_df)
    n_significant = reject.sum()
    print(f"Multiple comparison correction applied: {method.upper()}")
    print(f"  - Total tests: {n_total}")
    print(f"  - Significant after correction: {n_significant} ({100*n_significant/n_total:.1f}%)")
    print(f"  - Alpha level: {alpha}")
    
    return results_df


def add_percentage_columns(results_df: pd.DataFrame, 
                           df: pd.DataFrame,
                           pre_period: tuple, 
                           post_period: tuple,
                           pre_col: str = 'Pre-Period Mean',
                           post_col: str = 'Post-Period Mean') -> pd.DataFrame:
    """
    Add percentage columns for employment metrics relative to labour force.
    
    Parameters:
    -----------
    results_df : DataFrame with analysis results
    df : Original data DataFrame
    pre_period : Tuple of (start_date, end_date) for pre-period
    post_period : Tuple of (start_date, end_date) for post-period
    pre_col : Name of the pre-period mean column
    post_col : Name of the post-period mean column
    """
    results_df = results_df.copy()
    
    pre_mask = (df['REF_DATE'] >= pre_period[0]) & (df['REF_DATE'] <= pre_period[1])
    post_mask = (df['REF_DATE'] >= post_period[0]) & (df['REF_DATE'] <= post_period[1])
    
    pre_pct_list = []
    post_pct_list = []
    
    for _, row in results_df.iterrows():
        province = row['Province']
        metric = row['Metric']
        age_group = row.get('Age Group')
        
        if metric in ['Full-time employment', 'Part-time employment ']:
            # Get labour force data for percentage calculation
            pre_filter = pre_mask & (df['GEO'] == province)
            post_filter = post_mask & (df['GEO'] == province)
            
            if age_group is not None:
                pre_filter = pre_filter & (df['Age group'] == age_group)
                post_filter = post_filter & (df['Age group'] == age_group)
            
            pre_labour = df[pre_filter]['Labour force'].mean()
            post_labour = df[post_filter]['Labour force'].mean()
            
            pre_pct = round((row[pre_col] / pre_labour) * 100, 1)
            post_pct = round((row[post_col] / post_labour) * 100, 1)
        elif metric == 'Unemployment rate':
            pre_pct = row[pre_col]
            post_pct = row[post_col]
        else:
            pre_pct = None
            post_pct = None
        
        pre_pct_list.append(pre_pct)
        post_pct_list.append(post_pct)
    
    results_df['Pre-Period %'] = pre_pct_list
    results_df['Post-Period %'] = post_pct_list
    
    return results_df


# =============================================================================
# DATA EXPORT
# =============================================================================

def export_data(df: pd.DataFrame, filename: str, pivot_cols: dict = None) -> None:
    """
    Export DataFrame to CSV, optionally pivoting first.
    
    Parameters:
    -----------
    df : DataFrame to export
    filename : Output filename
    pivot_cols : If provided, dict with 'index', 'columns', 'values' for pivoting
    """
    if pivot_cols:
        df_export = df[list(pivot_cols.values())].pivot(
            index=pivot_cols['index'],
            columns=pivot_cols['columns'],
            values=pivot_cols['values']
        )
        df_export.reset_index(inplace=True)
    else:
        df_export = df
    
    df_export.to_csv(filename, index=False)
    print(f"Exported: {filename}")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def ensure_output_dir(dirname: str | Path = OUTPUT_DIR) -> Path:
    """Create output directory if it doesn't exist and return the path."""
    path = Path(dirname)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig, 
    filename: str, 
    output_dir: str | Path = OUTPUT_DIR, 
    dpi: int = DEFAULT_DPI
) -> Path:
    """Save figure to output directory and return the filepath."""
    dirpath = ensure_output_dir(output_dir)
    filepath = dirpath / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filepath}")
    return filepath


def plot_unemployment_trends(df: pd.DataFrame, 
                             start_date: str = None, 
                             end_date: str = None,
                             title_suffix: str = "") -> pd.DataFrame:
    """
    Plot unemployment rate trends by province for the working population (15+).
    
    Parameters:
    -----------
    df : DataFrame with unemployment data
    start_date : Optional start date filter (YYYY-MM-DD)
    end_date : Optional end date filter (YYYY-MM-DD)
    title_suffix : Additional text to add to the plot title
    
    Returns:
    --------
    Filtered DataFrame used for plotting
    """
    # Filter for working population
    df_filtered = df[df['Age group'] == '15 years and over'].copy()
    
    # Apply date filters if provided
    if start_date:
        df_filtered = df_filtered[df_filtered['REF_DATE'] >= start_date]
    if end_date:
        df_filtered = df_filtered[df_filtered['REF_DATE'] <= end_date]
    
    plt.figure(figsize=(15, 10))
    
    # Plot provinces (excluding Canada)
    sns.lineplot(
        data=df_filtered[df_filtered['GEO'] != 'Canada'], 
        x='REF_DATE', y='Unemployment rate', 
        hue='GEO', linewidth=1.5
    )
    
    # Plot Canada with emphasis
    sns.lineplot(
        data=df_filtered[df_filtered['GEO'] == 'Canada'], 
        x='REF_DATE', y='Unemployment rate', 
        color='black', label='Canada', linewidth=2.5
    )
    
    title = f'Unemployment Rate Trends by Province (15 years and over){title_suffix}'
    plt.title(title, fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Unemployment Rate (%)', fontsize=14)
    plt.legend(title='Province', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return df_filtered


def plot_unemployment_trends_with_save(df: pd.DataFrame, 
                                        start_date: str = None, 
                                        end_date: str = None,
                                        title_suffix: str = "",
                                        filename: str = None) -> pd.DataFrame:
    """
    Plot unemployment rate trends by province and optionally save to file.
    
    Parameters:
    -----------
    df : DataFrame with unemployment data
    start_date : Optional start date filter (YYYY-MM-DD)
    end_date : Optional end date filter (YYYY-MM-DD)
    title_suffix : Additional text to add to the plot title
    filename : Optional filename to save the figure
    
    Returns:
    --------
    Filtered DataFrame used for plotting
    """
    df_filtered = df[df['Age group'] == '15 years and over'].copy()
    
    if start_date:
        df_filtered = df_filtered[df_filtered['REF_DATE'] >= start_date]
    if end_date:
        df_filtered = df_filtered[df_filtered['REF_DATE'] <= end_date]
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    sns.lineplot(
        data=df_filtered[df_filtered['GEO'] != 'Canada'], 
        x='REF_DATE', y='Unemployment rate', 
        hue='GEO', linewidth=1.5, ax=ax
    )
    
    sns.lineplot(
        data=df_filtered[df_filtered['GEO'] == 'Canada'], 
        x='REF_DATE', y='Unemployment rate', 
        color='black', label='Canada', linewidth=2.5, ax=ax
    )
    
    title = f'Unemployment Rate Trends by Province (15 years and over){title_suffix}'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Unemployment Rate (%)', fontsize=14)
    ax.legend(title='Province', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()
    return df_filtered


def plot_statistical_heatmap(results_df: pd.DataFrame, 
                             value_col: str,
                             title: str,
                             filename: str = None,
                             cmap: str = 'RdYlGn_r',
                             fmt: str = '.1f') -> None:
    """
    Create a heatmap visualization of statistical results.
    
    Parameters:
    -----------
    results_df : DataFrame with results
    value_col : Column to use for heatmap values
    title : Plot title
    filename : Optional filename to save
    cmap : Colormap (default: red-yellow-green reversed)
    fmt : Format string for annotations
    """
    # Check if Age Group column exists
    if 'Age Group' in results_df.columns:
        pivot = results_df.pivot_table(
            index=['Province'], 
            columns=['Metric', 'Age Group'], 
            values=value_col
        )
    else:
        pivot = results_df.pivot_table(
            index=['Province'], 
            columns=['Metric'], 
            values=value_col
        )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, center=0, ax=ax,
                linewidths=0.5, cbar_kws={'label': value_col})
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()


def plot_change_summary(results_df: pd.DataFrame, 
                        title: str,
                        filename: str = None) -> None:
    """
    Create a bar chart summarizing significant changes by metric and direction.
    
    Parameters:
    -----------
    results_df : DataFrame with 'Significant' and 'Change' columns
    title : Plot title
    filename : Optional filename to save
    """
    # Count changes by metric and direction
    sig_results = results_df[results_df['Significant'] == True].copy()
    
    if len(sig_results) == 0:
        print("No significant results to plot.")
        return
    
    change_counts = sig_results.groupby(['Metric', 'Change']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    change_counts.plot(kind='bar', ax=ax, color=['#d62728', '#2ca02c'], edgecolor='black')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Number of Province-Age Group Combinations', fontsize=12)
    ax.legend(title='Direction')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fontsize=10)
    
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()


def plot_unemployment_rate_comparison(results_df: pd.DataFrame,
                                      pre_col: str,
                                      post_col: str,
                                      title: str,
                                      filename: str = None) -> None:
    """
    Create a grouped bar chart comparing unemployment rates before and after.
    
    Parameters:
    -----------
    results_df : DataFrame with unemployment rate results
    pre_col : Column name for pre-period values
    post_col : Column name for post-period values
    title : Plot title
    filename : Optional filename to save
    """
    # Filter for unemployment rate only
    unemp_df = results_df[results_df['Metric'] == 'Unemployment rate'].copy()
    
    if 'Age Group' in unemp_df.columns:
        unemp_df['Label'] = unemp_df['Province'] + '\n' + unemp_df['Age Group'].str.replace(' years', 'y')
    else:
        unemp_df['Label'] = unemp_df['Province']
    
    # Sort by province
    unemp_df = unemp_df.sort_values('Province')
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(unemp_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, unemp_df[pre_col], width, label='Pre-COVID', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, unemp_df[post_col], width, label='COVID/Post-COVID', color='#e74c3c', edgecolor='black')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Province / Age Group', fontsize=12)
    ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(unemp_df['Label'], rotation=90, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()


# ==============================================================================
# EFFECT SIZE AND CONFIDENCE INTERVAL VISUALIZATIONS
# ==============================================================================

def plot_effect_sizes(results_df: pd.DataFrame,
                      title: str,
                      filename: str = None,
                      metric_filter: str = None) -> None:
    """
    Create a horizontal bar chart showing effect sizes (Cohen's d) by province.
    
    Parameters:
    -----------
    results_df : DataFrame with 'Cohen's d' column
    title : Plot title
    filename : Optional filename to save
    metric_filter : If provided, filter to specific metric
    """
    df_plot = results_df.copy()
    
    if metric_filter:
        df_plot = df_plot[df_plot['Metric'] == metric_filter]
    
    if 'Age Group' in df_plot.columns:
        df_plot['Label'] = df_plot['Province'] + '\n' + df_plot['Age Group'].str.replace(' years', 'y')
    else:
        df_plot['Label'] = df_plot['Province']
    
    df_plot = df_plot.sort_values("Cohen's d", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_plot) * 0.3)))
    
    # Color bars based on effect size magnitude using dict mapping
    color_map = {
        EffectSize.NEGLIGIBLE: '#95a5a6',  # Gray
        EffectSize.SMALL: '#f39c12',       # Orange  
        EffectSize.MEDIUM: '#e74c3c',      # Red
        EffectSize.LARGE: '#c0392b',       # Dark red
    }
    colors = [color_map[EffectSize.from_cohens_d(d)] for d in df_plot["Cohen's d"]]
    
    bars = ax.barh(df_plot['Label'], df_plot["Cohen's d"], color=colors, edgecolor='black')
    
    # Add vertical lines for effect size thresholds
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Large (0.8)')
    ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.8, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linewidth=1)
    
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()


def plot_effect_size_heatmap(results_df: pd.DataFrame,
                             title: str,
                             filename: str = None) -> None:
    """
    Create a heatmap of effect sizes by province and metric/age group.
    """
    if 'Age Group' in results_df.columns:
        pivot = results_df.pivot_table(
            index='Province',
            columns=['Metric', 'Age Group'],
            values="Cohen's d"
        )
    else:
        pivot = results_df.pivot_table(
            index='Province',
            columns='Metric',
            values="Cohen's d"
        )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Custom colormap: blue (negative) -> white (zero) -> red (positive)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
                linewidths=0.5, cbar_kws={'label': "Cohen's d"},
                vmin=-2, vmax=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()


def plot_confidence_intervals(results_df: pd.DataFrame,
                              pre_col: str,
                              post_col: str,
                              pre_ci_col: str,
                              post_ci_col: str,
                              title: str,
                              filename: str = None,
                              metric_filter: str = None) -> None:
    """
    Create a forest plot showing means with confidence intervals.
    
    Parameters:
    -----------
    results_df : DataFrame with mean and CI columns
    pre_col : Column name for pre-period mean
    post_col : Column name for post-period mean
    pre_ci_col : Column name for pre-period CI (tuple)
    post_ci_col : Column name for post-period CI (tuple)
    title : Plot title
    filename : Optional filename to save
    metric_filter : If provided, filter to specific metric
    """
    df_plot = results_df.copy()
    
    if metric_filter:
        df_plot = df_plot[df_plot['Metric'] == metric_filter]
    
    if len(df_plot) == 0:
        print(f"No data for metric: {metric_filter}")
        return
    
    # Create labels
    if 'Age Group' in df_plot.columns:
        df_plot['Label'] = df_plot['Province'] + ' - ' + df_plot['Age Group'].str.replace(' years', 'y')
    else:
        df_plot['Label'] = df_plot['Province']
    
    df_plot = df_plot.sort_values('Label')
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_plot) * 0.4)))
    
    y_positions = np.arange(len(df_plot))
    
    # Extract CI bounds
    pre_lower = [ci[0] for ci in df_plot[pre_ci_col]]
    pre_upper = [ci[1] for ci in df_plot[pre_ci_col]]
    post_lower = [ci[0] for ci in df_plot[post_ci_col]]
    post_upper = [ci[1] for ci in df_plot[post_ci_col]]
    
    # Plot pre-period
    ax.errorbar(df_plot[pre_col], y_positions - 0.15, 
                xerr=[df_plot[pre_col] - pre_lower, pre_upper - df_plot[pre_col]],
                fmt='o', color='#3498db', label='Pre-COVID', capsize=3, markersize=6)
    
    # Plot post-period
    ax.errorbar(df_plot[post_col], y_positions + 0.15,
                xerr=[df_plot[post_col] - post_lower, post_upper - df_plot[post_col]],
                fmt='s', color='#e74c3c', label='COVID/Post-COVID', capsize=3, markersize=6)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_plot['Label'])
    ax.set_xlabel('Value (with 95% CI)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()


def print_effect_size_summary(results_df: pd.DataFrame) -> None:
    """Print a summary of effect sizes."""
    print("\n" + "="*60)
    print("EFFECT SIZE SUMMARY")
    print("="*60)
    
    for effect in ['large', 'medium', 'small', 'negligible']:
        count = len(results_df[results_df['Effect Size'] == effect])
        pct = 100 * count / len(results_df)
        print(f"  {effect.capitalize():12} effects: {count:3} ({pct:5.1f}%)")
    
    print("-"*60)
    cohens_d_col = results_df["Cohen's d"]
    print(f"  Mean |Cohen's d|: {cohens_d_col.abs().mean():.3f}")
    print(f"  Max  |Cohen's d|: {cohens_d_col.abs().max():.3f}")
    print("="*60)


# ==============================================================================
# CORRELATION ANALYSIS
# ==============================================================================

def calculate_metric_correlations(df: pd.DataFrame,
                                   metrics: list,
                                   province: str = 'Canada',
                                   age_group: str = None,
                                   time_period: tuple = None) -> pd.DataFrame:
    """
    Calculate pairwise correlations between employment metrics.
    
    Parameters:
    -----------
    df : DataFrame with employment data (wide format with metrics as columns)
    metrics : List of metric column names to correlate
    province : Province to filter (default 'Canada')
    age_group : Age group to filter (optional)
    time_period : Tuple of (start_date, end_date) to filter (optional)
    
    Returns:
    --------
    DataFrame with correlation matrix
    """
    # Filter data
    mask = df['GEO'] == province
    if age_group:
        mask &= df['Age group'] == age_group
    if time_period:
        mask &= (df['REF_DATE'] >= time_period[0]) & (df['REF_DATE'] <= time_period[1])
    
    df_filtered = df[mask].copy().sort_values('REF_DATE')
    
    # Filter to requested metrics that exist in the dataframe
    available_metrics = [m for m in metrics if m in df_filtered.columns]
    if len(available_metrics) < 2:
        print(f"Warning: Only {len(available_metrics)} metrics available for correlation")
        print(f"Available columns: {df_filtered.columns.tolist()}")
        return pd.DataFrame()
    
    # Calculate correlation matrix directly from metric columns
    corr_matrix = df_filtered[available_metrics].corr()
    
    return corr_matrix


def calculate_provincial_correlations(df: pd.DataFrame,
                                       metric: str,
                                       provinces: list = None,
                                       age_group: str = None,
                                       time_period: tuple = None) -> pd.DataFrame:
    """
    Calculate correlations of a metric across provinces.
    
    Parameters:
    -----------
    df : DataFrame with employment data (wide format)
    metric : The metric column to analyze
    provinces : List of provinces (optional, defaults to all)
    age_group : Age group to filter (optional)
    time_period : Tuple of (start_date, end_date) to filter (optional)
    
    Returns:
    --------
    DataFrame with correlation matrix between provinces
    """
    df_filtered = df.copy()
    
    if age_group:
        df_filtered = df_filtered[df_filtered['Age group'] == age_group]
    
    if time_period:
        df_filtered = df_filtered[
            (df_filtered['REF_DATE'] >= time_period[0]) & 
            (df_filtered['REF_DATE'] <= time_period[1])
        ]
    
    # Get unique provinces
    if provinces is None:
        provinces = df_filtered['GEO'].unique().tolist()
        if 'Canada' in provinces:
            provinces.remove('Canada')  # Often want to exclude aggregate
    
    # Pivot to get provinces as columns with the metric values
    df_pivot = df_filtered.pivot_table(
        index='REF_DATE',
        columns='GEO',
        values=metric,
        aggfunc='first'
    ).reset_index()
    
    # Filter to requested provinces
    available_provinces = [p for p in provinces if p in df_pivot.columns]
    
    # Calculate correlation matrix
    corr_matrix = df_pivot[available_provinces].corr()
    
    return corr_matrix


def calculate_time_lagged_correlation(
    df: pd.DataFrame,
    metric1: str,
    metric2: str,
    province: str = 'Canada',
    age_group: str | None = None,
    max_lag: int = 12
) -> pd.DataFrame:
    """
    Calculate time-lagged correlations between two metrics.
    
    Args:
        df: DataFrame with employment data (wide format).
        metric1: First metric column (will be lagged).
        metric2: Second metric column.
        province: Province to analyze.
        age_group: Age group to filter (optional).
        max_lag: Maximum lag in months.
    
    Returns:
        DataFrame with lag, correlation, p-value, and significance.
    """
    # Filter data
    df_filtered = df[df['GEO'] == province].copy()
    if age_group:
        df_filtered = df_filtered[df_filtered['Age group'] == age_group]
    
    df_filtered = df_filtered.sort_values('REF_DATE')
    
    if metric1 not in df_filtered.columns or metric2 not in df_filtered.columns:
        print(f"Metrics not found: {metric1}, {metric2}")
        print(f"Available columns: {df_filtered.columns.tolist()}")
        return pd.DataFrame()
    
    results = []
    series1 = df_filtered[metric1].dropna()
    series2 = df_filtered[metric2].dropna()
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # metric1 leads metric2
            s1 = series1.iloc[:lag].values
            s2 = series2.iloc[-lag:].values
        elif lag > 0:
            # metric1 lags metric2
            s1 = series1.iloc[lag:].values
            s2 = series2.iloc[:-lag].values
        else:
            s1 = series1.values
            s2 = series2.values
        
        # Ensure same length
        min_len = min(len(s1), len(s2))
        if min_len > 2:
            corr, p_value = pearsonr(s1[:min_len], s2[:min_len])
            results.append({
                'Lag (months)': lag,
                'Correlation': corr,
                'P-value': p_value,
                'Significant': p_value < 0.05
            })
    
    return pd.DataFrame(results)


def plot_correlation_matrix(corr_matrix: pd.DataFrame,
                            title: str,
                            filename: str = None,
                            figsize: tuple = (10, 8)) -> None:
    """
    Plot a correlation matrix heatmap.
    
    Parameters:
    -----------
    corr_matrix : Correlation matrix DataFrame
    title : Plot title
    filename : Optional filename to save
    figsize : Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                mask=mask, square=True,
                linewidths=0.5, cbar_kws={'label': 'Correlation'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()


def plot_lagged_correlation(lag_df: pd.DataFrame,
                            metric1: str,
                            metric2: str,
                            title: str = None,
                            filename: str = None) -> None:
    """
    Plot time-lagged correlation results.
    
    Parameters:
    -----------
    lag_df : DataFrame from calculate_time_lagged_correlation
    metric1 : Name of first metric (for label)
    metric2 : Name of second metric (for label)
    title : Optional custom title
    filename : Optional filename to save
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Color bars by significance
    colors = ['#e74c3c' if sig else '#95a5a6' for sig in lag_df['Significant']]
    
    bars = ax.bar(lag_df['Lag (months)'], lag_df['Correlation'], 
                  color=colors, edgecolor='black', alpha=0.8)
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Find max correlation
    max_idx = lag_df['Correlation'].abs().idxmax()
    max_lag = lag_df.loc[max_idx, 'Lag (months)']
    max_corr = lag_df.loc[max_idx, 'Correlation']
    ax.annotate(f'Max: r={max_corr:.2f} at lag={max_lag}',
                xy=(max_lag, max_corr),
                xytext=(max_lag + 2, max_corr + 0.1 if max_corr > 0 else max_corr - 0.1),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='black'))
    
    ax.set_xlabel('Lag (months)', fontsize=12)
    ax.set_ylabel('Correlation coefficient', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Time-Lagged Correlation: {metric1} vs {metric2}', 
                     fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Significant (p<0.05)'),
                       Patch(facecolor='#95a5a6', label='Not significant')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()


def analyze_covid_correlations(df: pd.DataFrame,
                                metrics: list,
                                province: str = 'Canada',
                                pre_period: tuple = None,
                                covid_period: tuple = None) -> dict:
    """
    Compare correlation structures before and during COVID.
    
    Parameters:
    -----------
    df : DataFrame with employment data
    metrics : List of metrics to correlate
    province : Province to analyze
    pre_period : Tuple of (start, end) dates for pre-COVID
    covid_period : Tuple of (start, end) dates for COVID period
    
    Returns:
    --------
    Dictionary with 'pre_covid' and 'covid' correlation matrices
    """
    pre_corr = calculate_metric_correlations(
        df, metrics, province=province, time_period=pre_period
    )
    
    covid_corr = calculate_metric_correlations(
        df, metrics, province=province, time_period=covid_period
    )
    
    return {
        'pre_covid': pre_corr,
        'covid': covid_corr,
        'difference': covid_corr - pre_corr if not pre_corr.empty else pd.DataFrame()
    }


def print_correlation_summary(corr_matrix: pd.DataFrame, title: str = "Correlation Summary") -> None:
    """Print a summary of correlation strengths."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    # Get upper triangle values (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_corrs = corr_matrix.where(mask).stack()
    
    if len(upper_corrs) == 0:
        print("  No correlations to summarize")
        return
    
    print(f"  Number of pairs: {len(upper_corrs)}")
    print(f"  Mean |correlation|: {upper_corrs.abs().mean():.3f}")
    print(f"  Max correlation: {upper_corrs.max():.3f}")
    print(f"  Min correlation: {upper_corrs.min():.3f}")
    
    # Categorize
    strong = ((upper_corrs.abs() >= 0.7)).sum()
    moderate = ((upper_corrs.abs() >= 0.4) & (upper_corrs.abs() < 0.7)).sum()
    weak = ((upper_corrs.abs() < 0.4)).sum()
    
    print("-"*60)
    print(f"  Strong (|r|≥0.7):   {strong:3} ({100*strong/len(upper_corrs):5.1f}%)")
    print(f"  Moderate (0.4-0.7): {moderate:3} ({100*moderate/len(upper_corrs):5.1f}%)")
    print(f"  Weak (|r|<0.4):     {weak:3} ({100*weak/len(upper_corrs):5.1f}%)")
    print("="*60)


# ==============================================================================
# INTERRUPTED TIME SERIES (ITS) ANALYSIS
# ==============================================================================

def prepare_its_data(df: pd.DataFrame,
                     metric: str,
                     province: str = 'Canada',
                     age_group: str = None,
                     intervention_date: str = '2020-03-01') -> pd.DataFrame:
    """
    Prepare data for Interrupted Time Series analysis.
    
    Parameters:
    -----------
    df : DataFrame with employment data
    metric : Column name of the outcome variable
    province : Province to analyze
    age_group : Age group to filter (optional)
    intervention_date : Date when COVID intervention began
    
    Returns:
    --------
    DataFrame with ITS variables: time, intervention, time_after_intervention
    """
    # Filter data
    mask = df['GEO'] == province
    if age_group:
        mask &= df['Age group'] == age_group
    
    df_its = df[mask][['REF_DATE', metric]].copy()
    df_its = df_its.sort_values('REF_DATE').reset_index(drop=True)
    df_its = df_its.dropna(subset=[metric])
    
    # Create ITS variables
    intervention_dt = pd.to_datetime(intervention_date)
    
    # Time variable (months since start)
    df_its['time'] = np.arange(len(df_its))
    
    # Intervention indicator (0 before, 1 after)
    df_its['intervention'] = (df_its['REF_DATE'] >= intervention_dt).astype(int)
    
    # Time after intervention (0 before, 1, 2, 3... after)
    df_its['time_after'] = np.where(
        df_its['REF_DATE'] >= intervention_dt,
        (df_its['REF_DATE'] - intervention_dt).dt.days // 30,  # Approximate months
        0
    )
    
    df_its = df_its.rename(columns={metric: 'outcome'})
    
    return df_its


def run_its_analysis(df_its: pd.DataFrame) -> ITSResult:
    """
    Run Interrupted Time Series regression analysis.
    
    Model: Y = β0 + β1*time + β2*intervention + β3*time_after + ε
    
    Where:
        β0: Baseline level
        β1: Pre-intervention trend (slope)
        β2: Immediate effect of intervention (level change)
        β3: Change in trend after intervention (slope change)
    
    Args:
        df_its: DataFrame from prepare_its_data with ITS variables.
    
    Returns:
        ITSResult dataclass with model, data, and extracted parameters.
    """
    import statsmodels.api as sm
    
    # Prepare regression variables
    X = sm.add_constant(df_its[['time', 'intervention', 'time_after']])
    y = df_its['outcome']
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    
    # Calculate predicted and counterfactual values
    df_its = df_its.copy()
    df_its['predicted'] = model.predict(X)
    
    X_counterfactual = X.assign(intervention=0, time_after=0)
    df_its['counterfactual'] = model.predict(X_counterfactual)
    
    return ITSResult(
        model=model,
        data=df_its,
        baseline_level=model.params['const'],
        pre_trend=model.params['time'],
        immediate_effect=model.params['intervention'],
        trend_change=model.params['time_after'],
        immediate_effect_pvalue=model.pvalues['intervention'],
        trend_change_pvalue=model.pvalues['time_after'],
        r_squared=model.rsquared,
        r_squared_adj=model.rsquared_adj,
    )


def plot_its_results(
    its_results: ITSResult | dict,
    title: str,
    ylabel: str = 'Outcome',
    intervention_label: str = 'COVID-19',
    filename: str | None = None
) -> None:
    """
    Plot ITS analysis results with observed, predicted, and counterfactual lines.
    
    Args:
        its_results: ITSResult dataclass or dict from run_its_analysis.
        title: Plot title.
        ylabel: Y-axis label.
        intervention_label: Label for the intervention line.
        filename: Optional filename to save.
    """
    # Support both dataclass and dict for backward compatibility
    df_its = its_results.data if isinstance(its_results, ITSResult) else its_results['data']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Find intervention point
    intervention_idx = df_its[df_its['intervention'] == 1].index[0]
    intervention_date = df_its.loc[intervention_idx, 'REF_DATE']
    
    # Plot observed data
    ax.scatter(df_its['REF_DATE'], df_its['outcome'], 
               alpha=0.5, s=20, color='#3498db', label='Observed')
    
    # Plot predicted (fitted) line
    ax.plot(df_its['REF_DATE'], df_its['predicted'], 
            color='#e74c3c', linewidth=2, label='Fitted (ITS model)')
    
    # Plot counterfactual
    post_intervention = df_its[df_its['intervention'] == 1]
    ax.plot(post_intervention['REF_DATE'], post_intervention['counterfactual'],
            color='#2ecc71', linewidth=2, linestyle='--', 
            label='Counterfactual (no intervention)')
    
    # Add vertical line at intervention
    ax.axvline(x=intervention_date, color='black', linestyle=':', 
               linewidth=2, alpha=0.7)
    ax.text(intervention_date, ax.get_ylim()[1], f' {intervention_label}',
            fontsize=10, ha='left', va='top')
    
    # Shade the difference (intervention effect)
    ax.fill_between(post_intervention['REF_DATE'],
                    post_intervention['predicted'],
                    post_intervention['counterfactual'],
                    alpha=0.3, color='#e74c3c', label='Intervention effect')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()


def print_its_summary(its_results: ITSResult | dict, metric_name: str = 'Outcome') -> None:
    """Print a summary of ITS analysis results."""
    # Support both dataclass and dict for backward compatibility
    if isinstance(its_results, ITSResult):
        r = its_results
    else:
        # Convert dict to namespace-like access
        from types import SimpleNamespace
        r = SimpleNamespace(**its_results)
    
    print("\n" + "="*70)
    print("INTERRUPTED TIME SERIES ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nMetric: {metric_name}")
    print(f"Model R²: {r.r_squared:.4f} (Adj. R²: {r.r_squared_adj:.4f})")
    
    print("\n" + "-"*70)
    print("PARAMETER ESTIMATES:")
    print("-"*70)
    
    def significance_stars(p: float) -> str:
        """Return significance stars based on p-value."""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""
    
    print(f"\n1. Pre-intervention trend (monthly change):")
    print(f"   β1 = {r.pre_trend:.4f}")
    
    print(f"\n2. Immediate effect (level change at intervention):")
    print(f"   β2 = {r.immediate_effect:.4f}")
    print(f"   p-value = {r.immediate_effect_pvalue:.4f} {significance_stars(r.immediate_effect_pvalue)}")
    
    print(f"\n3. Change in trend after intervention:")
    print(f"   β3 = {r.trend_change:.4f}")
    print(f"   p-value = {r.trend_change_pvalue:.4f} {significance_stars(r.trend_change_pvalue)}")
    
    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)
    
    if r.immediate_effect_pvalue < DEFAULT_ALPHA:
        direction = "increase" if r.immediate_effect > 0 else "decrease"
        print(f"  ✓ Significant immediate {direction} of {abs(r.immediate_effect):.2f}")
    else:
        print("  ✗ No significant immediate effect detected")
    
    if r.trend_change_pvalue < DEFAULT_ALPHA:
        direction = "steeper" if r.trend_change > 0 else "flatter"
        print(f"  ✓ Significant trend change: slope became {direction}")
    else:
        print("  ✗ No significant change in trend detected")
    
    print("="*70)


def run_its_for_provinces(df: pd.DataFrame,
                          metric: str,
                          provinces: list,
                          age_group: str = None,
                          intervention_date: str = '2020-03-01') -> pd.DataFrame:
    """
    Run ITS analysis for multiple provinces and compile results.
    
    Parameters:
    -----------
    df : DataFrame with employment data
    metric : Column name of the outcome variable
    provinces : List of provinces to analyze
    age_group : Age group to filter (optional)
    intervention_date : Date when COVID intervention began
    
    Returns:
    --------
    DataFrame with ITS results for each province
    """
    results_list = []
    
    for province in provinces:
        try:
            df_its = prepare_its_data(df, metric, province, age_group, intervention_date)
            
            if len(df_its) < 24:  # Need sufficient data
                continue
                
            its_result = run_its_analysis(df_its)
            
            results_list.append({
                'Province': province,
                'Immediate Effect': its_result['immediate_effect'],
                'Immediate Effect p-value': its_result['immediate_effect_pvalue'],
                'Trend Change': its_result['trend_change'],
                'Trend Change p-value': its_result['trend_change_pvalue'],
                'R²': its_result['r_squared'],
                'Significant Immediate': its_result['immediate_effect_pvalue'] < 0.05,
                'Significant Trend': its_result['trend_change_pvalue'] < 0.05
            })
        except Exception as e:
            print(f"Error processing {province}: {e}")
            continue
    
    return pd.DataFrame(results_list)


def plot_its_comparison(its_summary_df: pd.DataFrame,
                        title: str,
                        filename: str = None) -> None:
    """
    Plot comparison of ITS immediate effects across provinces.
    
    Parameters:
    -----------
    its_summary_df : DataFrame from run_its_for_provinces
    title : Plot title
    filename : Optional filename to save
    """
    df_plot = its_summary_df.sort_values('Immediate Effect', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_plot) * 0.4)))
    
    # Color by significance
    colors = ['#e74c3c' if sig else '#95a5a6' 
              for sig in df_plot['Significant Immediate']]
    
    bars = ax.barh(df_plot['Province'], df_plot['Immediate Effect'],
                   color=colors, edgecolor='black')
    
    ax.axvline(x=0, color='black', linewidth=1)
    
    ax.set_xlabel('Immediate Effect (Level Change)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Significant (p<0.05)'),
                       Patch(facecolor='#95a5a6', label='Not significant')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    plt.show()
