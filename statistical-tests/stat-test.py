import os
import yaml
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_classification_results(results_path):
    """
    Load classification results from YAML files.
    """
    aggregated_report_path = results_path / "aggregated_report.yaml"
    file_reports_path = results_path / "file_reports.yaml"
    
    try:
        with open(aggregated_report_path, 'r') as f:
            aggregated_report = yaml.safe_load(f)
        
        with open(file_reports_path, 'r') as f:
            file_reports = yaml.safe_load(f)
        
        return aggregated_report, file_reports
    except FileNotFoundError:
        logger.error(f"Classification results not found in {results_path}")
        raise

def check_normality(data, variable_name, alpha=0.05):
    """
    Perform normality tests and create Q-Q plot.
    Returns a dictionary of normality test results.
    """
    # Shapiro-Wilk test (best for small sample sizes)
    shapiro_stat, shapiro_p = stats.shapiro(data)
    
    # Anderson-Darling test
    anderson_result = stats.anderson(data, dist='norm')
    
    # Create Q-Q plot
    plt.figure(figsize=(10, 5))
    
    # Q-Q plot
    plt.subplot(1, 2, 1)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {variable_name}")
    
    # Histogram with normal distribution overlay
    plt.subplot(1, 2, 2)
    plt.hist(data, bins='auto', density=True, alpha=0.7, color='skyblue')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(data), np.std(data))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f"Histogram of {variable_name}")
    
    plt.tight_layout()
    plt.savefig(f"{variable_name.lower().replace(' ', '_')}_normality.png")
    plt.close()
    
    # Interpret results
    normality_results = {
        'shapiro_statistic': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'anderson_statistic': anderson_result.statistic,
        'anderson_critical_values': anderson_result.critical_values,
        'anderson_significance_levels': anderson_result.significance_level,
        'is_normal': shapiro_p > alpha
    }
    
    return normality_results

def calculate_deltas(df, column):
    """
    Calculate deltas between consecutive rows, with first delta set to 0.
    Assumes the DataFrame is sorted by period.
    """
    # Sort by period to ensure correct delta calculation
    df_sorted = df.sort_values('period')
    
    # Calculate deltas
    deltas = df_sorted[column].diff().fillna(0)
    
    return deltas

def perform_statistical_tests(config_file, use_deltas=False, normalize=False, mode='mean'):
    """
    Perform comprehensive statistical tests with optional delta and normalization analysis.
    
    Parameters:
    - config_file: Path to the configuration YAML file
    - use_deltas: Whether to analyze changes instead of raw values
    - normalize: Whether to perform min-max normalization before analysis
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Generate paths
    results_path = Path(config['results_root']) / (
        f"{config['split_type']}_range_{config['range']}/"
        f"{config['start_year']}-{config.get('start_month', 'all')}_"
        f"{config['end_year']}-{config.get('end_month', 'all')}/"
        f"{config['project_name']}"
    )
    
    # Load similarity scores
    similarity_scores_path = results_path / "similarity_analysis" / f"similarity_scores_{mode}.csv"
    similarity_df = pd.read_csv(similarity_scores_path)
    
    # Load classification results
    try:
        aggregated_report, file_reports = load_classification_results(results_path)
    except FileNotFoundError:
        logger.error("Could not load classification results.")
        return
    
    # Prepare performance metrics
    performance_metrics = []
    
    # Extract F1-scores from file reports or aggregated report
    if file_reports:
        for period, report in file_reports.items():
            # Macro F1-score across all classes
            period_name = os.path.splitext(period)[0]  # Remove .csv extension
            f1_macro = report.get('macro avg', {}).get('f1-score', np.nan)
            performance_metrics.append({
                'period': period_name,
                'f1_macro': f1_macro
            })
    else:
        # Fallback to aggregated report
        f1_macro = aggregated_report.get('macro avg', {}).get('f1-score', np.nan)
        performance_metrics.append({
            'period': 'aggregated',
            'f1_macro': f1_macro
        })
    
    # Convert to DataFrame
    performance_df = pd.DataFrame(performance_metrics)
    
    # Merge DataFrames
    merged_df = pd.merge(similarity_df, performance_df, on='period', how='inner')
    
    # Create results directory
    stat_tests_path = results_path / "statistical_tests"
    os.makedirs(stat_tests_path, exist_ok=True)
    
    # Determine analysis type
    if use_deltas:
        # Calculate deltas
        merged_df['similarity_delta'] = calculate_deltas(merged_df, 'similarity')
        merged_df['f1_macro_delta'] = calculate_deltas(merged_df, 'f1_macro')
        
        # Use delta columns for analysis
        similarity_column = 'similarity_delta'
        f1_column = 'f1_macro_delta'
        plot_title_suffix = " (Deltas)"
        test_title_suffix = " of Changes"
    else:
        # Use original columns
        similarity_column = 'similarity'
        f1_column = 'f1_macro'
        plot_title_suffix = ""
        test_title_suffix = ""
    
    # Optional Normalization
    if normalize:
        # Min-Max Normalization
        scaler = MinMaxScaler()
        merged_df[f'{similarity_column}_normalized'] = scaler.fit_transform(merged_df[[similarity_column]])
        merged_df[f'{f1_column}_normalized'] = scaler.fit_transform(merged_df[[f1_column]])
        
        # Update columns for analysis
        similarity_column = f'{similarity_column}_normalized'
        f1_column = f'{f1_column}_normalized'
        plot_title_suffix += " (Normalized)"
        test_title_suffix += " (Normalized)"
    
    # Perform normality tests
    similarity_normality = check_normality(merged_df[similarity_column], f"Cosine Similarity{plot_title_suffix}")
    f1_normality = check_normality(merged_df[f1_column], f"F1 Macro Score{plot_title_suffix}")
    
    # Save normality test results
    normality_results = {
        'cosine_similarity': similarity_normality,
        'f1_macro': f1_normality
    }
    
    with open(stat_tests_path / "normality_results.yaml", 'w') as f:
        yaml.dump(normality_results, f)
    
    # Determine appropriate correlation test based on normality
    correlation_method = "Pearson" if similarity_normality['is_normal'] and f1_normality['is_normal'] else "Spearman"
    
    # Perform correlation test
    if correlation_method == "Pearson":
        corr_func = stats.pearsonr
        test_name = "Pearson Correlation"
    else:
        corr_func = stats.spearmanr
        test_name = "Spearman Rank Correlation"
    
    # Perform correlation test
    correlation_coef, p_value = corr_func(merged_df[similarity_column], merged_df[f1_column])
    
    # Save correlation results
    correlation_results = {
        'test_method': correlation_method,
        'correlation_coefficient': correlation_coef,
        'p_value': p_value,
        'sample_size': len(merged_df)
    }
    
    with open(stat_tests_path / "correlation_results.yaml", 'w') as f:
        yaml.dump(correlation_results, f)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x=similarity_column, y=f1_column)
    plt.title(f"{test_name}{test_title_suffix}: Cosine Similarity vs F1-Score\n{config['project_name']} - {config['split_type']} split")
    plt.xlabel(f"Average Cosine Similarity{plot_title_suffix}")
    plt.ylabel(f"Macro F1-Score{plot_title_suffix}")
    
    # Annotate with correlation information
    plt.annotate(
        f"{test_name}{test_title_suffix}:\n"
        f"Coefficient: {correlation_coef:.4f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Sample Size: {len(merged_df)}",
        xy=(0.05, 0.95), 
        xycoords='axes fraction',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    plt.tight_layout()
    
    # Update filename to reflect normalization and delta usage
    filename_components = ['similarity_vs_performance']
    if use_deltas:
        filename_components.append('deltas')
    if normalize:
        filename_components.append('normalized')
    
    plt.savefig(stat_tests_path / f"{'_'.join(filename_components)}.png")
    plt.close()
    
    # Log comprehensive results
    logger.info("Statistical Analysis Results:")
    logger.info("Normality Tests:")
    logger.info(f"Cosine Similarity - Shapiro p-value: {similarity_normality['shapiro_p_value']:.4f}")
    logger.info(f"F1 Macro Score - Shapiro p-value: {f1_normality['shapiro_p_value']:.4f}")
    logger.info(f"Selected Correlation Test: {test_name}{test_title_suffix}")
    logger.info(f"Correlation Coefficient: {correlation_coef:.4f}")
    logger.info(f"p-value: {p_value:.4f}")
    logger.info(f"Results saved to {stat_tests_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform statistical tests on model performance and similarity.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--use-deltas",
        action="store_true",
        help="Analyze changes (deltas) in similarity and performance instead of raw values"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Perform min-max normalization before statistical tests"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Mode of aggregation for similarity scores"
    )
    args = parser.parse_args()
    
    perform_statistical_tests(args.config, args.use_deltas, args.normalize, args.mode)

if __name__ == "__main__":
    main()