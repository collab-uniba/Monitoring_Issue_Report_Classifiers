import os
import yaml
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Custom YAML representer for numpy types
def numpy_representer(dumper, data):
    """Custom representer for numpy scalars to convert to Python native types"""
    if isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return dumper.represent_int(int(data))
    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        return dumper.represent_float(float(data))
    elif isinstance(data, (np.ndarray,)):
        return dumper.represent_list(data.tolist())
    elif isinstance(data, np.bool_):
        return dumper.represent_bool(bool(data))
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

# Add custom representer to PyYAML
yaml.add_representer(np.ndarray, numpy_representer)
yaml.add_representer(np.float64, numpy_representer)
yaml.add_representer(np.float32, numpy_representer)
yaml.add_representer(np.int64, numpy_representer)
yaml.add_representer(np.int32, numpy_representer)
yaml.add_representer(np.bool_, numpy_representer)

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

def check_normality(data, variable_name, output_dir, alpha=0.05):
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
    plt.savefig(f"{output_dir}/{variable_name.lower().replace(' ', '_')}_normality.png")  
    plt.close()
    
    # Interpret results
    normality_results = {
        'shapiro_statistic': float(shapiro_stat),
        'shapiro_p_value': float(shapiro_p),
        'anderson_statistic': float(anderson_result.statistic),
        'anderson_critical_values': anderson_result.critical_values.tolist(),
        'anderson_significance_levels': anderson_result.significance_level.tolist(),
        'is_normal': bool(shapiro_p > alpha)
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
    results_path = Path(config['results_root'])

    # Build the path differently based on split_type
    if config['split_type'] == 'year':
        # For year splits, only include years in the path
        results_path = results_path / (
            f"{config['project_name']}/"
            f"{config['model_type']}/"
            f"{config['split_type']}_range_{config['range']}/"
            f"{config['start_year']}_{config['end_year']}"
        )
    elif config['split_type'] == 'month':
        # For month splits, include year and month
        results_path = results_path / (
            f"{config['project_name']}/"
            f"{config['model_type']}/"
            f"{config['split_type']}_range_{config['range']}/"
            f"{config['start_year']}-{config.get('start_month', 'all')}_"
            f"{config['end_year']}-{config.get('end_month', 'all')}"
        )
    elif config['split_type'] == 'day':
        # For day splits, include year, month, and day
        results_path = results_path / (
            f"{config['project_name']}/"
            f"{config['model_type']}/"
            f"{config['split_type']}_range_{config['range']}/"
            f"{config['start_year']}-{config.get('start_month', 'all')}-{config.get('start_day', 'all')}_"
            f"{config['end_year']}-{config.get('end_month', 'all')}-{config.get('end_day', 'all')}"
        )
    else:
        # Fallback for any other split type
        logger.warning(f"Unknown split_type: {config['split_type']}. Using generic path format.")
        results_path = results_path / (
            f"{config['project_name']}/"
            f"{config['model_type']}/"
            f"{config['split_type']}_range_{config['range']}/"
            f"{config['start_year']}-{config.get('start_month', 'all')}_"
            f"{config['end_year']}-{config.get('end_month', 'all')}"
        )
    
    # Load similarity scores
    similarity_scores_path = results_path / "similarity_analysis" / f"similarity_scores_{mode}.csv"
    similarity_df = pd.read_csv(similarity_scores_path)

    similarity_df['period'] = similarity_df['period'].astype(str)
    
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

    performance_df['period'] = performance_df['period'].astype(str)

    # Convert performance_df to match similarity_df
    # This means that if in period we have start and end date that match, we should only keep the end date

    if int(config['range']) == 1:
        performance_df['period'] = performance_df['period'].str.split('_').str[1]

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
    similarity_normality = check_normality(merged_df[similarity_column], f"Cosine Similarity{plot_title_suffix}", stat_tests_path)
    f1_normality = check_normality(merged_df[f1_column], f"F1 Macro Score{plot_title_suffix}", stat_tests_path)
    
    # Save normality test results
    normality_results = {
        'cosine_similarity': similarity_normality,
        'f1_macro': f1_normality
    }
    
    with open(stat_tests_path / f"normality_results_{mode}.yaml", 'w') as f:
        yaml.dump(normality_results, f, default_flow_style=False)
    
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
        'correlation_coefficient': float(correlation_coef),
        'p_value': float(p_value),
        'sample_size': len(merged_df)
    }

    with open(stat_tests_path / f"correlation_results_{mode}.yaml", 'w') as f:
        yaml.dump(correlation_results, f, default_flow_style=False)
    
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
    
    plt.savefig(stat_tests_path / f"{'_'.join(filename_components)}_{mode}.png")
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

        # Normalization and DTW Analysis
    def perform_dtw_analysis(similarity_series, f1_series):
        """
        Perform Dynamic Time Warping analysis with multiple normalization approaches
        """
        # Ensure series are numpy arrays
        similarity_series = np.array(similarity_series)
        f1_series = np.array(f1_series)
        
        # Normalization methods
        normalization_results = {}
        
        # 1. Min-Max Normalization
        minmax_scaler = MinMaxScaler()
        similarity_minmax = minmax_scaler.fit_transform(similarity_series.reshape(-1, 1)).flatten()
        f1_minmax = minmax_scaler.fit_transform(f1_series.reshape(-1, 1)).flatten()
        
        # 2. Z-score Standardization
        zscore_scaler = StandardScaler()
        similarity_zscore = zscore_scaler.fit_transform(similarity_series.reshape(-1, 1)).flatten()
        f1_zscore = zscore_scaler.fit_transform(f1_series.reshape(-1, 1)).flatten()
        
        # Convert to the correct format for fastdtw (sequences of 1D points)
        similarity_minmax = np.array([[x] for x in similarity_minmax])
        f1_minmax = np.array([[x] for x in f1_minmax])
        
        # Perform DTW for both normalization methods
        # Min-Max Normalized DTW
        distance_minmax, path_minmax = fastdtw(similarity_minmax, f1_minmax, dist=euclidean)
        normalization_results['minmax'] = {
            'distance': distance_minmax,
            'path': path_minmax,
            'similarity_normalized': similarity_minmax.tolist(),
            'f1_normalized': f1_minmax.tolist()
        }
        
        similarity_zscore = np.array([[x] for x in similarity_zscore])
        f1_zscore = np.array([[x] for x in f1_zscore])

        # Z-score Normalized DTW
        distance_zscore, path_zscore = fastdtw(similarity_zscore, f1_zscore, dist=euclidean)
        normalization_results['zscore'] = {
            'distance': distance_zscore,
            'path': path_zscore,
            'similarity_normalized': similarity_zscore.tolist(),
            'f1_normalized': f1_zscore.tolist()
        }
        
        return normalization_results
    
    # Perform analysis
    # Choose column for analysis
    if use_deltas:
        similarity_column = 'similarity_delta'
        f1_column = 'f1_macro_delta'
        delta_suffix = " (Deltas)"
    else:
        similarity_column = 'similarity'
        f1_column = 'f1_macro'
        delta_suffix = ""
    
    # Perform DTW analysis
    dtw_results = perform_dtw_analysis(
        merged_df[similarity_column], 
        merged_df[f1_column]
    )
    
    # Create results directory
    stat_tests_path = results_path / "statistical_tests"
    os.makedirs(stat_tests_path, exist_ok=True)
    
    # Save DTW results
    with open(stat_tests_path / f"fastdtw_results_{mode}.yaml", 'w') as f:
        yaml.dump(dtw_results, f)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Original and Normalized Series
    plt.subplot(2, 2, 1)
    plt.title(f'Original Series{delta_suffix}')
    plt.plot(merged_df['period'], merged_df[similarity_column], label='Cosine Similarity')
    plt.plot(merged_df['period'], merged_df[f1_column], label='F1 Macro Score')

    # Filter x-ticks to show only January of each year (or first month of each year)
    # Assuming period format is like '2020-01', '2020-02', etc.
    years = [period.split('-')[0] for period in merged_df['period']]
    months = [period.split('-')[1] if '-' in period else '01' for period in merged_df['period']]
    tick_positions = []
    tick_labels = []
    current_year = None

    for i, (year, month) in enumerate(zip(years, months)):
        if year != current_year and month == '01':  # Show only January (or first month in dataset)
            current_year = year
            tick_positions.append(i)
            tick_labels.append(year)

    plt.xticks(tick_positions, tick_labels, rotation=45)
    
    plt.subplot(2, 2, 2)
    plt.title(f'Min-Max Normalized Series{delta_suffix}')
    plt.plot(
        range(len(merged_df)), 
        np.array(dtw_results['minmax']['similarity_normalized']).flatten(), 
        label='Cosine Similarity'
    )
    plt.plot(
        range(len(merged_df)),
        np.array(dtw_results['minmax']['f1_normalized']).flatten(), 
        label='F1 Macro Score'
    )

    # Use the same tick positions and labels as in the first subplot
    plt.xticks(tick_positions, tick_labels, rotation=45)
    
    # DTW Alignment Path (Min-Max)
    plt.subplot(2, 2, 3)
    path_x, path_y = zip(*dtw_results['minmax']['path'])
    plt.title(f'DTW Alignment Path (Min-Max){delta_suffix}')
    plt.scatter(path_x, path_y, c='red', alpha=0.5)
    plt.xlabel('Cosine Similarity Index')
    plt.ylabel('F1 Macro Score Index')
    
    # DTW Metrics Comparison
    plt.subplot(2, 2, 4)
    dtw_methods = ['Min-Max', 'Z-Score']
    dtw_distances = [
        dtw_results['minmax']['distance'], 
        dtw_results['zscore']['distance']
    ]
    plt.bar(dtw_methods, dtw_distances)
    plt.title('DTW Distances')
    plt.ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig(stat_tests_path / f"fastdtw_analysis_{mode}.png")
    plt.close()
    
    # Log DTW results
    logger.info("FastDTW Analysis Results:")
    logger.info(f"DTW Distance (Min-Max): {dtw_results['minmax']['distance']:.4f}")
    logger.info(f"DTW Distance (Z-Score): {dtw_results['zscore']['distance']:.4f}")

    
def perform_yearly_correlation_analysis(config_file, use_deltas=False, normalize=False, mode='mean'):
    """
    Perform statistical tests for each year separately when split_type is month.
    Plot the correlations between F1 and cosine similarity for each year with different colors based on test type.
    
    Parameters:
    - config_file: Path to the configuration YAML file
    - use_deltas: Whether to analyze changes instead of raw values
    - normalize: Whether to perform min-max normalization before analysis
    - mode: Aggregation mode for similarity scores ('mean' or 'max')
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Only proceed if split_type is month
    if config['split_type'] != 'month':
        logger.info("Yearly correlation analysis is only applicable for month split_type.")
        return
    
    # Generate paths
    results_path = Path(config['results_root'])
    results_path = results_path / (
        f"{config['split_type']}_range_{config['range']}/"
        f"{config['start_year']}-{config.get('start_month', 'all')}_"
        f"{config['end_year']}-{config.get('end_month', 'all')}/"
        f"{config['project_name']}"
    )
    
    # Load similarity scores
    similarity_scores_path = results_path / "similarity_analysis" / f"similarity_scores_{mode}.csv"
    similarity_df = pd.read_csv(similarity_scores_path)
    similarity_df['period'] = similarity_df['period'].astype(str)
    
    # Extract year from period in similarity_df
    # Handle different period formats: YYYY-MM, YYYY_MM, or YYYY
    similarity_df['year'] = similarity_df['period'].str.extract(r'(\d{4})').astype(int)
    
    # Load classification results
    try:
        aggregated_report, file_reports = load_classification_results(results_path)
    except FileNotFoundError:
        logger.error("Could not load classification results.")
        return
    
    # Prepare performance metrics
    performance_metrics = []
    
    # Extract F1-scores from file reports
    if file_reports:
        for period, report in file_reports.items():
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
    performance_df['period'] = performance_df['period'].astype(str)
    
    # Convert performance_df to match similarity_df
    if int(config['range']) == 1:
        # If range is 1, consider converting period formats if needed
        if '_' in performance_df['period'].iloc[0]:
            performance_df['period'] = performance_df['period'].str.split('_').str[1]
    
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
    
    # Get unique years
    years = sorted(merged_df['year'].unique())
    
    # Initialize results dictionary
    yearly_correlations = {}
    
    # Initialize plot
    plt.figure(figsize=(12, 8))
    
    # Colors for different test types
    pearson_color = 'blue'
    spearman_color = 'red'
    
    # Legend elements
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pearson_color, markersize=10, label='Pearson'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=spearman_color, markersize=10, label='Spearman')
    ]
    
    # For the scatter plot
    xs, ys, colors, labels = [], [], [], []
    
    # Perform analysis for each year
    for year in years:
        year_data = merged_df[merged_df['year'] == year]
        
        # Skip years with too few data points
        if len(year_data) < 3:
            logger.warning(f"Year {year} has fewer than 3 data points. Skipping.")
            continue
        
        # Check normality
        similarity_series = year_data[similarity_column]
        f1_series = year_data[f1_column]
        
        # Determine if data is normally distributed
        # We'll use Shapiro-Wilk test with alpha=0.05
        _, similarity_p = stats.shapiro(similarity_series)
        _, f1_p = stats.shapiro(f1_series)
        
        is_normal = similarity_p > 0.05 and f1_p > 0.05
        
        # Choose appropriate correlation method
        if is_normal:
            corr_func = stats.pearsonr
            test_name = "Pearson"
            point_color = pearson_color
        else:
            corr_func = stats.spearmanr
            test_name = "Spearman"
            point_color = spearman_color
        
        # Calculate correlation
        correlation_coef, p_value = corr_func(similarity_series, f1_series)
        
        # Store results
        yearly_correlations[str(year)] = {
            'test_method': test_name,
            'correlation_coefficient': float(correlation_coef),
            'p_value': float(p_value),
            'sample_size': len(year_data),
            'is_significant': bool(p_value < 0.05)
        }
        
        # Collect data for scatter plot
        xs.append(year)
        ys.append(correlation_coef)
        colors.append(point_color)
        labels.append(f"{year}")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Plot correlation coefficients
    for i, (x, y, color, label) in enumerate(zip(xs, ys, colors, labels)):
        # Check if the year exists in yearly_correlations
        if str(x) in yearly_correlations:
            p_value = yearly_correlations[str(x)]['p_value']  # Access p-value for the year
            plt.scatter(x, y, color=color, s=100, alpha=0.7)
            plt.annotate(
                f"{label}\n(p = {p_value:.4f})",  # Include p-value in the annotation
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
        else:
            logger.warning(f"Skipping year {x} as it was not processed due to insufficient data.")
        
    # Add significance threshold lines
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(y=0.3, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=-0.3, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.3)
    plt.axhline(y=-0.5, color='gray', linestyle='-.', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Correlation Coefficient')
    plt.title(f'Yearly Correlation: F1-Score vs Similarity{plot_title_suffix}\n{config["project_name"]}')
    
    # Adjust x-axis
    plt.xticks(years)
    plt.ylim(-1, 1)
    
    # Add legend
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(stat_tests_path / f"yearly_correlations_{mode}.png")
    plt.close()
    
    # Save correlation results
    with open(stat_tests_path / f"yearly_correlation_results_{mode}.yaml", 'w') as f:
        yaml.dump(yearly_correlations, f, default_flow_style=False)
    
    # Log results
    logger.info("Yearly Correlation Analysis Results:")
    for year, results in yearly_correlations.items():
        logger.info(f"Year {year}: {results['test_method']} correlation = {results['correlation_coefficient']:.4f} (p = {results['p_value']:.4f})")
    
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
    
    # Perform the standard statistical tests
    perform_statistical_tests(
        args.config,
        args.use_deltas,
        args.normalize,
        args.mode
    )
    
    # Load the config file to check if split_type is 'month'
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Only perform yearly correlation analysis if split_type is 'month'
    if config['split_type'] == 'month':
        logger.info("Split type is 'month'. Performing yearly correlation analysis...")
        perform_yearly_correlation_analysis(
            args.config,
            args.use_deltas,
            args.normalize,
            args.mode
        )
    else:
        logger.info(f"Split type is '{config['split_type']}'. Skipping yearly correlation analysis.")

if __name__ == "__main__":
    main()