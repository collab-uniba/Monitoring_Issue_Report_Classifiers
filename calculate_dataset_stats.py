#!/usr/bin/env python3
"""
Script to calculate dataset statistics for temporal segmentation.

This script reads configuration files and calculates:
- Number of training months (time windows with data)
- Number of test months (time windows with data for testing)
- Number of training instances (total training samples)
- Number of test instances (total test samples)

The script reuses existing code from the classification module to load
and process data in the same way as the classification pipeline.

Usage:
    # Process all default configs (apache-m1, jira-m1, redhat-m1)
    python calculate_dataset_stats.py

    # Process specific config files
    python calculate_dataset_stats.py --configs config/config-apache-m1.yaml

    # Save LaTeX table to file
    python calculate_dataset_stats.py --output dataset_stats.tex

Note: If data files are not available, the script will provide estimates
      based on configuration. Actual numbers require the data directory:
      data/windows/{split_type}_range_{range}/{project_name}/
"""

import argparse
import csv
import logging
from pathlib import Path
import pandas as pd
import yaml

from classification.config_manager import ConfigManager
from classification.data_handlers import DataHandler
from classification.label_mapper import LabelMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_months(start_year, start_month, end_year, end_month):
    """
    Calculate the number of months between two dates (inclusive).
    
    Args:
        start_year: Starting year
        start_month: Starting month (1-12)
        end_year: Ending year
        end_month: Ending month (1-12)
    
    Returns:
        Number of months in the range (inclusive)
    """
    months = (end_year - start_year) * 12 + (end_month - start_month) + 1
    return months


def get_dataset_stats(config_path):
    """
    Calculate dataset statistics from a configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with training_months, test_months, training_instances, test_instances,
        training_month_files (list of file names), and test_month_files (list of file names)
    """
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    project_name = config['project_name']
    split_type = config['split_type']
    range_val = config['range']
    start_year = config['start_year']
    end_year = config['end_year']
    start_month = config.get('start_month', 1)
    end_month = config.get('end_month', 12)
    
    logger.info(f"Processing {project_name}: {start_year}-{start_month} to {end_year}-{end_month}")
    
    # Calculate total months
    total_months = calculate_months(start_year, start_month, end_year, end_month)
    logger.info(f"Total months in range: {total_months}")
    
    # Initialize statistics
    stats = {
        'project_name': project_name,
        'training_months': 0,
        'test_months': 0,
        'training_instances': 0,
        'test_instances': 0,
        'training_month_files': [],
        'test_month_files': []
    }
    
    # Check if data directory exists
    data_dir = Path(f"data/windows/{split_type}_range_{range_val}/{project_name}")
    
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        logger.warning("Cannot calculate actual statistics without data files.")
        logger.info("")
        logger.info("To get accurate statistics, ensure data is prepared by running:")
        logger.info(f"  python data-preparation/create_time_windows.py --split-type {split_type} --range {range_val}")
        logger.info("")
        logger.info(f"Total months in configuration range: {total_months}")
        logger.info("However, actual training/test splits depend on data availability.")
        logger.info("")
        # Return zero counts to indicate missing data
        stats['training_months'] = 0
        stats['test_months'] = 0
        return stats
    
    # Initialize components
    label_mapper = LabelMapper(config.get('label_set', []))
    data_handler = DataHandler(data_dir)
    
    try:
        # Load training data
        logger.info("Loading training data...")
        df_train = data_handler.load_data(
            split_type,
            range_val,
            project_name,
            start_year,
            end_year,
            label_mapper,
            start_month,
            end_month,
            test=False
        )
        
        # Count training instances and months
        stats['training_instances'] = len(df_train)
        
        # Count unique time periods in training data
        if 'file_name' in df_train.columns:
            training_files = sorted(df_train['file_name'].unique())
            stats['training_months'] = len(training_files)
            stats['training_month_files'] = list(training_files)
            logger.info(f"Training: {stats['training_months']} months, {stats['training_instances']} instances")
            logger.info(f"Training month files: {', '.join(training_files)}")
        
    except Exception as e:
        logger.warning(f"Error loading training data: {e}")
    
    try:
        # Load test data
        logger.info("Loading test data...")
        df_test = data_handler.load_data(
            split_type,
            range_val,
            project_name,
            start_year,
            end_year,
            label_mapper,
            start_month,
            end_month,
            test=True
        )
        
        # Count test instances and months
        stats['test_instances'] = len(df_test)
        
        # Count unique time periods in test data
        if 'file_name' in df_test.columns:
            test_files = sorted(df_test['file_name'].unique())
            stats['test_months'] = len(test_files)
            stats['test_month_files'] = list(test_files)
            logger.info(f"Test: {stats['test_months']} months, {stats['test_instances']} instances")
            logger.info(f"Test month files: {', '.join(test_files)}")
            
    except Exception as e:
        logger.warning(f"Error loading test data: {e}")
    
    return stats


def get_monthly_label_distribution(config_path):
    """
    Calculate label distribution per month and cumulative statistics.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        DataFrame with monthly distribution and cumulative counts
    """
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    project_name = config['project_name']
    split_type = config['split_type']
    range_val = config['range']
    start_year = config['start_year']
    end_year = config['end_year']
    start_month = config.get('start_month', 1)
    end_month = config.get('end_month', 12)
    
    # Check if data directory exists
    data_dir = Path(f"data/windows/{split_type}_range_{range_val}/{project_name}")
    
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return None
    
    # Initialize components
    label_mapper = LabelMapper(config.get('label_set', []))
    data_handler = DataHandler(data_dir)
    
    try:
        # Load training data
        df_train = data_handler.load_data(
            split_type,
            range_val,
            project_name,
            start_year,
            end_year,
            label_mapper,
            start_month,
            end_month,
            test=False
        )
        
        if 'file_name' not in df_train.columns or 'label' not in df_train.columns:
            logger.warning("Missing required columns in training data")
            return None
        
        # Get unique labels
        labels = sorted(df_train['label'].unique())
        
        # Sort files chronologically
        month_files = sorted(df_train['file_name'].unique())
        
        # Calculate distribution per month
        monthly_data = []
        cumulative_counts = {label: 0 for label in labels}
        cumulative_total = 0
        
        for month_file in month_files:
            month_df = df_train[df_train['file_name'] == month_file]
            
            # Count per label for this month
            label_counts = {}
            for label in labels:
                count = len(month_df[month_df['label'] == label])
                label_counts[label] = count
                cumulative_counts[label] += count
            
            # Total for this month
            month_total = len(month_df)
            cumulative_total += month_total
            
            # Create row
            row = {
                'month_file': month_file,
                'month_total': month_total,
                'cumulative_total': cumulative_total
            }
            
            # Add per-label counts
            for label in labels:
                row[f'label_{label}_count'] = label_counts[label]
                row[f'label_{label}_cumulative'] = cumulative_counts[label]
            
            monthly_data.append(row)
        
        df_monthly = pd.DataFrame(monthly_data)
        return df_monthly, labels
        
    except Exception as e:
        logger.error(f"Error calculating monthly distribution: {e}", exc_info=True)
        return None


def format_monthly_distribution_table(df_monthly, labels, project_name):
    """
    Format the monthly label distribution as a text table.
    
    Args:
        df_monthly: DataFrame with monthly statistics
        labels: List of label values
        project_name: Name of the project
        
    Returns:
        String with formatted table
    """
    if df_monthly is None or df_monthly.empty:
        return f"No monthly distribution data available for {project_name}\n"
    
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"MONTHLY LABEL DISTRIBUTION - {project_name}")
    lines.append(f"{'='*80}")
    
    # Create header
    header_parts = ["Month"]
    for label in labels:
        header_parts.append(f"L{label}")
        header_parts.append(f"CumL{label}")
    header_parts.extend(["Total", "CumTotal"])
    
    header_line = " | ".join(f"{h:>12}" for h in header_parts)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Add data rows
    for _, row in df_monthly.iterrows():
        month = row['month_file'].replace('.csv', '')
        row_parts = [f"{month:>12}"]
        
        for label in labels:
            count = int(row[f'label_{label}_count'])
            cum_count = int(row[f'label_{label}_cumulative'])
            row_parts.append(f"{count:>12}")
            row_parts.append(f"{cum_count:>12}")
        
        total = int(row['month_total'])
        cum_total = int(row['cumulative_total'])
        row_parts.append(f"{total:>12}")
        row_parts.append(f"{cum_total:>12}")
        
        lines.append(" | ".join(row_parts))
    
    lines.append("="*80)
    # Create legend dynamically based on number of labels
    label_legend = "/".join([f"L{label}" for label in labels])
    lines.append(f"Legend: {label_legend} = Label counts, CumL = Cumulative label counts")
    lines.append(f"Total = Total samples in month, CumTotal = Cumulative total samples")
    lines.append("")
    
    return "\n".join(lines)


def save_monthly_distribution_csv(df_monthly, labels, project_name, output_path):
    """
    Save monthly distribution to CSV file.
    
    Args:
        df_monthly: DataFrame with monthly statistics
        labels: List of label values
        project_name: Name of the project
        output_path: Path to save CSV file
    """
    if df_monthly is None or df_monthly.empty:
        logger.warning(f"No monthly distribution data to save for {project_name}")
        return
    
    # Prepare data for CSV
    csv_data = []
    for _, row in df_monthly.iterrows():
        csv_row = {
            'project': project_name,
            'month_file': row['month_file'],
            'month_total': int(row['month_total']),
            'cumulative_total': int(row['cumulative_total'])
        }
        
        for label in labels:
            csv_row[f'label_{label}_count'] = int(row[f'label_{label}_count'])
            csv_row[f'label_{label}_cumulative'] = int(row[f'label_{label}_cumulative'])
        
        csv_data.append(csv_row)
    
    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(output_path, index=False)
    logger.info(f"Monthly distribution saved to: {output_path}")


def format_latex_table(stats_list):
    """
    Format the statistics as a LaTeX table.
    
    Args:
        stats_list: List of statistics dictionaries
        
    Returns:
        String with LaTeX table content
    """
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Temporal segmentation summary for selected ecosystems.}")
    lines.append("\\label{tab:temporal-slices-summary}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{rrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Ecosystem} & \\textbf{Training Months} & \\textbf{Test Months} & \\textbf{Training Instances} \\\\")
    lines.append("\\midrule")
    
    for stats in stats_list:
        project = stats['project_name']
        train_months = stats['training_months']
        test_months = stats['test_months']
        train_instances = stats['training_instances']
        lines.append(f"{project} & \\texttt{{{train_months}}} & \\texttt{{{test_months}}} & \\texttt{{{train_instances}}} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate dataset statistics for temporal segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default configs (apache-m1, jira-m1, redhat-m1)
  python calculate_dataset_stats.py
  
  # Process specific config files
  python calculate_dataset_stats.py --configs config/config-apache-m1.yaml
  
  # Save LaTeX table to file
  python calculate_dataset_stats.py --output dataset_stats.tex
  
  # Save statistics to CSV
  python calculate_dataset_stats.py --output-csv dataset_stats.csv
        """
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "config/config-apache-m1.yaml",
            "config/config-jira-m1.yaml", 
            "config/config-redhat-m1.yaml"
        ],
        help="List of configuration files to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for LaTeX table (optional)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Output file for CSV format (optional)"
    )
    parser.add_argument(
        "--monthly-distribution",
        action="store_true",
        help="Calculate and display monthly label distribution with cumulative counts"
    )
    parser.add_argument(
        "--monthly-output-csv",
        type=str,
        help="Output file for monthly distribution CSV (optional)"
    )
    
    args = parser.parse_args()
    
    stats_list = []
    
    for config_path in args.configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {config_path}")
        logger.info(f"{'='*60}")
        
        try:
            stats = get_dataset_stats(config_path)
            stats_list.append(stats)
        except Exception as e:
            logger.error(f"Error processing {config_path}: {e}", exc_info=True)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET STATISTICS SUMMARY")
    print("="*60)
    
    for stats in stats_list:
        print(f"\n{stats['project_name']}:")
        print(f"  Training Months: {stats['training_months']}")
        print(f"  Test Months: {stats['test_months']}")
        print(f"  Training Instances: {stats['training_instances']}")
        print(f"  Test Instances: {stats['test_instances']}")
        
        # Display training month files for debugging
        if stats['training_month_files']:
            print(f"  Training Month Files:")
            for i, month_file in enumerate(stats['training_month_files'], 1):
                print(f"    {i}. {month_file}")
        
        # Display test month files for debugging
        if stats['test_month_files']:
            print(f"  Test Month Files:")
            for i, month_file in enumerate(stats['test_month_files'], 1):
                print(f"    {i}. {month_file}")
    
    # Generate LaTeX table
    latex_table = format_latex_table(stats_list)
    
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    print(latex_table)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex_table)
        logger.info(f"\nLaTeX table saved to: {args.output}")
    
    # Save CSV if requested
    if args.output_csv:
        with open(args.output_csv, 'w', newline='') as f:
            # Prepare data for CSV (convert file lists to strings)
            csv_data = []
            for stats in stats_list:
                csv_row = {
                    'project_name': stats['project_name'],
                    'training_months': stats['training_months'],
                    'test_months': stats['test_months'],
                    'training_instances': stats['training_instances'],
                    'test_instances': stats['test_instances'],
                    'training_month_files': ';'.join(stats['training_month_files']),
                    'test_month_files': ';'.join(stats['test_month_files'])
                }
                csv_data.append(csv_row)
            
            writer = csv.DictWriter(f, fieldnames=['project_name', 'training_months', 'test_months', 'training_instances', 'test_instances', 'training_month_files', 'test_month_files'])
            writer.writeheader()
            writer.writerows(csv_data)
        logger.info(f"\nCSV saved to: {args.output_csv}")
    
    # Calculate and display monthly distribution if requested
    if args.monthly_distribution:
        for config_path in args.configs:
            try:
                config_manager = ConfigManager(config_path)
                config = config_manager.get_config()
                project_name = config['project_name']
                
                logger.info(f"\nCalculating monthly distribution for {project_name}...")
                result = get_monthly_label_distribution(config_path)
                
                if result is not None:
                    df_monthly, labels = result
                    
                    # Display the table
                    table_output = format_monthly_distribution_table(df_monthly, labels, project_name)
                    print(table_output)
                    
                    # Save to CSV if requested
                    if args.monthly_output_csv:
                        # Create filename with project name
                        output_path = args.monthly_output_csv
                        if len(args.configs) > 1:
                            # Multiple configs - add project name to filename
                            base, ext = output_path.rsplit('.', 1) if '.' in output_path else (output_path, 'csv')
                            output_path = f"{base}_{project_name}.{ext}"
                        
                        save_monthly_distribution_csv(df_monthly, labels, project_name, output_path)
                else:
                    logger.warning(f"Could not calculate monthly distribution for {project_name}")
                    
            except Exception as e:
                logger.error(f"Error calculating monthly distribution for {config_path}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
