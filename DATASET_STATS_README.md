# Dataset Statistics Calculator

This document describes how to use the `calculate_dataset_stats.py` script to generate statistics about the temporal segmentation of datasets.

## Purpose

The script calculates dataset statistics for filling the temporal segmentation summary table in research papers. It computes:

- **Training Months**: Number of time windows (months) used for training
- **Test Months**: Number of time windows (months) used for testing
- **Training Instances**: Total number of training samples
- **Test Instances**: Total number of test samples (optional, not shown in default table)

## Requirements

The script reuses existing code from the classification module and requires:

1. Python 3.x with dependencies from `requirements.txt`
2. Configuration files (e.g., `config-apache-m1.yaml`)
3. Data files in `data/windows/{split_type}_range_{range}/{project_name}/`

## Usage

### Basic Usage

Process the default configurations (Apache, Jira, RedHat with month range 1):

```bash
python calculate_dataset_stats.py
```

### Process Specific Configs

```bash
python calculate_dataset_stats.py --configs config/config-apache-m1.yaml config/config-jira-m1.yaml
```

### Save LaTeX Table

```bash
python calculate_dataset_stats.py --output dataset_stats.tex
```

### Save CSV Format

```bash
python calculate_dataset_stats.py --output-csv dataset_stats.csv
```

## Data Preparation

If the data files are not available, the script will display warnings and instructions. To prepare the data:

```bash
# Navigate to data-preparation directory
cd data-preparation

# Create time windows for month split with range 1
python create_time_windows.py --split-type month --range 1
```

Make sure you have the source data files in the expected location before running `create_time_windows.py`.

## How It Works

The script:

1. Loads each configuration file using `ConfigManager`
2. Determines the data directory path based on config parameters
3. Uses `DataHandler` to load training and test data (same logic as classification pipeline)
4. Counts unique time windows and instances
5. Formats output as LaTeX table or CSV

## Output Format

### Console Output

The script prints:
- Processing logs for each ecosystem
- Summary table with all statistics
- LaTeX-formatted table ready to paste into papers

### LaTeX Table

```latex
\begin{table}[t]
\centering
\caption{Temporal segmentation summary for selected ecosystems.}
\label{tab:temporal-slices-summary}
\resizebox{\linewidth}{!}{%
\begin{tabular}{rrrr}
\toprule
\textbf{Ecosystem} & \textbf{Training Months} & \textbf{Test Months} & \textbf{Training Instances} \\
\midrule
Apache & \texttt{51} & \texttt{9} & \texttt{12345} \\
Jira & \texttt{59} & \texttt{1} & \texttt{23456} \\
RedHat & \texttt{57} & \texttt{3} & \texttt{34567} \\
\bottomrule
\end{tabular}%
}
\end{table}
```

### CSV Format

```csv
project_name,training_months,test_months,training_instances,test_instances
Apache,51,9,12345,678
Jira,59,1,23456,890
RedHat,57,3,34567,1234
```

## Understanding the Calculation

The script calculates statistics by:

1. **Training Data**: Uses `DataHandler.load_data()` with `test=False` to load all time windows that fall within or overlap the configured date range
2. **Test Data**: Uses `DataHandler.load_data()` with `test=True` to load time windows after the configured range
3. **Counting**: 
   - Months are counted by unique filenames (each file represents one time window)
   - Instances are counted by total number of rows in the loaded DataFrames

## Notes

- The actual training/test split depends on available data files, not just the configuration
- Some months in the configured range might not have data (e.g., no issues reported)
- The test set includes data AFTER the configured training period
- The script reuses the same data loading logic as the classification pipeline for consistency

## Troubleshooting

### "Data directory not found"

If you see this warning, the data files haven't been prepared yet. Run the data preparation pipeline:

```bash
cd data-preparation
python create_time_windows.py --split-type month --range 1
```

### Different numbers than expected

The actual counts depend on:
- Which months have data files
- How many issues were reported in each month
- The train/test splitting logic (test uses data AFTER the configured range)

### Import errors

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```
