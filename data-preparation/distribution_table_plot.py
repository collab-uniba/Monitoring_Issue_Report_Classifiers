import os
import pandas as pd
import argparse
import subprocess
from pathlib import Path
from typing import Optional
from config import Config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TABLES_PATH = Path("distribution-table/tables")
WINDOWS_TABLES_PATH = TABLES_PATH / "windows"

def parse_timestamp(timestamp_str: str) -> pd.Timestamp:
    """Parse timestamp in format '2021-12-14T16:45:26.000+0000'"""
    return pd.to_datetime(timestamp_str, format='%Y-%m-%dT%H:%M:%S.%f%z', utc=True)

def ensure_time_windows(split_type: str, range_size: int, allow_window_creation: bool = True) -> bool:
    """
    Ensure time windows exist, create them if needed and allowed.
    """
    window_path = Config.WINDOWS_PATH / f"{split_type}_range_{range_size}"
    
    if window_path.exists():
        return True
        
    if not allow_window_creation:
        logger.error(f"Time windows not found at {window_path} and creation not allowed")
        return False
        
    logger.info("Time windows not found. Creating them...")
    try:
        subprocess.run([
            "python", 
            "data-preparation/create_time_windows.py",
            "--split-type", split_type,
            "--range", str(range_size)
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create time windows: {e}")
        return False

def generate_overall_distribution(mapped_folder: Path):
    """
    Generate overall distribution table from mapped data.
    """
    file_info_list = []
    total_bug_count = 0
    total_feature_count = 0
    total_question_count = 0

    for file_path in mapped_folder.glob('*.csv'):
        df = pd.read_csv(file_path)
        try:
            df['date'] = df['date'].apply(parse_timestamp)
        except Exception as e:
            logger.error(f"Error parsing dates in {file_path}: {e}")
            continue

        min_year = df['date'].dt.year.min(skipna=True)
        max_year = df['date'].dt.year.max(skipna=True)

        bug_count = (df['label'].str.lower() == 'bug').sum()
        feature_count = (df['label'].str.lower() == 'feature').sum()
        question_count = (df['label'].str.lower() == 'question').sum()

        total_bug_count += bug_count
        total_feature_count += feature_count
        total_question_count += question_count

        file_info_list.append({
            'Project': file_path.stem,
            'Year First Issue': int(min_year),
            'Year Last Issue': int(max_year),
            'Bug': bug_count,
            'Feature': feature_count,
            'Question': question_count
        })

    # Add totals row
    file_info_list.insert(0, {
        'Project': 'Total',
        'Year First Issue': '',
        'Year Last Issue': '',
        'Bug': total_bug_count,
        'Feature': total_feature_count,
        'Question': total_question_count
    })

    distribution_table = pd.DataFrame(file_info_list)
    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    distribution_table.to_csv(TABLES_PATH / 'overall.csv', index=False)
    print(distribution_table)

def generate_windows_distribution(windows_path: Path):
    """
    Generate distribution tables for each project across their time windows.
    """
    WINDOWS_TABLES_PATH.mkdir(parents=True, exist_ok=True)
    
    # Process each project directory
    for project_dir in windows_path.iterdir():
        if not project_dir.is_dir():
            continue
            
        project_name = project_dir.name
        logger.info(f"Processing windows distribution for project: {project_name}")
        
        # Dictionary to store distributions for each time window
        window_distributions = []
        
        # Process each window file
        total_bug_count = 0
        total_feature_count = 0
        total_question_count = 0
        
        for file_path in sorted(project_dir.glob('*.csv')):
            window_period = file_path.stem
            df = pd.read_csv(file_path)
            
            # Count labels for this window (case insensitive)
            label_counts = df['label'].str.lower().value_counts()
            bug_count = label_counts.get('bug', 0)
            feature_count = label_counts.get('feature', 0)
            question_count = label_counts.get('question', 0)
            
            # Update totals
            total_bug_count += bug_count
            total_feature_count += feature_count
            total_question_count += question_count
            
            # Add window distribution
            window_distributions.append({
                'Period': window_period,
                'Bug': bug_count,
                'Feature': feature_count,
                'Question': question_count
            })
        
        if window_distributions:
            # Create DataFrame for this project's windows
            project_df = pd.DataFrame(window_distributions)
            
            # Add total row at the beginning
            total_row = pd.DataFrame([{
                'Period': 'Total',
                'Bug': total_bug_count,
                'Feature': total_feature_count,
                'Question': total_question_count
            }])
            
            project_df = pd.concat([total_row, project_df], ignore_index=True)
            
            # Save project distribution
            output_file = WINDOWS_TABLES_PATH / f'{project_name}.csv'
            project_df.to_csv(output_file, index=False)
            logger.info(f"Created distribution file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate distribution tables based on the provided mode.')
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['overall', 'windows'],
                      help='Mode to generate distribution tables')
    parser.add_argument('--split-type', type=str, choices=['year', 'month', 'day'],
                      help='Type of time split to use (required for windows mode)')
    parser.add_argument('--range', type=int,
                      help='Size of the time window range (required for windows mode)')
    parser.add_argument('--allow-window-creation', type=bool, default=True,
                      help='Allow creation of time windows if they don\'t exist')
    
    args = parser.parse_args()

    if args.mode == 'windows':
        if not args.split_type or not args.range:
            parser.error("Windows mode requires --split-type and --range parameters")
            
        if not ensure_time_windows(args.split_type, args.range, args.allow_window_creation):
            return
            
        windows_path = Config.WINDOWS_PATH / f"{args.split_type}_range_{args.range}"
        generate_windows_distribution(windows_path)
    else:  # overall
        generate_overall_distribution(Config.MAPPED_PATH)

if __name__ == "__main__":
    main()