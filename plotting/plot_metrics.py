import yaml
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from datetime import datetime

def extract_date_info(filename):
    """
    Extract the starting and ending dates from a filename.
    Returns a tuple of (start_year, start_month, start_day, end_year, end_month, end_day)
    with None for unspecified values.
    """
    # First, remove the .csv extension if present
    if filename.endswith('.csv'):
        filename = filename[:-4]
        
    # Pattern 1: YYYY-YYYY
    pattern1 = r'^(\d{4})-(\d{4})$'
    match = re.match(pattern1, filename)
    if match:
        start_year, end_year = int(match.group(1)), int(match.group(2))
        return (start_year, 1, 1, end_year, 12, 31)  # Assume full years
        
    # Pattern 2: YYYY-MM_YYYY-MM (handle format like 2000-1_2000-1)
    pattern2 = r'^(\d{4})-(\d{1,2})_(\d{4})-(\d{1,2})$'
    match = re.match(pattern2, filename)
    if match:
        start_year, start_month, end_year, end_month = map(int, match.groups())
        return (start_year, start_month, 1, end_year, end_month, None)
        
    # Pattern 3: YYYY_MM-YYYY_MM
    pattern3 = r'^(\d{4})_(\d{2})-(\d{4})_(\d{2})$'
    match = re.match(pattern3, filename)
    if match:
        start_year, start_month, end_year, end_month = map(int, match.groups())
        return (start_year, start_month, 1, end_year, end_month, None)

    # Pattern 4: YYYY_MM_DD-YYYY_MM_DD
    pattern4 = r'^(\d{4})_(\d{2})_(\d{2})-(\d{4})_(\d{2})_(\d{2})$'
    match = re.match(pattern4, filename)
    if match:
        return tuple(map(int, match.groups()))
        
    # Debug output
    print(f"Failed to parse date from filename: {filename}")
    return None

def calculate_date_value(date_info):
    """
    Calculate the appropriate date value for plotting.
    For same year and month (e.g., 2000-1_2000-1), returns the exact year.month.
    Otherwise returns the midpoint between start and end dates.
    Returns a decimal year (e.g., 2020.5 for mid-2020).
    """
    if not date_info:
        return None

    start_year, start_month, start_day, end_year, end_month, end_day = date_info
    
    # If it's the same year and month, return that point exactly
    if start_year == end_year and start_month == end_month:
        return float(start_year) + (start_month - 1) / 12.0
    
    # If it's the same year with full-year range, return that year exactly
    if start_year == end_year and start_month == 1 and end_month == 12:
        return float(start_year)
    
    # Handle missing day values
    if start_day is None:
        start_day = 1
    if end_day is None:
        # Estimate last day of month (simplified)
        end_day = 30 if end_month in [4, 6, 9, 11] else 31
        if end_month == 2:
            end_day = 28  # Simplified, ignoring leap years

    # Create datetime objects
    try:
        start_date = datetime(start_year, start_month, start_day)
        end_date = datetime(end_year, end_month, end_day)

        # Calculate midpoint
        mid_date = start_date + (end_date - start_date) / 2

        # Convert to decimal year
        year_start = datetime(mid_date.year, 1, 1)
        year_length = datetime(mid_date.year + 1, 1, 1) - year_start

        # Calculate fraction of year
        fraction = (mid_date - year_start).total_seconds() / year_length.total_seconds()
        decimal_year = mid_date.year + fraction

        return decimal_year

    except ValueError as e:
        print(f"Error with date calculation for {date_info}: {e}")
        return float(start_year) + (start_month - 1) / 12.0  # Fallback

def parse_yaml_and_plot(yaml_file, debug=False):
    """Parse the YAML file and plot the f1-scores over time with precise date handling."""
    try:
        # Get the directory of the input YAML file
        yaml_dir = os.path.dirname(os.path.abspath(yaml_file))
        yaml_basename = os.path.basename(yaml_file)
        plot_filename = os.path.join(yaml_dir, 'performance_timeline.png')

        # Load YAML file
        with open(yaml_file, 'r') as file:
            content = file.read()
            # Debug the raw content if needed
            if debug:
                print("Raw YAML content sample:", content[:200])
            data = yaml.safe_load(content)

        if not data:
            print(f"No data found in {yaml_file}")
            return

        if debug:
            print(f"YAML data type: {type(data)}")
            print(f"First few keys: {list(data.keys())[:3] if isinstance(data, dict) else 'Not a dict'}")

        # Extract dates and f1-scores
        date_scores = []

        # Process the data structure where keys are filenames and values are report dictionaries
        for filename, file_info in data.items():
            if debug:
                print(f"Processing file: {filename}")
                
            # Skip any entries that don't look like data
            if not isinstance(file_info, dict):
                print(f"Skipping {filename} - not a dictionary")
                continue

            # Get f1-score from macro avg section
            if 'macro avg' in file_info and isinstance(file_info['macro avg'], dict):
                if 'f1-score' in file_info['macro avg']:
                    f1_score = file_info['macro avg']['f1-score']
                    
                    # Extract date info from the filename
                    date_info = extract_date_info(filename)
                    if debug:
                        print(f"Extracted date info from {filename}: {date_info}")

                    if date_info:
                        decimal_date = calculate_date_value(date_info)
                        if decimal_date:
                            date_scores.append((decimal_date, f1_score, filename))
                            if debug:
                                print(f"Added data point: {decimal_date}, {f1_score}, {filename}")
                        else:
                            print(f"Could not calculate decimal date for {filename} with date_info {date_info}")
                    else:
                        print(f"Could not extract date info from filename: {filename}")
                else:
                    print(f"'f1-score' not found in 'macro avg' for {filename}")
            else:
                print(f"'macro avg' not found in file_info for {filename}")

        if not date_scores:
            print("No valid data points found with dates and f1-scores")
            return

        print(f"Successfully extracted {len(date_scores)} data points")

        # Sort by date
        date_scores.sort(key=lambda x: x[0])

        # Extract sorted data
        dates = [item[0] for item in date_scores]
        scores = [item[1] for item in date_scores]
        filenames = [item[2] for item in date_scores]

        # Create the plot
        plt.figure(figsize=(14, 7))

        # Plot points and line
        plt.plot(dates, scores, 'o-', color='blue', linewidth=2, markersize=8)

        # Set axis labels and title
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Macro Avg F1-Score', fontsize=12)
        plt.title('Performance (Macro Avg F1-Score) Over Time', fontsize=14)

        # Set y-axis limits with some padding
        min_score = min(scores) - 0.05
        max_score = max(scores) + 0.05
        plt.ylim(max(0, min_score), min(1, max_score))

        # Format x-axis with appropriate ticks
        # Get the whole years present in the data
        all_years = sorted(set(int(date) for date in dates))
        plt.xticks(all_years)

        # Add minor ticks for quarters if data spans less than 10 years
        if len(all_years) < 10:
            minor_ticks = []
            for year in all_years:
                for quarter in [0.25, 0.5, 0.75]:
                    minor_ticks.append(year + quarter)
            plt.gca().set_xticks(minor_ticks, minor=True)

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.grid(True, which='minor', linestyle=':', alpha=0.4)

        # Add table below the chart (without displaying in the plot)
        table_data = []
        for date, score, filename in date_scores:
            # Format date for display
            year = int(date)
            if date.is_integer():
                month = 1  # For whole years, show as January
            else:
                month = int((date % 1) * 12) + 1
            date_str = f"{year}-{month:02d}"  # Always use zero-padded month


            table_data.append([date_str, f"{score:.3f}", filename])

        print("\nPerformance Data:")
        print(f"{'Date':<10} {'F1-Score':<10} {'Files'}")
        print("-" * 80)
        for row in table_data:
            print(f"{row[0]:<10} {row[1]:<10} {row[2]}")

        # Adjust layout and save the plot in the same directory as the input YAML
        plt.tight_layout()
        plt.savefig(plot_filename)

        print(f"\nPlot saved as '{plot_filename}' in the same directory as '{yaml_basename}'")

        # Show the plot
        plt.show()

    except Exception as e:
        import traceback
        print(f"Error processing the YAML file: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot performance metrics from a YAML file.')
    parser.add_argument('yaml_file', help='Path to the YAML file containing performance data')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    parse_yaml_and_plot(args.yaml_file, args.debug)