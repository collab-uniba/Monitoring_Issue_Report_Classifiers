import pandas as pd
from pathlib import Path
from typing import List, Optional
import argparse
from datetime import datetime
import logging
from config import Config
from utils import setup_logging

setup_logging()

class TimeWindowCreator:
    def __init__(self, split_type: str, range_size: int):
        """
        Initialize the TimeWindowCreator.
        
        Args:
            split_type: Type of split ('year', 'month', or 'day')
            range_size: Size of the time window range
        """
        self.split_type = split_type.lower()
        self.range_size = range_size
        
        if self.split_type not in ['year', 'month', 'day']:
            raise ValueError("Split type must be 'year', 'month', or 'day'")
            
        # Create base windows directory
        self.windows_base = Config.WINDOWS_PATH / f"{split_type}_range_{range_size}"
        self.windows_base.mkdir(parents=True, exist_ok=True)

    def get_period_key(self, date: datetime) -> str:
        """
        Get the period key for a given date based on split type and range.
        
        Args:
            date: Datetime object to get period for
            
        Returns:
            String representing the period (e.g., '2009-2010' for year range 2)
        """
        if self.split_type == 'year':
            base_period = date.year
        elif self.split_type == 'month':
            base_period = date.year * 12 + date.month - 1
        else:  # day
            base_period = (date - datetime(1970, 1, 1)).days

        # Calculate range start and end
        start_period = base_period - (base_period % self.range_size)
        end_period = start_period + self.range_size - 1
        
        # Convert back to human-readable format
        if self.split_type == 'year':
            return f"{start_period}_{end_period}"
        elif self.split_type == 'month':
            start_year, start_month = divmod(start_period, 12)
            end_year, end_month = divmod(end_period, 12)
            return f"{start_year}-{start_month+1:02d}_{end_year}-{end_month+1:02d}"
        else:  # day
            start_date = datetime(1970, 1, 1) + pd.Timedelta(days=start_period)
            end_date = datetime(1970, 1, 1) + pd.Timedelta(days=end_period)
            return f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"

    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse timestamp string in format '2021-12-14T16:45:26.000+0000'
        
        Args:
            timestamp_str: Timestamp string to parse
            
        Returns:
            Datetime object
        """
        # Remove the timezone offset and milliseconds for consistent parsing
        timestamp_str = timestamp_str.split('.')[0]
        return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')

    def process_collection(self, collection: str):
        """
        Process a single collection and create time window files.
        
        Args:
            collection: Name of the collection to process
        """
        input_path = Config.MAPPED_PATH / f"{collection}.csv"
        if not input_path.exists():
            logging.warning(f"No data file found for collection {collection}")
            return

        logging.info(f"Processing collection: {collection}")
        
        # Create collection directory
        collection_dir = self.windows_base / collection
        collection_dir.mkdir(exist_ok=True)
        
        # Read data
        df = pd.read_csv(input_path)
        
        # Parse dates using the custom parser
        try:
            df['date'] = df['date'].apply(self.parse_timestamp)
        except Exception as e:
            logging.error(f"Error parsing dates for collection {collection}: {str(e)}")
            return
        
        # Group by time periods
        df['period'] = df['date'].apply(self.get_period_key)
        
        # Save each period to a separate file
        for period, period_df in df.groupby('period'):
            output_file = collection_dir / f"{period}.csv"
            period_df.drop('period', axis=1).to_csv(output_file, index=False)
            logging.info(f"Created window file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create time windows from mapped data')
    parser.add_argument('--split-type', required=True, choices=['year', 'month', 'day'],
                      help='Type of time split to perform')
    parser.add_argument('--range', type=int, required=True,
                      help='Size of the time window range')
    parser.add_argument('--collections', nargs='*',
                      help='Specific collections to process (default: use config)')
    
    args = parser.parse_args()
    
    # Get collections to process
    collections = Config.get_collections(args.collections)
    if collections is None:
        # If None, process all collections found in mapped directory
        collections = [p.stem for p in Config.MAPPED_PATH.glob('*.csv')]
    
    window_creator = TimeWindowCreator(args.split_type, args.range)
    
    for collection in collections:
        window_creator.process_collection(collection)

if __name__ == "__main__":
    main()