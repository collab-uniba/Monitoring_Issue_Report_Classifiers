import csv
from config import Config
from utils import setup_logging, ensure_directories
import sys
import logging
from typing import Optional, List
from pathlib import Path

csv.field_size_limit(sys.maxsize)

class CSVCleaner:
    def __init__(self, collections: Optional[List[str]] = None, remove_duplicates: bool = True):
        """
        Initialize CSVCleaner with optional collection filter and duplicate removal option.
        
        Args:
            collections: Optional list of collection names to clean. If None, cleans all collections.
            remove_duplicates: If True, duplicate rows will be removed. Defaults to True.
        """
        self.config = Config()
        self.collections = collections
        self.remove_duplicates = remove_duplicates
        setup_logging()
        ensure_directories([self.config.CLEANED_PATH])
        
    def get_files_to_clean(self) -> List[Path]:
        """
        Get list of CSV files to clean based on collections filter.
        
        Returns:
            List of Path objects for CSV files to clean
        """
        all_files = list(self.config.EXPORT_PATH.glob('*.csv'))
        
        if not self.collections:
            return all_files
            
        # Filter files based on collection names
        collection_files = []
        for collection in self.collections:
            file_path = self.config.EXPORT_PATH / f"{collection}.csv"
            if file_path.exists():
                collection_files.append(file_path)
            else:
                logging.warning(f"Collection file not found: {file_path}")
                
        return collection_files
        
    def clean_file(self, input_file: Path):
        """
        Clean a single CSV file.
        
        Args:
            input_file: Path object for the input CSV file
        """
        output_path = self.config.CLEANED_PATH / input_file.name
        
        with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
             open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader((line.replace('\0', '') for line in infile))
            writer = csv.DictWriter(outfile, fieldnames=self.config.CSV_FIELD_MAPPINGS['output_fields'])
            
            writer.writeheader()
            
            # Set to keep track of seen rows if remove_duplicates is True
            seen_rows = set()
            
            for row in reader:
                cleaned_row = {
                    'title': row[self.config.CSV_FIELD_MAPPINGS['input_fields']['summary']],
                    'body': row[self.config.CSV_FIELD_MAPPINGS['input_fields']['description']],
                    'label': row[self.config.CSV_FIELD_MAPPINGS['input_fields']['issue_type']],
                    'date': row[self.config.CSV_FIELD_MAPPINGS['input_fields']['created_date']]
                }
                
                # Create a tuple of the cleaned row values to check for duplicates
                row_tuple = tuple(cleaned_row.values())
                
                if self.remove_duplicates:
                    if row_tuple not in seen_rows:
                        seen_rows.add(row_tuple)
                        writer.writerow(cleaned_row)
                else:
                    writer.writerow(cleaned_row)
    
    def run(self):
        """Clean selected CSV files in the export directory."""
        files_to_clean = self.get_files_to_clean()
        
        if not files_to_clean:
            logging.warning("No files found to clean!")
            return
            
        logging.info(f"Starting cleaning of files: {[f.name for f in files_to_clean]}")
        
        for file in files_to_clean:
            self.clean_file(file)
            logging.info(f"Cleaned file: {file.name}")
            
        logging.info("Cleaning process completed")