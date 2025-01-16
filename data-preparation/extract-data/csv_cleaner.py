import csv
from config import Config
from utils import setup_logging, ensure_directories
import sys
import logging

csv.field_size_limit(sys.maxsize)

class CSVCleaner:
    def __init__(self):
        self.config = Config()
        setup_logging()
        ensure_directories([self.config.CLEANED_PATH])
        
    def clean_file(self, input_file):
        """Clean a single CSV file."""
        input_path = self.config.EXPORT_PATH / input_file
        output_path = self.config.CLEANED_PATH / input_file
        
        with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
             open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader((line.replace('\0', '') for line in infile))
            writer = csv.DictWriter(outfile, fieldnames=self.config.CSV_FIELD_MAPPINGS['output_fields'])
            
            writer.writeheader()
            for row in reader:
                cleaned_row = {
                    'title': row[self.config.CSV_FIELD_MAPPINGS['input_fields']['summary']],
                    'body': row[self.config.CSV_FIELD_MAPPINGS['input_fields']['description']],
                    'label': row[self.config.CSV_FIELD_MAPPINGS['input_fields']['issue_type']],
                    'date': row[self.config.CSV_FIELD_MAPPINGS['input_fields']['created_date']]
                }
                writer.writerow(cleaned_row)
    
    def run(self):
        """Clean all CSV files in the export directory."""
        for file in self.config.EXPORT_PATH.glob('*.csv'):
            self.clean_file(file.name)
            logging.info(f"Cleaned file: {file.name}")