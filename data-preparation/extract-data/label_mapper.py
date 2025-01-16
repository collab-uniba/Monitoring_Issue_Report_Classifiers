import csv
import re
from config import Config
from utils import setup_logging, ensure_directories
import logging

class LabelMapper:
    def __init__(self):
        self.config = Config()
        setup_logging()
        ensure_directories([self.config.MAPPED_PATH])
        self.compile_patterns()
        
    def compile_patterns(self):
        """Compile regex patterns for label mapping."""
        self.patterns = {
            re.compile(pattern, flags=re.IGNORECASE): replacement
            for pattern, replacement in self.config.LABEL_PATTERNS.items()
        }
    
    def map_label(self, label):
        """Map a single label using the defined patterns."""
        for pattern, replacement in self.patterns.items():
            if pattern.match(label):
                return replacement, True
        return label, False
    
    def map_file(self, input_file):
        """Map labels in a single CSV file."""
        input_path = self.config.CLEANED_PATH / input_file
        output_path = self.config.MAPPED_PATH / input_file
        
        with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
             open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            
            writer.writeheader()
            for row in reader:
                new_label, was_mapped = self.map_label(row['label'])
                if was_mapped:
                    row['label'] = new_label
                    writer.writerow(row)
    
    def run(self):
        """Map labels in all CSV files."""
        for file in self.config.CLEANED_PATH.glob('*.csv'):
            self.map_file(file.name)
            logging.info(f"Mapped labels in file: {file.name}")