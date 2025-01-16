import csv
import re
from config import Config
from utils import ensure_directories
import logging
from typing import Optional, List, Tuple
from pathlib import Path

class LabelMapper:
    def __init__(self, collections: Optional[List[str]] = None):
        """
        Initialize LabelMapper with optional collection filter.
        
        Args:
            collections: Optional list of collection names to process. If None, processes all collections.
        """
        self.config = Config()
        self.collections = collections
        ensure_directories([self.config.MAPPED_PATH])
        self.compile_patterns()
        
    def compile_patterns(self):
        """Compile regex patterns for label mapping."""
        self.patterns = {
            re.compile(pattern, flags=re.IGNORECASE): replacement
            for pattern, replacement in self.config.LABEL_PATTERNS.items()
        }
    
    def get_files_to_map(self) -> List[Path]:
        """
        Get list of CSV files to map based on collections filter.
        
        Returns:
            List of Path objects for CSV files to process
        """
        all_files = list(self.config.CLEANED_PATH.glob('*.csv'))
        
        if not self.collections:
            return all_files
            
        # Filter files based on collection names
        collection_files = []
        for collection in self.collections:
            file_path = self.config.CLEANED_PATH / f"{collection}.csv"
            if file_path.exists():
                collection_files.append(file_path)
            else:
                logging.warning(f"Collection file not found: {file_path}")
                
        return collection_files
    
    def map_label(self, label: str) -> Tuple[str, bool]:
        """
        Map a single label using the defined patterns.
        
        Args:
            label: Original label to map
            
        Returns:
            Tuple of (mapped label, whether it was mapped)
        """
        for pattern, replacement in self.patterns.items():
            if pattern.match(label):
                return replacement, True
        return label, False
    
    def map_file(self, input_file: Path):
        """
        Map labels in a single CSV file.
        
        Args:
            input_file: Path object for the input CSV file
        """
        output_path = self.config.MAPPED_PATH / input_file.name
        
        with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
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
        """Map labels in selected CSV files."""
        files_to_map = self.get_files_to_map()
        
        if not files_to_map:
            logging.warning("No files found to map!")
            return
            
        logging.info(f"Starting label mapping for files: {[f.name for f in files_to_map]}")
        
        for file in files_to_map:
            self.map_file(file)
            logging.info(f"Mapped labels in file: {file.name}")
            
        logging.info("Label mapping completed")