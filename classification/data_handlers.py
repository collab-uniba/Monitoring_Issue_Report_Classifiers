# classification/data_handlers.py
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from datasets import Dataset

logger = logging.getLogger(__name__)

@dataclass
class TimeWindow:
    start_year: int
    end_year: int
    start_month: Optional[int] = None
    end_month: Optional[int] = None
    start_day: Optional[int] = None
    end_day: Optional[int] = None

    def __lt__(self, other):
        self_tuple = (self.start_year, self.start_month or 0, self.start_day or 0)
        other_tuple = (other.start_year, other.start_month or 0, other.start_day or 0)
        return self_tuple < other_tuple

    def __str__(self):
        if self.start_month is None:
            return f"{self.start_year}-{self.end_year}"
        elif self.start_day is None:
            return f"{self.start_year}-{self.start_month:02d} to {self.end_year}-{self.end_month:02d}"
        else:
            return f"{self.start_year}-{self.start_month:02d}-{self.start_day:02d} to {self.end_year}-{self.end_month:02d}-{self.end_day:02d}"

class DataHandler:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.file_cache = {}

    def _parse_filename(self, filename: str) -> TimeWindow:
        """Parse filename to extract time window information."""
        filename = filename.replace('.csv', '')
        
        if '-' not in filename:  # Year pattern
            start_year, end_year = map(int, filename.split('_'))
            return TimeWindow(start_year=start_year, end_year=end_year)
        
        start_part, end_part = filename.split('_')
        
        if '-' in start_part:  # Month or Day pattern
            parts = start_part.split('-')
            if len(parts) == 2:  # Month pattern
                start_year, start_month = map(int, parts)
                end_year, end_month = map(int, end_part.split('-'))
                return TimeWindow(
                    start_year=start_year, end_year=end_year,
                    start_month=start_month, end_month=end_month
                )
            elif len(parts) == 3:  # Day pattern
                start_year, start_month, start_day = map(int, parts)
                end_year, end_month, end_day = map(int, end_part.split('-'))
                return TimeWindow(
                    start_year=start_year, end_year=end_year,
                    start_month=start_month, end_month=end_month,
                    start_day=start_day, end_day=end_day
                )
        
        raise ValueError(f"Unrecognized filename pattern: {filename}")

    def _get_files_in_range(self, target: TimeWindow, split_type: str, test: bool = False) -> List[str]:
        """Find files within or after a specified time range."""
        logger.info(f"Searching for files in range: {target}, split_type: {split_type}, test: {test}")
        
        if split_type not in self.file_cache:
            files = list(self.data_dir.glob("*.csv"))
            logger.info(f"Found {len(files)} total files in directory")
            self.file_cache[split_type] = {
                file.name: self._parse_filename(file.name) 
                for file in files
            }

        matching_files = []
        for filename, window in self.file_cache[split_type].items():
            logger.info(f"Checking file: {filename}, window: {window}")
            if test:
                # For test data, only include the exact period we're looking for
                if split_type == "month":
                    if (window.start_year == target.start_year and 
                        window.start_month == target.start_month and
                        window.end_year == target.end_year and 
                        window.end_month == target.end_month):
                        matching_files.append(filename)
                        logger.info(f"Including test file: {filename}")
                elif split_type == "day":
                    if (window.start_year == target.start_year and 
                        window.start_month == target.start_month and
                        window.start_day == target.start_day and
                        window.end_year == target.end_year and 
                        window.end_month == target.end_month and
                        window.end_day == target.end_day):
                        matching_files.append(filename)
                        logger.info(f"Including test file: {filename}")
                else:  # year
                    if (window.start_year == target.start_year and 
                        window.end_year == target.end_year):
                        matching_files.append(filename)
                        logger.info(f"Including test file: {filename}")
            else:
                # For training data, include files that overlap with the target window
                if split_type == "month":
                    overlaps = not (window.end_year < target.start_year or 
                                  window.start_year > target.end_year or
                                  (window.end_year == target.start_year and window.end_month < target.start_month) or
                                  (window.start_year == target.end_year and window.start_month > target.end_month))
                    if overlaps:
                        matching_files.append(filename)
                        logger.info(f"Including training file: {filename}")
                elif split_type == "day":
                    overlaps = not (
                        window.end_year < target.start_year or
                        window.start_year > target.end_year or
                        (window.end_year == target.start_year and 
                         (window.end_month < target.start_month or
                          (window.end_month == target.start_month and window.end_day < target.start_day))) or
                        (window.start_year == target.end_year and
                         (window.start_month > target.end_month or
                          (window.start_month == target.end_month and window.start_day > target.end_day)))
                    )
                    if overlaps:
                        matching_files.append(filename)
                        logger.info(f"Including training file: {filename}")
                else:  # year
                    if window.end_year >= target.start_year and window.start_year <= target.end_year:
                        matching_files.append(filename)
                        logger.info(f"Including training file: {filename}")
        
        return sorted(matching_files)

    def _get_files_after_range(self, target: TimeWindow, split_type: str, test: bool = False) -> List[str]:
        logger.info(f"Searching for files after range: {target}, split_type: {split_type}, test: {test}")
        
        if split_type not in self.file_cache:
            files = list(self.data_dir.glob("*.csv"))
            logger.info(f"Found {len(files)} total files in directory")
            self.file_cache[split_type] = {
                file.name: self._parse_filename(file.name) 
                for file in files
            }
            logger.info(f"Cached {len(self.file_cache[split_type])} files")
        
        matching_files = []
        for filename, window in self.file_cache[split_type].items():
            logger.debug(f"Checking file {filename} with window {window}")
            
            if test:
                if split_type == "year":
                    matches = window.start_year > target.end_year
                    logger.debug(f"Test condition for {filename}: {matches}")
                    if matches:
                        matching_files.append(filename)
                        logger.info(f"Including test file: {filename}")
                    else:
                        logger.debug(f"Skipping test file: {filename}") 
                elif split_type == "month":
                    matches = (window.start_year > target.end_year or 
                             (window.start_year == target.end_year and window.start_month > target.end_month) or
                             (window.start_year == target.end_year and window.start_month == target.end_month))
                    logger.debug(f"Test condition for {filename}: {matches}")
                    if matches:
                        matching_files.append(filename)
                        logger.info(f"Including test file: {filename}")
                    else:
                        logger.debug(f"Skipping test file: {filename}")
            else:
                if split_type == "month":
                    overlaps = not (window.end_year < target.start_year or 
                                  window.start_year > target.end_year or
                                  (window.end_year == target.start_year and window.end_month < target.start_month) or
                                  (window.start_year == target.end_year and window.start_month > target.end_month))
                    if overlaps:
                        matching_files.append(filename)
                        logger.info(f"Including training file: {filename}")
                    else:
                        logger.debug(f"Skipping training file: {filename}")
        
        logger.info(f"Found {len(matching_files)} matching files")
        return sorted(matching_files)

    def load_data(self, 
                  split_type: str, 
                  range_val: str, 
                  project_name: str, 
                  start_year: int, 
                  end_year: int, 
                  label_mapper,
                  start_month: Optional[int] = None, 
                  end_month: Optional[int] = None,
                  start_day: Optional[int] = None, 
                  end_day: Optional[int] = None,
                  test: bool = False, 
                  exact_date: bool = False) -> pd.DataFrame:
        """
        Centralized data loading method with flexible time window support.
        """
        data_dir = Path(f"data/windows/{split_type}_range_{range_val}/{project_name}")
        
        # Validate and process data directory
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Create time window
        target = TimeWindow(
            start_year=start_year,
            end_year=end_year,
            start_month=start_month,
            end_month=end_month,
            start_day=start_day,
            end_day=end_day
        )
        
        # Determine matching files
        if test:
            if exact_date:
                matching_files = self._get_files_in_range(target, split_type, test=test)
            else:
                matching_files = self._get_files_after_range(target, split_type, test=test)
        else:
            matching_files = self._get_files_in_range(target, split_type, test=test)
        
        if not matching_files:
            raise ValueError(f"No data files found for the specified range in {data_dir}")
        
        # Load and combine matching files
        dfs = []
        for filename in matching_files:
            df = pd.read_csv(data_dir / filename)
            df['file_name'] = filename
            dfs.append(df)
        
        df_all = pd.concat(dfs, ignore_index=True)
        
        # Preprocess data
        df_all['date'] = pd.to_datetime(df_all['date'], errors='coerce')
        if label_mapper.label_to_id:
            df_all = df_all[df_all['label'].isin(label_mapper.label_to_id.keys())]
        
        df_all['text'] = df_all['title'] + " " + df_all['body']
        df_all['labels'] = label_mapper.map_labels(df_all['label']).astype(int)

        # Remove label column and rename labels column to label
        df_all.drop(columns=['label'], inplace=True)
        df_all.rename(columns={'labels': 'label'}, inplace=True)

        # Fill na values for text column    
        df_all['text'] = df_all['text'].fillna("")
        
        return df_all

    @staticmethod
    def sample_training_data(df, 
                              sampling_strategy='random', 
                              samples_per_class=50, 
                              seed=42) -> pd.DataFrame:
        """
        Sample training data with different strategies.
        
        Args:
            df: DataFrame with training data
            sampling_strategy: Strategy to use ('random', 'balanced', 'stratified')
            samples_per_class: Number of samples per class
            seed: Random seed for reproducibility
        
        Returns:
            DataFrame with sampled training data
        """
        np.random.seed(seed)
        sampled_data = []
        
        for label in df['label'].unique():
            class_data = df[df['label'] == label]
            
            if sampling_strategy == 'random':
                sampled_data.append(class_data.sample(
                    min(len(class_data), samples_per_class), 
                    random_state=seed
                ))
            elif sampling_strategy == 'balanced':
                sampled_data.append(class_data.sample(
                    min(len(class_data), samples_per_class), 
                    random_state=seed
                ))
            elif sampling_strategy == 'stratified':
                class_proportion = len(class_data) / len(df)
                n_samples = max(
                    int(samples_per_class * class_proportion * len(df['label'].unique())), 
                    min(3, len(class_data))
                )
                sampled_data.append(
                    class_data.sample(min(n_samples, len(class_data)), random_state=seed)
                )
        
        return pd.concat(sampled_data, ignore_index=True)

    def prepare_dataset(self, 
                        df, 
                        label_mapper, 
                        tokenizer=None, 
                        max_token_len=512, 
                        use_validation=True, 
                        split_size=0.3):
        """
        Prepare dataset for training, including optional validation split.
        
        Args:
            df: Input DataFrame
            label_mapper: Label mapping utility
            tokenizer: Tokenizer for encoding (optional)
            max_token_len: Maximum token length
            use_validation: Whether to create a validation split
            split_size: Size of validation split
        
        Returns:
            Tuple of training and validation datasets (or single dataset)
        """
        if use_validation:
            train_df, validation_df = train_test_split(
                df, 
                test_size=split_size, 
                random_state=42, 
                stratify=df['labels']
            )
        else:
            train_df = df
            validation_df = None
        
        # Convert to Hugging Face datasets if tokenizer provided
        if tokenizer:
            train_dataset = self._convert_to_dataset(train_df, tokenizer, max_token_len)
            validation_dataset = (
                self._convert_to_dataset(validation_df, tokenizer, max_token_len) 
                if validation_df is not None 
                else None
            )
            return train_dataset, validation_dataset
        
        return train_df, validation_df

    def _convert_to_dataset(self, df, tokenizer, max_token_len):
        """
        Convert DataFrame to Hugging Face Dataset with tokenization.
        
        Args:
            df: Input DataFrame
            tokenizer: Tokenizer for encoding
            max_token_len: Maximum token length
        
        Returns:
            Hugging Face Dataset
        """
        df['text'] = df['text'].fillna("")
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'], 
                padding=True, 
                truncation=True, 
                max_length=max_token_len
            )
        
        return Dataset.from_pandas(df).map(tokenize_function, batched=True)