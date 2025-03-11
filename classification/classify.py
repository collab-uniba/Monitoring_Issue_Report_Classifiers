import os
import argparse
import yaml
import pandas as pd
import torch
import logging
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LabelMapper:
    """
    Maintains consistent label mapping across training and testing.
    Labels are sorted alphabetically and mapped to sequential integers starting from 0.
    """
    def __init__(self, label_set=None):
        if label_set:
            # Sort labels alphabetically
            sorted_labels = sorted(label_set)
            # Create mappings with sequential integers
            self.label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
            self.id_to_label = dict(enumerate(sorted_labels))
        else:
            self.label_to_id = {}
            self.id_to_label = {}
    
    def map_labels(self, labels):
        return labels.map(self.label_to_id)
    
    def inverse_map(self, ids):
        return [self.id_to_label[id] for id in ids]
    
    @property
    def num_labels(self):
        return len(self.label_to_id)
    
    
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_token_len=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if pd.isna(text):
            text = ""
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

logging.basicConfig(level=logging.INFO)
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

class DataLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.file_cache = {}
        
    def _parse_filename(self, filename: str) -> TimeWindow:
        logger.debug(f"Parsing filename: {filename}")
        if '-' not in filename:
            raise ValueError(f"Invalid filename format: {filename}")
            
        filename = filename.replace('.csv', '')
        
        if '_' not in filename:  # Year pattern
            start_year, end_year = map(int, filename.split('-'))
            return TimeWindow(start_year=start_year, end_year=end_year)
        
        start_part, end_part = filename.split('_')
        
        if '-' in start_part:  # Month pattern
            if len(start_part.split('-')) == 2:
                start_year, start_month = map(int, start_part.split('-'))
                end_year, end_month = map(int, end_part.split('-'))
                window = TimeWindow(
                    start_year=start_year, end_year=end_year,
                    start_month=start_month, end_month=end_month
                )
                logger.debug(f"Parsed month window: {window}")
                return window
            else:  # Day pattern
                start_year, start_month, start_day = map(int, start_part.split('-'))
                end_year, end_month, end_day = map(int, end_part.split('-'))
                return TimeWindow(
                    start_year=start_year, end_year=end_year,
                    start_month=start_month, end_month=end_month,
                    start_day=start_day, end_day=end_day
                )
        
        raise ValueError(f"Unrecognized filename pattern: {filename}")

    def _get_files_after_range(self, target: TimeWindow, split_type: str, test: bool = False) -> List[str]:
        logger.info(f"Searching for files in range: {target}, split_type: {split_type}, test: {test}")
        
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
                if split_type == "month":
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
    
    def _get_files_in_range(self, target: TimeWindow, split_type: str, test: bool = False) -> List[str]:
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

    
def generate_results_path(root_path, split_type, range_val, project_name, start_year, end_year, 
                         start_month=None, end_month=None, start_day=None, end_day=None):
    path = Path(root_path) / f"{split_type}_range_{range_val}"
    if split_type == "year":
        path /= f"{start_year}_{end_year}"
    elif split_type == "month":
        path /= f"{start_year}-{start_month}_{end_year}-{end_month}"
    elif split_type == "day":
        path /= f"{start_year}-{start_month}-{start_day}_{end_year}-{end_month}-{end_day}"

    path /= project_name
    return path

def load_data(split_type: str, range_val: str, project_name: str, 
              start_year: int, end_year: int, label_mapper,
              start_month: Optional[int] = None, end_month: Optional[int] = None,
              start_day: Optional[int] = None, end_day: Optional[int] = None,
              test: bool = False, exact_date: bool = False) -> pd.DataFrame:
    """
    Load data based on split type and range with optimized file searching.
    """
    data_dir = Path(f"data/windows/{split_type}_range_{range_val}/{project_name}")
    logger.info(f"Loading data from {data_dir}")
    logger.info(f"Parameters: split_type={split_type}, start_year={start_year}, end_year={end_year}, "
                f"start_month={start_month}, end_month={end_month}, test={test}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    loader = DataLoader(data_dir)
    
    target = TimeWindow(
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
        start_day=start_day,
        end_day=end_day
    )
    
    logger.info(f"Searching for data in time window: {target}")
    if not test:
        matching_files = loader._get_files_in_range(target, split_type, test=False)
    else:
        if exact_date:
            matching_files = loader._get_files_in_range(target, split_type, test=True)
        else:
            matching_files = loader._get_files_after_range(target, split_type, test=True)
    
    if not matching_files:
        logger.error(f"No matching files found for window {target}")
        # List all files in directory for debugging
        all_files = list(data_dir.glob("*.csv"))
        logger.error(f"Available files in directory: {[f.name for f in all_files]}")
        raise ValueError(f"No data files found for the specified range in {data_dir}")
    
    dfs = []
    for filename in matching_files:
        logger.info(f"Loading file: {filename}")
        df = pd.read_csv(data_dir / filename)
        df['file_name'] = filename
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df_all)} total rows from {len(matching_files)} files")
    
    df_all['date'] = pd.to_datetime(df_all['date'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%f+0000')
    if label_mapper.label_to_id:
        df_all = df_all[df_all['label'].isin(label_mapper.label_to_id.keys())]
    df_all['text'] = df_all['title'] + " " + df_all['body']
    df_all['labels'] = label_mapper.map_labels(df_all['label']).astype(int)
    
    if df_all['labels'].isna().any():
        raise ValueError("Some labels could not be mapped to integers")
    
    return df_all

def train_model(df_train_val, results_path, model_save_path, config, label_mapper, use_validation=True, split_size=0.3):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    if use_validation:
        train_df, validation_df = train_test_split(
            df_train_val, 
            test_size=split_size, 
            random_state=42, 
            stratify=df_train_val['labels']
        )
        train_dataset = CustomDataset(train_df['text'].to_numpy(), train_df['labels'].to_numpy(), tokenizer)
        validation_dataset = CustomDataset(
            validation_df['text'].to_numpy(),
            validation_df['labels'].to_numpy(),
            tokenizer
        )
    else:
        # Use entire dataset for training when validation is disabled
        train_dataset = CustomDataset(df_train_val['text'].to_numpy(), df_train_val['labels'].to_numpy(), tokenizer)
        validation_dataset = None
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=label_mapper.num_labels
        )

    training_args = TrainingArguments(
        output_dir=results_path,
        num_train_epochs=config['training_args']['num_train_epochs'],
        per_device_train_batch_size=config['training_args']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training_args']['per_device_eval_batch_size'],
        warmup_steps=config['training_args']['warmup_steps'],
        weight_decay=config['training_args']['weight_decay'],
        logging_dir=config['training_args']['logging_dir'],
        eval_strategy="epoch" if use_validation else "no",
        save_strategy="no",
        learning_rate=float(config['training_args']['learning_rate']),
        adam_epsilon=float(config['training_args']['adam_epsilon']),
        eval_steps=config['training_args']['eval_steps'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=lambda pred: classification_report(
            pred.label_ids,
            pred.predictions.argmax(-1),
            output_dict=True
        ),
    )

    trainer.train()
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save label mapping along with the model
    label_mapping_path = model_save_path / "label_mapping.yaml"
    with open(label_mapping_path, 'w') as f:
        yaml.dump({
            'label_to_id': label_mapper.label_to_id,
            'id_to_label': label_mapper.id_to_label
        }, f)
    
    logger.info(f"Model and label mapping saved to {model_save_path}")
    return trainer, model, tokenizer

def evaluate_model(model, tokenizer, test_df, results_path, label_mapper):
    test_dataset = CustomDataset(test_df['text'].to_numpy(), test_df['labels'].to_numpy(), tokenizer)
    trainer = Trainer(model=model)

    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = test_df['labels'].to_numpy()

    # Convert numeric predictions and labels back to original label names
    pred_labels = label_mapper.inverse_map(preds)
    true_labels = label_mapper.inverse_map(labels)

    # Generate classification report for each file
    file_reports = {}
    if 'file_name' in test_df.columns:
        for file in test_df['file_name'].unique():
            file_mask = test_df['file_name'] == file
            file_true_labels = [true_labels[i] for i, mask in enumerate(file_mask) if mask]
            file_pred_labels = [pred_labels[i] for i, mask in enumerate(file_mask) if mask]
            file_reports[file] = classification_report(
                file_true_labels,
                file_pred_labels,
                output_dict=True
            )

    # Generate aggregated classification report
    aggregated_report = classification_report(true_labels, pred_labels, output_dict=True)

    # Save reports and predictions
    with open(results_path / "file_reports.yaml", 'w') as f:
        yaml.dump(file_reports, f)

    with open(results_path / "aggregated_report.yaml", 'w') as f:
        yaml.dump(aggregated_report, f)

    test_df['predicted_label'] = pred_labels
    test_df.to_csv(results_path / "predictions.csv", index=False)

    logger.info(f"Reports and predictions saved to {results_path}")

def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    results_path = generate_results_path(
        config['results_root'],
        config['split_type'],
        config['range'],
        config['project_name'],
        config['start_year'],
        config['end_year'],
        config.get('start_month'),
        config.get('end_month'),
        config.get('start_day'),
        config.get('end_day')
    )

    model_save_path = results_path / "model"
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    # Initialize label mapper with the label set from config
    label_list = sorted(config.get('label_set', []))
    label_mapper = LabelMapper(label_set=label_list)

    if presaved_model_path := config.get('presaved_model_path'):
        logger.info(f"Using pre-saved model from {presaved_model_path}. Loading saved label mapping.")
        model_save_path = Path(presaved_model_path)
        # Load saved label mapping
        with open(model_save_path / "label_mapping.yaml", 'r') as f:
            mapping_data = yaml.safe_load(f)
            label_mapper = LabelMapper(label_set=mapping_data['label_to_id'].keys())
    else:
        # Load and train with consistent label mapping
        df_train_val = load_data(
            config['split_type'],
            config['range'],
            config['project_name'],
            config['start_year'],
            config['end_year'],
            label_mapper,
            config.get('start_month'),
            config.get('end_month'),
            config.get('start_day'),
            config.get('end_day')
        )

        train_model(
            df_train_val,
            results_path,
            model_save_path,
            config,
            label_mapper,
            use_validation=config.get('use_validation', True),
            split_size=config.get('split_size', 0.3)
        )

    # Load test data with the same label mapping
    df_test = load_data(
        config['split_type'],
        config['range'],
        config['project_name'],
        config['start_year'],
        config['end_year'],
        label_mapper,
        config.get('start_month'),
        config.get('end_month'),
        config.get('start_day'),
        config.get('end_day'),
        test=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    evaluate_model(model, tokenizer, df_test, results_path, label_mapper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model using a YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    main(args.config)