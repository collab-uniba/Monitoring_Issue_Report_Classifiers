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

def generate_results_path(root_path, split_type, range_val, project_name, start_year, end_year, 
                         start_month=None, end_month=None, start_day=None, end_day=None):
    path = Path(root_path) / f"{split_type}_range_{range_val}"
    if split_type == "year":
        path /= f"{start_year}-{end_year}"
    elif split_type == "month":
        path /= f"{start_year}-{start_month}_{end_year}-{end_month}"
    elif split_type == "day":
        path /= f"{start_year}-{start_month}-{start_day}_{end_year}-{end_month}-{end_day}"

    path /= project_name
    return path

def load_data(split_type, range_val, project_name, start_year, end_year, label_mapper, 
              start_month=None, end_month=None, start_day=None, end_day=None, test=False):
    """
    Load data based on split type and range with consistent label mapping.
    """
    def overlaps_year(file_start, file_end):
        return not (file_end < start_year or file_start > end_year)

    def overlaps_month(file_start, file_end):
        return not (
            file_end[0] < start_year or file_start[0] > end_year or
            (file_end[0] == start_year and file_end[1] < start_month) or
            (file_start[0] == end_year and file_start[1] > end_month)
        )

    def overlaps_day(file_start, file_end):
        return not (
            file_end[0] < start_year or file_start[0] > end_year or
            (file_end[0] == start_year and (file_end[1], file_end[2]) < (start_month, start_day)) or
            (file_start[0] == end_year and (file_start[1], file_start[2]) > (end_month, end_day))
        )

    def parse_filename(file, split_type):
        if split_type == "year":
            start_year, end_year = map(int, file.replace('.csv', '').split('-'))
            return start_year, end_year
        
        elif split_type == "month":
            start_part, end_part = file.replace('.csv', '').split('_')
            start_year, start_month = map(int, start_part.split('-'))
            end_year, end_month = map(int, end_part.split('-'))
            return (start_year, start_month), (end_year, end_month)
        
        elif split_type == "day":
            start_part, end_part = file.replace('.csv', '').split('_')
            start_year, start_month, start_day = map(int, start_part.split('-'))
            end_year, end_month, end_day = map(int, end_part.split('-'))
            return (start_year, start_month, start_day), (end_year, end_month, end_day)
        
        else:
            raise ValueError("Invalid split_type. Must be 'year', 'month', or 'day'")

    data_dir = Path(f"data/windows/{split_type}_range_{range_val}/{project_name}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    df_all = pd.DataFrame()
    file_names = []

    for file in data_dir.glob("*.csv"):
        file_name = file.name
        file_start, file_end = parse_filename(file_name, split_type)

        if test:
            if (split_type == "year" and file_start > end_year) or \
               (split_type == "month" and (file_start[0] > end_year or (file_start[0] == end_year and file_start[1] > end_month))) or \
               (split_type == "day" and (file_start[0] > end_year or (file_start[0] == end_year and (file_start[1] > end_month or (file_start[1] == end_month and file_start[2] > end_day))))):
                df = pd.read_csv(file)
                df['file_name'] = file_name
                df_all = pd.concat([df_all, df], ignore_index=True)
                file_names.append(file_name)
        else:
            if (split_type == "year" and overlaps_year(file_start, file_end)) or \
               (split_type == "month" and overlaps_month(file_start, file_end)) or \
               (split_type == "day" and overlaps_day(file_start, file_end)):
                df = pd.read_csv(file)
                df['file_name'] = file_name
                df_all = pd.concat([df_all, df], ignore_index=True)
                file_names.append(file_name)

    if df_all.empty:
        raise ValueError(f"No data found for the specified range in {data_dir}")

    if test:
        logger.info(f"Files used for testing: {file_names}")
    else:
        logger.info(f"Files used for training: {file_names}")

       # Data preprocessing with consistent label mapping
    df_all['date'] = pd.to_datetime(df_all['date'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%f+0000')
    if label_mapper.label_to_id:
        df_all = df_all[df_all['label'].isin(label_mapper.label_to_id.keys())]
    df_all['text'] = df_all['title'] + " " + df_all['body']
    
    # Ensure labels are properly mapped to integers
    df_all['labels'] = label_mapper.map_labels(df_all['label']).astype(int)
    
    # Add validation check
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