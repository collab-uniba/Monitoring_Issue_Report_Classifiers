import os
import argparse
import yaml
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
from pathlib import Path

# Define the CustomDataset class
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


def generate_results_path(root_path, split_type, range_val, project_name, start_year, end_year, start_month=None, end_month=None, start_day=None, end_day=None):
    path = Path(root_path) / f"{split_type}_range_{range_val}"
    if split_type == "year":
        path /= f"{start_year}-{end_year}"
    elif split_type == "month":
        path /= f"{start_year}-{start_month}_{end_year}-{end_month}"
    elif split_type == "day":
        path /= f"{start_year}-{start_month}-{start_day}_{end_year}-{end_month}-{end_day}"

    path /= project_name
    return path


def load_data(split_type, range_val, project_name, start_year, end_year, start_month=None, end_month=None, start_day=None, end_day=None, label_set=None, test=False):
    """
    Load data based on split type and range.

    Parameters:
        split_type (str): The split type ('year', 'month', or 'day').
        range_val (int): The range value for the split type.
        project_name (str): Name of the project directory.
        start_year, end_year (int): Start and end years for filtering.
        start_month, end_month, start_day, end_day (int, optional): Additional filters for months and days.
        label_set (set, optional): Labels to include in the dataset.
        test (bool): If True, load data after the end date for testing.

    Returns:
        pd.DataFrame: Combined data filtered by the specified range and labels.
    """
    def overlaps_year(file_start, file_end):
        return file_end >= start_year and file_start <= end_year

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
            return int(file.split('-')[0]), int(file.split('-')[1].split('.')[0])
        elif split_type == "month":
            start_year, start_month = map(int, file.split('-')[:2])
            end_year, end_month = map(int, file.split('_')[1].split('-')[:2])
            return (start_year, start_month), (end_year, end_month)
        elif split_type == "day":
            start_parts = list(map(int, file.split('-')[0:3]))
            end_parts = list(map(int, file.split('_')[1].split('-')))
            return tuple(start_parts), tuple(end_parts)

    # Define data directory
    data_dir = Path(f"data/windows/{split_type}_range_{range_val}/{project_name}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Initialize dataframe
    df_all = pd.DataFrame()

    # Process each file
    for file in data_dir.glob("*.csv"):
        file_name = file.name
        file_start, file_end = parse_filename(file_name, split_type)

        if test:
            if (split_type == "year" and file_start > end_year) or \
               (split_type == "month" and (file_start[0] > end_year or (file_start[0] == end_year and file_start[1] > end_month))) or \
               (split_type == "day" and (file_start[0] > end_year or (file_start[0] == end_year and (file_start[1] > end_month or (file_start[1] == end_month and file_start[2] > end_day))))):
                df = pd.read_csv(file)
                df_all = pd.concat([df_all, df], ignore_index=True)
        else:
            if (split_type == "year" and overlaps_year(file_start, file_end)) or \
               (split_type == "month" and overlaps_month(file_start, file_end)) or \
               (split_type == "day" and overlaps_day(file_start, file_end)):
                df = pd.read_csv(file)
                df_all = pd.concat([df_all, df], ignore_index=True)

    if df_all.empty:
        raise ValueError(f"No data found for the specified range in {data_dir}")

    # Data preprocessing
    df_all['date'] = pd.to_datetime(df_all['date'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%f+0000')
    if label_set is not None:
        df_all = df_all[df_all['label'].isin(label_set)]
    df_all['text'] = df_all['title'] + " " + df_all['body']
    df_all['labels'] = df_all['label'].map({label: i for i, label in enumerate(label_set)})

    return df_all


def train_model(df_train_val, results_path, model_save_path, config, use_validation=True, split_size=0.3):
    train_df, validation_df = train_test_split(df_train_val, test_size=split_size, random_state=42, stratify=df_train_val['labels'])

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    train_dataset = CustomDataset(train_df['text'].to_numpy(), train_df['labels'].to_numpy(), tokenizer)
    validation_dataset = None

    if use_validation:
        validation_dataset = CustomDataset(validation_df['text'].to_numpy(), validation_df['labels'].to_numpy(), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=len(df_train_val['labels'].unique()))

    training_args = TrainingArguments(
        output_dir=results_path,
        num_train_epochs=config['training_args']['num_train_epochs'],
        per_device_train_batch_size=config['training_args']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training_args']['per_device_eval_batch_size'],
        warmup_steps=config['training_args']['warmup_steps'],
        weight_decay=config['training_args']['weight_decay'],
        logging_dir=config['training_args']['logging_dir'],
        eval_strategy="epoch" if use_validation else "no",
        save_strategy="epoch",
        learning_rate=float(config['training_args']['learning_rate']),
        adam_epsilon=float(config['training_args']['adam_epsilon']),
        eval_steps=config['training_args']['eval_steps'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=lambda pred: classification_report(pred.label_ids, pred.predictions.argmax(-1), output_dict=True),
    )

    trainer.train()
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")


def evaluate_model(model, tokenizer, test_df, results_path):
    test_dataset = CustomDataset(test_df['text'].to_numpy(), test_df['labels'].to_numpy(), tokenizer)
    trainer = Trainer(model=model)

    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = test_df['labels'].to_numpy()

    # Generate classification report for each file
    file_reports = {}
    for file in test_df['file_name'].unique():
        file_df = test_df[test_df['file_name'] == file]
        file_labels = file_df['labels'].to_numpy()
        file_preds = preds[test_df['file_name'] == file]
        file_reports[file] = classification_report(file_labels, file_preds, output_dict=True)

    # Generate aggregated classification report
    aggregated_report = classification_report(labels, preds, output_dict=True)

    # Save reports and predictions
    with open(results_path / "file_reports.yaml", 'w') as f:
        yaml.dump(file_reports, f)

    with open(results_path / "aggregated_report.yaml", 'w') as f:
        yaml.dump(aggregated_report, f)

    test_df['predictions'] = preds
    test_df.to_csv(results_path / "predictions.csv", index=False)

    print(f"Reports and predictions saved to {results_path}")


# Main function
def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Generate results path
    results_path = generate_results_path(
        config['results_root'], config['split_type'], config['range'], config['project_name'],
        config['start_year'], config['end_year'],
        config.get('start_month'), config.get('end_month'),
        config.get('start_day'), config.get('end_day')
    )

    # Define model save path
    model_save_path = results_path / "model"

    # Create directories if they do not exist
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    # Load the label set from config or default to an empty set
    label_set = set(config.get('labels', []))

    # Load the training and validation data
    df_train_val = load_data(
        config['split_type'], config['range'], config['project_name'],
        config['start_year'], config['end_year'],
        config.get('start_month'), config.get('end_month'),
        config.get('start_day'), config.get('end_day'),
        label_set=label_set
    )

    # Train the model with optional validation and custom split size
    train_model(
        df_train_val,
        results_path,
        model_save_path,
        config,
        use_validation=config.get('use_validation', True),
        split_size=config.get('split_size', 0.3)
    )

    # Load the test data
    df_test = load_data(
        config['split_type'], config['range'], config['project_name'],
        config['start_year'], config['end_year'],
        config.get('start_month'), config.get('end_month'),
        config.get('start_day'), config.get('end_day'),
        label_set=label_set,
        test=True
    )

    # Evaluate the model on the test set
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    evaluate_model(model, tokenizer, df_test, results_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a classification model using a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    main(args.config)