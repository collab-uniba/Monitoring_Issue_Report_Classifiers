# classification/model_manager.py
import os
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, label_mapper):
        """
        Initialize ModelManager with label mapping.
        
        Args:
            label_mapper: Utility for mapping labels
        """
        self.label_mapper = label_mapper

    def create_model(self, config):
        """
        Create model based on configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Initialized model
        """
        model_type = config.get('model_type', 'roberta').lower()
        
        if model_type == 'setfit':
            return self._create_setfit_model(config)
        else:
            return self._create_transformer_model(config)

    def _create_setfit_model(self, config):
        """Create SetFit model."""
        if config.get('presaved_model_path'):
            return SetFitModel.from_pretrained(
                Path(config['presaved_model_path']),
                num_classes=self.label_mapper.num_labels
            )
        # Default to using a sentence transformer model
        sentence_transformer_model = config.get(
            'sentence_transformer_model', 
            'all-MiniLM-L6-v2'
        )
        
        return SetFitModel.from_pretrained(
            sentence_transformer_model,
            num_classes=self.label_mapper.num_labels
        )
    

    def _create_transformer_model(self, config):
        """Create Transformer-based classification model."""
        if config.get('presaved_model_path'):
            return AutoModelForSequenceClassification.from_pretrained(
                Path(config['presaved_model_path']),
                num_labels=self.label_mapper.num_labels
            )
        return AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=self.label_mapper.num_labels
        )

    def train_model(self, train_dataset, validation_dataset, config, results_path):
        """
        Train model with flexible configuration.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            config: Training configuration
            results_path: Path to save results
        
        Returns:
            Trained model and trainer
        """
        model_type = config.get('model_type', 'roberta').lower()
        
        if model_type == 'setfit':
            return self._train_setfit(train_dataset, validation_dataset, config, results_path)
        else:
            return self._train_transformer(train_dataset, validation_dataset, config, results_path)

    def _train_setfit(self, train_dataset, validation_dataset, config, results_path):
        """Train SetFit model."""
        batch_size = config.get('training_args', {}).get('per_device_train_batch_size', 16)
        num_iterations = config.get('setfit_args', {}).get('num_iterations', 20)
        num_epochs = config.get('training_args', {}).get('num_train_epochs', 1)

        # Convert dataframe to Dataset
        if isinstance(train_dataset, pd.DataFrame):
            train_dataset = Dataset.from_pandas(train_dataset)
        if validation_dataset and isinstance(validation_dataset, pd.DataFrame):
            validation_dataset = Dataset.from_pandas(validation_dataset)

        logger.info(f"Column names in train dataset: {train_dataset.column_names}")
        
        model = self.create_model(config)
        
        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            batch_size=batch_size,
            num_iterations=num_iterations,
            num_epochs=num_epochs
        )
        
        trainer.train()
        return trainer, model

    def _train_transformer(self, train_dataset, validation_dataset, config, results_path):
        """Train Transformer-based model."""
        model = self.create_model(config)
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(
                examples['text'], 
                padding='max_length', 
                truncation=True, 
                max_length=config['max_length']
            )
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        if validation_dataset:
            validation_dataset = validation_dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=results_path,
            num_train_epochs=config['training_args']['num_train_epochs'],
            per_device_train_batch_size=config['training_args']['per_device_train_batch_size'],
            per_device_eval_batch_size=config['training_args']['per_device_eval_batch_size'],
            warmup_steps=config['training_args']['warmup_steps'],
            weight_decay=config['training_args']['weight_decay'],
            logging_dir=config['training_args']['logging_dir'],
            eval_strategy="epoch" if validation_dataset else "no",
            save_strategy="no",
            learning_rate=float(config['training_args']['learning_rate']),
            adam_epsilon=float(config['training_args']['adam_epsilon']),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
        )

        trainer.train()
        return trainer, model

    def save_model(self, model, tokenizer, model_save_path, model_type):
        """
        Save model and associated metadata.
        
        Args:
            model: Trained model
            tokenizer: Model tokenizer (optional)
            model_save_path: Path to save model
            model_type: Type of model (setfit or transformer)
        """
        if model_type == 'setfit':
            model.save_pretrained(model_save_path)
        else:
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)

        # Save label mapping
        label_mapping_path = Path(model_save_path) / "label_mapping.yaml"
        with open(label_mapping_path, 'w') as f:
            yaml.dump({
                'label_to_id': self.label_mapper.label_to_id,
                'id_to_label': self.label_mapper.id_to_label,
                'model_type': model_type
            }, f)

    def evaluate_model(self, model, test_df, results_path, model_type='transformer'):
        """
        Evaluate model on test dataset.
        
        Args:
            model: Trained model
            test_df: Test DataFrame
            results_path: Path to save results
            model_type: Type of model
        
        Returns:
            Evaluation reports and predictions
        """
        if model_type == 'setfit':
            pred_labels, true_labels = self._evaluate_setfit(model, test_df)
        else:
            pred_labels, true_labels = self._evaluate_transformer(model, test_df)

        # Generate classification reports
        file_reports = self._generate_file_reports(test_df, true_labels, pred_labels)
        aggregated_report = classification_report(true_labels, pred_labels, output_dict=True)

        # Save reports
        with open(results_path / "file_reports.yaml", 'w') as f:
            yaml.dump(file_reports, f)

        with open(results_path / "aggregated_report.yaml", 'w') as f:
            yaml.dump(aggregated_report, f)

        # Save predictions
        test_df['predicted_label'] = pred_labels
        test_df.to_csv(results_path / "predictions.csv", index=False)

        return file_reports, aggregated_report

    def _evaluate_setfit(self, model, test_df):
        """Evaluate SetFit model."""
        test_df['text'] = test_df['text'].fillna("")
        texts = test_df['text'].tolist()
        predictions = model.predict(texts)
        
        pred_labels = [self.label_mapper.inverse_map([pred.item()])[0] for pred in predictions]
        true_labels = self.label_mapper.inverse_map(test_df['label'].to_numpy())
        
        return pred_labels, true_labels

    def _evaluate_transformer(self, model, test_df):
        """Evaluate Transformer model."""
        trainer = Trainer(model=model)
        predictions = trainer.predict(test_df)
        
        preds = predictions.predictions.argmax(-1)
        labels = test_df['labels'].to_numpy()

        pred_labels = self.label_mapper.inverse_map(preds)
        true_labels = self.label_mapper.inverse_map(labels)
        
        return pred_labels, true_labels

    def _generate_file_reports(self, test_df, true_labels, pred_labels):
        """Generate classification reports for each file."""
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
        return file_reports