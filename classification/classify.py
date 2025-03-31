# classification/classify.py
import os
import argparse
import logging
from pathlib import Path

from config_manager import ConfigManager
from data_handlers import DataHandler, TimeWindow
from model_manager import ModelManager
from label_mapper import LabelMapper

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_results_path(config):
    """
    Generate results path based on configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path to results directory
    """
    model_type = config.get('model_type', 'roberta').lower()
    split_type = config['split_type']
    range_val = config['range']
    project_name = config['project_name']
    
    base_path = Path(config.get('results_root', 'results'))
    date_folder = f"{config['start_year']}-{config.get('start_month', '01')}_" \
                  f"{config['end_year']}-{config.get('end_month', '12')}"
    
    results_path = base_path / project_name / model_type / f"{split_type}_range_{range_val}" / date_folder
    os.makedirs(results_path, exist_ok=True)
    
    return results_path

def main(config_file):
    # Load and validate configuration
    config_manager = ConfigManager(config_file)
    config = config_manager.get_config()
    
    # Initialize components
    label_mapper = LabelMapper(config.get('label_set', []))
    data_handler = DataHandler(Path(f"data/windows/{config['split_type']}_range_{config['range']}/{config['project_name']}"))
    model_manager = ModelManager(label_mapper)
    
    # Generate results path
    results_path = generate_results_path(config)
    model_save_path = results_path / "model"
    os.makedirs(model_save_path, exist_ok=True)
    
    # Check for pre-saved model
    if config.get('presaved_model_path'):
        logger.info(f"Using pre-saved model from {config['presaved_model_path']}")
        model = model_manager.create_model(config) 
    else:
        # Load training data
        df_train_val = data_handler.load_data(
            config['split_type'],
            config['range'],
            config['project_name'],
            config['start_year'],
            config['end_year'],
            label_mapper,
            config.get('start_month'),
            config.get('end_month')
        )

        # Prepare dataset
        train_dataset, validation_dataset = data_handler.prepare_dataset(
            df_train_val, 
            label_mapper,
            tokenizer=None,  # Tokenizer will be created in model training
            use_validation=config.get('use_validation', True),
            split_size=config.get('split_size', 0.3)
        )

        # If the model is SetFit, sample a subset of the training data
        if config.get('model_type', 'roberta').lower() == 'setfit':
            train_dataset = data_handler.sample_training_data(train_dataset)

        # Train model
        _, model = model_manager.train_model(
            train_dataset, 
            validation_dataset, 
            config, 
            results_path
        )

        # Save model
        model_manager.save_model(
            model, 
            None,  # Tokenizer (will be None for SetFit)
            model_save_path, 
            config.get('model_type', 'roberta').lower()
        )

    # Load test data
    df_test = data_handler.load_data(
        config['split_type'],
        config['range'],
        config['project_name'],
        config['start_year'],
        config['end_year'],
        label_mapper,
        config.get('start_month'),
        config.get('end_month'),
        test=True
    )

    # Evaluate model
    model_manager.evaluate_model(
        model, 
        df_test, 
        results_path, 
        config.get('model_type', 'roberta').lower()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate classification models")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    main(args.config)