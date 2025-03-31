# classification/similarity.py  
import os
import argparse
import yaml
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from our modules instead of classify.py
from data_handlers import DataHandler, TimeWindow
from model_manager import ModelManager
from label_mapper import LabelMapper

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_text(text):
    """
    Preprocess text data to ensure it's in the correct format for encoding.
    """
    return "" if pd.isna(text) or text is None else str(text).strip()

def calculate_average_similarity(train_embeddings, test_embeddings, mode="mean", batch_size=1000):
    """
    Calculate average or max cosine similarity between two sets of embeddings using batching.
    The mode can be 'max' (default) or 'mean'.
    """
    total_similarity = 0
    total_comparisons = 0
    
    if not isinstance(train_embeddings, torch.Tensor):
        train_embeddings = torch.tensor(train_embeddings)
    if not isinstance(test_embeddings, torch.Tensor):
        test_embeddings = torch.tensor(test_embeddings)
    
    for i in range(0, len(test_embeddings), batch_size):
        test_batch = test_embeddings[i:i + batch_size]
        similarities = util.pytorch_cos_sim(test_batch, train_embeddings)
        
        if mode == "max":
            batch_similarity, _ = torch.max(similarities, dim=1)
        elif mode == "mean":
            batch_similarity = torch.mean(similarities, dim=1)
        else:
            raise ValueError("Invalid mode. Choose 'max' or 'mean'.")

        total_similarity += torch.sum(batch_similarity).item()
        total_comparisons += len(test_batch)
    
    return total_similarity / total_comparisons


def calculate_similarity_distribution(train_embeddings, test_embeddings, mode="max", batch_size=1000):
    """
    Calculate distribution metrics (percentiles) of cosine similarity.
    The mode can be 'max' (default) or 'mean'.
    """
    all_similarities = []

    if not isinstance(train_embeddings, torch.Tensor):
        train_embeddings = torch.tensor(train_embeddings)
    if not isinstance(test_embeddings, torch.Tensor):
        test_embeddings = torch.tensor(test_embeddings)

    for i in range(0, len(test_embeddings), batch_size):
        test_batch = test_embeddings[i:i + batch_size]
        similarities = util.pytorch_cos_sim(test_batch, train_embeddings)

        if mode == "max":
            selected_similarities, _ = torch.max(similarities, dim=1)
        elif mode == "mean":
            selected_similarities = torch.mean(similarities, dim=1)
        else:
            raise ValueError("Invalid mode. Choose 'max' or 'mean'.")

        all_similarities.extend(selected_similarities.cpu().numpy())

    return {
        'mean': np.mean(all_similarities),
        'std': np.std(all_similarities),
        'min': np.min(all_similarities),
        'p10': np.percentile(all_similarities, 10),
        'p25': np.percentile(all_similarities, 25),
        'median': np.percentile(all_similarities, 50),
        'p75': np.percentile(all_similarities, 75),
        'p90': np.percentile(all_similarities, 90),
        'max': np.max(all_similarities),
    }

def compute_embeddings(texts, model, batch_size=32):
    """
    Compute embeddings for a list of texts using batching.
    """
    processed_texts = [preprocess_text(text) for text in texts]
    # Only count as empty if the final processed text is empty
    empty_texts = sum(not text.strip() for text in processed_texts)
    if empty_texts > 0:
        logger.warning(f"Found {empty_texts} completely empty texts after preprocessing")

    return model.encode(processed_texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

def get_test_periods(data_handler, split_type, end_cutoff):
    """
    Get all available test periods from the data directory that come strictly after the end_cutoff.
    For monthly splits, ensures the end_cutoff month itself is excluded.
    Uses DataHandler to find the files.
    """
    # Log training cutoff information
    if split_type == "year":
        logger.info(f"Training data cutoff: Year {end_cutoff}")
    elif split_type == "month":
        logger.info(f"Training data cutoff: Year {end_cutoff.start_year}, Month {end_cutoff.start_month}")
    elif split_type == "day":
        logger.info(f"Training data cutoff: Year {end_cutoff.start_year}, Month {end_cutoff.start_month}, Day {end_cutoff.start_day}")
    
    test_periods = []
    
    # Use the file_cache from data_handler which was populated during data loading
    for filename, window in data_handler.file_cache.get(split_type, {}).items():
        if split_type == "year":
            if window.start_year > end_cutoff.end_year:
                test_periods.append(TimeWindow(start_year=window.start_year, end_year=window.end_year))
        
        elif split_type == "month":
            if (window.start_year > end_cutoff.end_year or 
                (window.start_year == end_cutoff.end_year and window.start_month > end_cutoff.end_month)):
                test_periods.append(TimeWindow(
                    start_year=window.start_year, 
                    end_year=window.end_year,
                    start_month=window.start_month, 
                    end_month=window.end_month
                ))
        
        elif split_type == "day":
            if (window.start_year > end_cutoff.end_year or 
                (window.start_year == end_cutoff.end_year and window.start_month > end_cutoff.end_month) or
                (window.start_year == end_cutoff.end_year and window.start_month == end_cutoff.end_month and 
                 window.start_day > end_cutoff.end_day)):
                test_periods.append(TimeWindow(
                    start_year=window.start_year, 
                    end_year=window.end_year,
                    start_month=window.start_month, 
                    end_month=window.end_month,
                    start_day=window.start_day, 
                    end_day=window.end_day
                ))
    
    # Sort test periods
    test_periods.sort()
    
    # Log test periods information
    logger.info(f"Found {len(test_periods)} test periods after the training cutoff")
    if test_periods:
        if split_type == "year":
            logger.info(f"Test periods: Years {test_periods[0].start_year} to {test_periods[-1].end_year}")
        elif split_type == "month":
            start = test_periods[0]
            end = test_periods[-1]
            logger.info(f"Test periods: {start.start_year}-{start.start_month:02d} to {end.end_year}-{end.end_month:02d}")
        elif split_type == "day":
            start = test_periods[0]
            end = test_periods[-1]
            logger.info(f"Test periods: {start.start_year}-{start.start_month:02d}-{start.start_day:02d} to "
                        f"{end.end_year}-{end.end_month:02d}-{end.end_day:02d}")
    else:
        logger.warning("No test periods found after the cutoff date")
    
    return test_periods

def generate_results_path(results_root, split_type, range_val, project_name, model_type,
                         start_year, end_year, start_month=None, end_month=None,
                         start_day=None, end_day=None):
    """
    Generate a path for storing results.
    """
    base_path = Path(results_root) / f"{split_type}_range_{range_val}" / project_name / model_type
    
    if split_type == "year":
        time_str = f"{start_year}_{end_year}"
    elif split_type == "month":
        time_str = f"{start_year}-{start_month:02d}_{end_year}-{end_month:02d}"
    else:  # day
        time_str = f"{start_year}-{start_month:02d}-{start_day:02d}_{end_year}-{end_month:02d}-{end_day:02d}"
    
    return base_path / time_str

def analyze_similarity(config_file, include_distribution=False, mode="max"):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Log configuration information
    logger.info("Analysis Configuration:")
    logger.info(f"Project: {config['project_name']}")
    logger.info(f"Split type: {config['split_type']}")
    logger.info(f"Including distribution metrics: {include_distribution}")
    logger.info("Training period:")
    logger.info(f"  Start: Year {config['start_year']}" + 
                (f", Month {config['start_month']}" if 'start_month' in config else "") +
                (f", Day {config['start_day']}" if 'start_day' in config else ""))
    logger.info(f"  End: Year {config['end_year']}" +
                (f", Month {config['end_month']}" if 'end_month' in config else "") +
                (f", Day {config['end_day']}" if 'end_day' in config else ""))

    model_type = config.get('model_type', 'roberta').lower()

    results_path = generate_results_path(
        config['results_root'],
        config['split_type'],
        config['range'],
        config['project_name'],
        model_type,
        config['start_year'],
        config['end_year'],
        config.get('start_month'),
        config.get('end_month'),
        config.get('start_day'),
        config.get('end_day')
    )

    similarity_path = results_path / "similarity_analysis"
    os.makedirs(similarity_path, exist_ok=True)

    label_mapper = LabelMapper(label_set=config.get('label_set', []))
    
    # Initialize DataHandler
    data_dir = Path(f"data/windows/{config['split_type']}_range_{config['range']}/{config['project_name']}")
    data_handler = DataHandler(data_dir)

    # Load training data using DataHandler
    df_train_val = data_handler.load_data(
        config['split_type'],
        config['range'],
        config['project_name'],
        config['start_year'],
        config['end_year'],
        label_mapper,
        start_month=config.get('start_month'),
        end_month=config.get('end_month'),
        start_day=config.get('start_day'),
        end_day=config.get('end_day')
    )

    train_df, _ = data_handler.prepare_dataset(
            df_train_val, 
            label_mapper,
            tokenizer=None,  # Tokenizer will be created in model training
            use_validation=config.get('use_validation', True),
            split_size=config.get('split_size', 0.3)
    )

    # If the model is SetFit, sample a subset of the training data
    if config.get('model_type', 'roberta').lower() == 'setfit':
        train_df = data_handler.sample_training_data(train_df)


    logger.info("Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    logger.info("Computing embeddings for training data...")
    train_embeddings = compute_embeddings(train_df['text'].tolist(), model)

    # Create TimeWindow for end cutoff
    end_cutoff = TimeWindow(
        start_year=config['end_year'],
        end_year=config['end_year'],
        start_month=config.get('end_month'),
        end_month=config.get('end_month'),
        start_day=config.get('end_day'),
        end_day=config.get('end_day')
    )

    # Get all available test periods using data_handler
    test_periods = get_test_periods(data_handler, config['split_type'], end_cutoff)
    logger.info(f"Found {len(test_periods)} test periods after the training cutoff")

    similarity_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_embeddings = train_embeddings.to(device)

    for period in test_periods:
        # Load test data using DataHandler
        test_df = data_handler.load_data(
            config['split_type'],
            config['range'],
            config['project_name'],
            period.start_year,
            period.end_year,
            label_mapper,
            start_month=period.start_month,
            end_month=period.end_month,
            start_day=period.start_day,
            end_day=period.end_day,
            test=True,
            exact_date=True
        )

        # Generate period string for display/logs
        if config['split_type'] == 'year':
            period_str = str(period.start_year)
        elif config['split_type'] == 'month':
            period_str = f"{period.start_year}-{period.start_month:02d}"
        else:  # day
            period_str = f"{period.start_year}-{period.start_month:02d}-{period.start_day:02d}"

        if not test_df.empty:
            # Ensure text column exists
            if 'text' not in test_df.columns:
                test_df['text'] = test_df.apply(
                    lambda row: f"{preprocess_text(row.get('title', ''))} {preprocess_text(row.get('body', ''))}".strip(),
                    axis=1
                )

            logger.info(f"Computing embeddings for test period {period_str}...")
            test_embeddings = compute_embeddings(test_df['text'].tolist(), model).to(device)

            # Initialize result dictionary
            result = {
                'period': period_str,
                'num_samples': len(test_df)
            }

            # Always calculate average cosine similarity
            logger.info(f"Calculating average cosine similarity for test period {period_str}...")
            result['similarity'] = calculate_average_similarity(train_embeddings, test_embeddings, mode=mode)

            # Optionally calculate distribution metrics
            if include_distribution:
                logger.info(f"Calculating similarity distribution for test period {period_str}...")
                dist_metrics = calculate_similarity_distribution(train_embeddings, test_embeddings, mode=mode)
                for k, v in dist_metrics.items():
                    result[f'dist_{k}'] = v

            similarity_results.append(result)

            if torch.cuda.is_available():
                del test_embeddings
                torch.cuda.empty_cache()

    # Create DataFrame
    results_df = pd.DataFrame(similarity_results)
    results_df.to_csv(similarity_path / f"similarity_scores_{mode}.csv", index=False)

    # Plot average similarity
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['period'], results_df['similarity'], marker='o')
    plt.title(f"Average Cosine Similarity Over Time\n{config['project_name']} - {config['split_type']} split")
    plt.xlabel(config['split_type'].capitalize())
    plt.ylabel("Average Cosine Similarity")

    # Set x-ticks appropriately
    if len(results_df) > 20:
        if config['split_type'] == "month":
            plt.xticks(results_df['period'][::12], rotation=45)
        elif config['split_type'] == "day":
            plt.xticks(results_df['period'][::30], rotation=45)
        else:
            plt.xticks(rotation=45)
    else:
        plt.xticks(rotation=45)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(similarity_path / f"cosine_similarity_trend_{mode}.png")
    plt.close()

    # Plot distribution metrics if requested
    if include_distribution:
        dist_cols = [col for col in results_df.columns if col.startswith('dist_') and col != 'dist_std']
        if dist_cols:
            plt.figure(figsize=(12, 8))
            for col in dist_cols:
                plt.plot(results_df['period'], results_df[col], marker='o', label=col.replace('dist_', ''))

            plt.title(f"Similarity Distribution Metrics\n{config['project_name']} - {config['split_type']} split")
            plt.xlabel(config['split_type'].capitalize())
            plt.ylabel("Similarity Score")
            plt.legend()

            if len(results_df) > 20:
                if config['split_type'] == "month":
                    plt.xticks(results_df['period'][::12], rotation=45)
                elif config['split_type'] == "day":
                    plt.xticks(results_df['period'][::30], rotation=45)
                else:
                    plt.xticks(rotation=45)
            else:
                plt.xticks(rotation=45)

            plt.grid(True)
            plt.tight_layout()
            plt.savefig(similarity_path / f"similarity_distribution_{mode}.png")
            plt.close()

    logger.info(f"Results saved to {similarity_path}")
    logger.info(f"Similarity results saved to {similarity_path / f'similarity_scores_{mode}.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze similarity between train and test sets.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--include-distribution",
        action="store_true",
        help="Include similarity distribution metrics in addition to average similarity."
    )
    parser.add_argument(
        "--mode",
        choices=["max", "mean"],
        default="max",
        help="Select whether to use max or mean similarity."
    )
    args = parser.parse_args()
    
    analyze_similarity(args.config, args.include_distribution, args.mode)