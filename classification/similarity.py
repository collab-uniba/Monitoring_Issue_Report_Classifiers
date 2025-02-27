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

# Reuse functions from classify.py
from classify import load_data, generate_results_path, LabelMapper

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_text(text):
    """
    Preprocess text data to ensure it's in the correct format for encoding.
    """
    return "" if pd.isna(text) or text is None else str(text).strip()

def calculate_average_similarity(train_embeddings, test_embeddings, batch_size=1000):
    """
    Calculate average cosine similarity between two sets of embeddings using batching.
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
        batch_avg = torch.mean(similarities, dim=1)
        total_similarity += torch.sum(batch_avg).item()
        total_comparisons += len(test_batch)
    
    return total_similarity / total_comparisons

def calculate_similarity_distribution(train_embeddings, test_embeddings, batch_size=1000):
    """
    Calculate distribution metrics (percentiles) of cosine similarity.
    Returns a dictionary with percentiles.
    """
    all_similarities = []

    if not isinstance(train_embeddings, torch.Tensor):
        train_embeddings = torch.tensor(train_embeddings)
    if not isinstance(test_embeddings, torch.Tensor):
        test_embeddings = torch.tensor(test_embeddings)

    for i in range(0, len(test_embeddings), batch_size):
        test_batch = test_embeddings[i:i + batch_size]
        similarities = util.pytorch_cos_sim(test_batch, train_embeddings)
        # For each test embedding, get the max similarity to any training embedding
        max_similarities, _ = torch.max(similarities, dim=1)
        all_similarities.extend(max_similarities.cpu().numpy())

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

def get_test_periods(data_dir, split_type, end_cutoff):
    """
    Get all available test periods from the data directory that come strictly after the end_cutoff.
    For monthly splits, ensures the end_cutoff month itself is excluded.
    """
    # Log training cutoff information
    if split_type == "year":
        logger.info(f"Training data cutoff: Year {end_cutoff}")
    elif split_type == "month":
        logger.info(f"Training data cutoff: Year {end_cutoff[0]}, Month {end_cutoff[1]}")
    elif split_type == "day":
        logger.info(f"Training data cutoff: Year {end_cutoff[0]}, Month {end_cutoff[1]}, Day {end_cutoff[2]}")
    
    test_periods = []
    
    for file_path in sorted(data_dir.glob("*.csv")):
        file_name = file_path.stem  # Get filename without extension
        
        if split_type == "year":
            start_year = int(file_name.split('-')[0])
            if start_year > end_cutoff:
                test_periods.append(start_year)
        
        elif split_type == "month":
            start_part = file_name.split('_')[0]
            start_year, start_month = map(int, start_part.split('-'))
            # Strictly after the end cutoff - if end_cutoff is (2005, 12), start with 2006-01
            if start_year > end_cutoff[0] or (start_year == end_cutoff[0] and start_month > end_cutoff[1]):
                test_periods.append((start_year, start_month))
        
        elif split_type == "day":
            start_part = file_name.split('_')[0]
            start_year, start_month, start_day = map(int, start_part.split('-'))
            if (start_year > end_cutoff[0] or 
                (start_year == end_cutoff[0] and start_month > end_cutoff[1]) or
                (start_year == end_cutoff[0] and start_month == end_cutoff[1] and start_day > end_cutoff[2])):
                test_periods.append((start_year, start_month, start_day))
    
    # Log test periods information
    logger.info(f"Found {len(test_periods)} test periods after the training cutoff")
    if test_periods:
        if split_type == "year":
            logger.info(f"Test periods: Years {min(test_periods)} to {max(test_periods)}")
        elif split_type == "month":
            start_year, start_month = min(test_periods)
            end_year, end_month = max(test_periods)
            logger.info(f"Test periods: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        elif split_type == "day":
            start_year, start_month, start_day = min(test_periods)
            end_year, end_month, end_day = max(test_periods)
            logger.info(f"Test periods: {start_year}-{start_month:02d}-{start_day:02d} to {end_year}-{end_month:02d}-{end_day:02d}")
    else:
        logger.warning("No test periods found after the cutoff date")
    
    return sorted(test_periods)

def analyze_similarity(config_file, include_distribution=False):
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

    similarity_path = results_path / "similarity_analysis"
    os.makedirs(similarity_path, exist_ok=True)

    label_mapper = LabelMapper(label_set=config.get('label_set', []))

    # Load training data
    train_df = load_data(
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

    # Ensure text column exists
    train_df['text'] = train_df.apply(
        lambda row: f"{preprocess_text(row.get('title', ''))} {preprocess_text(row.get('body', ''))}".strip(),
        axis=1
    )

    logger.info("Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    logger.info("Computing embeddings for training data...")
    train_embeddings = compute_embeddings(train_df['text'].tolist(), model)

    # Get data directory and end cutoff based on split type
    data_dir = Path(f"data/windows/{config['split_type']}_range_{config['range']}/{config['project_name']}")

    if config['split_type'] == 'year':
        end_cutoff = config['end_year']
    elif config['split_type'] == 'month':
        end_cutoff = (config['end_year'], config['end_month'])
    else:  # day
        end_cutoff = (config['end_year'], config['end_month'], config['end_day'])

    # Get all available test periods
    test_periods = get_test_periods(data_dir, config['split_type'], end_cutoff)
    logger.info(f"Found {len(test_periods)} test periods after the training cutoff")

    similarity_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_embeddings = train_embeddings.to(device)

    for period in test_periods:
        if config['split_type'] == 'year':
            test_df = load_data(
                config['split_type'],
                config['range'],
                config['project_name'],
                period,
                period,
                label_mapper,
                test=True,
                exact_date=True
            )
            period_str = str(period)
        elif config['split_type'] == 'month':
            year, month = period
            test_df = load_data(
                config['split_type'],
                config['range'],
                config['project_name'],
                year,
                year,
                label_mapper,
                start_month=month,
                end_month=month,
                test=True,
                exact_date=True
            )
            period_str = f"{year}-{month:02d}"
        else:  # day
            year, month, day = period
            test_df = load_data(
                config['split_type'],
                config['range'],
                config['project_name'],
                year,
                year,
                label_mapper,
                start_month=month,
                end_month=month,
                start_day=day,
                end_day=day,
                test=True,
                exact_date=True
            )
            period_str = f"{year}-{month:02d}-{day:02d}"

        if not test_df.empty:
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
            result['similarity'] = calculate_average_similarity(train_embeddings, test_embeddings)
            
            # Optionally calculate distribution metrics
            if include_distribution:
                logger.info(f"Calculating similarity distribution for test period {period_str}...")
                dist_metrics = calculate_similarity_distribution(train_embeddings, test_embeddings)
                for k, v in dist_metrics.items():
                    result[f'dist_{k}'] = v
            
            similarity_results.append(result)

            if torch.cuda.is_available():
                del test_embeddings
                torch.cuda.empty_cache()

    # Create DataFrame
    results_df = pd.DataFrame(similarity_results)
    results_df.to_csv(similarity_path / "similarity_scores.csv", index=False)

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
    plt.savefig(similarity_path / "cosine_similarity_trend.png")
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
            plt.savefig(similarity_path / "similarity_distribution.png")
            plt.close()

    logger.info(f"Results saved to {similarity_path}")
    logger.info(f"Similarity results saved to {similarity_path / 'similarity_scores.csv'}")

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
    args = parser.parse_args()
    
    analyze_similarity(args.config, args.include_distribution)