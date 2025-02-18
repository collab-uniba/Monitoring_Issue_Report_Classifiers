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
    if pd.isna(text) or text is None:
        return ""
    return str(text).strip()

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

def compute_embeddings(texts, model, batch_size=32):
    """
    Compute embeddings for a list of texts using batching.
    """
    processed_texts = [preprocess_text(text) for text in texts]
    empty_texts = sum(1 for text in processed_texts if not text)
    if empty_texts > 0:
        logger.warning(f"Found {empty_texts} empty texts after preprocessing")
    
    return model.encode(processed_texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

def get_test_periods(data_dir, split_type, end_cutoff):
    """
    Get all available test periods from the data directory that come after the end_cutoff.
    """
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
            if (start_year, start_month) > end_cutoff:
                test_periods.append((start_year, start_month))
        
        elif split_type == "day":
            start_part = file_name.split('_')[0]
            start_year, start_month, start_day = map(int, start_part.split('-'))
            if (start_year, start_month, start_day) > end_cutoff:
                test_periods.append((start_year, start_month, start_day))
    
    return sorted(test_periods)

def analyze_similarity(config_file):
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
    if 'text' not in train_df.columns:
        train_df['text'] = train_df.apply(
            lambda row: f"{preprocess_text(row.get('title', ''))} {preprocess_text(row.get('body', ''))}",
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
                test=True
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
                test=True
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
                test=True
            )
            period_str = f"{year}-{month:02d}-{day:02d}"
        
        if not test_df.empty:
            if 'text' not in test_df.columns:
                test_df['text'] = test_df.apply(
                    lambda row: f"{preprocess_text(row.get('title', ''))} {preprocess_text(row.get('body', ''))}",
                    axis=1
                )
            
            logger.info(f"Computing embeddings for test period {period_str}...")
            test_embeddings = compute_embeddings(test_df['text'].tolist(), model).to(device)
            
            logger.info(f"Calculating similarity for test period {period_str}...")
            avg_similarity = calculate_average_similarity(train_embeddings, test_embeddings)
            
            similarity_results.append({
                'period': period_str,
                'similarity': avg_similarity,
                'num_samples': len(test_df)
            })
            
            if torch.cuda.is_available():
                del test_embeddings
                torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(similarity_results)
    results_df.to_csv(similarity_path / "similarity_scores.csv", index=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['period'], results_df['similarity'], marker='o')
    plt.title(f"Average Similarity Over Time\n{config['project_name']} - {config['split_type']} split")
    plt.xlabel(config['split_type'].capitalize())
    plt.ylabel("Average Cosine Similarity")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(similarity_path / "similarity_trend.png")
    logger.info(f"Results saved to {similarity_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze similarity between train and test sets.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    
    analyze_similarity(args.config)