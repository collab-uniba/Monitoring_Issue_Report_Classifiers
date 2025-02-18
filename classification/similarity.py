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

# Reuse the load_data and generate_results_path functions from classify.py
from classify import load_data, generate_results_path, LabelMapper

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def calculate_average_similarity(train_embeddings, test_embeddings, batch_size=1000):
    """
    Calculate average cosine similarity between two sets of embeddings using batching
    and optimized sentence-transformers utilities.
    """
    total_similarity = 0
    total_comparisons = 0
    
    # Convert embeddings to PyTorch tensors if they aren't already
    if not isinstance(train_embeddings, torch.Tensor):
        train_embeddings = torch.tensor(train_embeddings)
    if not isinstance(test_embeddings, torch.Tensor):
        test_embeddings = torch.tensor(test_embeddings)
    
    # Process in batches to handle memory efficiently
    for i in range(0, len(test_embeddings), batch_size):
        test_batch = test_embeddings[i:i + batch_size]
        
        # Use sentence-transformers' optimized cosine similarity function
        similarities = util.pytorch_cos_sim(test_batch, train_embeddings)
        
        # Calculate mean similarity for each test example
        batch_avg = torch.mean(similarities, dim=1)
        total_similarity += torch.sum(batch_avg).item()
        total_comparisons += len(test_batch)
    
    return total_similarity / total_comparisons

def compute_embeddings(texts, model, batch_size=32):
    """
    Compute embeddings for a list of texts using batching.
    Returns embeddings as PyTorch tensors.
    """
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

def analyze_similarity(config_file):
    # Load configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize paths and label mapper
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
        config.get('start_month'),
        config.get('end_month'),
        config.get('start_day'),
        config.get('end_day')
    )
    
    # Initialize sentence transformer model
    logger.info("Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute embeddings for training data
    logger.info("Computing embeddings for training data...")
    train_embeddings = compute_embeddings(train_df['text'].tolist(), model)
    
    # Initialize results storage
    similarity_results = []
    
    # Calculate the range of test periods based on split type
    if config['split_type'] == 'year':
        test_periods = range(config['end_year'] + 1, config['end_year'] + 6)  # Test next 5 years
        period_format = lambda x: str(x)
    elif config['split_type'] == 'month':
        test_periods = pd.date_range(
            start=f"{config['end_year']}-{config['end_month']}",
            periods=12,
            freq='M'
        )
        period_format = lambda x: f"{x.year}-{x.month:02d}"
    elif config['split_type'] == 'day':
        test_periods = pd.date_range(
            start=f"{config['end_year']}-{config['end_month']}-{config['end_day']}",
            periods=30,
            freq='D'
        )
        period_format = lambda x: f"{x.year}-{x.month:02d}-{x.day:02d}"
    
    # Calculate similarity for each test period
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
        elif config['split_type'] == 'month':
            test_df = load_data(
                config['split_type'],
                config['range'],
                config['project_name'],
                period.year,
                period.year,
                period.month,
                period.month,
                label_mapper,
                test=True
            )
        else:  # day
            test_df = load_data(
                config['split_type'],
                config['range'],
                config['project_name'],
                period.year,
                period.year,
                period.month,
                period.month,
                period.day,
                period.day,
                label_mapper,
                test=True
            )
        
        if not test_df.empty:
            logger.info(f"Computing embeddings for test period {period_format(period)}...")
            test_embeddings = compute_embeddings(test_df['text'].tolist(), model).to(device)
            
            logger.info(f"Calculating similarity for test period {period_format(period)}...")
            avg_similarity = calculate_average_similarity(train_embeddings, test_embeddings)
            
            similarity_results.append({
                'period': period_format(period),
                'similarity': avg_similarity,
                'num_samples': len(test_df)
            })
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(similarity_results)
    results_df.to_csv(similarity_path / "similarity_scores.csv", index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['period'], results_df['similarity'], marker='o')
    plt.title(f"Average Similarity Over Time\n{config['project_name']} - {config['split_type']} split")
    plt.xlabel(config['split_type'].capitalize())
    plt.ylabel("Average Cosine Similarity")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
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