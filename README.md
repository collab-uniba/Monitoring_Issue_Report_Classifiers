# Monitoring Issue Report Classifiers

## Project Overview

This project facilitates the classification of issue reports into Bugs and Enhancements using the RoBERTa model, leveraging the Hugging Face transformers library. It includes scripts for data extraction from MongoDB, preparation of distribution tables with temporal windows, and a detailed Jupyter Notebook for training and evaluating the classification model.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25181517/183423507-c056a6f9-1ba8-4312-a350-19bcbc5a8697.png" alt="Python Logo" width="50" height="50"/>
  <img src="https://github.com/mongodb/mongo/raw/master/docs/leaf.svg" alt="MongoDB Logo" width="50" height="50"/>
  <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="PyTorch Logo" width="50" height="50"/>
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Transformers Logo" width="50" />
</p>

## Getting Started

### Prerequisites

- MongoDB with access to the necessary databases.
- Python 3.x and Jupyter Notebook or JupyterLab installed.
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `torch`, `transformers`, `scikit-learn`, `accelerate`.

### Installation

1. Clone the repository or download the ZIP and extract its contents.
2. Ensure MongoDB is running and accessible.
3. Install the Python dependencies listed above, preferably in a virtual environment:
```
   pip install -r requirements.txt
```

### Execution Workflow

To correctly use this project and generate the desired outputs, follow the steps in the order provided:

1. **Standard Extraction**:
    - Navigate to `SCRIPTS/STANDARD_EXTRACTION`.
    - Run `StandardStarter.py` to begin the standard extraction process. This script extracts Jira Repos from MongoDB and prepares them for further analysis.

2. **Generate Distribution Tables with Temporal Windows**:
    - Run `DistributionTableWindows.py` located in `SCRIPTS/DISTRIBUTION`. This script generates distribution tables considering different temporal windows, essential for the subsequent analysis.

3. **Model Training and Evaluation**:
    - Open and execute the `NOTEBOOKS/train-test.ipynb` Jupyter Notebook for training the RoBERTa model. This notebook includes steps for:
        - Data loading and preparation.
        - Model configuration and training.
        - Evaluation of the model's performance in classifying issue reports.

## Key Components

- `SCRIPTS/`: Contains Python scripts for data extraction and preparation.
- `NOTEBOOKS/`: Includes the `train-test.ipynb` Jupyter Notebook for model training and evaluation.
- `CSV/`: Directory for CSV files used in model training, as referenced in the notebook.


### Additional Scripts

The project also includes various utility scripts for data cleaning, label mapping, and removing duplicated rows. These are located in `SCRIPTS/TOOLS` and `SCRIPTS/ID-MAP-EXTRACTION/processes`. These scripts support the main extraction and analysis processes and may be used as needed.

## Outputs

- Extracted and processed data ready for model training.
- A trained RoBERTa model capable of classifying issue reports into bugs and enhancements.
- Evaluation metrics and visualizations to evaluate model performance.

#

<p align="center">
Author: Simone Le Noci <br><br>
<a href="https://github.com/SimoneNuts"><img src="https://github.com/SimoneNuts.png?" width="100px"/></a>
</p>