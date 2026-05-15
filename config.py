"""
SCG-GELP configuration.

All paths are relative to the project root (the directory containing this file).
Model weight files are distributed via GitHub Releases; download them and place
them in the weights/ directory before running.
"""
import os

# ---------------------------------------------------------------------------
# Project root — auto-detected from this fileʼs location
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model weight paths (large files — download from GitHub Releases)
# ---------------------------------------------------------------------------
# SCG-Transformer checkpoint (~148 MB)
SCG_TRANSFORMER_PT = os.path.join(PROJECT_ROOT, 'weights', 'scg_checkpoint.pt')

# DNABERT-2 fine-tuned checkpoint directory (~447 MB, HuggingFace format)
DNABERT2_CHECKPOINT = os.path.join(PROJECT_ROOT, 'dnabert2')

# Directory containing individually trained best sklearn models (~21 MB total)
BEST_MODELS_DIR = os.path.join(PROJECT_ROOT, 'weights', 'sk_best_models')

# ---------------------------------------------------------------------------
# Execution configuration
# ---------------------------------------------------------------------------
# Whether to apply MinMaxScaler to DNABERT-2 embeddings before prediction.
# Set to False when models were trained on unscaled embeddings (default).
APPLY_DNABERT_SCALER = False

# MinMaxScaler fitted on the NESG training-set DNABERT-2 embeddings
# NESG_DNABERT_SCALER = os.path.join(PROJECT_ROOT, 'weights', 'nesg_dnabert_scaler.pickle')
NESG_DNABERT_SCALER = ''


# Default sklearn models to use for ensemble prediction.
# Set to None to auto-detect all *_best_model.pickle files in BEST_MODELS_DIR.
#
# Available algorithms (14 total):
#   AdaBoost  - AdaBoost
#   Bagging   - Bagging ensemble
#   DT        - Decision Tree
#   KNN       - K-Nearest Neighbors
#   LDA       - Linear Discriminant Analysis
#   LR        - Logistic Regression
#   MLP       - Multi-Layer Perceptron
#   NB        - Naive Bayes
#   QDA       - Quadratic Discriminant Analysis
#   RF        - Random Forest
#   SGD       - Stochastic Gradient Descent
#   SVM       - Support Vector Machine
#   XGBoost   - Extreme Gradient Boosting
#   GB        - Gradient Boosting
DEFAULT_SELECTED_MODELS = ['SVM', 'LR', 'MLP']

# Beam-search hyper-parameters for the SCG-Transformer generator.
# Each list must have the same length; corresponding entries are paired as
# (batch_size, width, size) for successive beam-search rounds.

# test
# BEAM_BATCH_SIZES = [4]
# BEAM_WIDTHS = [5]
# BEAM_SIZES = [80]

BEAM_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
BEAM_WIDTHS = [2, 2, 3, 4, 5, 5]
BEAM_SIZES = [val * 20 for val in BEAM_BATCH_SIZES]

# Toggle individual pipeline stages on/off.
EXEC_FUNC = {
    'Exec SCG Model': True,
    'Exec DNABERT-2 Model': True,
    'Exec Sklearn model': True
}
