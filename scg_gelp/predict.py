"""
Sklearn ensemble prediction module.

Loads multiple individually trained best models (SVM, LR, MLP, RF, XGBoost,
etc.) and performs ensemble prediction on DNABERT-2 embeddings to rank
sequences by their predicted soluble expression probability.
"""

import os
import glob
import pickle

import config


def Sklearn_model(dnabert_2_dataset, selected_models=None):
    """Run ensemble prediction using DNABERT-2 embeddings.

    Each model in the ensemble was trained independently on the same training
    set but with different algorithms/hyperparameters. The final ranking is
    produced per-model; aggregation (averaging ranks/scores across models) is
    handled downstream in the pipeline.

    Args:
        dnabert_2_dataset: path to DNABERT-2 embedding pickle file.
        selected_models: list of model names to use (e.g. ['SVM', 'LR']).
                         If None, auto-detects all *_best_model.pickle files
                         in config.BEST_MODELS_DIR.

    Returns:
        dict: {model_name: {sequence_name: [probability, rank]}}
              where probability is rounded to 3 decimal places and rank is
              1-indexed (1 = highest expression probability).
    """
    # ---- Load DNABERT-2 embeddings ----
    with open(dnabert_2_dataset, 'rb') as file:
        dnabert_embedding_data = pickle.load(file)

    # Separate names and feature vectors
    names, seq_embedding_info = [], []
    for gene_name in dnabert_embedding_data:
        data = dnabert_embedding_data[gene_name]
        names.append(gene_name)
        seq_embedding_info.append(data['embedding'])

    # ---- Auto-detect available models if not specified ----
    if selected_models is None:
        model_files = glob.glob(f'{config.BEST_MODELS_DIR}/*_best_model.pickle')
        selected_models = sorted([
            os.path.basename(f).replace('_best_model.pickle', '')
            for f in model_files
        ])

    print(f'[Sklearn] Using {len(selected_models)} models: {selected_models}')

    # ---- Predict with each model ----
    res = {}
    for model_name in selected_models:
        model_path = os.path.join(config.BEST_MODELS_DIR,
                                   f'{model_name}_best_model.pickle')
        if not os.path.exists(model_path):
            print(f'[Warning] Model not found: {model_path}, skipping.')
            continue

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f'  - {model_name}')

        # predict_proba returns (n_samples, 2) array: [P(class 0), P(class 1)]
        y_ppba = model.predict_proba(seq_embedding_info)
        y_score = y_ppba[:, 1]  # probability of high-expression class

        # Build {name: score} mapping
        name_score = {}
        for output, name in zip(y_score, names):
            name_score[name] = output

        # Sort by score descending, then assign ranks (1 = best)
        name_score = sorted(name_score.items(), key=lambda x: x[1], reverse=True)

        name_score_ranked = {}
        for i, (k, v) in enumerate(name_score, start=1):
            name_score_ranked[k] = [round(float(v), 3), i]

        res[model_name] = name_score_ranked

    return res
