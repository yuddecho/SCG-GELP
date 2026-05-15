"""
SCG-GELP main pipeline orchestration.

Integrates three stages:
  1. SCG-Transformer  → synonymous codon DNA generation
  2. DNABERT-2        → 768-dim embedding extraction
  3. Sklearn Ensemble → soluble-expression probability prediction

The pipeline produces ranked DNA candidates sorted by predicted expression level.
"""

import csv
import os
import pickle

import config
from .scg import SCG_Transformer_predict
from .dnabert2 import DNABERT_2_Embedding_Func
from .predict import Sklearn_model


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run(protein_name, protein_seq, reference_dnas=None, output_dir=None):
    """Run the full SCG-GELP pipeline on a single protein sequence.

    Args:
        protein_name: protein identifier (used for output file naming).
        protein_seq: amino-acid sequence string, must end with '*'.
        reference_dnas: optional dict {name: dna_seq} of reference sequences
                        to include alongside generated candidates.
        output_dir: directory for output files (default: {protein_name}-scg-gelp-res/).

    Returns:
        dict: {model_name: {sequence_name: [probability, rank]}}
    """
    # ---- Validate protein sequence ----
    protein_seq = _validate_protein_seq(protein_seq)

    # ---- Defaults ----
    if reference_dnas is None:
        reference_dnas = {}
    if output_dir is None:
        output_dir = os.path.join('data', 'example', 'outputs',
                                   f'{protein_name}-scg-gelp-res')

    os.makedirs(output_dir, exist_ok=True)

    # ---- Output file paths ----
    dna_file = os.path.join(output_dir, f'{protein_name}_scg_dnas.csv')
    embedding_pickle = os.path.join(
        output_dir, f'{protein_name}_scg_dnas_dnabert_embedding.pickle')
    predict_pickle = os.path.join(output_dir, f'{protein_name}_predict_res.pickle')
    result_csv = os.path.join(output_dir, f'{protein_name}_predict_res.csv')

    # =======================================================================
    # Stage 1: Generate synonymous codon DNA sequences via SCG-Transformer
    # =======================================================================
    dnas = []
    if config.EXEC_FUNC.get('Exec SCG Model', True):
        dnas = SCG_Transformer_predict(
            protein_seq,
            config.BEAM_BATCH_SIZES,
            config.BEAM_WIDTHS,
            config.BEAM_SIZES
        )

    # Always write DNA CSV (SCG candidates + reference genes),
    # so that reference sequences can be evaluated even when SCG is disabled.
    n_ref = len(reference_dnas)
    with open(dna_file, 'w', encoding='utf-8') as w:
        w.write('name,nucle-seq\n')
        for i, dna in enumerate(dnas):
            w.write(f'{protein_name}_scg_{i},{dna}\n')
        for k in reference_dnas:
            w.write(f'{k},{reference_dnas[k]}\n')

    print(f'[Pipeline] {len(dnas)} SCG + {n_ref} reference -> {dna_file}\n')

    # =======================================================================
    # Stage 2: Extract DNABERT-2 embeddings
    # =======================================================================
    if config.EXEC_FUNC.get('Exec DNABERT-2 Model', True):
        dna_dnabert_embedding = DNABERT_2_Embedding_Func(
            dna_file, embedding_pickle)
        print(f'[Pipeline] Embeddings extracted for '
              f'{len(dna_dnabert_embedding)} sequences\n')

    # =======================================================================
    # Stage 3: Predict expression probability via sklearn ensemble
    # =======================================================================
    if config.EXEC_FUNC.get('Exec Sklearn model', True):
        res = Sklearn_model(embedding_pickle,
                            selected_models=config.DEFAULT_SELECTED_MODELS)
    else:
        res = {}

    # ---- Persist results ----
    with open(predict_pickle, 'wb') as w:
        pickle.dump(res, w)

    _export_ranking_csv(res, dna_file, result_csv)

    print(f'[Pipeline] Results saved to {result_csv}')
    return res


# ===========================================================================
# Result export
# ===========================================================================

def _validate_protein_seq(seq):
    """Validate and normalize a protein amino-acid sequence.

    - Ensures the sequence ends with '*' (stop codon).
    - Ensures all letters are uppercase.

    Returns the normalized sequence.
    """
    if not seq.endswith('*'):
        print('[Validate] Sequence missing stop codon — appending "*"')
        seq = seq + '*'

    # Check for lowercase letters (valid amino-acid chars + *)
    lower_found = any(c.islower() for c in seq)
    if lower_found:
        print('[Validate] Sequence contains lowercase letters — converting to uppercase')
        seq = seq.upper()

    return seq


def _read_dna_map(dna_file):
    """Read the DNA candidates CSV and return a {name: sequence} mapping."""
    dna_map = {}
    with open(dna_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                dna_map[row[0].strip()] = row[1].strip()
    return dna_map


def _export_ranking_csv(res, dna_file, result_csv):
    """Export prediction results as a ranked CSV file with DNA sequences.

    Columns: name, sequence, {model} Acc, {model} Rank, ...,
             Total Acc, Total Rank
    """
    if not res:
        return

    # Load DNA sequences from Stage 1 output
    dna_map = _read_dna_map(dna_file)

    model_names = list(res.keys())
    seq_names = list(res[model_names[0]].keys())

    # Build rows with computed averages
    rows = []
    for seq_name in seq_names:
        row_str = ''
        acc_sum, rank_sum = 0, 0

        for model_name in model_names:
            prob, rank = res[model_name][seq_name]
            row_str += f'{prob},{rank},'
            acc_sum += prob
            rank_sum += rank

        avg_acc = round(acc_sum / len(model_names), 3)
        avg_rank = round(rank_sum / len(model_names), 3)
        seq = dna_map.get(seq_name, '')
        rows.append((avg_rank, f'{seq_name},{row_str}{avg_acc},{avg_rank},{seq}\n'))

    # Sort by total rank ascending (best first)
    rows.sort(key=lambda x: x[0])

    with open(result_csv, 'w', encoding='utf-8') as w:
        # Header row
        header_cols = ''
        for model_name in model_names:
            header_cols += f'{model_name} acc,{model_name} rank,'
        w.write(f'name,{header_cols}total acc,total rank,sequence\n')

        for _, row in rows:
            w.write(row)


# ===========================================================================
# Result query helpers
# ===========================================================================

def get_ranking(res, model_id=0):
    """Extract a sorted ranking from prediction results.

    Args:
        res: dict returned by pipeline.run().
        model_id: which model's ranking to use (0 = first model).

    Returns:
        list[tuple]: [(sequence_name, probability, rank), ...]
                     sorted by rank ascending (best first).
    """
    model_names = list(res.keys())
    seq_names = list(res[model_names[model_id]].keys())

    ranking = []
    for seq_name in seq_names:
        prob, rank = res[model_names[model_id]][seq_name]
        ranking.append((seq_name, prob, rank))

    ranking.sort(key=lambda x: x[2])
    return ranking
