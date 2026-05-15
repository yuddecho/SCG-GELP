"""
SCG-GELP — Synonymous Codon Generator for Gene Expression Level Prediction.

A toolkit for synonymous codon optimization that, given a protein sequence:
  1. Generates multiple synonymous DNA sequences via constrained beam search
     with the SCG-Transformer.
  2. Extracts 768-dim embeddings for each DNA sequence using DNABERT-2.
  3. Predicts soluble expression probability using an ensemble of sklearn models.
  4. Ranks all candidates by predicted expression level.
"""

from .pipeline import run, get_ranking
from .models import (
    Vocab,
    Seq2SeqTransformer,
    PositionalEncoding,
    CODON_TABLE,
    translate_dna_to_protein,
    seed_everything,
)
from .scg import SCG_Transformer_predict
from .dnabert2 import DNABERT_2_Embedding_Func

__all__ = [
    'run',
    'get_ranking',
    'SCG_Transformer_predict',
    'DNABERT_2_Embedding_Func',
]
