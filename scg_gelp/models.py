"""
Core model components for SCG-GELP.

Contains:
- Vocab: amino-acid / nucleotide sequence tokenizer
- PositionalEncoding: sinusoidal position embeddings for the Transformer
- Seq2SeqTransformer: protein-to-DNA synonymous codon generator
- Genetic code table (codon -> amino acid) and translation utilities
- Mask generation helpers for autoregressive decoding
"""

import os
import math
import random
import collections
import json

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ===========================================================================
# Reproducibility
# ===========================================================================

def seed_everything(seed=42):
    """Set random seeds across all libraries for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ===========================================================================
# Genetic code — standard codon -> amino-acid mapping
# ===========================================================================

CODON_TABLE = {
    # Isoleucine
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I',
    # Methionine (start)
    'ATG': 'M',
    # Threonine
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    # Asparagine
    'AAC': 'N', 'AAT': 'N',
    # Lysine
    'AAA': 'K', 'AAG': 'K',
    # Serine
    'AGC': 'S', 'AGT': 'S', 'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    # Arginine
    'AGA': 'R', 'AGG': 'R', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    # Leucine
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L', 'TTA': 'L', 'TTG': 'L',
    # Proline
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    # Histidine
    'CAC': 'H', 'CAT': 'H',
    # Glutamine
    'CAA': 'Q', 'CAG': 'Q',
    # Valine
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    # Alanine
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    # Aspartate
    'GAC': 'D', 'GAT': 'D',
    # Glutamate
    'GAA': 'E', 'GAG': 'E',
    # Glycine
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    # Phenylalanine
    'TTC': 'F', 'TTT': 'F',
    # Tyrosine
    'TAC': 'Y', 'TAT': 'Y',
    # Stop codons
    'TAA': '*', 'TAG': '*', 'TGA': '*',
    # Cysteine
    'TGC': 'C', 'TGT': 'C',
    # Tryptophan
    'TGG': 'W',
}


def translate_dna_to_protein(dna_sequence):
    """Translate a DNA nucleotide sequence into a protein amino-acid sequence.

    Walks the DNA in 3-base steps (codons). Unknown codons are mapped to 'X'.
    """
    protein_sequence = ''
    for i in range(0, len(dna_sequence), 3):
        codon = dna_sequence[i:i + 3].upper()
        protein_sequence += CODON_TABLE.get(codon, 'X')
    return protein_sequence


# ===========================================================================
# Vocab — bidirectional token <-> id mapping for sequences
# ===========================================================================

class Vocab:
    """Vocabulary for amino-acid (single-letter) or DNA (3-mer codon) tokens.

    Special tokens are always indexed as:
        UNK_IDX = 0   (<unk>)
        PAD_IDX = 1   (<pad>)
        BOS_IDX = 2   (<bos>)
        EOS_IDX = 3   (<eos>)

    The remaining tokens are sorted by descending frequency in the training set.
    """

    def __init__(self):
        self.token2id = {}    # str -> int
        self.id2token = {}    # int -> str
        self.count = 0        # next available id
        self.unk_token = '<unk>'

    def init(self, seqs, token_type,
             special_tokens=['<unk>', '<pad>', '<bos>', '<eos>']):
        """Build vocabulary from a list of sequences.

        Args:
            seqs: list of raw sequences (strings).
            token_type: 'dna' (split into 3-mer codons) or 'protein' (per-char).
            special_tokens: tokens to insert first (in order, before frequency sort).
        """
        self.token2id = {}
        self.id2token = {}
        self.count = 0

        # Reserve special tokens with fixed low ids
        if special_tokens:
            for token in special_tokens:
                self.add_token(token)

        # Tokenize based on type
        if token_type == 'dna':
            # Split each sequence into non-overlapping 3-base codons
            tokens = [seq[i:i + 3] for seq in seqs for i in range(0, len(seq), 3)]
        elif token_type == 'protein':
            # Single-character amino-acid tokens
            tokens = [token for seq in seqs for token in seq]

        # Sort by frequency (most common first) so high-frequency tokens get low ids
        counter = collections.Counter(tokens)
        token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        for token, _ in token_freqs:
            if token not in self.token2id:
                self.token2id[token] = self.count
                self.id2token[self.count] = token
                self.count += 1

    def add_token(self, token):
        """Add a single token if not already present."""
        if token not in self.token2id:
            self.token2id[token] = self.count
            self.id2token[self.count] = token
            self.count += 1

    def tokens_to_ids(self, tokens):
        """Convert a list of tokens to their integer ids (unknown -> 0)."""
        return [self.token2id.get(token, 0) for token in tokens]

    def ids_to_tokens(self, ids):
        """Convert a list of integer ids back to tokens (unknown id -> '<unk>')."""
        return [self.id2token.get(str(_id), self.unk_token) for _id in ids]

    def save(self, vocab_path):
        """Persist vocabulary to a JSON file."""
        res = {'token2id': self.token2id, 'id2token': self.id2token}
        with open(vocab_path, 'w', encoding='utf-8') as json_file:
            json.dump(res, json_file)

    def load(self, vocab_path):
        """Load vocabulary from a JSON file."""
        with open(vocab_path, 'r') as json_file:
            loaded_data = json.load(json_file)
        self.token2id = loaded_data['token2id']
        self.id2token = loaded_data['id2token']

    def __len__(self):
        return len(self.token2id)

    def __str__(self):
        return str(list(self.token2id.items()))


def get_vocab(protein_vocab_path, dna_vocab_path, log_info=True):
    """Load protein and DNA vocabularies from disk.

    Raises FileNotFoundError if either file is missing.
    """
    if not os.path.exists(protein_vocab_path):
        raise FileNotFoundError(f'Protein vocab not found: {protein_vocab_path}')
    if not os.path.exists(dna_vocab_path):
        raise FileNotFoundError(f'DNA vocab not found: {dna_vocab_path}')

    protein_vocab, dna_vocab = Vocab(), Vocab()
    protein_vocab.load(protein_vocab_path)
    dna_vocab.load(dna_vocab_path)

    if log_info:
        print(f'  Protein vocab: {protein_vocab_path} ({len(protein_vocab)} tokens)')
        print(f'  DNA vocab:     {dna_vocab_path} ({len(dna_vocab)} tokens)')

    return protein_vocab, dna_vocab


# ===========================================================================
# PositionalEncoding — sinusoidal position information for Transformer
# ===========================================================================

class PositionalEncoding(nn.Module):
    """Inject sinusoidal positional information into token embeddings.

    Uses sine/cosine functions of different frequencies as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    The encoding matrix is registered as a non-trainable buffer so it is
    automatically moved to the correct device and saved with the model.
    """

    def __init__(self, d_model, dropout, max_len=5000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the full positional encoding matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model).to(device)

        # Position indices: [[0], [1], [2], ..., [max_len-1]]  shape (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)

        # Frequency term: 1 / (10000^(2i/d_model)) computed efficiently via log/exp
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # Even indices (0, 2, 4, ...) get sine
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices (1, 3, 5, ...) get cosine
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer: saved with state_dict but not updated by optimizer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Add positional encoding up to the current sequence length
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


# ===========================================================================
# Seq2SeqTransformer — the SCG-Transformer model
# ===========================================================================

class Seq2SeqTransformer(nn.Module):
    """Protein-to-DNA synonymous codon generator.

    Encoder-decoder Transformer that maps a protein amino-acid sequence to
    multiple synonymous DNA codon sequences via constrained beam search.

    Architecture:
        - Learned token embeddings (src + tgt)
        - Sinusoidal positional encoding
        - Stacked encoder / decoder Transformer layers
        - Linear projection head (generator) to target vocabulary
    """

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 emb_size: int,
                 nhead: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_length: int = 5000,
                 pad_idx: int = 1):
        super(Seq2SeqTransformer, self).__init__()

        # Token embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, emb_size,
                                          padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_size,
                                          padding_idx=pad_idx)

        # Shared positional encoding
        self.positional_encoding = PositionalEncoding(emb_size, dropout,
                                                      max_len=max_length)

        # Core Transformer (PyTorch built-in, batch_first=True)
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Linear projection from d_model to target vocabulary size
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        """Full forward pass (used during training)."""
        # Embed and add positional encoding
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # Transformer forward
        outs = self.transformer(src, tgt, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        """Encode the source (protein) sequence into memory.

        Used during inference: called once per input protein.
        """
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        return self.transformer.encoder(src, src_mask)

    def decode(self, tgt, memory, tgt_mask):
        """Decode one step given the encoder memory and current target prefix.

        Used during inference: called autoregressively in beam search.
        """
        tgt = self.tgt_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        return self.transformer.decoder(tgt, memory, tgt_mask)


# ===========================================================================
# Mask utilities for autoregressive Transformer decoding
# ===========================================================================

def generate_square_subsequent_mask(sz, device):
    """Create an upper-triangular mask for causal (autoregressive) attention.

    Position i can attend to positions 0..i (inclusive).
    Returns a float mask where -inf = disallowed, 0.0 = allowed.
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (mask
            .float()
            .masked_fill(mask == False, float('-inf'))
            .masked_fill(mask == True, float(0.0)))
    return mask


def create_mask(src, tgt, pad_idx, device):
    """Build all masks required for a single Transformer forward pass.

    Returns:
        src_mask: all-zero (encoder sees full source, no causal masking).
        tgt_mask: causal subsequent mask for autoregressive decoding.
        src_padding_mask: True where source tokens are <pad>.
        tgt_padding_mask: True where target tokens are <pad>.
    """
    src_seq_len = src.shape[-1]
    tgt_seq_len = tgt.shape[-1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    # Encoder can attend to all source positions (no causal restriction)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
