"""
DNABERT-2 transfer-learning feature extraction.

Uses a pre-trained (or fine-tuned) DNABERT-2 model to extract 768-dimensional
embedding vectors from DNA sequences. These embeddings serve as input features
for the downstream sklearn ensemble predictors.
"""

import os
import csv
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import config


# ===========================================================================
# Dataset — reads CSV files with columns: name, nucle-seq[, label]
# ===========================================================================

class SeqenceDataset(Dataset):
    """PyTorch Dataset for DNA sequences stored in CSV format.

    Expected CSV columns:
        name       — sequence identifier
        nucle-seq  — DNA nucleotide string (A/T/C/G)
        label      — (optional) numerical label; defaults to -1 if absent
    """

    def __init__(self, data_file: str):
        super(SeqenceDataset, self).__init__()
        with open(data_file, "r") as f:
            data = list(csv.reader(f))[1:]  # skip header row

        self.names = [d[0] for d in data]           # sequence IDs
        self.texts = [d[1] for d in data]            # DNA strings
        self.labels = [d[2] if len(d) > 2 else -1   # labels (or -1)
                       for d in data]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self.names[i], self.texts[i], self.labels[i]


# ===========================================================================
# Collation — tokenizes and pads a batch for DNABERT-2
# ===========================================================================

def collate_fn(batch_samples, tokenizer):
    """Custom collate function: tokenize DNA strings and batch them together.

    Returns:
        batch_names: list of sequence identifiers.
        X: tokenized inputs dict (input_ids, attention_mask).
        batch_labels: list of integer labels.
    """
    batch_names, batch_texts, batch_labels = [], [], []
    for sample in batch_samples:
        batch_names.append(sample[0])
        batch_texts.append(sample[1])
        batch_labels.append(int(sample[2]))

    # Tokenize with padding and truncation
    X = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False  # single-sequence, no need for type ids
    )
    return batch_names, X, batch_labels


# ===========================================================================
# Embedding extraction loop
# ===========================================================================

def dnabert_embedding(model, dataloader, scaler, embedding_data_file, device):
    """Extract per-sequence DNABERT-2 embeddings.

    For each sequence, the 768-dim hidden states from the last layer are
    averaged over all non-special tokens (positions 1 to seq_len-1, excluding
    [CLS] and [SEP]) to produce a fixed-size representation.

    Args:
        model: DNABERT-2 HuggingFace model.
        dataloader: DataLoader yielding (names, tokenized_X, labels).
        scaler: fitted sklearn Scaler or None.
        embedding_data_file: path to save the resulting pickle.
        device: torch device.

    Returns:
        dict: {sequence_name: {'embedding': ndarray(768,), 'label': int}}
    """
    embedding_data = {}

    bar = tqdm(total=len(dataloader), desc='DNABERT-2 embedding')

    for gene_names, batch_X, labels in dataloader:
        bar.update(1)

        # Effective length of each sequence (excluding padding)
        batch_lens = [attention_mask.sum().item()
                      for attention_mask in batch_X['attention_mask']]

        inputs = batch_X['input_ids'].to(device)

        with torch.no_grad():
            # hidden_states: last_hidden_state of shape (batch, seq_len, 768)
            hidden_states = model(inputs)[0]

        # Average pooling over real tokens (skip [CLS] at position 0 and
        # [SEP] / padding at the end)
        for i, seq_len in enumerate(batch_lens):
            rep = hidden_states[i, 1:seq_len - 1].mean(0).cpu().numpy()
            embedding_data[gene_names[i]] = {
                'embedding': rep,
                'label': labels[i]
            }

    bar.close()

    # Optionally apply a pre-fitted scaler (e.g. MinMaxScaler)
    if scaler is not None:
        for k in embedding_data:
            embedding_data[k]['embedding'] = scaler.transform(
                [embedding_data[k]['embedding']])[0]

    # Persist embeddings to disk for reuse
    with open(embedding_data_file, 'wb') as w:
        pickle.dump(embedding_data, w)

    return embedding_data


# ===========================================================================
# Public API
# ===========================================================================

def DNABERT_2_Embedding_Func(dna_file, dna_dnabert_embedding_pickle):
    """Load DNABERT-2 and extract embeddings for all sequences in a CSV file.

    Args:
        dna_file: path to input CSV (columns: name, nucle-seq).
        dna_dnabert_embedding_pickle: path to output pickle file.

    Returns:
        dict: {sequence_name: {'embedding': ndarray(768,), 'label': int}}
    """
    checkpoint = config.DNABERT2_CHECKPOINT
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f'DNABERT-2 checkpoint not found: {checkpoint}\n'
            f'Download from HuggingFace or GitHub Releases and place in weights/'
        )

    print(f'[DNABERT-2] Checkpoint: {checkpoint}')

    # DNABERT-2 was trained with a maximum sequence length of 636 tokens
    model_max_length = 636

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, model_max_length=model_max_length, trust_remote_code=True)
    dna_bert_2 = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[DNABERT-2] Device: {device}')

    dna_bert_2 = dna_bert_2.to(device)
    dna_bert_2.eval()

    batch_size = 16
    dataset = SeqenceDataset(dna_file)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=lambda x: collate_fn(x, tokenizer))

    # Load scaler only if the downstream models were trained on scaled features
    scaler = None
    if config.APPLY_DNABERT_SCALER:
        with open(config.NESG_DNABERT_SCALER, 'rb') as w:
            nesg_dnabert_scaler = pickle.load(w)
        scaler = nesg_dnabert_scaler['TT Scaler']

    return dnabert_embedding(dna_bert_2, dataloader, scaler,
                              dna_dnabert_embedding_pickle, device)
