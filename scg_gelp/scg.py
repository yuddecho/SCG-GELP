"""
SCG-Transformer synonymous codon generator.

Uses constrained beam search to generate multiple synonymous DNA sequences
from a single input protein sequence. The constraint enforces that each
decoded codon must translate to the expected amino acid at that position.
"""

import os
import torch

from tqdm import tqdm

from .models import (
    get_vocab,
    Seq2SeqTransformer,
    generate_square_subsequent_mask,
    CODON_TABLE,
)
import config


def translate_single_beam_search_decode(model, src, src_vocab, tgt_vocab,
                                         batch_size, beam_width, beam_size, device):
    """Constrained beam-search decoder for synonymous codon generation.

    The key constraint: at each decoding step i, the predicted codon must
    translate to the amino acid at position i in the source protein sequence.
    This guarantees the output DNA encodes the *same* protein.

    Args:
        model: trained Seq2SeqTransformer.
        src: raw protein sequence string (e.g. 'MKI...*').
        src_vocab: Vocab for protein (amino-acid) tokens.
        tgt_vocab: Vocab for DNA (codon) tokens.
        batch_size: max beam hypotheses processed in one forward pass.
        beam_width: top-k candidates kept at each decoding step.
        beam_size: total beam queue size (pruned after each step).
        device: torch device.

    Returns:
        list[str]: decoded DNA sequences (without <bos>/<eos> markers).
    """
    model.eval()
    model = model.to(device)

    # ---- Prepare source (protein) sequence ----
    src_tokens = ['<bos>'] + list(src) + ['<eos>']  # wrap with special tokens
    src_token_ids = src_vocab.tokens_to_ids(src_tokens)
    src_tensor = torch.tensor(src_token_ids).unsqueeze(0).to(device)  # (1, src_len)

    # Encoder mask: full bidirectional attention (all zeros)
    src_len = len(src_tokens)
    src_mask = torch.zeros((src_len, src_len), device=device).type(torch.bool)

    # Padding mask: True where src == <pad>
    pad_idx = src_vocab.token2id['<pad>']
    src_padding_mask = (src_tensor == pad_idx)

    # Encode the entire protein sequence once
    memory = model.encode(src_tensor, src_mask)  # (1, src_len, d_model)

    # ---- Initialize beam search ----
    tgt_bos = tgt_vocab.token2id['<bos>']
    tgt_eos = tgt_vocab.token2id['<eos>']
    tgt_start = torch.ones(1, 1).fill_(tgt_bos).type(torch.long).to(device)

    # Beam queue: list of (target_tensor, cumulative_log_prob, is_finished)
    beam_queue = [(tgt_start, 0.0, False)]

    # ---- Autoregressive decoding loop ----
    total_steps = len(src_tokens) - 1
    pbar = tqdm(total=total_steps, desc='Beam search')
    for i in range(1, len(src_tokens)):
        next_beam_queue = []
        seen_tgt = set()  # deduplicate within this step

        bq_len = len(beam_queue)

        # Process beam hypotheses in mini-batches
        for bq_start_i in range(0, bq_len, batch_size):
            bq_end_i = min(bq_start_i + batch_size, bq_len)
            sub_beam_queue = beam_queue[bq_start_i:bq_end_i]

            # Collect targets for this sub-batch
            tgt_list = [item[0] for item in sub_beam_queue]
            tgt_cat = torch.cat(tgt_list, dim=0).to(device)

            # Causal mask for current target length
            tgt_mask = generate_square_subsequent_mask(
                tgt_cat.size(-1), device).type(torch.bool)

            # Repeat memory for each hypothesis in the sub-batch
            memory_bs = memory.repeat(len(sub_beam_queue), 1, 1).to(device)

            # Decode and predict next-token logits
            out = model.decode(tgt_cat, memory_bs, tgt_mask)
            probs = model.generator(out[:, -1])  # (batch, tgt_vocab_size)

            # Extract scores and states for the current sub-batch
            scores = [item[1] for item in sub_beam_queue]
            states = [item[2] for item in sub_beam_queue]

            for tgt, prob, score, state in zip(tgt_list, probs, scores, states):
                # Top-k next tokens for this hypothesis
                topk_probs, topk_indices = torch.topk(prob, k=beam_width, dim=-1)

                for k in range(beam_width):
                    # Build candidate target sequence
                    # tgt is (1, seq_len) — squeeze to 1D for cat with scalar token
                    next_tgt = torch.cat(
                        [tgt.squeeze(0), topk_indices[k].unsqueeze(0)], dim=-1)
                    next_tgt = next_tgt.unsqueeze(0)  # (1, seq_len+1)

                    # Deduplicate: skip if this exact prefix was already seen
                    if next_tgt in seen_tgt:
                        continue

                    # Cumulative log-probability (sum of log-probs)
                    next_score = score + topk_probs[k].item()

                    end_char_id = topk_indices[k].item()

                    # ---- Constraint check: codon must match source amino acid ----
                    if end_char_id != tgt_eos:
                        nucle_char = tgt_vocab.id2token[str(end_char_id)]
                        if nucle_char in ['<unk>', '<bos>', '<eos>', '<pad>']:
                            acid_char = nucle_char
                        else:
                            acid_char = CODON_TABLE[nucle_char]
                        next_state = False
                    else:
                        acid_char = '<eos>'
                        next_state = True

                    # Enforce synonymous constraint
                    if acid_char != src_tokens[i]:
                        continue

                    # Accept candidate
                    seen_tgt.add(next_tgt)
                    next_beam_queue.append((next_tgt, next_score, next_state))

                    # Prune beam queue to top `beam_size` by cumulative score
                    if len(next_beam_queue) > beam_size:
                        next_beam_queue = sorted(
                            next_beam_queue, key=lambda x: x[1], reverse=True)
                        next_beam_queue = next_beam_queue[:beam_size]

        # Update active beam for next step
        beam_queue = next_beam_queue
        pbar.update(1)

        # Early stopping: all beams have either reached <eos> or max length
        all_done = True
        for (tgt, score, state) in beam_queue:
            if not (state is True or tgt.size(-1) >= len(src_tokens)):
                all_done = False
                break
        if all_done:
            pbar.update(pbar.total - pbar.n)  # fill to 100%
            break

    pbar.close()

    # ---- Convert final beam hypotheses to DNA strings ----
    res = []
    for (tgt, score, state) in beam_queue:
        tgt_ids = tgt.tolist()[0]
        tgt_tokens = tgt_vocab.ids_to_tokens(tgt_ids)
        tgt_str = ''.join(tgt_tokens)
        # Strip special tokens
        tgt_str = tgt_str.replace("<bos>", "").replace("<eos>", "")
        res.append(tgt_str)

    return res


def SCG_Transformer_predict(protein_seq, beam_batch_sizes, beam_widths, beam_sizes):
    """Run the SCG-Transformer to generate synonymous codon DNA sequences.

    Loads the pretrained checkpoint from config.SCG_TRANSFORMER_PT and runs
    constrained beam search.

    Args:
        protein_seq: amino-acid sequence string (e.g. 'MKI...*').
        beam_batch_sizes: list of batch sizes for each beam-search round.
        beam_widths: list of beam widths for each round.
        beam_sizes: list of beam sizes for each round.

    Returns:
        list[str]: all generated DNA sequences (deduplicated across rounds).
    """
    # Hyper-parameters — must match the training configuration
    d_model = 512
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    max_length = 1024
    PAD_IDX = 1

    # Validate checkpoint exists
    checkpoint_path = config.SCG_TRANSFORMER_PT
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'SCG-Transformer checkpoint not found: {checkpoint_path}\n'
            f'Download the weight file from GitHub Releases and place it in weights/'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[SCG] Device: {device}')
    print(f'[SCG] Checkpoint: {checkpoint_path}')

    # Load vocabularies (shipped inside the scg_gelp package)
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    protein_vocab_path = os.path.join(_pkg_dir, 'ref_vocab_protein.json')
    dna_vocab_path = os.path.join(_pkg_dir, 'ref_vocab_dna.json')
    protein_vocab, dna_vocab = get_vocab(
        protein_vocab_path, dna_vocab_path, log_info=False)

    # Build the model with the same architecture used during training
    model = Seq2SeqTransformer(
        len(protein_vocab), len(dna_vocab),
        emb_size=d_model, nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout,
        max_length=max_length, pad_idx=PAD_IDX
    )

    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'[SCG] Loaded weights from epoch {checkpoint.get("epoch", "?")}, '
          f'loss {checkpoint.get("loss", "?"):.4f}')

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[SCG] Trainable parameters: {n_params:,}')

    # Run beam search with each parameter combination
    dnas_set = set()
    print(f'[SCG] Beam search: batch_sizes={beam_batch_sizes}, '
          f'widths={beam_widths}, sizes={beam_sizes}')
    for beam_batch_size, beam_width, beam_size in zip(
            beam_batch_sizes, beam_widths, beam_sizes):
        dnas = translate_single_beam_search_decode(
            model, protein_seq, protein_vocab, dna_vocab,
            beam_batch_size, beam_width, beam_size, device)
        for d in dnas:
            dnas_set.add(d)

    dnas = list(dnas_set)
    print(f'[SCG] Generated {len(dnas)} unique DNA sequences')
    return dnas
