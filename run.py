#!/usr/bin/env python3
"""
SCG-GELP command-line entry point.

Usage (single protein):
    python run.py --protein_name MK --protein_seq "MKVWLVG...*"

Usage (batch from CSV):
    python run.py --input data/example/proteins.csv

    CSV format:  protein_name,protein_sequence
                MRI,MRIINVKD...*
                MKV,MKVWLVG...*

Usage (include reference sequences for comparison):
    python run.py --protein_name MK --protein_seq "MKVWLVG...*" --reference refs.csv

    Reference CSV format:  protein_name,gene_name,gene_sequence
                           MK,MK-WT,ATGCGTATC...
                           MK,MK-GS,ATGAAAGTT...

For Colab usage, set the HF_HOME environment variable to a writable directory
before importing:

    import os
    os.environ['HF_HOME'] = '/tmp/hf-cache'
    from scg_gelp import run
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

# Set HuggingFace cache directory — prefer home, fall back to /tmp
_cache_home = os.path.expanduser('~')
if not os.path.isdir(_cache_home) or not os.access(_cache_home, os.W_OK):
    _cache_home = '/tmp'
os.environ.setdefault('HF_HOME', os.path.join(_cache_home, '.cache', 'huggingface'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_cache_home, '.cache', 'matplotlib'))

from scg_gelp import run, get_ranking


def run_single(protein_name, protein_seq, output_dir, reference_dnas=None):
    """Process a single protein and print ranking results."""
    res = run(
        protein_name=protein_name,
        protein_seq=protein_seq,
        reference_dnas=reference_dnas,
        output_dir=output_dir,
    )
    _print_results(res, protein_name)
    return res


def run_batch(input_file, output_dir, reference_map=None):
    """Process all proteins listed in a CSV file.

    CSV columns: ``protein_name, protein_sequence`` (or ``name, seq``)

    Args:
        input_file: path to the input CSV.
        output_dir: base output directory.
        reference_map: optional dict {protein_name: {gene_name: dna_seq}}.
    """
    proteins = _read_input_csv(input_file)

    if not proteins:
        print(f'[Error] No proteins found in {input_file}')
        sys.exit(1)

    if reference_map:
        n_refs = sum(len(v) for v in reference_map.values())
        print(f'Batch mode: {len(proteins)} protein(s) from {input_file}, '
              f'+ {n_refs} reference gene(s)\n')
    else:
        print(f'Batch mode: {len(proteins)} protein(s) from {input_file}\n')

    all_results = {}
    for i, (name, seq) in enumerate(proteins, 1):
        refs = reference_map.get(name, {}) if reference_map else {}
        ref_info = f' + {len(refs)} references' if refs else ''

        print(f'\n{"=" * 60}')
        print(f'[{i}/{len(proteins)}] Processing: {name}{ref_info}')
        print(f'{"=" * 60}')

        try:
            res = run_single(name, seq, output_dir, refs if refs else None)
            all_results[name] = res
        except Exception as e:
            print(f'[Error] Failed to process {name}: {e}')
            continue

    _print_batch_summary(all_results)
    return all_results


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def _read_input_csv(input_file):
    """Parse an input CSV with columns ``protein_name, protein_sequence``
    (or ``name, seq``).

    Automatically skips a header row.
    """
    proteins = []
    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue
            if first and row[0].strip().lower() in ('name', 'protein_name'):
                first = False
                continue
            first = False
            if len(row) >= 2:
                proteins.append((row[0].strip(), row[1].strip()))
    return proteins


def _read_reference_csv(reference_file):
    """Parse a reference gene CSV.

    Columns: ``protein_name, gene_name, gene_sequence``

    Returns:
        dict: {protein_name: {gene_name: dna_sequence}}
    """
    ref_map = defaultdict(dict)
    with open(reference_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue
            # Auto-detect and skip header
            if first and row[0].strip().lower() == 'protein_name':
                first = False
                continue
            first = False
            if len(row) >= 3:
                protein = row[0].strip()
                gene = row[1].strip()
                seq = row[2].strip()
                ref_map[protein][gene] = seq
    return dict(ref_map)


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def _print_results(res, protein_name):
    """Print ranked prediction results for a single protein.

    Shows top 10 SCG-generated candidates and all reference gene results.
    """
    ranking = get_ranking(res)

    # Split into SCG-generated and reference sequences
    scg_prefix = f'{protein_name}_scg_'
    scg_seqs = [(n, p, r) for n, p, r in ranking if n.startswith(scg_prefix)]
    ref_seqs = [(n, p, r) for n, p, r in ranking if not n.startswith(scg_prefix)]

    print('\n' + '-' * 60)
    print('Prediction Ranking Results')
    print('-' * 60)

    # Print top 10 SCG candidates
    top_n = min(10, len(scg_seqs))
    print(f'\nTop {top_n} SCG Candidates:')
    print(f'  {"Name":<42} {"Score":<8} {"Rank":<6}')
    print(f'  {"-"*56}')
    for seq_name, prob, rank in scg_seqs[:top_n]:
        print(f'  {seq_name:<42} {prob:<8.3f} {rank:<6}')

    # Print reference gene results
    if ref_seqs:
        print(f'\nReference Genes:')
        print(f'  {"Name":<42} {"Score":<8} {"Rank":<6}')
        print(f'  {"-"*56}')
        for seq_name, prob, rank in ref_seqs:
            print(f'  {seq_name:<42} {prob:<8.3f} {rank:<6}')


def _print_batch_summary(all_results):
    """Print a summary of batch processing results."""
    if not all_results:
        return

    print('\n' + '=' * 60)
    print('Batch Summary')
    print('=' * 60)

    for name, res in all_results.items():
        ranking = get_ranking(res)
        if ranking:
            best = ranking[0]
            print(f'  {name:30s}  best: {best[0]:35s}  score={best[1]:.3f}')
        else:
            print(f'  {name:30s}  no results')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SCG-GELP: Synonymous Codon Optimization & Gene Expression '
                    'Level Prediction'
    )

    # Mutually exclusive: single protein via CLI args, or batch via CSV
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', type=str, default=None,
        help='CSV file with columns "name,seq" for batch processing')
    input_group.add_argument(
        '--protein_name', type=str, default=None,
        help='Protein identifier (used for output file naming)')

    parser.add_argument('--protein_seq', type=str, default=None,
                        help='Amino-acid sequence, must end with * (stop codon)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: {protein_name}-scg-gelp-res/)')
    parser.add_argument(
        '--reference', type=str, default=None,
        help='Reference gene CSV (columns: protein_name,gene_name,gene_sequence). '
             'Known optimized sequences are merged with SCG candidates and '
             'evaluated together for comparison.')

    args = parser.parse_args()

    # ---- Determine output directory ----
    if args.output_dir:
        output_dir = args.output_dir
    elif args.input:
        # Batch mode: output alongside the input CSV
        output_dir = os.path.dirname(os.path.abspath(args.input))
    else:
        # Single protein: output to ./outputs/
        output_dir = os.path.join(os.getcwd(), 'outputs')

    # Load reference gene sequences (optional)
    reference_map = None
    if args.reference:
        if not os.path.exists(args.reference):
            print(f'[Error] Reference CSV not found: {args.reference}')
            sys.exit(1)
        reference_map = _read_reference_csv(args.reference)
        n_genes = sum(len(v) for v in reference_map.values())
        print(f'Loaded {n_genes} reference gene(s) for '
              f'{len(reference_map)} protein(s) from {args.reference}\n')

    # Execute
    if args.input:
        if not os.path.exists(args.input):
            print(f'[Error] Input CSV not found: {args.input}')
            sys.exit(1)
        run_batch(args.input, output_dir, reference_map)
    else:
        if not args.protein_name:
            parser.error('--protein_name is required when not using --input')
        if not args.protein_seq:
            parser.error('--protein_seq is required when not using --input')

        refs = reference_map.get(args.protein_name, {}) if reference_map else {}
        run_single(args.protein_name, args.protein_seq, output_dir,
                   refs if refs else None)


if __name__ == '__main__':
    main()
