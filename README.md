# SCG-GELP

**S**ynonymous **C**odon **G**enerator for **G**ene **E**xpression **L**evel **P**rediction

A deep learning-guided synonymous codon optimization toolkit. Given a protein
sequence, it generates multiple synonymous DNA sequences and predicts their
soluble expression probability in *Escherichia coli* using a three-stage pipeline.

---

![SCG-GELP Pipeline](images/Figure_1.png)


## Installation

```bash
docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

docker run -id --name scg_gelp \
    -w /opt \
    --gpus device=0 \
    nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
    
docker exec -it scg_gelp bash

# install tools
apt-get update && apt-get install -y git vim wget

# install conda
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" &&
bash Miniforge3-$(uname)-$(uname -m).sh -b -u -p /opt/miniconda3 &&
/opt/miniconda3/bin/conda init bash &&
source ~/.bashrc

conda create -n scg_gelp python=3.12 -y
conda activate scg_gelp 

# https://pytorch.org/get-started/previous-versions/
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

conda install tqdm scikit-learn==1.7.1 transformers==4.55.3 -y

pip install xgboost==3.2.0 lightgbm==4.6.0

conda install einops==0.8.1 -y

# uninstall triton https://github.com/MAGICS-LAB/DNABERT_2/issues/57
pip uninstall triton

git clone https://github.com/yuddecho/SCG-GELP.git
cd SCG-GELP
```

## Quick Start

Example data is provided in `data/example/`:

```bash
# Batch run on example proteins with reference gene sequences
python run.py --input data/example/proteins.csv --reference data/example/ref_genes.csv
```

Output is saved to `data/example/outputs/` by default.

## Usage

### Command Line

```bash
# Single protein
python run.py --protein_name MK --protein_seq "MKVWLVGAYGIVSTTAMVGARAIERGIAPKIGLVSELPHFEGIEK...*"

# Single protein with reference genes for comparison
python run.py --protein_name MK --protein_seq "MKVWLVG...*" --reference data/example/ref_genes.csv

# Batch processing from CSV
python run.py --input data/example/proteins.csv

# Batch with reference genes
python run.py --input data/example/proteins.csv --reference data/example/ref_genes.csv
```

**Input CSV** (`protein_name,protein_sequence`):

| protein_name | protein_sequence |
|--------------|------------------|
| MRI | MRIINVKD...\* |
| MKV | MKVWLVG...\* |

**Reference CSV** (`protein_name,gene_name,gene_sequence`):

| protein_name | gene_name | gene_sequence |
|--------------|-----------|---------------|
| MRI | WT | ATGCGTATC... |
| MRI | GS | ATGCGTATC... |
| MKV | GS | ATGAAAGTT... |

Reference sequences are merged with SCG-generated candidates and evaluated
together, allowing direct comparison of known optimized sequences against the
modelʼs output. Reference genes are always evaluated even when the SCG
generation stage is disabled in `config.py`.

**Output CSV** (sorted by total rank, best first):

| name | SVM acc | SVM rank | LR acc | LR rank | ... | total acc | total rank | sequence |
|------|---------|----------|--------|---------|-----|-----------|------------|----------|
| MRI_scg_3 | 0.937 | 1 | 0.958 | 1 | ... | 0.956 | 1.667 | ATGAGA... |
| WT | 0.949 | 2 | 0.928 | 3 | ... | 0.931 | 6.250 | ATGCGT... |

### Python API

```python
from scg_gelp import run, get_ranking

# Run the full pipeline
res = run(
    protein_name='MK',
    protein_seq='MKVWLVGAYGIVSTTAMVGARAIERGIAPKIGLVSELPHFEGIEK...*',
    reference_dnas={'WT': 'ATGAAAGTT...'},   # optional
)

# Get ranked results
ranking = get_ranking(res)
for seq_name, prob, rank in ranking:
    print(f'Rank {rank}: {seq_name} (score={prob:.3f})')
```

## Citation

Yu, D.; Geng, N.; Fan, L.; Qin, Y.; Sun, S.; Chen, H.; Wang, R.; Liao, X.; You, C. Deep Learning-Guided Reverse Translation Enhances Soluble Expression of Recombinant Proteins in *Escherichia coli*. Preprints 2026, 2026051014. https://doi.org/10.20944/preprints202605.1014.v1