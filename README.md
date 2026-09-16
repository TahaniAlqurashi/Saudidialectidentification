# A Comprehensive Evaluation of Transformer Models for Saudi Dialect Identification

Code and data accompanying the paper *"A Comprehensive Evaluation of Transformer Models for Saudi Dialect Identification"* (Alqurashi, submitted). This repository provides the training/evaluation notebooks, the exact data splits used, and instructions to reproduce the reported results.

## Overview

This project evaluates transformer-based language models on four-way Saudi Arabic dialect identification (Hijazi, Najdi, Hasawi, Janobi), comparing:

- **Fine-tuned encoders**: QARIB, MARBERT, AraBERTv2, CAMeLBERT-DA, ALLaM-7B, Qwen2.5-7B
- **Zero-shot / in-context learning**: Mistral-7B-Instruct, ALLaM-7B-Instruct, Qwen2.5-7B-Instruct, under five prompting conditions (5-shot, 10-shot, rules-only zero-shot, chain-of-thought, hand-crafted few-shot)
- **Task complexity**: binary, leave-one-dialect-out (3-way), and full 4-way classification

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── train_dialectdf.csv                          # 3,014 samples (80%)
├── val_dialectdf.csv                             # 377 samples (10%)
├── test_dialectdf.csv                            # 377 samples (10%)
├── Language_Identification_Fine_Tuning_LLM.ipynb # Main experiments: 4-way, 3-way, binary fine-tuning
└── Few-Shot_CoT_Zero-Shot-R.ipynb                # Zero-shot / in-context learning sweep (Table 3b)
```

## Dataset

The corpus (3,768 entries; Hijazi 1,033, Najdi 1,007, Hasawi 920, Janobi 808) originates from Alqurashi (2022), *"Applying a Character-Level Model to a Short Arabic Dialect Sentence,"* and consists predominantly of single words and short phrases. Splits (80/10/10, stratified, seed=42) are provided directly as CSVs — no separate download or preprocessing step is required.

35 corpus entries (≈1%) carry internally inconsistent dialect labels across duplicate occurrences, disproportionately involving Hasawi. These are retained in the released splits and discussed in the paper (Section 3.1, 4.1).

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt --break-system-packages
```

Experiments were run on Google Colab with an NVIDIA A100 GPU. The 7B-parameter models (ALLaM-7B, Qwen2.5-7B) require 8-bit quantization (`bitsandbytes`) and were fine-tuned with LoRA; the smaller encoders were fully fine-tuned.

## Usage

1. Clone this repository and ensure the three CSV files are in the working directory.
2. Open `Language_Identification_Fine_Tuning_LLM.ipynb` for fine-tuning experiments (4-way, leave-one-dialect-out, binary pairs).
3. Open `Few-Shot_CoT_Zero-Shot.ipynb` for the zero-shot / in-context learning sweep. This notebook is resumable: it skips any condition whose result file already exists in `./results/`.
4. All notebooks include a pre-flight check confirming no prompt example or vocabulary item overlaps with the validation or test sets.

## Key Results

| Model | Parameters | Accuracy | Macro F1 |
|---|---|---|---|
| QARIB | 110M | 43.77 ± 1.06% | 43.06 ± 1.21% |
| MARBERT | 135M | 42.35 ± 1.34% | 41.65 ± 2.03% |
| CAMeLBERT-DA | 110M | 36.52 ± 2.86% | 36.42 ± 2.79% |
| AraBERTv2 | 135M | 36.52 ± 1.10% | 35.93 ± 1.33% |
| ALLaM-7B | 7B | 33.51 ± 1.46% | 33.10 ± 1.53% |
| Qwen2.5-7B | 7B | 28.21 ± 1.51% | 27.92 ± 1.75% |

Mean ± standard deviation across three seeds (42, 123, 2024). Full results, statistical significance testing, and task-complexity analysis (binary, leave-one-dialect-out) are reported in the paper.

## Citation

If you use this code or data, please cite:

```bibtex
@article{alqurashi2026saudi,
  author  = {Alqurashi, Tahani},
  title   = {A Comprehensive Evaluation of Transformer Models for Saudi Dialect Identification},
  journal = {Journal Not Specified},
  year    = {2026}
}
```

## License

See LICENSE file. Corpus originally introduced by Alqurashi (2022); consult the original source for corpus-specific usage terms.

## Contact

Tahani Alqurashi — Department of Data Science, College of Computing, Umm Al-Qura University — tmqurashi@uqu.edu.sa
