# CS331 — Knowledge Distillation-Based Lightweight Industrial Anomaly Detection

## Research Topic
Knowledge Distillation-Based Lightweight Industrial Anomaly Detection

## Research Objective
Establish a clean, reproducible research foundation to study and improve lightweight unsupervised anomaly detection and localization on industrial surfaces. Two students will independently investigate two complementary improvement methods (`PENDING`) based on the primary reference paper.

## Reference Paper
- **Title:** Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection
- **arXiv:** `2402.00448v1`
- **Link:** https://arxiv.org/abs/2402.00448

## Dataset
- **MVTec AD** (15 industrial categories: 10 object categories, 5 texture categories; unsupervised one-class protocol using `train/good` only).

## Experiment Environment
- **Compute:** Kaggle GPU
- **Version Control:** GitHub

## Report Structure
1. Introduction
2. Dataset
3. Theoretical Background
4. Methods (`PENDING`)
5. Experiments (`PENDING`)
6. Conclusion (`PENDING`)

## Current Project Status
- **Repository Reset:** Clean baseline workspace prepared on branch `chore/repository-reset`.
- **Reference Baseline Implementation:** `PENDING`
- **Improvement Method A (Student 1):** `PENDING`
- **Improvement Method B (Student 2):** `PENDING`
- **Kaggle GPU Experiments:** `PENDING` (no experiments executed yet)

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .gitattributes
├── LICENSE
├── configs/
│   └── mvtec_ad.yaml
├── docs/
│   ├── report_outline.md
│   ├── research_plan.md
│   └── experiment_protocol.md
├── notebooks/
│   └── README.md
├── scripts/
│   ├── check_environment.py
│   └── validate_dataset.py
├── src/
│   └── anomaly_detection/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── mvtec.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── profiling.py
│           └── seed.py
├── tests/
│   ├── test_dataset.py
│   └── test_metrics.py
├── results/
│   └── README.md
└── archive/
    ├── README.md
    ├── legacy_src/
    ├── legacy_scripts/
    └── legacy_notebooks/
```
