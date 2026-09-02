# SPOC ESMFOLD

*A False Positive Scoring Method for ESMFOLD2-fast Outputs.*

A **domain–domain interaction (DDI) classifier / interaction scorer** built on
**ESMFold2 predicted structures + public biological databases**, shipped as a
self-contained GitHub repository with a batch-inference script and a notebook
that reproduces the paper's evaluation figures.

- **Models**: Random Forest, three variants
  - `SPOC ESMFOLD` (39 features after iterative pruning: 25 structural + 14 biological)
  - `Structural classifier` (49 structural features after iterative pruning)
  - `degree-match RF` (pure interaction-network degree features, used as a
    data-leakage check baseline)
- **Inference**: `batch_inference.py` — scores a set of dimers (CIF/NPZ) and
  outputs RF probabilities plus a full feature table
- **Reproduction**: `notebooks/visualization.ipynb` — reproduces the paper's
  figures (Fig 1D / 3A–3D / 4A–4B + supplementary AUPR and hyperparameter-sweep
  figures) from the **final datasets only** (no intermediate pipeline data required)

---

## Repository layout

```
classifier_package/
├── README.md
├── requirements.txt
├── batch_inference.py              # batch inference entry point
├── modules/                        # feature-computation modules
│   ├── structure_feature_export.py # structural features (pLDDT/PAE/contacts/ipSAE…)
│   ├── compute_contacts.py         # positive-contact (C+) computation
│   └── biology_feature_export.py   # biological features (BioGRID/Coexpr/CRISPR/DepMap/T5/AM)
├── notebooks/
│   └── visualization.ipynb         # paper-figure reproduction notebook (English)
├── data/
│   ├── models/                     # inference models (shipped)
│   ├── deeploc_output/             # DeepLoc / ID-mapping caches (shipped; AlphaMissense cache excluded)
│   ├── fasta/                      # human proteome reference (shipped)
│   ├── analysis/                   # plotting data for the notebook (shipped, self-contained)
│   │   ├── test.tsv / train.tsv
│   │   ├── rf_all_feat_model.pkl / rf_struct_feat_model.pkl / rf_degree_match_model.pkl
│   │   ├── rf_permutation_importance_{allfeat,structfeat}.csv
│   │   ├── domain_pairs_stats.csv  # pre-extracted data for Fig 1D
│   │   └── param_sweep_aupr_results_{all,struct}.csv  # hyperparameter-sweep results
│   └── spoc/                       # 11 GB biological DB (NOT shipped; see Data)
├── scripts/
│   ├── check_spoc_db.py            # verify data/spoc completeness
│   └── param_sweep_aupr_heatmap.py # RF hyperparameter sweep (recomputes the CSVs)
├── test_input/                     # example inference input (3 dimers)
└── test_out.tsv                    # example inference output (78-col feature table)
```

---

## Installation

```bash
# Python 3.9+
pip install -r requirements.txt
```

Dependencies: `numpy pandas scipy scikit-learn biopython h5py tqdm requests
matplotlib seaborn py3Dmol ipykernel`

> Inference needs only the first set (`numpy/pandas/scipy/scikit-learn/biopython/
> h5py/tqdm/requests`); `matplotlib/seaborn/py3Dmol/ipykernel` are for the
> reproduction notebook.

---

## 1. Model inference (`batch_inference.py`)

Score a batch of dimers (each with `.cif` + `.npz` from an ESMFold2 multimer
prediction) and write a table with RF probabilities + all 78 features.

```bash
python batch_inference.py \
    --input_dir <dimer_dir> \
    --target_uniprot <target_UniProt> \
    --output <out.tsv>

# Example: the 3 ARF6 candidates in test_input/ (matches test_out.tsv)
python batch_inference.py --input_dir test_input --target_uniprot P62330 --output ARF6_0.tsv

# Optional: structural features only (skips biological DB loading)
python batch_inference.py --input_dir test_input --target_uniprot P62330 --output ARF6_0.tsv --skip_bio
```

| Argument | Description |
|---|---|
| `--input_dir` | Required. Directory containing the dimers' `.cif` + `.npz` |
| `--target_uniprot` | Required. Target protein UniProt ID (e.g. ARF6 → `P62330`) |
| `--output` | Output tsv path (default `batch_results.tsv`) |
| `--skip_bio` | Skip biological features (use only the structure model) |

**Input format**: ESMFold2 multimer predictions — per dimer a `<dimer_id>.cif`
(structure) and `<dimer_id>.npz` (containing `plddt`/`pae`/`iptm`, …). If an NPZ
lacks `sample_atom_coords`/`input_asym_id`/…, the script auto-adapts it from the
CIF into `<dimer_id>_ad.npz`.

**Output columns** (78 total): `gene / uniprot_A / uniprot_B / target / fname /
n_c+ / score_all_feat / score_struct_only` + 54 structural features (pLDDT, PAE,
contacts, chemistry, pDockQ, ipTM, 19 ipSAE terms, …) + 14 biological features
(BioGRID, co-expression, CRISPR, DepMap, ProtT5, AlphaMissense).

> The first run loads `data/spoc/` biological databases (~5–10 min) and
> auto-generates `data/deeploc_output/cache_alphamissense.pkl` (~3 min the first
> time; seconds afterwards).

---

## 2. Paper-figure reproduction (`notebooks/visualization.ipynb`)

A single **English** notebook, organized **one cell per figure**, that
reproduces the paper's evaluation figures from the final datasets only:

| Cell | Figure | Content |
|------|--------|---------|
| Setup | — | paths, imports, plotting style, unified data loading |
| Fig 1D | **Figure 1D** | Crosslink retention after domain splitting |
| Fig 3A | **Figure 3A** | ROC of key structural scores + RF models (test set) |
| Fig 3B | **Figure 3B** | Recall at FDR=5% (1:1 sampling), all feature scores |
| Fig 3C | **Figure 3C** | Recall at FDR=5% across ratios 1:1–1:128 (core 4 scores) |
| Fig 3D | **Figure 3D** | Fixed-threshold recall scatter at 1:128 subset |
| Fig 4A/4B | **Figure 4A/4B** | Gini importance (SPOC ESMFOLD / Structural classifier) |
| Supplementary | — | AUPR (Precision-Recall) of the 3 models |
| Supplementary | — | RF hyperparameter sweep (3-fold CV AUPR heatmaps) |

**How to run**

```bash
cd notebooks
jupyter notebook            # or open notebooks/visualization.ipynb in VS Code
```

- Run cells top to bottom;
- The Setup cell auto-points `DATA` to `data/analysis/` (repo root inferred as
  the parent of the working directory; override with `CLASSIFIER_PACKAGE`);
- Figures are written to `figures/` at the repo root (auto-created);
- **Figure 3D** reuses the 1:128 sampled subsets built by the Figure 3C cell —
  run cells in order.

---

## 3. Data

### Shipped with the repository

| Path | Description |
|---|---|
| `data/models/` | Inference models: `rf_all_feat_model.pkl`, `rf_struct_feat_model.pkl` |
| `data/fasta/` | `human_proteomes_reviewed.fasta` |
| `data/deeploc_output/` | `cache_id_mapping.pkl`, `cache_deeploc.pkl` |
| `data/analysis/` | Plotting data for the notebook (train/test, 3 RF models, permutation-importance CSVs, `domain_pairs_stats.csv` for Fig 1D, `param_sweep_aupr_results_*.csv` for the sweep figure) |
| `test_input/`, `test_out.tsv` | Example inference input & output |

### Not shipped — `data/spoc/` (~11 GB)

The biological databases required for the biological features are **not**
shipped (excluded via `.gitignore`). Download and place them as follows:

```
data/spoc/
├── AlphaMissence/AlphaMissense_aa_substitutions.tsv      # AlphaMissense (full proteome)
├── CoexpressDB/                                           # human co-expression DB
├── DepMap/CRISPRGeneEffect.csv                            # DepMap CRISPR gene effect
├── ProtT5_embedding/per-protein.h5                        # ProtT5 embeddings
├── biogrid/BIOGRID-ALL-5.0.258.tab3.txt                   # BioGRID interactions
└── biogrid/biogrid_ORCS/protein_hit_screens.pkl           # CRISPR ORCS screens
```

Run `python scripts/check_spoc_db.py` to verify completeness.

The RF hyperparameter-sweep results (`data/analysis/param_sweep_aupr_results_*.csv`)
are generated by `scripts/param_sweep_aupr_heatmap.py`:
`python scripts/param_sweep_aupr_heatmap.py` (or `--mode all` / `--mode struct`).
The notebook's sweep figure reads these CSVs and redraws the heatmaps.

> If some databases are missing, inference still runs — the corresponding
> biological features are imputed (NaN); `--skip_bio` skips them entirely.
> The notebook's reproduction does **not** need `data/spoc/` at all.

### Auto-generated at runtime (no need to ship)

- `data/deeploc_output/cache_alphamissense.pkl` (~132 MB, rebuilt from
  `data/spoc/AlphaMissence`, git-ignored)
- `figures/` (notebook output, git-ignored)

---

## License

MIT — see `LICENSE` (replace with your preferred license if needed).

For questions, please open an issue.
