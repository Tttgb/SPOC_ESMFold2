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
│   │   ├── rf_importance_{allfeat,structfeat}.csv          # Gini importance (feature, importance) → Fig 4
│   │   ├── rf_permutation_importance_{allfeat,structfeat}.csv  # permutation drop (train/test)
│   │   ├── domain_pairs_stats.csv  # pre-extracted data for Fig 1D
│   │   └── param_sweep_aupr_results_{all,struct}.csv  # hyperparameter-sweep results
│   └── spoc/                       # 11 GB biological DB (NOT shipped; see Data)
├── scripts/
│   ├── check_spoc_db.py            # verify data/spoc completeness
│   ├── download_spoc_db.sh         # download biological DBs from Zenodo
│   ├── esmfold2_predict_save_npz.py  # reference ESMFold2 run -> .cif + .npz
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

### Optional — DeepLoc 2.1 (only needed for brand-new proteins)

`colocalization_match_score` uses **DeepLoc 2.1** subcellular localisation.
DeepLoc 2.1 is a **standalone prediction tool** (not a PyPI package), so it is
deliberately **not** listed in `requirements.txt`. The shipped cache
`data/deeploc_output/cache_deeploc.pkl` already covers **all 20,416 proteins** in
the datasets, so inference **never** calls DeepLoc in normal use.

Only if you score a protein **absent from the cache** will `batch_inference.py`
invoke the `deeploc2` command-line tool on the fly (see the DeepLoc note in the
Data section). `batch_inference.py` calls `deeploc2` simply as a subprocess, so
install it into **whatever Python environment you run `batch_inference.py` in**
— it only needs to be on `$PATH`; no particular conda environment is required.
Install per the official instructions:

- https://services.healthtech.dtu.dk/services/DeepLoc-2.1/
- Publication: Ødum *et al.*, *Nucleic Acids Research*, 2024, doi:10.1093/nar/gkae237

```bash
# in the same environment used to run batch_inference.py (GPU + PyTorch needed)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e <path/to/deeploc2_package>   # from the official DeepLoc 2.1 distribution
```


---

## 1. Model inference (`batch_inference.py`)

Score **one** protein dimer (a `.cif` + its matching `.npz` from an ESMFold2
multimer prediction) with the two chains' UniProt IDs, and write a single row
with RF probabilities (SPOC ESMFOLD / Structural classifier) + all features.
No "target" or batch-scanning concept is needed.

```bash
python batch_inference.py \
    --cif <dimer.cif> --npz <dimer.npz> \
    --uniprot_A <UP_A> --uniprot_B <UP_B> \
    [--output out.tsv] [--skip_bio]

# Example: score the IQSEC1–ARF6 dimer in test_input/
python batch_inference.py \
    --cif 'test_input/IQSEC1;Q6DN90;1_ARF6.cif' \
    --npz 'test_input/IQSEC1;Q6DN90;1_ARF6.npz' \
    --uniprot_A Q6DN90 --uniprot_B P62330 \
    --output IQSEC1_ARF6.tsv

# Optional: structural features only (skips biological DB loading)
python batch_inference.py --cif <dimer.cif> --npz <dimer.npz> \
    --uniprot_A <UP_A> --uniprot_B <UP_B> --output out.tsv --skip_bio
```

| Argument | Description |
|---|---|
| `--cif` | Required. Dimer structure `.cif` file |
| `--npz` | Required. Matching `.npz` file (see `scripts/esmfold2_predict_save_npz.py`) |
| `--uniprot_A` | Required. UniProt ID of chain A (for biological-feature mapping) |
| `--uniprot_B` | Required. UniProt ID of chain B |
| `--output` | Output tsv path (default `inference_result.tsv`) |
| `--skip_bio` | Skip biological features (use only the structure model) |

**Input format**: a `<dimer_id>.cif` (structure) and a matching `<dimer_id>.npz`
carrying the per-residue confidence arrays (`plddt`/`pae`/`iptm`). The two
chains' UniProt IDs (`--uniprot_A` / `--uniprot_B`) are used only to map the
biological features (BioGRID, co-expression, CRISPR, DepMap, ProtT5,
AlphaMissense).

> **The stock ESMFold2 example does not write an `.npz`** — it only saves the
> `.cif`. `batch_inference.py` needs the `.npz` companion because the structural
> features are computed from `plddt`/`pae`/`iptm` and atom coordinates, not from
> the CIF alone. Use the reference runner `scripts/esmfold2_predict_save_npz.py`
> (the same BioHub ESMFold2 API as the official example) to produce both files:
>
> ```bash
> # run in an ESMFold2 (biohub esm 3.x) environment, not the classifier env
> python scripts/esmfold2_predict_save_npz.py \
>     --seq_a M...  --seq_b M...  --out my_dimer
> # → my_dimer.cif + my_dimer.npz (plddt / pae / iptm / ptm)
> ```
>
> Only `plddt`/`pae`/`iptm` (plus `ptm`) are required in the `.npz`. If atom
> coordinates / chain assignment are absent, `batch_inference.py` auto-derives
> them from the `.cif` into `<dimer_id>_ad.npz`.

**Output columns**: `uniprot_A / uniprot_B / fname / cif / npz / n_c+ /
score_all_feat / score_struct_only` + 54 structural features (pLDDT, PAE,
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

The Setup cell auto-points `DATA` to `data/analysis/` (override with
`CLASSIFIER_PACKAGE`); figures are written to `figures/` at the repo root
(auto-created).

---

## 3. Data

### Shipped with the repository

| Path | Description |
|---|---|
| `data/models/` | Inference models: `rf_all_feat_model.pkl`, `rf_struct_feat_model.pkl` |
| `data/fasta/` | `human_proteomes_reviewed.fasta` |
| `data/deeploc_output/` | `cache_id_mapping.pkl`, `cache_deeploc.pkl` (DeepLoc 2.1 localisation cache, covers all 20.4k involved proteins — see note below) |
| `data/analysis/` | Plotting data for the notebook (train/test, 3 RF models, Gini-importance CSVs for Fig 4, permutation-importance CSVs, `domain_pairs_stats.csv` for Fig 1D, `param_sweep_aupr_results_*.csv` for the sweep figure) |
| `test_input/`, `test_out.tsv` | Example inference input & output |

> **Note on DeepLoc.** The `colocalization_match_score` biological feature uses
> DeepLoc 2.1 subcellular localisation. DeepLoc is **not** a static data file —
> it is a prediction tool (the `deeploc2` CLI, run on GPU). The pre-computed
> cache `data/deeploc_output/cache_deeploc.pkl` (shipped above) already covers
> **all 20,416 involved proteins**, so inference reads the cache and never needs
> to run DeepLoc. Only if you score a brand-new protein (not in the cache) will
> the script invoke `deeploc2` on the fly — install it in the environment used
> to run `batch_inference.py` (see the install note above).

### Not shipped — `data/spoc/` (~11 GB raw, ~3.5 GB compressed)

The biological databases required for the biological features are **not** shipped
(excluded via `.gitignore`). **Ready-to-use** archives are hosted on Zenodo
(DOI: *to be added after upload*). Install them with a single command:

```bash
bash scripts/download_spoc_db.sh
```

This downloads the five archives (`AlphaMissence`, `biogrid`, `CoexpressDB`,
`DepMap`, `ProtT5_embedding`), verifies their SHA256 checksums and extracts
them into `data/spoc/`:

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
