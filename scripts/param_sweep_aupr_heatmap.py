#!/usr/bin/env python3
"""RF hyperparameter sweep -> 3-fold CV AUPR heatmaps (per model).

Scans n_estimators x max_depth x min_samples_split (4 x 5 x 4 = 80 combos),
each evaluated with 3-fold StratifiedKFold CV AUPR.

This script is the source of the notebook's *Supplementary Figure - RF
hyperparameter sweep*. It saves the structured results (CSV) that the
notebook reads to redraw the heatmaps, and also writes a 4-panel PDF per model.

Usage:
    python scripts/param_sweep_aupr_heatmap.py               # both models
    python scripts/param_sweep_aupr_heatmap.py --mode all    # only SPOC ESMFOLD
    python scripts/param_sweep_aupr_heatmap.py --mode struct # only Structural classifier

Outputs:
    data/analysis/param_sweep_aupr_results_{all,struct}.csv
    figures/supp_param_sweep_aupr_{model}.pdf  (4 panels, one per min_samples_split)
"""
import os
import pickle
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

REPO = os.environ.get(
    'CLASSIFIER_PACKAGE',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, 'data', 'analysis')
FIG = os.path.join(REPO, 'figures')
os.makedirs(FIG, exist_ok=True)

RANDOM_SEED = 42
N_EST = [200, 400, 600, 1000]
DEPTH = [8, 12, 16, 20, 24]
SPLIT = [2, 5, 10, 20]

MODELS = {
    'all':    {'name': 'SPOC ESMFOLD',          'pkl': 'rf_all_feat_model.pkl'},
    'struct': {'name': 'Structural classifier', 'pkl': 'rf_struct_feat_model.pkl'},
}


def run(mode):
    cfg = MODELS[mode]
    name = cfg['name']
    print('=' * 72)
    print(f'RF hyperparameter sweep: {name}  (mode={mode})')
    print('=' * 72)

    df_train = pd.read_csv(os.path.join(DATA, 'train.tsv'), sep='\t')
    y = df_train['label'].values
    mdl = pickle.load(open(os.path.join(DATA, cfg['pkl']), 'rb'))
    cols = [c for c in mdl['features'] if c in df_train.columns]
    X = SimpleImputer(strategy='median').fit_transform(df_train[cols].values)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    rows = []
    t0 = time.time()
    for ne in N_EST:
        for md in DEPTH:
            for ms in SPLIT:
                scores = []
                for tr, va in cv.split(X, y):
                    rf = RandomForestClassifier(
                        n_estimators=ne, max_depth=md, min_samples_split=ms,
                        criterion='log_loss', random_state=RANDOM_SEED, n_jobs=-1)
                    rf.fit(X[tr], y[tr])
                    p = rf.predict_proba(X[va])[:, 1]
                    scores.append(average_precision_score(y[va], p))
                rows.append({'n_estimators': ne, 'max_depth': md,
                             'min_samples_split': ms,
                             'aupr_0': scores[0], 'aupr_1': scores[1], 'aupr_2': scores[2],
                             'mean': float(np.mean(scores)), 'std': float(np.std(scores))})
    res = pd.DataFrame(rows)
    csv_path = os.path.join(DATA, f'param_sweep_aupr_results_{mode}.csv')
    res.to_csv(csv_path, index=False)
    print(f'[{name}] {len(res)} combos in {time.time()-t0:.0f}s -> {csv_path}')

    # ---- 4-panel heatmap PDF (one panel per min_samples_split) ----
    vmin, vmax = float(res['mean'].min()), float(res['mean'].max())
    pdf_path = os.path.join(FIG, f'supp_param_sweep_aupr_{name.replace(" ", "_")}.pdf')
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path) as pdf:
        for ms in SPLIT:
            sub = res[res['min_samples_split'] == ms]
            mat = sub.pivot(index='max_depth', columns='n_estimators', values='mean')
            std = sub.pivot(index='max_depth', columns='n_estimators', values='std')
            fig, ax = plt.subplots(figsize=(10, 7))
            im = ax.imshow(mat.values, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(N_EST)))
            ax.set_xticklabels([str(n) for n in N_EST])
            ax.set_yticks(range(len(DEPTH)))
            ax.set_yticklabels([str(d) for d in DEPTH])
            ax.set_xlabel('n_estimators')
            ax.set_ylabel('max_depth')
            ax.set_title(f'3-fold CV AUPR (mean±std) - {name} - min_samples_split={ms}')
            for i in range(len(DEPTH)):
                for j in range(len(N_EST)):
                    c = 'white' if mat.values[i, j] > (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{mat.values[i, j]:.3f}\n±{std.values[i, j]:.3f}',
                            ha='center', va='center', fontsize=9, color=c)
            fig.colorbar(im, ax=ax, label='mean AUPR')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'[{name}] PDF saved: {pdf_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['all', 'struct', 'both'], default='both')
    args = ap.parse_args()
    modes = ['all', 'struct'] if args.mode == 'both' else [args.mode]
    for m in modes:
        run(m)
    print('DONE')
