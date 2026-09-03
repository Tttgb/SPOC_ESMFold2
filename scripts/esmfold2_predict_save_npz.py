#!/usr/bin/env python3
"""
ESMFold2 dimer prediction -> .cif + .npz for batch_inference.py
================================================================
The official BioHub ESMFold2 example only writes a .cif and does NOT save the
per-residue arrays (pLDDT / PAE / ipTM) that classifier_package needs for its
structural features. This reference script uses the exact same ESMFold2 API as
the official example (``ESMFold2Model.from_pretrained`` +
``ESMFold2InputBuilder().fold(...)``) and additionally dumps the required
arrays into a matching .npz:

    <out>.cif   mmCIF structure  (identical to the official example's output)
    <out>.npz   plddt / pae / iptm / ptm   (consumed by batch_inference.py)

Run it in an ESMFold2 environment (biohub "esm" 3.x + transformers
ESMFold2Model) -- NOT in the classifier_package requirements environment.

Usage
-----
    python scripts/esmfold2_predict_save_npz.py \\
        --seq_a MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAE... \\
        --seq_b MKQLEDKVEELLSKNYHLENEVARLKKLVGER \\
        --out my_dimer

    # then score the dimer with the classifier:
    python batch_inference.py --cif my_dimer.cif --npz my_dimer.npz \\
        --uniprot_A Q6DN90 --uniprot_B P62330 --output out.tsv

Notes
-----
* batch_inference.py needs only plddt/pae/iptm/ptm in the .npz. If atom
  coordinates / chain assignment are absent it re-derives them from the .cif
  (auto-writing <out>_ad.npz), so this minimal set is sufficient.
* ``--lm_dropout 0`` makes the fold deterministic. The official default is 0.3
  (randomized LM dropout), which produces slightly different structures per run.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model


def main():
    ap = argparse.ArgumentParser(
        description="Run ESMFold2 on one protein dimer and save .cif + .npz "
                    "for classifier_package/batch_inference.py")
    ap.add_argument("--seq_a", required=True,
                    help="amino-acid sequence of chain A")
    ap.add_argument("--seq_b", required=True,
                    help="amino-acid sequence of chain B")
    ap.add_argument("--out", default="dimer",
                    help="output prefix: writes <out>.cif and <out>.npz")
    ap.add_argument("--model", default="biohub/ESMFold2",
                    help="ESMFold2 model id or local model path")
    ap.add_argument("--device", default="cuda",
                    help="torch device ('cuda' or 'cpu')")
    ap.add_argument("--num_loops", type=int, default=3,
                    help="number of structure-module recycle loops")
    ap.add_argument("--num_sampling_steps", type=int, default=50,
                    help="number of diffusion sampling steps")
    ap.add_argument("--lm_dropout", type=float, default=0.0,
                    help="LM dropout; 0 = deterministic fold")
    ap.add_argument("--ccd", default=None,
                    help="path to the CCD cache directory (parent of ccd.pkl); "
                         "defaults to the parent of $ESMCFOLD_CCD_PATH if set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- load the model (same call as the official example) ----
    model = ESMFold2Model.from_pretrained(args.model).to(args.device).eval()

    # ---- define the dimer (chain A, then chain B) ----
    spi = StructurePredictionInput(sequences=[
        ProteinInput(id="A", sequence=args.seq_a),
        ProteinInput(id="B", sequence=args.seq_b),
    ])

    # ---- fold (same call as the official example) ----
    ccd_cache = args.ccd
    if ccd_cache is None and os.environ.get("ESMCFOLD_CCD_PATH"):
        ccd_cache = str(Path(os.environ["ESMCFOLD_CCD_PATH"]).parent)
    result = ESMFold2InputBuilder(ccd_cache=ccd_cache).fold(
        model, spi,
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=1,
        lm_dropout=args.lm_dropout,
        seed=args.seed,
    )

    # ---- 1) structure: identical to the official example's output ----
    with open(f"{args.out}.cif", "w") as f:
        f.write(result.complex.to_mmcif())
    print(f"[save] {args.out}.cif")

    # ---- 2) feature arrays consumed by batch_inference.py ----
    plddt = np.asarray(result.plddt.float().cpu()).reshape(-1)
    pae = np.asarray(result.pae.float().cpu())
    iptm = float(result.iptm) if result.iptm is not None else 0.0
    ptm = float(result.ptm) if result.ptm is not None else 0.0

    np.savez_compressed(
        f"{args.out}.npz",
        plddt=plddt.astype(np.float32),   # [N_res]
        pae=pae.astype(np.float32),       # [N_res, N_res]
        iptm=iptm,                        # scalar
        ptm=ptm,                          # scalar
    )
    print(f"[save] {args.out}.npz  (N_res={plddt.size}, "
          f"mean pLDDT={plddt.mean():.3f}, pTM={ptm:.3f}, ipTM={iptm:.3f})")


if __name__ == "__main__":
    main()
