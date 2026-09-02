#!/usr/bin/env python3
"""
Compute inter-chain contact-positive (C+) metrics from ESMFold2 predictions (.npz).

C+ criteria (PMID reference):
  1. at least 1 inter-chain heavy-atom pair with distance < 5 Å
  2. both residues have pLDDT > 0.5
  3. no heavy-atom pair with distance < 1 Å (clash filter)
  4. PAE(x,y) < 15 and PAE(y,x) < 15

Outputs:
  - contact_positive_results.pkl: dict[dimer_id] = {n_positive_contacts, plddt_A, plddt_B, iptm, ...}
  - contact_positive_results.tsv: label\tsource\tcomplex_name\tiptm\tplddt_A\tplddt_B\tn_c+
"""

import os, sys, pickle, time
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


PAE_THRESH = 15.0
PLDDT_THRESH = 0.5
DIST_CONTACT = 5.0
DIST_CLASH = 1.0


def compute_contacts(npz_path: str) -> dict | None:
    """Compute C+ metrics from an npz file. Returns a dict or None."""
    try:
        d = np.load(npz_path, allow_pickle=False)
    except Exception:
        return None

    plddt = d['plddt']                               # [N_res]
    pae = d['pae']                                   # [N_res, N_res]
    asym_id = d['input_asym_id'].flatten()           # [N_res]
    iptm = float(d['iptm'])
    atom2res = d['input_atom_to_token'][0]           # [N_atoms]
    coords = d['sample_atom_coords']                 # [N_atoms, 3]

    # ── group atoms by chain ──
    a_mask = asym_id[atom2res] == 0
    b_mask = asym_id[atom2res] == 1
    if not a_mask.any() or not b_mask.any():
        return None

    coords_A = coords[a_mask]
    coords_B = coords[b_mask]
    res_A = atom2res[a_mask].astype(int)
    res_B = atom2res[b_mask].astype(int)

    # ── KDTree: find all inter-chain atom pairs within 5 Å ──
    tree_A = cKDTree(coords_A)
    pairs = tree_A.query_ball_tree(cKDTree(coords_B), r=DIST_CONTACT)
    # pairs[a_idx] = list of b_idx within distance r

    # ── aggregate to residue pairs ──
    res_pair_dist = {}  # (ri, rj) → min_dist
    res_pair_clash = set()

    for i_a, nb in enumerate(pairs):
        ri = res_A[i_a]
        for i_b in nb:
            rj = res_B[i_b]
            dist = float(np.linalg.norm(coords_A[i_a] - coords_B[i_b]))
            key = (ri, rj)

            if key not in res_pair_dist or dist < res_pair_dist[key]:
                res_pair_dist[key] = dist

            if dist < DIST_CLASH:
                res_pair_clash.add(key)

    # ── filter C+ pairs ──
    c_plus = 0
    c_clash = 0 
    for (ri, rj), _ in res_pair_dist.items():
        # clash filter
        if (ri, rj) in res_pair_clash:
            c_clash +=1
            continue

        # pLDDT
        if plddt[ri] <= PLDDT_THRESH or plddt[rj] <= PLDDT_THRESH:
            continue

        # PAE (both directions < 15)
        if pae[ri, rj] >= PAE_THRESH or pae[rj, ri] >= PAE_THRESH:
            continue

        c_plus += 1

    # ── chain pLDDT ──
    plddt_A = float(plddt[asym_id == 0].mean())
    plddt_B = float(plddt[asym_id == 1].mean())

    return {
        'n_positive_contacts': c_plus,
        'plddt_A': plddt_A,
        'plddt_B': plddt_B,
        'iptm': iptm,
        'is_contact_positive': c_plus >= 5,
        'n_clash' : c_clash
    }


def _process_one(args: tuple) -> tuple:
    """Pool worker: (dimer_id, npz_dir) → (dimer_id, metrics_dict | None)"""
    did, npz_dir = args
    npz_path = os.path.join(npz_dir, f"{did}_features.npz")
    if not os.path.exists(npz_path):
        return did, None
    metrics = compute_contacts(npz_path)
    return did, metrics


def main():
    t0 = time.time()

    data_dir = '/home/data/xjd/ESMFOLD_filter'
    pkl_path = os.path.join(data_dir, 'domain_pairs.pkl')
    npz_dir = os.path.join(data_dir, 'output_domain')
    out_dir = os.path.join(data_dir, 'contact_positive')
    os.makedirs(out_dir, exist_ok=True)

    # ── load PKL ──
    print("Loading domain_pairs.pkl ...")
    with open(pkl_path, 'rb') as f:
        raw_pairs = pickle.load(f)
    print(f"  {len(raw_pairs)} records")

    # ── build dimer_id list ──
    dimer_ids = []
    for entry in raw_pairs:
        dimer_id = (f"{entry['uniprot_A']}_{entry['dom_A_start']}_{entry['dom_A_end']}__"
                    f"{entry['uniprot_B']}_{entry['dom_B_start']}_{entry['dom_B_end']}")
        dimer_ids.append(dimer_id)

    # unique dimer mapping
    unique_ids, first_idx = [], {}
    for i, did in enumerate(dimer_ids):
        if did not in first_idx:
            first_idx[did] = i
            unique_ids.append(did)

    print(f"  unique dimer: {len(unique_ids)}")

    # ── parallel computation ──
    n_workers = min(cpu_count(), 32)
    print(f"  Using {n_workers} workers ...")

    results = {}
    n_contact_pos = 0

    tasks = [(did, npz_dir) for did in unique_ids]
    with Pool(n_workers) as pool:
        for did, metrics in tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=100),
            total=len(tasks), desc="Computing C+", unit="pair"
        ):
            if metrics is None:
                continue
            results[did] = metrics
            if metrics['is_contact_positive']:
                n_contact_pos += 1

    # ── save pkl ──
    pkl_out = os.path.join(out_dir, 'contact_positive_results.pkl')
    with open(pkl_out, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── write TSV ──
    tsv_out = os.path.join(out_dir, 'contact_positive_results.tsv')
    with open(tsv_out, 'w') as f:
        f.write("label\tsource\tcomplex_name\tiptm\tplddt_A\tplddt_B\t"
                "n_positive_contacts\tn_clash\n")
        for entry in raw_pairs:
            did = (f"{entry['uniprot_A']}_{entry['dom_A_start']}_{entry['dom_A_end']}__"
                   f"{entry['uniprot_B']}_{entry['dom_B_start']}_{entry['dom_B_end']}")
            cname = entry.get('pair_id', did)
            m = results.get(did, {})
            f.write(f"{entry.get('label', '')}\t{entry.get('source', '')}\t"
                    f"{cname}\t"
                    f"{m.get('iptm', '')}\t{m.get('plddt_A', '')}\t"
                    f"{m.get('plddt_B', '')}\t"
                    f"{m.get('n_positive_contacts', '')}\t"
                    f"{m.get('n_clash', '')}\n"
                    )
                    

    # ── summary ──
    elapsed = time.time() - t0
    n_ok = len(results)
    n_total = len(unique_ids)
    print(f"\n{'='*50}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Computed: {n_ok}/{n_total}")
    print(f"  Contact positive (C+ >= 5): {n_contact_pos} "
          f"({100*n_contact_pos/n_total:.1f}%)")
    print(f"  pkl saved: {pkl_out}")
    print(f"  tsv saved: {tsv_out}")


if __name__ == '__main__':
    main()
