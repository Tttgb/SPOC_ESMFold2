#!/usr/bin/env python3
"""
Extract structural features from ESMFold2-predicted .npz + .cif files.
CIF is used for: pDockQ CB coordinates + atom-level chemical features (atom names read directly)
NPZ is used for: pLDDT, PAE, coords (all atoms), residue types
"""

import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree # type: ignore
from tqdm import tqdm
from Bio.PDB.MMCIFParser import MMCIFParser

# Resolve paths relative to the repository root; override with CLASSIFIER_PACKAGE if needed.
BASE = os.environ.get(
    'CLASSIFIER_PACKAGE',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_DIR = os.path.join(BASE, 'homology_reduce_2nd')
OUT_DIR = os.path.join(BASE, 'FP_classifier')
NPZ_DIR = os.path.join(BASE, 'output_domain')
FASTA_PATH = os.path.join(BASE, 'dataset/random_pair/human_proteomes_reviewed.fasta')

SOURCES = ['random', 'PDB_decoy', 'XL_MS', 'PDB_contact', 'XL_MS_random']

DIST_CONTACT = 5.0   # general contact criterion
DIST_CLASH = 1.0
DIST_HBOND = 3.0
DIST_PDOCKQ = 8.0    # pDockQ-specific

# AA charge: 0=neutral, 1=positive, -1=negative
AA3_CHARGE = {
    'ALA':0,'ARG':1,'ASN':0,'ASP':-1,'CYS':0,'GLN':0,'GLU':-1,'GLY':0,
    'HIS':1,'ILE':0,'LEU':0,'LYS':1,'MET':0,'PHE':0,'PRO':0,'SER':0,
    'THR':0,'TRP':0,'TYR':0,'VAL':0,
}


def load_full_lengths(fasta_path):
    lengths = {}
    uid, seq = None, []
    with open(fasta_path) as f:
        for line in f:
            l = line.strip()
            if not l: continue
            if l.startswith('>'):
                if uid: lengths[uid] = len(''.join(seq))
                uid = l.split('|')[1]; seq = []
            else:
                seq.append(l)
        if uid: lengths[uid] = len(''.join(seq))
    return lengths


def read_cif(cif_path):
    """Returns {chain: {coords:[N,3], elements:[N], names:[N], res_types:[N], res_id:[N]}}

    Compatible with ESMFold2 outputs (which lack _atom_site.occupancy, etc.).
    """
    # try the standard MMCIFParser first
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure('x', cif_path)
        return _parse_structure(structure)
    except Exception:
        pass
    # fallback: parse manually with MMCIF2Dict (tolerates missing fields)
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        mmcif_dict = MMCIF2Dict(cif_path)
        return _parse_mmcif_dict(mmcif_dict)
    except Exception:
        return {}


def _parse_structure(structure):
    """Parse using a Biopython Structure object"""
    info = {}
    global_rid = 0
    for chain in structure[0]:
        coords, elements, names, res_types, res_ids = [], [], [], [], []
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            for atom in residue:
                coords.append(atom.coord)
                elements.append(atom.element.strip() if atom.element else '')
                names.append(atom.name.strip())
                res_types.append(residue.resname)
                res_ids.append(global_rid)
            global_rid += 1
        if coords:
            info[chain.id] = {
                'coords': np.array(coords, dtype=np.float32),
                'elements': np.array(elements, dtype='<U2'),
                'names': np.array(names, dtype='<U4'),
                'res_types': np.array(res_types, dtype='<U4'),
                'res_id': np.array(res_ids, dtype=np.int32),
            }
    return info


def _parse_mmcif_dict(mmcif_dict):
    """Parse from a raw MMCIF2Dict (handles incomplete ESMFold2 CIFs)"""
    info = {}

    # check required fields
    for key in ['_atom_site.Cartn_x', '_atom_site.Cartn_y', '_atom_site.Cartn_z',
                '_atom_site.label_asym_id', '_atom_site.label_comp_id',
                '_atom_site.label_atom_id', '_atom_site.type_symbol']:
        if key not in mmcif_dict:
            return info

    x = np.array(mmcif_dict['_atom_site.Cartn_x'], dtype=np.float32)
    y = np.array(mmcif_dict['_atom_site.Cartn_y'], dtype=np.float32)
    z = np.array(mmcif_dict['_atom_site.Cartn_z'], dtype=np.float32)
    coords_all = np.stack([x, y, z], axis=1)

    chains = np.array(mmcif_dict['_atom_site.label_asym_id'], dtype='<U2')
    res_types = np.array(mmcif_dict['_atom_site.label_comp_id'], dtype='<U4')
    atom_names = np.array(mmcif_dict['_atom_site.label_atom_id'], dtype='<U4')
    elements = np.array(mmcif_dict['_atom_site.type_symbol'], dtype='<U2')

    # parse residue IDs (may have _atom_site.label_seq_id)
    if '_atom_site.label_seq_id' in mmcif_dict:
        raw_seq_ids = mmcif_dict['_atom_site.label_seq_id']
        seq_ids = np.array([int(s) if s != '.' else -1 for s in raw_seq_ids], dtype=np.int32)
    else:
        seq_ids = np.arange(len(chains), dtype=np.int32)

    # group by chain
    unique_chains = np.unique(chains)
    global_rid = 0
    for ch_id in unique_chains:
        mask = chains == ch_id
        chain_coords = coords_all[mask]
        chain_elements = elements[mask]
        chain_names = atom_names[mask]
        chain_res_types = res_types[mask]
        chain_seq_ids = seq_ids[mask]

        # assign global rid (incrementing over unique seq_ids)
        unique_seq = np.unique(chain_seq_ids)
        rid_map = {s: i + global_rid for i, s in enumerate(unique_seq)}
        rid_per_atom = np.array([rid_map[s] for s in chain_seq_ids.tolist()], dtype=np.int32)
        global_rid += len(unique_seq)

        info[ch_id] = {
            'coords': chain_coords.astype(np.float32),
            'elements': chain_elements,
            'names': chain_names,
            'res_types': chain_res_types,
            'res_id': rid_per_atom,
        }

    return info


def get_cb_mask(info, chain_id):
    """Boolean mask of CB atoms (CA for GLY) - pDockQ v1"""
    atom_names = info[chain_id]['names']
    res_types = info[chain_id]['res_types']
    return np.array([(name == 'CB' or (rt == 'GLY' and name == 'CA'))
                     for name, rt in zip(atom_names, res_types)], dtype=bool)




def compute_pdockq(info, plddt):
    """pDockQ v1"""
    chain_ids = list(info.keys())
    if len(chain_ids) < 2: return 0.0
    mask1 = get_cb_mask(info, chain_ids[0])
    mask2 = get_cb_mask(info, chain_ids[1])
    coords1 = info[chain_ids[0]]['coords'][mask1];  res_ids1 = info[chain_ids[0]]['res_id'][mask1]
    coords2 = info[chain_ids[1]]['coords'][mask2];  res_ids2 = info[chain_ids[1]]['res_id'][mask2]
    if not len(coords1) or not len(coords2): return 0.0

    all_coords = np.concatenate([coords1, coords2])
    coord_diffs = all_coords[:, None] - all_coords[None, :]
    distances = np.sqrt(np.sum(coord_diffs**2, axis=2)).T
    n_coords1 = len(coords1)
    contacts = np.argwhere(distances[:n_coords1, n_coords1:] <= DIST_PDOCKQ)
    if not len(contacts): return 0.0

    interface_res_ids = set(res_ids1[contacts[:, 0]].tolist() + res_ids2[contacts[:, 1]].tolist())
    interface_res_ids = [r for r in interface_res_ids if r < len(plddt)]
    avg_plddt = float(np.mean(plddt[interface_res_ids])) * 100 if interface_res_ids else 0.0
    logit_x = avg_plddt * np.log10(len(contacts) + 1)
    return 0.724 / (1 + np.exp(-0.052 * (logit_x - 152.611))) + 0.018


def compute_pdockq_v2(info, plddt, pae):
    """pDockQ v2 - aligned with the ipsae_esmfold2.py reference:
       CB distances (GLY->CA), PAE of contact pairs normalized at d0=10, take the max over both directions"""
    chain_ids = list(info.keys())
    if len(chain_ids) < 2:
        return 0.0

    def _one_direction(anchor_idx, target_idx):
        mask1 = get_cb_mask(info, chain_ids[anchor_idx])
        mask2 = get_cb_mask(info, chain_ids[target_idx])
        coords1 = info[chain_ids[anchor_idx]]['coords'][mask1]
        res_ids1 = info[chain_ids[anchor_idx]]['res_id'][mask1]
        coords2 = info[chain_ids[target_idx]]['coords'][mask2]
        res_ids2 = info[chain_ids[target_idx]]['res_id'][mask2]
        if not len(coords1) or not len(coords2):
            return 0.0

        all_coords = np.concatenate([coords1, coords2])
        coord_diffs = all_coords[:, None] - all_coords[None, :]
        distances = np.sqrt(np.sum(coord_diffs ** 2, axis=2)).T
        n_coords1 = len(coords1)
        contacts = np.argwhere(distances[:n_coords1, n_coords1:] <= DIST_PDOCKQ)
        if not len(contacts):
            return 0.0

        ri = res_ids1[contacts[:, 0]]
        rj = res_ids2[contacts[:, 1]]
        pae_contact = pae[ri, rj]
        mean_ptm = float(np.mean(1.0 / (1.0 + (pae_contact / 10.0) ** 2)))

        interface_res_ids = list(set(res_ids1[contacts[:, 0]].tolist() + res_ids2[contacts[:, 1]].tolist()))
        interface_res_ids = [r for r in interface_res_ids if r < len(plddt)]
        mean_plddt = float(np.mean(plddt[interface_res_ids])) * 100 if interface_res_ids else 0.0

        x = mean_plddt * mean_ptm
        return 1.31 / (1.0 + np.exp(-0.075 * (x - 84.733))) + 0.005

    score_ab = _one_direction(0, 1)
    score_ba = _one_direction(1, 0)
    return max(score_ab, score_ba)


# ══════════════════════════════════════════════════════════════
# ipSAE - reference: scripts/ipsae_esmfold2.py (biorxiv 2025.02.10.637595)
# For a dimer (chains A/B), compute TM-score-style scores from the interface
# PAE subset (pae < pae_cutoff); ipsae_d0res / d0chn / d0dom (asym + max of both directions)
# ══════════════════════════════════════════════════════════════
def _ipsae_calc_d0(L):
    """calc_d0 (array-friendly). L = number of residues involved"""
    L = np.maximum(np.asarray(L, dtype=np.float64), 26.0)
    d0 = 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8
    return np.maximum(1.0, d0)


def compute_ipsae(pae, asym_id, pae_cutoff=10.0, dist_cutoff=15.0):
    """Compute ipSAE and all interface metrics from the ipsae_esmfold2.py reference (chains A/B).
    Covers: ipsae_d0res/d0chn/d0dom (asym+max), iptm_d0chn, LIS,
            n0res/n0dom/n0chn, d0res/d0dom/d0chn, interface residue counts.
    Returns a dict or None."""
    mask_a = asym_id == 0
    mask_b = asym_id == 1
    ia = np.where(mask_a)[0]
    ib = np.where(mask_b)[0]
    if not len(ia) or not len(ib):
        return None
    pae_ab = pae[np.ix_(ia, ib)].astype(np.float64)  # [na, nb]
    na, nb = pae_ab.shape

    def _per_res_mean(ptm_mat, valid_mat):
        denom = valid_mat.sum(axis=1)
        num = (ptm_mat * valid_mat).sum(axis=1)
        return np.where(denom > 0, num / np.maximum(denom, 1), 0.0)

    def _one_direction(pae_dir):
        n1, n2 = pae_dir.shape
        valid = pae_dir < pae_cutoff
        # d0chn: based on the total residue count of both chains
        d0chn = float(_ipsae_calc_d0(n1 + n2))
        ptm_all = 1.0 / (1.0 + (pae_dir / d0chn) ** 2)
        # ipTM_d0chn: mean over all B residues (not just the interface)
        r_iptm = _per_res_mean(ptm_all, np.ones_like(valid, dtype=bool))
        iptm_d0chn_asym = float(np.max(r_iptm)) if n1 else 0.0
        # ipsae_d0chn: interface only
        r_chn = _per_res_mean(ptm_all, valid)
        ipsae_d0chn_asym = float(np.max(r_chn)) if n1 else 0.0
        # d0dom: based on the number of unique interface residues
        n0dom = int(valid.any(axis=1).sum()) + int(valid.any(axis=0).sum())
        d0dom = float(_ipsae_calc_d0(n0dom))
        ptm_dom = 1.0 / (1.0 + (pae_dir / d0dom) ** 2)
        r_dom = _per_res_mean(ptm_dom, valid)
        ipsae_d0dom_asym = float(np.max(r_dom)) if n1 else 0.0
        # d0res: interface counts per residue
        n0res_rows = valid.sum(axis=1)
        d0res_rows = _ipsae_calc_d0(n0res_rows)[:, None]
        ptm_res = 1.0 / (1.0 + (pae_dir / d0res_rows) ** 2)
        r_res = _per_res_mean(ptm_res, valid)
        ipsae_d0res_asym = float(np.max(r_res)) if n1 else 0.0
        # n0res/d0res taken at the residue of max ipsae_d0res (aligned with the reference)
        if n1 and r_res.max() > 0:
            mi = int(np.argmax(r_res))
            n0res_asym = int(n0res_rows[mi])
            d0res_asym = float(d0res_rows[mi, 0])
        else:
            n0res_asym, d0res_asym = 0, 0.0
        # interface residue counts (nres1/nres2)
        nres_a = int(valid.any(axis=1).sum())
        nres_b = int(valid.any(axis=0).sum())
        return (iptm_d0chn_asym, ipsae_d0chn_asym, ipsae_d0dom_asym,
                ipsae_d0res_asym, n0res_asym, d0res_asym, nres_a, nres_b)

    pae_ba = pae[np.ix_(ib, ia)]  # B->A direction (PAE is asymmetric; can't just use pae_ab.T)
    a = _one_direction(pae_ab)      # A->B: pae[A rows, B cols]
    b = _one_direction(pae_ba)      # B->A: pae[B rows, A cols]

    # LIS: mean of (12-PAE)/12 over inter-chain PAE<12 (average of both directions)
    sel_ab = pae_ab[pae_ab < 12.0]
    sel_ba = pae_ba[pae_ba < 12.0]
    lis_ab = float(np.mean((12.0 - sel_ab) / 12.0)) if sel_ab.size else 0.0
    lis_ba = float(np.mean((12.0 - sel_ba) / 12.0)) if sel_ba.size else 0.0
    lis = 0.5 * (lis_ab + lis_ba)

    d0chn = float(_ipsae_calc_d0(na + nb))
    valid_ab = pae_ab < pae_cutoff
    n0dom = int(valid_ab.any(axis=1).sum()) + int(valid_ab.any(axis=0).sum())
    d0dom = float(_ipsae_calc_d0(n0dom))

    return {
        'ipsae_d0res_asym': a[3], 'ipsae_d0chn_asym': a[1], 'ipsae_d0dom_asym': a[2],
        'ipsae_d0res_max': max(a[3], b[3]),
        'ipsae_d0chn_max': max(a[1], b[1]),
        'ipsae_d0dom_max': max(a[2], b[2]),
        'iptm_d0chn_asym': a[0], 'iptm_d0chn_max': max(a[0], b[0]),
        'lis': lis,
        'n0res_asym': a[4], 'n0res_max': max(a[4], b[4]),
        'd0res_asym': a[5], 'd0res_max': max(a[5], b[5]),
        'n0dom': n0dom, 'd0dom': d0dom, 'd0chn': d0chn, 'n0chn': na + nb,
        'iface_residues_a': a[6], 'iface_residues_b': a[7],
    }


def compute_features(npz_path, cif_path, full_lengths, row):
    try:
        npz_data = np.load(npz_path, allow_pickle=False)
        cif = read_cif(cif_path)
        if not cif or len(cif) < 2:
            return None
    except:
        return None

    plddt = npz_data['plddt'].astype(np.float32)
    pae = npz_data['pae'].astype(np.float32)
    asym_id = npz_data['input_asym_id'].flatten()
    atom_to_res = npz_data['input_atom_to_token'][0].astype(np.int32)
    coords_npz = npz_data['sample_atom_coords'].astype(np.float32)
    iptm = float(npz_data['iptm'])
    ptm = float(npz_data.get('ptm', 0.0))
    N_res = len(plddt)

    mask_chain_a = asym_id == 0  # residue-level: belongs to chain A
    mask_chain_b = asym_id == 1  # residue-level: belongs to chain B
    if not mask_chain_a.any() or not mask_chain_b.any():
        return None

    chain_ids = list(cif.keys())
    chain_A, chain_B = chain_ids[0], chain_ids[1]

    # ── all pLDDT ──
    all_disorder_percent = float((plddt < 0.5).mean())
    all_plddt_mean = float(plddt.mean())
    all_plddt_max = float(plddt.max())
    all_plddt_min = float(plddt.min())
    all_pae_mean = float(pae.mean())

    # ── domain coverage ──
    uniprot_a, uniprot_b = row['uniprot_A'], row['uniprot_B']
    dom_len_a = row['dom_A_end'] - row['dom_A_start'] + 1
    dom_len_b = row['dom_B_end'] - row['dom_B_start'] + 1
    domain_coverage = (dom_len_a/max(full_lengths.get(uniprot_a, dom_len_a), 1) +
                       dom_len_b/max(full_lengths.get(uniprot_b, dom_len_b), 1)) / 2

    # ── pDockQ ──
    pdockq_e = compute_pdockq(cif, plddt)
    pdockq_e_v2 = compute_pdockq_v2(cif, plddt, pae)

    # ── inter-chain atom contacts (from NPZ) ──
    atom_mask_a = mask_chain_a[atom_to_res].astype(bool)  # atom-level: belongs to chain A
    atom_mask_b = mask_chain_b[atom_to_res].astype(bool)  # atom-level: belongs to chain B
    if not atom_mask_a.any() or not atom_mask_b.any():
        return None

    # ── CIF/NPZ coordinate consistency ──
    cif_coords_a = cif[chain_A]['coords']; cif_coords_b = cif[chain_B]['coords']
    npz_coords_a = coords_npz[atom_mask_a]; npz_coords_b = coords_npz[atom_mask_b]
    n_check = min(5, len(cif_coords_a), len(npz_coords_a), len(cif_coords_b), len(npz_coords_b))
    if n_check:
        diff_a = np.max(np.abs(cif_coords_a[:n_check] - npz_coords_a[:n_check]))
        diff_b = np.max(np.abs(cif_coords_b[:n_check] - npz_coords_b[:n_check]))
        if diff_a > 1e-3 or diff_b > 1e-3:
            raise RuntimeError(f'CIF/NPZ mismatch: {npz_path} dA={diff_a:.6f} dB={diff_b:.6f}')

    coords_a = coords_npz[atom_mask_a]  # all atoms of chain A
    coords_b = coords_npz[atom_mask_b]  # all atoms of chain B
    res_idx_a = atom_to_res[atom_mask_a]  # atom->residue map for chain A
    res_idx_b = atom_to_res[atom_mask_b]  # atom->residue map for chain B

    tree = cKDTree(coords_a)
    pairs_list = tree.query_ball_tree(cKDTree(coords_b), r=DIST_CONTACT)  # all A-B atom pairs within 5 Å

    res_pair_data = {}
    for atom_idx_a, neighbors in enumerate(pairs_list):
        res_i = res_idx_a[atom_idx_a]
        for atom_idx_b in neighbors:
            res_j = res_idx_b[atom_idx_b]
            key = (int(res_i), int(res_j))
            if key not in res_pair_data:
                res_pair_data[key] = {'dists': [], 'atom_pairs': []}  # dists[0] characterizes the res-res pair
            res_pair_data[key]['atom_pairs'].append((atom_idx_a, atom_idx_b))  # count atom pairs per res-res pair

    if not res_pair_data:
        return None

    # ── atom-level classification (from CIF atom names + elements) ──
    cif_a = cif[chain_A]
    cif_b = cif[chain_B]

    total_atom_contacts = 0
    backbone_count = 0
    hbond_count = 0
    salt_count = 0
    rep_count = 0
    clash_residues = set()

    backbone_names = {'N', 'CA', 'C', 'O'}

    for (res_i, res_j), data in res_pair_data.items():  # data = {'dists':..., 'atom_pairs':...}
        for atom_idx_a, atom_idx_b in data['atom_pairs']:
            dist = float(np.linalg.norm(coords_a[atom_idx_a] - coords_b[atom_idx_b]))
            total_atom_contacts += 1  # atom pairs within 5 Å (already filtered by the cKDTree query)

            # clash
            if dist < DIST_CLASH:
                clash_residues.add(res_i)
                clash_residues.add(res_j)

            # CIF atom info (atom_idx_a/atom_idx_b are global atom indices within chain A/B)
            if atom_idx_a < len(cif_a['names']) and atom_idx_b < len(cif_b['names']):
                atom_name_a = cif_a['names'][atom_idx_a]
                atom_name_b = cif_b['names'][atom_idx_b]
                element_a = cif_a['elements'][atom_idx_a]
                element_b = cif_b['elements'][atom_idx_b]

                # backbone
                if atom_name_a in backbone_names or atom_name_b in backbone_names:
                    backbone_count += 1

                # H-bond: N...O < 3A
                if dist < DIST_HBOND and ((element_a == 'N' and element_b == 'O') or (element_a == 'O' and element_b == 'N')):
                    hbond_count += 1

                # salt bridge & repulsive (from CIF residue types); note dist is <= 5 Å here
                res_type_a = cif_a['res_types'][atom_idx_a]
                res_type_b = cif_b['res_types'][atom_idx_b]
                charge_a = AA3_CHARGE.get(res_type_a, 0)
                charge_b = AA3_CHARGE.get(res_type_b, 0)
                if charge_a * charge_b == -1:
                    salt_count += 1  # attractive -> salt bridge
                elif charge_a * charge_b == 1:
                    rep_count += 1  # repulsive -> repulsive contact

    # ── non-clash interfacial pairs (only exclude clash; no pLDDT/PAE filter) ──
    filtered_pairs = []
    for (res_i, res_j), data in res_pair_data.items():
        if res_i in clash_residues or res_j in clash_residues: continue
        filtered_pairs.append((res_i, res_j, data))

    num_residue_contacts = len(filtered_pairs)

    # ── interface residue count & ratio (exclude clash; atom contact <= 5 Å counts as interface) ──
    all_if_residues = set()
    for ri, rj, _ in filtered_pairs:
        all_if_residues.add(ri); all_if_residues.add(rj)
    if_residues_num = len(all_if_residues)
    if_residues_percent = if_residues_num / N_res if N_res else 0

    # ── interfacial pLDDT (all interface residues, unfiltered) ──
    plddt_if = [plddt[r] for r in all_if_residues] if all_if_residues else [0]
    if_plddt_avg = float(np.mean(plddt_if))
    if_plddt_max = float(np.max(plddt_if))
    if_plddt_min = float(np.min(plddt_if))

    # ── pLDDT difference (filtered pairs) ──
    diffs = [abs(float(plddt[ri])-float(plddt[rj])) for ri, rj, _ in filtered_pairs]
    if_plddt_diff_mean = float(np.mean(diffs)) if diffs else 0.0

    # ── PAE (clash-filtered pairs) ──
    pae_values, pae_diffs = [], []
    for ri, rj, _ in filtered_pairs:
        pae_ij, pae_ji = float(pae[ri, rj]), float(pae[rj, ri])
        pae_values.append(pae_ij); pae_diffs.append(abs(pae_ij-pae_ji))
    pae_avg = float(np.mean(pae_values)) if pae_values else 0.0
    pae_max = float(np.max(pae_values)) if pae_values else 0.0
    pae_min = float(np.min(pae_values)) if pae_values else 0.0
    pae_diff_mean = float(np.mean(pae_diffs)) if pae_diffs else 0.0

    # ── unfiltered PAE ──
    uf_pae = [float(pae[ri, rj]) for (ri, rj) in res_pair_data]
    unfiltered_if_pae_mean = float(np.mean(uf_pae)) if uf_pae else 0.0

    # ── contact scores (clash-filtered pairs) ──
    contact_scores = []
    for ri, rj, data in filtered_pairs:
        n_atom_pairs = len(data['atom_pairs'])
        avg_plddt_pair = (float(plddt[ri])+float(plddt[rj]))/2
        avg_pae_pair = (float(pae[ri,rj])+float(pae[rj,ri]))/2
        contact_scores.append(n_atom_pairs * avg_plddt_pair / (1 + avg_pae_pair))
    contact_score_avg = float(np.mean(contact_scores)) if contact_scores else 0.0
    contact_score_max = float(np.max(contact_scores)) if contact_scores else 0.0
    contact_score_median = float(np.median(contact_scores)) if contact_scores else 0.0

    # ── atom-level percentages ──
    total_contacts = total_atom_contacts or 1
    backbone_percent = backbone_count / total_contacts
    h_bond_percent = hbond_count / total_contacts
    salt_bridge_percent = salt_count / total_contacts
    repulsive_contact_percent = rep_count / total_contacts
    # ── other contact percent ──
    other_contact_percent = max(0.0, 1.0 - h_bond_percent -
                                salt_bridge_percent - repulsive_contact_percent)
    atom_contacts_per_residue_avg = total_atom_contacts / if_residues_num if if_residues_num else 0

    feats = {
        'all_disorder_percent': all_disorder_percent,
        'all_plddt_max': all_plddt_max, 'all_plddt_mean': all_plddt_mean,
        'all_plddt_min': all_plddt_min,
        'if_plddt_avg': if_plddt_avg, 'if_plddt_diff_mean': if_plddt_diff_mean,
        'if_plddt_max': if_plddt_max, 'if_plddt_min': if_plddt_min,
        'contact_score_avg': contact_score_avg, 'contact_score_max': contact_score_max,
        'contact_score_median': contact_score_median,
        'if_residues_num': if_residues_num, 'if_residues_percent': if_residues_percent,
        'num_clash_residues': len(clash_residues),
        'num_residue_contacts': num_residue_contacts,
        'backbone_percent': backbone_percent, 'all_pae_mean': all_pae_mean,
        'unfiltered_if_pae_mean': unfiltered_if_pae_mean,
        'pae_avg': pae_avg, 'pae_diff_mean': pae_diff_mean,
        'pae_max': pae_max, 'pae_min': pae_min,
        'repulsive_contact_percent': repulsive_contact_percent,
        'ptm': ptm, 'iptm': iptm, 'domain_coverage': domain_coverage,
        'h_bond_percent': h_bond_percent,
        'salt_bridge_percent': salt_bridge_percent,
        'other_contact_percent': other_contact_percent,
        'atom_contacts_per_residue_avg': atom_contacts_per_residue_avg,
        'pdockq_e': pdockq_e, 'pdockq_e_v2': pdockq_e_v2,
    }
    # ── ipSAE (19 interface metrics, matching the current RF model features) ──
    ipsae = compute_ipsae(pae, asym_id, pae_cutoff=10.0)
    if ipsae:
        feats.update(ipsae)
    return feats


def process_one(args):
    npz_path, cif_path, full_lengths, row = args
    feats = compute_features(npz_path, cif_path, full_lengths, row)
    if feats is None:
        return None
    result = row.to_dict()
    result.update(feats)
    return result


def main():
    print("Loading full protein lengths ...")
    full_lengths = load_full_lengths(FASTA_PATH)
    print(f"  {len(full_lengths)} proteins")

    num_workers = min(cpu_count(), 48)

    for source in SOURCES:
        tsv_in = os.path.join(IN_DIR, source,
            'XL_MS_random_filtered.tsv' if source == 'XL_MS_random' else f'{source}_filtered.tsv')
        if not os.path.exists(tsv_in):
            print(f'\n{source}: SKIP'); continue

        df_input = pd.read_csv(tsv_in, sep='\t')
        print(f'\n{source}: {len(df_input)} pairs')

        tasks = []
        skipped = 0
        for _, row in df_input.iterrows():
            dimer_id_val = row.get('dimer_id', '')
            if not dimer_id_val:
                dimer_id_val = f"{row['uniprot_A']}_{row['dom_A_start']}_{row['dom_A_end']}__{row['uniprot_B']}_{row['dom_B_start']}_{row['dom_B_end']}"
            npz_path = os.path.join(NPZ_DIR, f'{dimer_id_val}_features.npz')
            cif_path = os.path.join(NPZ_DIR, f'{dimer_id_val}.cif')
            if not os.path.exists(npz_path) or not os.path.exists(cif_path):
                skipped += 1; continue
            tasks.append((npz_path, cif_path, full_lengths, row))
        if skipped:
            print(f'  Missing: {skipped}')

        results = []
        with Pool(num_workers) as pool:
            for r in tqdm(pool.imap_unordered(process_one, tasks, chunksize=10),
                          total=len(tasks), desc=f'  {source}'):
                if r is not None: results.append(r)

        df_output = pd.DataFrame(results)
        src_dir = os.path.join(OUT_DIR, source)
        os.makedirs(src_dir, exist_ok=True)
        tsv_out = os.path.join(src_dir, f'{source}_features.tsv')
        df_output.to_csv(tsv_out, sep='\t', index=False)
        print(f'  Saved: {tsv_out} ({len(df_output)})')

    print('\nDone.')


if __name__ == '__main__':
    main()
