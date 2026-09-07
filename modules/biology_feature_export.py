#!/usr/bin/env python3
"""
Biological-feature export: append 14 biological columns to the existing
5 structural-feature TSVs.
Feature list:
  biogrid_detect_count, co_expression_score, colocalization_match_score,
  crispr_jaccard, crispr_shared_hit_count,
  depmap_abs_diff, depmap_cosine_dist, depmap_euclidian_dist,
  t5_domain_embedding_cosine_dist, t5_domain_embedding_euclidian_dist,
  avg_af_missense_score_diff, af_missense_score_mean,
  af_missense_score_mean_diff, num_significant_afm_scores
"""

import os, sys, pickle, gzip, re, subprocess
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm
import requests, time

# Resolve paths relative to the repository root; override with CLASSIFIER_PACKAGE if needed.
BASE = os.environ.get(
    'CLASSIFIER_PACKAGE',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT_DIR = os.path.join(BASE, 'FP_classifier')
CACHE_DIR = os.path.join(FEAT_DIR, 'deeploc_output')
SPOC_DIR = os.path.join(BASE, 'SPOC_related_dataset')
NPZ_DIR = os.path.join(BASE, 'output_domain')

SOURCES = ['random', 'PDB_decoy', 'XL_MS', 'PDB_contact', 'XL_MS_random']

DIST_CONTACT = 5.0  # interface criterion distance

# ============================================================
# 1. ID mapping (UniProt -> Symbol / Entrez)
# ============================================================
UNIPROT_RE = re.compile(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]([0-9]{2})?$')

def detect_id_type(q):
    q = q.strip()
    if q.isdigit(): return "entrezgene"
    if UNIPROT_RE.match(q): return "uniprot"
    if q.startswith("ENS"): return "ensembl.gene"
    return "symbol"


def build_id_mapping(uniprot_ids, cache_path):
    """Batch-map UniProt -> Symbol/Entrez via the MyGene API; cache the result"""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    uniprot_ids = sorted(set(uid for uid in uniprot_ids if uid and uid.strip()))
    print(f"  Mapping {len(uniprot_ids)} UniProt IDs via MyGene API ...")
    mapping = {}
    batch_size = 500
    for i in tqdm(range(0, len(uniprot_ids), batch_size), desc='  MyGene'):
        batch = uniprot_ids[i:i+batch_size]
        try:
            resp = requests.post('https://mygene.info/v3/query', json={
                'q': ' '.join(batch), 'scopes': 'uniprot',
                'fields': 'symbol,entrezgene', 'species': 'human',
                'size': len(batch) * 2,
            }, timeout=60)
            resp.raise_for_status()
            for hit in resp.json():
                uid = hit.get('query', '').strip()
                if uid:
                    mapping[uid] = {
                        'symbol': hit.get('symbol', ''),
                        'entrezgene': str(hit.get('entrezgene', '')),
                    }
            time.sleep(0.3)
        except Exception as e:
            print(f'  API error: {e}')
            time.sleep(2)
    with open(cache_path, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"  Mapped {len(mapping)}/{len(uniprot_ids)}")
    return mapping


# ============================================================
# 2. BioGRID detect_count
# ============================================================
def load_biogrid(biogrid_path, id_mapping):
    """Returns {(uniprot_a, uniprot_b): detect_count} (alphabetically sorted)"""
    print("Loading BioGRID ...")
    counts = {}
    with open(biogrid_path) as f:
        for line in tqdm(f, desc='  BioGRID'):
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < 28: continue
            org_a, org_b = cols[15], cols[16]
            if org_a != '9606' or org_b != '9606':
                continue  # keep human only
            up_a = cols[23].strip()  # SWISS-PROT Accessions
            up_b = cols[26].strip()
            if not up_a or not up_b or up_a == '-' or up_b == '-':
                continue
            # the Swiss-Prot field in BioGRID may contain multiple IDs; take the first
            up_a = up_a.split('|')[0]
            up_b = up_b.split('|')[0]
            key = tuple(sorted([up_a, up_b]))
            counts[key] = counts.get(key, 0) + 1
    print(f"  {len(counts)} unique human protein pairs in BioGRID")
    return counts


# ============================================================
# 3. CoexpressDB co_expression_score
# ============================================================
def load_coexpressdb(coexpr_dir, id_mapping):
    """Returns {entrez_gene_id: coexpression_score} - one overall co-expression score per gene"""
    print("Loading CoexpressDB (human) ...")
    import zipfile
    path = os.path.join(coexpr_dir,
        'Hsa-u.v22-05.G16651-S245698.combat_pca.subagging.z.d.zip')
    scores = {}
    with zipfile.ZipFile(path) as zf:
        fname = zf.namelist()[0]
        with zf.open(fname) as f:
            for line in tqdm(f, desc='  CoexpressDB'):
                line = line.decode() if isinstance(line, bytes) else line
                parts = line.strip().split()
                if len(parts) != 2: continue
                entrez, score = parts[0], float(parts[1])
                scores[entrez] = score
    print(f"  {len(scores)} genes")
    return scores


# ============================================================
# 4. DeepLoc 2.1 localization
# ============================================================
DEEPLOC_DIR = os.path.join(SPOC_DIR, 'DeepLoc_2.1/deeploc2_package')
DEEPLOC_LOC_COLS = [
    'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
    'Mitochondrion', 'Plastid', 'Endoplasmic reticulum',
    'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome',
    'Peripheral', 'Transmembrane', 'Lipid anchor', 'Soluble',
]


def run_deeploc(uniprot_ids, fasta_path, cache_path):
    """Run DeepLoc2 and return {uniprot_id: np.array(14,)}.
       Reads from cache if it exists."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # extract sequences from FASTA
    print("Extracting sequences for DeepLoc ...")
    seq_map = {}
    uid, seq = None, []
    with open(fasta_path) as f:
        for line in f:
            l = line.strip()
            if not l: continue
            if l.startswith('>'):
                if uid and uid in uniprot_ids:
                    seq_map[uid] = ''.join(seq)
                uid = l.split('|')[1] if '|' in l else l[1:].split()[0]
                seq = []
            else:
                seq.append(l)
        if uid and uid in uniprot_ids:
            seq_map[uid] = ''.join(seq)

    print(f"  Found {len(seq_map)}/{len(uniprot_ids)} sequences in FASTA")

    # write a temporary FASTA
    tmp_fasta = os.path.join(CACHE_DIR, 'deeploc_input.fasta')
    with open(tmp_fasta, 'w') as f:
        for uid, seq_str in seq_map.items():
            f.write(f'>{uid}\n{seq_str}\n')

    # run DeepLoc2 (uses the installed CLI)
    out_dir = CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)
    print("Running DeepLoc 2.1 (Fast model) ...")
    subprocess.run([
        'deeploc2',
        '--fasta', tmp_fasta,
        '--model', 'Fast',
        '--device', 'cuda',
        '-o', out_dir,
    ], check=True, env={**os.environ, 'TORCH_HOME': '/tmp/torch_cache'})

    # parse the output
    csv_files = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV output found in {out_dir}")
    result_df = pd.read_csv(os.path.join(out_dir, csv_files[-1]))

    # build the {uniprot_id: vector} dict
    loc_data = {}
    for _, row in result_df.iterrows():
        protein_id = row.iloc[0]  # first column: Protein_ID / ACC
        # try several ID formats
        uid_match = protein_id
        if '|' in str(protein_id):
            uid_match = str(protein_id).split('|')[1]
        vec = np.array([float(row.get(c, 0)) for c in DEEPLOC_LOC_COLS], dtype=np.float32)
        loc_data[uid_match] = vec

    with open(cache_path, 'wb') as f:
        pickle.dump(loc_data, f)
    print(f"  DeepLoc completed: {len(loc_data)} proteins")
    return loc_data


# ============================================================
# 5. CRISPR ORCS
# ============================================================
def load_crispr_orcs(pkl_path):
    """Returns {gene_symbol: set(screen_ids)}"""
    print("Loading CRISPR ORCS ...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # data[symbol] = set of screen IDs (as ints)
    data = {k: set(int(s) for s in v) for k, v in data.items()}
    print(f"  {len(data)} genes in CRISPR ORCS")
    return data


# ============================================================
# 6. ProtT5 Embeddings (pre-computed; domains inherit the full-length protein)
# ============================================================
def load_t5_embeddings(h5_path):
    """Return {uniprot_id: np.array(1024, float16)}; only needed IDs are read later"""
    print("Loading ProtT5 embeddings ...")
    f = h5py.File(h5_path, 'r')
    # return the open file handle for lazy access
    print(f"  {len(f.keys())} proteins available")
    return f  # h5py File; read on demand later


# ============================================================
# 7. DepMap
# ============================================================
def load_depmap(csv_path):
    """Returns {gene_symbol: np.array([N_cell_lines])}"""
    print("Loading DepMap ...")
    df = pd.read_csv(csv_path, index_col=0)
    # column format: "TP53 (7157)" -> extract symbol
    gene_vectors = {}
    for col in df.columns:
        symbol = col.split(' (')[0] if ' (' in col else col
        gene_vectors[symbol] = df[col].values.astype(np.float32)
    print(f"  {len(gene_vectors)} genes, {df.shape[0]} cell lines")
    return gene_vectors


# ============================================================
# 8. AlphaMissense
# ============================================================
def load_alphamissense(am_path, target_ids, cache_path):
    """Returns {uniprot_id: {position: avg_am_score}} keeping only target_ids; caches the result"""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("Loading AlphaMissense (filtered, ~3 min first time) ...")
    target_set = set(target_ids)
    data = {}
    variant_re = re.compile(r'^[A-Z](\d+)[A-Z]$')
    with open(am_path) as f:
        for line in tqdm(f, desc='  AlphaMissense'):
            if line.startswith('#') or line.startswith('uniprot'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < 4: continue
            uid = cols[0]
            if uid not in target_set:
                continue  # skip proteins we don't need
            variant, score = cols[1], float(cols[2])
            m = variant_re.match(variant)
            if not m: continue
            pos = int(m.group(1))
            if uid not in data:
                data[uid] = {}
            if pos not in data[uid]:
                data[uid][pos] = []
            data[uid][pos].append(score)
    for uid in data:
        data[uid] = {p: np.mean(scores) for p, scores in data[uid].items()}
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  {len(data)} proteins loaded (filtered + cached)")
    return data


# ============================================================
# 9. Interface residue detection (from NPZ)
# ============================================================
def get_interface_residues(npz_path):
    """Returns (set of res_ids_a, set of res_ids_b, n_residues_a) or None.
       res_ids are global residue indices (0-based, chain A first)."""
    try:
        d = np.load(npz_path, allow_pickle=False)
        asym_id = d['input_asym_id'].flatten()
        atom_to_res = d['input_atom_to_token'][0].astype(np.int32)
        coords = d['sample_atom_coords'].astype(np.float32)
    except:
        return None

    n_res_a = int((asym_id == 0).sum())

    mask_a = (asym_id == 0)[atom_to_res].astype(bool)
    mask_b = (asym_id == 1)[atom_to_res].astype(bool)
    if not mask_a.any() or not mask_b.any():
        return None

    tree = cKDTree(coords[mask_a])
    pairs = tree.query_ball_tree(cKDTree(coords[mask_b]), r=DIST_CONTACT)

    res_a = atom_to_res[mask_a]
    res_b = atom_to_res[mask_b]

    iface_a, iface_b = set(), set()
    for atom_idx_a, nbrs in enumerate(pairs):
        if nbrs:
            iface_a.add(int(res_a[atom_idx_a]))
            for atom_idx_b in nbrs:
                iface_b.add(int(res_b[atom_idx_b]))
    return iface_a, iface_b, n_res_a


# ============================================================
# 10. Main feature-computation function
# ============================================================
def compute_biology_features(row, biogrid, coexpr, crispr, depmap,
                              am_data, deeploc_data, t5_file, id_map, npz_dir,
                              npz_path=None):
    """If npz_path is given, use it directly (batch inference); otherwise build it from npz_dir + dimer_id"""
    ua, ub = row['uniprot_A'], row['uniprot_B']
    result = {}

    # ── ID mapping ──
    ma, mb = id_map.get(ua, {}), id_map.get(ub, {})
    sym_a, sym_b = ma.get('symbol', ''), mb.get('symbol', '')
    ent_a, ent_b = ma.get('entrezgene', ''), mb.get('entrezgene', '')

    # ── BioGRID detect_count ──
    key_bg = tuple(sorted([ua, ub]))
    result['biogrid_detect_count'] = biogrid.get(key_bg, 0)

    # ── CoexpressionDB ──
    coexpr_score = np.nan
    if ent_a and ent_b and ent_a in coexpr and ent_b in coexpr:
        coexpr_score = (coexpr[ent_a] + coexpr[ent_b]) / 2.0
    result['co_expression_score'] = coexpr_score

    # ── DeepLoc colocalization ──
    v_a = deeploc_data.get(ua)
    v_b = deeploc_data.get(ub)
    if v_a is not None and v_b is not None and v_a.any() and v_b.any():
        cos_sim = np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b))
        result['colocalization_match_score'] = float(cos_sim)
    else:
        result['colocalization_match_score'] = np.nan

    # ── CRISPR ──
    set_a = crispr.get(sym_a, set())
    set_b = crispr.get(sym_b, set())
    if set_a and set_b:
        intersection = set_a & set_b
        union = set_a | set_b
        result['crispr_jaccard'] = len(intersection) / len(union) if union else 0.0
        result['crispr_shared_hit_count'] = len(intersection)
    else:
        result['crispr_jaccard'] = 0.0
        result['crispr_shared_hit_count'] = 0

    # ── DepMap ──
    vec_a = depmap.get(sym_a)
    vec_b = depmap.get(sym_b)
    if vec_a is not None and vec_b is not None:
        mask = np.isfinite(vec_a) & np.isfinite(vec_b)
        va, vb = vec_a[mask], vec_b[mask]
        if len(va) > 10:
            result['depmap_abs_diff'] = float(np.mean(np.abs(va - vb)))
            result['depmap_cosine_dist'] = float(cosine(va, vb))
            result['depmap_euclidian_dist'] = float(euclidean(va, vb))
        else:
            result['depmap_abs_diff'] = np.nan
            result['depmap_cosine_dist'] = np.nan
            result['depmap_euclidian_dist'] = np.nan
    else:
        result['depmap_abs_diff'] = np.nan
        result['depmap_cosine_dist'] = np.nan
        result['depmap_euclidian_dist'] = np.nan

    # ── ProtT5 embedding distances (domains inherit the full-length protein) ──
    try:
        emb_a = np.array(t5_file[ua][()], dtype=np.float32) if ua in t5_file else None
        emb_b = np.array(t5_file[ub][()], dtype=np.float32) if ub in t5_file else None
    except:
        emb_a = emb_b = None
    if emb_a is not None and emb_b is not None:
        result['t5_domain_embedding_cosine_dist'] = float(cosine(emb_a, emb_b))
        result['t5_domain_embedding_euclidian_dist'] = float(euclidean(emb_a, emb_b))
    else:
        result['t5_domain_embedding_cosine_dist'] = np.nan
        result['t5_domain_embedding_euclidian_dist'] = np.nan

    # ── AlphaMissense ──
    am_a = am_data.get(ua, {})
    am_b = am_data.get(ub, {})

    # avg_af_missense_score_diff: domain-level
    da_start, da_end = int(row['dom_A_start']), int(row['dom_A_end'])
    db_start, db_end = int(row['dom_B_start']), int(row['dom_B_end'])
    scores_a = [am_a.get(p, np.nan) for p in range(da_start, da_end+1)]
    scores_b = [am_b.get(p, np.nan) for p in range(db_start, db_end+1)]
    scores_a = [s for s in scores_a if not np.isnan(s)]
    scores_b = [s for s in scores_b if not np.isnan(s)]
    mean_a = np.mean(scores_a) if scores_a else np.nan
    mean_b = np.mean(scores_b) if scores_b else np.nan
    result['avg_af_missense_score_diff'] = (
        abs(mean_a - mean_b) if not np.isnan(mean_a) and not np.isnan(mean_b) else np.nan)

    # AlphaMissense interface features
    if npz_path is None:
        dimer_id = row.get('dimer_id', '')
        if not dimer_id:
            dimer_id = f"{ua}_{da_start}_{da_end}__{ub}_{db_start}_{db_end}"
        npz_path = os.path.join(npz_dir, f'{dimer_id}_features.npz')
    iface_info = get_interface_residues(npz_path) if os.path.exists(npz_path) else None

    if iface_info and (iface_info[0] or iface_info[1]):
        iface_a, iface_b, n_res_a = iface_info
        # global residue index -> UniProt position
        # chain A: res 0..(n_res_a-1) -> dom_A_start..dom_A_end
        # chain B: res n_res_a..(n_res_a+n_res_b-1) -> dom_B_start..dom_B_end
        pair_means = []
        pair_diffs = []
        sig_count = 0
        for ri in iface_a:
            pos_a = da_start + ri               # ri < n_res_a
            score_a = am_a.get(pos_a, np.nan)
            if np.isnan(score_a):
                continue
            for rj in iface_b:
                pos_b = db_start + (rj - n_res_a)  # rj >= n_res_a
                score_b = am_b.get(pos_b, np.nan)
                if np.isnan(score_b):
                    continue
                mean_pair = (score_a + score_b) / 2.0
                pair_means.append(mean_pair)
                pair_diffs.append(abs(score_a - score_b))
                if mean_pair > 0.8:
                    sig_count += 1

        result['af_missense_score_mean'] = float(np.mean(pair_means)) if pair_means else np.nan
        result['af_missense_score_mean_diff'] = float(np.mean(pair_diffs)) if pair_diffs else np.nan
        result['num_significant_afm_scores'] = sig_count
    else:
        result['af_missense_score_mean'] = np.nan
        result['af_missense_score_mean_diff'] = np.nan
        result['num_significant_afm_scores'] = 0

    return result


# ============================================================
# 11. Main entry
# ============================================================
def main():
    # ── collect all UniProt IDs ──
    all_up_ids = set()
    for src in SOURCES:
        tsv = os.path.join(FEAT_DIR, src, f'{src}_features.tsv')
        if not os.path.exists(tsv): continue
        df = pd.read_csv(tsv, sep='\t')
        all_up_ids.update(df['uniprot_A'].dropna().unique())
        all_up_ids.update(df['uniprot_B'].dropna().unique())
    print(f"Unique UniProt IDs: {len(all_up_ids)}")

    # ── load databases ──
    id_map = build_id_mapping(all_up_ids,
        os.path.join(CACHE_DIR, 'cache_id_mapping.pkl'))

    biogrid = load_biogrid(
        os.path.join(SPOC_DIR, 'biogrid/BIOGRID-ALL-5.0.258.tab3.txt'),
        id_map)

    coexpr = load_coexpressdb(
        os.path.join(SPOC_DIR, 'CoexpressDB'), id_map)

    crispr = load_crispr_orcs(
        os.path.join(SPOC_DIR, 'biogrid/biogrid_ORCS/protein_hit_screens.pkl'))

    depmap = load_depmap(
        os.path.join(SPOC_DIR, 'DepMap/CRISPRGeneEffect.csv'))

    am_data = load_alphamissense(
        os.path.join(SPOC_DIR, 'AlphaMissence/AlphaMissense_aa_substitutions.tsv'),
        all_up_ids,
        os.path.join(CACHE_DIR, 'cache_alphamissense.pkl'))

    deeploc_data = run_deeploc(all_up_ids,
        os.path.join(BASE, 'dataset/random_pair/human_proteomes_reviewed.fasta'),
        os.path.join(CACHE_DIR, 'cache_deeploc.pkl'))

    t5_file = load_t5_embeddings(
        os.path.join(SPOC_DIR, 'ProtT5_embedding/per-protein.h5'))

    # ── process each source ──
    for src in SOURCES:
        tsv_in = os.path.join(FEAT_DIR, src, f'{src}_features.tsv')
        if not os.path.exists(tsv_in):
            print(f'{src}: SKIP')
            continue
        df = pd.read_csv(tsv_in, sep='\t')
        print(f'\n{src}: {len(df)} pairs')

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'  {src}'):
            feats = compute_biology_features(row, biogrid, coexpr, crispr, depmap,
                                              am_data, deeploc_data, t5_file, id_map, NPZ_DIR)
            results.append(feats)

        bio_df = pd.DataFrame(results)
        df_out = pd.concat([df, bio_df], axis=1)
        tsv_out = os.path.join(FEAT_DIR, src, f'{src}_features_biology.tsv')
        df_out.to_csv(tsv_out, sep='\t', index=False)
        print(f'  Saved: {tsv_out} ({len(df_out)} cols={list(bio_df.columns)})')

    print('\nDone.')


if __name__ == '__main__':
    main()
