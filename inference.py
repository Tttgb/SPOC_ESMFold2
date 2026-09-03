#!/usr/bin/env python3
"""
SPOC ESMFOLD single-dimer inference
========================================
Score ONE protein dimer (a .cif + its matching .npz) and write a single row
with RF probabilities (SPOC ESMFOLD / Structural classifier) + all features.
Biological features are mapped via the two chains' UniProt IDs (--uniprot_A /
--uniprot_B); no "target" or batch-scanning concept is needed.

Usage:
    python inference.py \
        --cif <dimer.cif> --npz <dimer.npz> \
        --uniprot_A <UP_A> --uniprot_B <UP_B> \
        [--output out.tsv] [--skip_bio]

Output: one-row TSV (RF scores + full feature table)
"""

import os, sys, json, time, gc, pickle, re, glob, argparse, subprocess
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════
# Package path configuration (override with CLASSIFIER_PACKAGE if needed)
# ══════════════════════════════════════════════════════════
PACKAGE_DIR = os.environ.get('CLASSIFIER_PACKAGE',
                             os.path.dirname(os.path.abspath(__file__)))
MODULES_DIR = os.path.join(PACKAGE_DIR, 'modules')
DATA_DIR    = os.path.join(PACKAGE_DIR, 'data')
SPOC_DIR    = os.path.join(DATA_DIR, 'spoc')
CACHE_DIR   = os.path.join(DATA_DIR, 'deeploc_output')
MODEL_DIR   = os.path.join(DATA_DIR, 'models')
FASTA_PATH  = os.path.join(DATA_DIR, 'fasta/human_proteomes_reviewed.fasta')
NPZ_DIR     = os.path.join(DATA_DIR, 'output_domain')  # 仅作为回退，batch 推理直接传 npz_path

sys.path.insert(0, MODULES_DIR)

from structure_feature_export import compute_features as compute_struct, load_full_lengths, read_cif
from compute_contacts import compute_contacts as compute_cplus
import biology_feature_export as bio

# ── 覆盖 biology_feature_export 里的路径常量（指向打包数据）──
bio.SPOC_DIR   = SPOC_DIR
bio.CACHE_DIR  = CACHE_DIR
bio.NPZ_DIR    = NPZ_DIR
bio.FEAT_DIR   = DATA_DIR


# ── 全局数据库缓存（进程级，只加载一次）──
_DB_CACHE = {}

# ── 生物特征数据库完整性检查 + 自动下载 ──
# 生物特征依赖 data/spoc/ 下的公共数据库；若缺失则自动运行
# scripts/download_spoc_db.sh（从 Zenodo 拉取约 3.5 GB）。
_SPOC_REQUIRED_DIRS = ['CoexpressDB']
_SPOC_REQUIRED_FILES = [
    'AlphaMissence/AlphaMissense_aa_substitutions.tsv',
    'DepMap/CRISPRGeneEffect.csv',
    'ProtT5_embedding/per-protein.h5',
    'biogrid/BIOGRID-ALL-5.0.258.tab3.txt',
    'biogrid/biogrid_ORCS/protein_hit_screens.pkl',
]


def _spoc_complete():
    """data/spoc/ 生物数据库是否齐备（所需目录与关键文件都在）"""
    if not os.path.isdir(SPOC_DIR):
        return False
    for d in _SPOC_REQUIRED_DIRS:
        if not os.path.isdir(os.path.join(SPOC_DIR, d)):
            return False
    for f in _SPOC_REQUIRED_FILES:
        if not os.path.isfile(os.path.join(SPOC_DIR, f)):
            return False
    return True


def ensure_spoc_db():
    """若 data/spoc/ 生物数据库缺失/不完整，自动运行下载脚本；
    下载后仍不完整则给出错误并退出（可用 --skip_bio 跳过生物特征）。"""
    if _spoc_complete():
        return
    print(f"[DB] data/spoc/ 生物数据库不完整，自动下载约 3.5 GB（需网络）...")
    script = os.path.join(PACKAGE_DIR, 'scripts', 'download_spoc_db.sh')
    if not os.path.isfile(script):
        sys.exit(f"[DB] 未找到下载脚本: {script}\n"
                 f"     请按 README 手动准备 data/spoc/，或使用 --skip_bio 只做结构推理。")
    print(f"[DB] 运行: bash {script}")
    try:
        rc = subprocess.run(['bash', script]).returncode
    except Exception as e:
        rc = -1
        print(f"[DB] 运行下载脚本失败: {e}")
    if rc != 0:
        print(f"[DB] 下载脚本退出码: {rc}")
    if not _spoc_complete():
        sys.exit(f"[DB] data/spoc/ 下载后仍不完整。请检查网络/磁盘后重试，"
                 f"或使用 --skip_bio 只做结构特征推理。")


def load_databases():
    """加载生物数据库，结果缓存在全局 dict 中"""
    if _DB_CACHE:
        return _DB_CACHE

    ensure_spoc_db()  # 缺失则自动下载

    import biology_feature_export as bio
    print("[DB] 加载数据库 (仅一次) ...")

    # ID 映射缓存
    id_cache = os.path.join(bio.CACHE_DIR, 'cache_id_mapping.pkl')
    _DB_CACHE['id_map'] = pickle.load(open(id_cache, 'rb')) if os.path.exists(id_cache) else {}

    # AlphaMissense 缓存
    am_cache = os.path.join(bio.CACHE_DIR, 'cache_alphamissense.pkl')
    _DB_CACHE['am_data'] = pickle.load(open(am_cache, 'rb')) if os.path.exists(am_cache) else {}

    # DeepLoc 缓存
    dl_cache = os.path.join(bio.CACHE_DIR, 'cache_deeploc.pkl')
    _DB_CACHE['dl_data'] = pickle.load(open(dl_cache, 'rb')) if os.path.exists(dl_cache) else {}

    # 全局数据库
    t0 = time.time()
    _DB_CACHE['biogrid'] = bio.load_biogrid(
        os.path.join(bio.SPOC_DIR, 'biogrid/BIOGRID-ALL-5.0.258.tab3.txt'), _DB_CACHE['id_map'])
    _DB_CACHE['coexpr'] = bio.load_coexpressdb(
        os.path.join(bio.SPOC_DIR, 'CoexpressDB'), _DB_CACHE['id_map'])
    _DB_CACHE['crispr'] = bio.load_crispr_orcs(
        os.path.join(bio.SPOC_DIR, 'biogrid/biogrid_ORCS/protein_hit_screens.pkl'))
    _DB_CACHE['depmap'] = bio.load_depmap(
        os.path.join(bio.SPOC_DIR, 'DepMap/CRISPRGeneEffect.csv'))
    _DB_CACHE['t5'] = bio.load_t5_embeddings(
        os.path.join(bio.SPOC_DIR, 'ProtT5_embedding/per-protein.h5'))
    print(f"[DB] 加载完成 ({time.time()-t0:.0f}s)")

    return _DB_CACHE


# ══════════════════════════════════════════════════════════
# 适配 + 特征提取
# ══════════════════════════════════════════════════════════

def adapt_npz(npz_path, cif_path):
    d = dict(np.load(npz_path, allow_pickle=False))
    needed = ['sample_atom_coords', 'input_asym_id', 'input_atom_to_token']
    if all(k in d for k in needed):
        return npz_path, d

    cif = read_cif(cif_path)
    if not cif or len(cif) < 2:
        return None, None

    chains_ids = list(cif.keys())
    N_res = len(d['plddt'])
    all_coords, all_res_ids = [], []
    asym_id = np.zeros(N_res, dtype=np.int32)

    for ci, chain_id in enumerate(chains_ids):
        ch = cif[chain_id]
        all_coords.append(ch['coords'])
        all_res_ids.append(ch['res_id'])
        for rid in np.unique(ch['res_id']):
            if rid < N_res:
                asym_id[rid] = ci

    coords = np.concatenate(all_coords).astype(np.float32)
    res_id = np.concatenate(all_res_ids).astype(np.int32)
    d['sample_atom_coords'] = coords
    d['input_asym_id'] = asym_id.reshape(1, -1)
    d['input_atom_to_token'] = res_id.reshape(1, -1)
    for k in ['atom_pad_mask', 'plddt_per_atom']:
        if k not in d: d[k] = np.zeros(coords.shape[0], dtype=np.float32)
    for k in ['input_ref_pos', 'input_ref_atom_name_chars', 'input_ref_element',
              'input_mol_type', 'input_res_type', 'input_entity_id',
              'input_residue_index', 'input_token_attention_mask']:
        if k not in d: d[k] = np.zeros(1, dtype=np.float32)

    adapted_path = npz_path.replace('.npz', '_ad.npz')
    np.savez_compressed(adapted_path, **d)
    return adapted_path, d


def extract_bio_features(row, db, npz_path=None):
    import biology_feature_export as bio
    return bio.compute_biology_features(
        row, db['biogrid'], db['coexpr'], db['crispr'], db['depmap'],
        db['am_data'], db['dl_data'], db['t5'], db['id_map'], bio.NPZ_DIR,
        npz_path=npz_path)


def predict_rf(feats, model_path):
    from sklearn.impute import SimpleImputer
    b = pickle.load(open(model_path, 'rb'))
    # 在子进程内禁止 RF 内部再开多线程，避免 Loky 警告
    model = b['model']
    if hasattr(model, 'n_jobs'):
        model.n_jobs = 1
    X = np.array([[feats.get(f, np.nan) for f in b['features']]], dtype=np.float64)
    return float(model.predict_proba(b['imputer'].transform(X))[:, 1][0])


# ══════════════════════════════════════════════════════════
# 批量缓存预热（数据库已预缓存，通常是幂等空操作）
# ══════════════════════════════════════════════════════════

def _batch_am_data(db, ids_to_query=None):
    """一次性扫描 AlphaMissense TSV，补齐所有未缓存的蛋白
       若 ids_to_query 全部已在缓存中，直接跳过（全人类蛋白组缓存已就绪）"""
    am_path = os.path.join(bio.SPOC_DIR, 'AlphaMissence/AlphaMissense_aa_substitutions.tsv')
    if not os.path.exists(am_path):
        return

    already = set(db['am_data'].keys())
    import re
    vr = re.compile(r'^[A-Z](\d+)[A-Z]$')

    # 只需补齐缺失的蛋白
    target = None
    if ids_to_query is not None:
        need = [u for u in ids_to_query if u not in already]
        if not need:
            print(f"[AM] 无需扫描 (缓存已覆盖全部 {len(ids_to_query)} 个目标蛋白)")
            return
        target = set(need)

    tmp = {}
    print("[AM] 扫描 TSV (仅一次) ...")
    t0 = time.time()
    with open(am_path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('uniprot'):
                continue
            c = line.strip().split('\t')
            if len(c) < 4:
                continue
            uid = c[0]
            if uid in already:
                continue
            if target is not None and uid not in target:
                continue
            m = vr.match(c[1])
            if m:
                pos = int(m.group(1))
                score = float(c[2])
                tmp.setdefault(uid, {}).setdefault(pos, []).append(score)

    for uid, pos_scores in tmp.items():
        db['am_data'][uid] = {p: float(np.mean(v)) for p, v in pos_scores.items()}

    n_new = len(tmp)
    if n_new:
        print(f"[AM] 完成 ({time.time()-t0:.0f}s), 新增 {n_new} 个蛋白")
        am_cache = os.path.join(bio.CACHE_DIR, 'cache_alphamissense.pkl')
        pickle.dump(db['am_data'], open(am_cache, 'wb'))
    else:
        print(f"[AM] 无需新增 (已有 {len(already)} 个)")


def _batch_mygene(ids_to_query, db):
    """批量查询 MyGene.info（querymany），替代逐个 HTTP 请求"""
    missing = [uid for uid in ids_to_query if uid not in db['id_map']]
    if not missing:
        return
    print(f"[MyGene] 批量查询 {len(missing)} 个 ID ...")
    import requests as req

    chunk_size = 1000
    for i in range(0, len(missing), chunk_size):
        chunk = missing[i:i + chunk_size]
        try:
            r = req.post('https://mygene.info/v3/query', json={
                'q': chunk,
                'scopes': 'uniprot',
                'fields': 'symbol,entrezgene',
                'species': 'human'
            }, timeout=120)
            results = r.json()
            if isinstance(results, dict) and 'hits' in results:
                results = results['hits']
            for item in results:
                if 'query' not in item:
                    continue
                uid = item['query']
                db['id_map'][uid] = {
                    'symbol': item.get('symbol', ''),
                    'entrezgene': str(item.get('entrezgene', ''))
                }
        except Exception as e:
            print(f"    MyGene batch error: {e}")
        time.sleep(0.5)

    n_new = len(missing) - sum(1 for uid in missing if uid not in db['id_map'])
    print(f"[MyGene] 完成 (成功 {n_new}/{len(missing)})")
    id_cache = os.path.join(bio.CACHE_DIR, 'cache_id_mapping.pkl')
    pickle.dump(db['id_map'], open(id_cache, 'wb'))


def _batch_deeploc(ids_to_query, db, cache_dir):
    """批量运行 DeepLoc2：多个蛋白写一个 FASTA，一次 GPU 推理"""
    missing = [uid for uid in ids_to_query if uid not in db['dl_data']]
    if not missing:
        return

    print(f"[DeepLoc] 批量提取 {len(missing)} 个蛋白的亚细胞定位 ...")

    # 从 FASTA 提取所有需要的序列
    sequences = {}
    with open(FASTA_PATH) as f:
        cur, sq = None, []
        for line in f:
            l = line.strip()
            if not l:
                continue
            if l.startswith('>'):
                if cur:
                    sequences[cur] = ''.join(sq)
                cur = l.split('|')[1] if '|' in l else l[1:].split()[0]
                sq = []
            else:
                sq.append(l)
        if cur:
            sequences[cur] = ''.join(sq)

    # 只保留能在 FASTA 中找到的
    have_seq = [uid for uid in missing if uid in sequences]
    if not have_seq:
        print(f"[DeepLoc] 在 FASTA 中未找到任何序列，跳过")
        return

    # 写多序列 FASTA
    tmp_fasta = os.path.join(cache_dir, f'dl_batch_{int(time.time())}.fasta')
    with open(tmp_fasta, 'w') as f:
        for uid in have_seq:
            f.write(f'>{uid}\n{sequences[uid]}\n')

    import subprocess
    t0 = time.time()
    # deeploc2 在 esmfold2 conda 环境中，需用 conda run 调用
    deeploc_cmd = ['conda', 'run', '-n', 'esmfold2', 'deeploc2',
                   '--fasta', tmp_fasta, '--model', 'Fast',
                   '--device', 'cuda', '-o', cache_dir]
    subprocess.run(deeploc_cmd, check=True,
                   env={**os.environ, 'TORCH_HOME': '/tmp/torch_cache'})

    # 解析输出 CSV
    csvs = sorted(glob.glob(os.path.join(cache_dir, '*.csv')),
                  key=os.path.getmtime, reverse=True)
    if csvs:
        df = pd.read_csv(csvs[0])
        for _, r in df.iterrows():
            pid = str(r.iloc[0]).split('|')[1] if '|' in str(r.iloc[0]) else str(r.iloc[0])
            db['dl_data'][pid] = np.array([float(r.get(c, 0)) for c in bio.DEEPLOC_LOC_COLS],
                                          dtype=np.float32)

    os.remove(tmp_fasta)
    print(f"[DeepLoc] 完成 ({time.time()-t0:.0f}s), 处理 {len(have_seq)} 个蛋白")

    dl_cache = os.path.join(bio.CACHE_DIR, 'cache_deeploc.pkl')
    pickle.dump(db['dl_data'], open(dl_cache, 'wb'))


# ══════════════════════════════════════════════════════════
# 并行 Worker（模块级别，可被 multiprocessing pickle）
# ══════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════
# 单 dimer 计算
# ══════════════════════════════════════════════════════════

def predict_models(feats, model_paths):
    """用两个 RF 模型（SPOC ESMFOLD / Structural classifier）对特征打分。"""
    out = {}
    for name, mp in model_paths.items():
        b = pickle.load(open(mp, 'rb'))
        model, imputer, feature_names = b['model'], b['imputer'], b['features']
        X = np.array([[feats.get(f, np.nan) for f in feature_names]], dtype=np.float64)
        if np.isnan(X).all():
            out[f'score_{name}'] = np.nan
            continue
        out[f'score_{name}'] = float(model.predict_proba(imputer.transform(X))[:, 1][0])
    return out


def score_dimer(cif_path, npz_path, uniprot_A, uniprot_B,
                db, skip_bio, full_lengths, model_paths):
    """计算单个 dimer：adapt + C+ + 结构特征 + 生物特征 + RF 分数。

    返回一个 dict（元信息 + RF 分数 + 全部特征），可直接转为一行 DataFrame。
    """
    out = {'uniprot_A': uniprot_A, 'uniprot_B': uniprot_B,
           'fname': os.path.splitext(os.path.basename(cif_path))[0],
           'cif': cif_path, 'npz': npz_path, 'n_c+': 0}
    try:
        npz_path, d = adapt_npz(npz_path, cif_path)
        if npz_path is None:
            out['error'] = 'adapt_failed'
            return out

        cpm = compute_cplus(npz_path)
        n_cplus = cpm['n_positive_contacts'] if cpm else 0
        out['n_c+'] = n_cplus

        # 折叠链实际长度（npz）作为 domain 区间兜底
        _asym = d['input_asym_id'].flatten()
        n_res_a = int((_asym == 0).sum())
        n_res_b = int((_asym == 1).sum())
        len_a = full_lengths.get(uniprot_A, n_res_a)
        len_b = full_lengths.get(uniprot_B, n_res_b)

        row = pd.Series({'uniprot_A': uniprot_A, 'uniprot_B': uniprot_B,
                         'dom_A_start': 1, 'dom_A_end': len_a,
                         'dom_B_start': 1, 'dom_B_end': len_b,
                         'dimer_id': out['fname']})
        feats = compute_struct(npz_path, cif_path, full_lengths, row)
        if feats is None:
            out['error'] = 'struct_fail'
            return out
        feats['n_positive_contacts'] = n_cplus
        if cpm is not None:
            # Structural classifier 需要 plddt_A/plddt_B（compute_contacts 已返回）
            feats['plddt_A'] = cpm['plddt_A']
            feats['plddt_B'] = cpm['plddt_B']

        if not skip_bio:
            row2 = pd.Series({'uniprot_A': uniprot_A, 'uniprot_B': uniprot_B,
                              'dom_A_start': 1, 'dom_A_end': len_a,
                              'dom_B_start': 1, 'dom_B_end': len_b,
                              'dimer_id': out['fname']})
            feats.update(extract_bio_features(row2, db, npz_path=npz_path))

        out.update(predict_models(feats, model_paths))
        out.update(feats)
        return out
    except Exception as e:
        out['error'] = str(e)[:100]
        return out


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Score a single protein dimer (CIF + NPZ) with the two '
                    "chains' UniProt IDs (SPOC ESMFOLD / Structural classifier).")
    p.add_argument('--cif', required=True, help='dimer structure .cif file')
    p.add_argument('--npz', required=True, help='matching .npz file from ESMFold2')
    p.add_argument('--uniprot_A', required=True, help='UniProt ID of chain A')
    p.add_argument('--uniprot_B', required=True, help='UniProt ID of chain B')
    p.add_argument('--output', default='inference_result.tsv', help='output tsv path')
    p.add_argument('--skip_bio', action='store_true',
                   help='skip biological features (use only the structure model)')
    args = p.parse_args()

    if not os.path.exists(args.cif):
        sys.exit(f'CIF not found: {args.cif}')
    if not os.path.exists(args.npz):
        sys.exit(f'NPZ not found: {args.npz}')

    # ── 生物数据库（--skip_bio 时跳过）──
    if not args.skip_bio:
        db = load_databases()
        # 缓存预热：只需两个链的 UniProt ID
        uniprots = {args.uniprot_A, args.uniprot_B}
        _batch_am_data(db, uniprots)
        _batch_mygene(uniprots, db)
        _batch_deeploc(uniprots, db, bio.CACHE_DIR)
        t5_path = os.path.join(bio.SPOC_DIR, 'ProtT5_embedding/per-protein.h5')
        db['t5'] = bio.load_t5_embeddings(t5_path)   # 单进程直接加载（h5py 不可 pickle）
    else:
        db = {}

    # ── 结构特征依赖 + RF 模型 ──
    full_lengths = load_full_lengths(FASTA_PATH)
    models = {
        'all_feat': os.path.join(MODEL_DIR, 'rf_all_feat_model.pkl'),
        'struct_only': os.path.join(MODEL_DIR, 'rf_struct_feat_model.pkl'),
    }

    # ── 打分（单 dimer）──
    print(f'Scoring: {os.path.basename(args.cif)}  ({args.uniprot_A} / {args.uniprot_B})')
    row = score_dimer(args.cif, args.npz, args.uniprot_A, args.uniprot_B,
                      db, args.skip_bio, full_lengths, models)
    df = pd.DataFrame([row])
    df.to_csv(args.output, sep='\t', index=False)

    print(f'\n完成: 1 条 → {args.output} ({len(df.columns)} 列)')
    for c in ['score_all_feat', 'score_struct_only']:
        if c in df.columns:
            v = df[c].iloc[0]
            print(f'  {c}: {v:.6f}' if pd.notna(v) else f'  {c}: NaN')
    if 'error' in df.columns and pd.notna(df['error']).any():
        print('  注意:', df['error'].iloc[0])

    # 关闭 T5
    if 't5' in db and hasattr(db['t5'], 'close'):
        db['t5'].close()


if __name__ == '__main__':
    main()
