#!/usr/bin/env python3
"""
Batch 推理（打包版）
========================================
独立运行包：所有支持数据库 / 模型 / 缓存都在 classifier_package/ 内，
无需依赖原始 ESMFOLD_filter 目录结构。

用法:
    python batch_inference.py --input_dir <dimer目录> --target_uniprot <目标> --output <out.tsv>
    python batch_inference.py --input_dir example/ARF6_0 --target_uniprot P62330 --output ARF6_0.tsv

输出: 一个大表（RF 分数 + 全部特征）
"""

import os, sys, json, time, gc, pickle, re, glob, argparse
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════
# 打包路径配置（可按需通过环境变量覆盖）
# ══════════════════════════════════════════════════════════
PACKAGE_DIR = os.environ.get('CLASSIFIER_PACKAGE',
                             '/home/data/xjd/ESMFOLD_filter/classifier_package')
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

def load_databases():
    """加载生物数据库，结果缓存在全局 dict 中"""
    if _DB_CACHE:
        return _DB_CACHE

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

# 全局共享数据（每个 worker 进程通过 initializer 设置一次）
_WORKER_DB = None
_WORKER_CONFIG = {}

def _init_worker(db_data, t5_path, uniprot_target, skip_bio, full_lengths):
    """每个 worker 进程初始化时调用一次，加载 T5 嵌入"""
    global _WORKER_DB, _WORKER_CONFIG
    _WORKER_DB = db_data
    _WORKER_DB['t5'] = bio.load_t5_embeddings(t5_path)  # 只加载一次
    _WORKER_CONFIG = {
        'uniprot_target': uniprot_target,
        'skip_bio': skip_bio,
        'full_lengths': full_lengths,
    }

def _full_worker(pair):
    """处理单个 pair：adapt + C+ + 结构 + 生物，返回 (info_dict, feats_dict) 不含 RF"""
    db = _WORKER_DB
    cfg = _WORKER_CONFIG
    try:
        npz_path, d = adapt_npz(pair['npz'], pair['cif'])
        if npz_path is None:
            return {'gene': pair['gene'], 'uniprot_A': pair['uniprot'],
                    'uniprot_B': cfg['uniprot_target'], 'target': pair['target'],
                    'fname': pair['fname'], 'npz': pair['npz'], 'cif': pair['cif'],
                    'n_c+': 0, 'error': 'adapt_failed'}, None
        cpm = compute_cplus(npz_path)
        n_cplus = cpm['n_positive_contacts'] if cpm else 0
        # 折叠链实际长度（npz）作为 domain 区间兜底，避免全长未知时误用 99999
        _asym = d['input_asym_id'].flatten()
        n_res_a = int((_asym == 0).sum())
        n_res_b = int((_asym == 1).sum())
        len_a = cfg['full_lengths'].get(pair['uniprot'], n_res_a)
        len_b = cfg['full_lengths'].get(cfg['uniprot_target'], n_res_b)
        row = pd.Series({'uniprot_A': pair['uniprot'],
                         'uniprot_B': cfg['uniprot_target'],
                         'dom_A_start': 1, 'dom_A_end': len_a,
                         'dom_B_start': 1, 'dom_B_end': len_b,
                         'dimer_id': pair['fname']})
        feats = compute_struct(npz_path, pair['cif'], cfg['full_lengths'], row)
        if feats is None:
            return {'gene': pair['gene'], 'uniprot_A': pair['uniprot'],
                    'uniprot_B': cfg['uniprot_target'], 'target': pair['target'],
                    'fname': pair['fname'], 'npz': npz_path, 'cif': pair['cif'],
                    'n_c+': 0, 'error': 'struct_fail'}, None
        feats['n_positive_contacts'] = n_cplus
        if cpm is not None:
            # 当前 rf_struct 模型需要 plddt_A/plddt_B（compute_contacts 已返回）
            feats['plddt_A'] = cpm['plddt_A']
            feats['plddt_B'] = cpm['plddt_B']

        if not cfg['skip_bio']:
            row2 = pd.Series({'uniprot_A': pair['uniprot'],
                              'uniprot_B': cfg['uniprot_target'],
                              'dom_A_start': 1, 'dom_A_end': len_a,
                              'dom_B_start': 1, 'dom_B_end': len_b,
                              'dimer_id': pair['fname']})
            feats.update(extract_bio_features(row2, db, npz_path=npz_path))

        info = {'gene': pair['gene'], 'uniprot_A': pair['uniprot'],
                'uniprot_B': cfg['uniprot_target'], 'target': pair['target'],
                'fname': pair['fname'], 'npz': npz_path, 'cif': pair['cif'],
                'n_c+': n_cplus}
        return info, feats
    except Exception as e:
        info = {'gene': pair['gene'], 'uniprot_A': pair['uniprot'],
                'uniprot_B': pair.get('uniprot_B', _WORKER_CONFIG.get('uniprot_target', '')),
                'target': pair['target'],
                'fname': pair['fname'], 'npz': pair['npz'], 'cif': pair['cif'],
                'n_c+': 0, 'error': str(e)[:100]}
        return info, None


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True,
                   help='包含 dimer 的 .cif + .npz 目录')
    p.add_argument('--output', default='batch_results.tsv')
    p.add_argument('--target_uniprot', required=True,
                   help='目标蛋白的 UniProt ID（如 ARF6→P62330）')
    p.add_argument('--skip_bio', action='store_true', help='跳过生物特征（只用 struct 模型）')
    args = p.parse_args()

    # ── 加载数据库 ──
    if not args.skip_bio:
        db = load_databases()
    else:
        db = {}

    # ── 扫描 pairs ──
    cif_files = sorted(glob.glob(os.path.join(args.input_dir, '*.cif')))
    pairs = []
    for cf in cif_files:
        base = os.path.splitext(cf)[0]
        nz = base + '.npz'
        if not os.path.exists(nz):
            print(f"[跳过] 无 NPZ: {cf}")
            continue
        # 解析文件名: {gene};{uniprot};{model}_{target}
        fname = os.path.splitext(os.path.basename(cf))[0]
        parts = fname.split(';')
        if len(parts) < 2:
            print(f"[跳过] 无法解析: {fname}")
            continue
        gene, uniprot = parts[0], parts[1]
        target = fname.rsplit('_', 1)[-1]
        pairs.append({'gene': gene, 'uniprot': uniprot, 'target': target,
                      'cif': cf, 'npz': nz, 'fname': fname})

    print(f"共 {len(pairs)} 个 pair")

    # ── 预加载结构特征依赖 ──
    full_lengths = load_full_lengths(FASTA_PATH)

    # ── 批量缓存预热（数据库已预缓存，通常是空操作）──
    uniprot_target = args.target_uniprot
    all_uniprots = {uniprot_target} | {p['uniprot'] for p in pairs}
    if not args.skip_bio:
        print(f"[缓存] 批量预热 {len(all_uniprots)} 个 UniProt ID ...")
        t_cache = time.time()
        _batch_am_data(db, all_uniprots)
        _batch_mygene(all_uniprots, db)
        _batch_deeploc(all_uniprots, db, bio.CACHE_DIR)
        print(f"[缓存] 完成 ({time.time()-t_cache:.0f}s)")

    # ── 加载 RF 模型 ──
    models = {
        'all_feat': os.path.join(MODEL_DIR, 'rf_all_feat_model.pkl'),
        'struct_only': os.path.join(MODEL_DIR, 'rf_struct_feat_model.pkl'),
    }

    # ── 全并行：每 worker 提取特征（不含 RF），RF 留到主进程批量预测 ──
    from multiprocessing import Pool, cpu_count
    n_workers = min(cpu_count(), 32)

    # T5 (h5py) 不可 pickle，用 initializer 让每个 worker 加载一次
    t5_path = os.path.join(bio.SPOC_DIR, 'ProtT5_embedding/per-protein.h5')
    db_for_workers = {k: v for k, v in db.items() if k != 't5'}

    print(f"[并行] {len(pairs)} pairs, {n_workers} workers ...")
    results = []
    all_feats = []
    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker,
              initargs=(db_for_workers, t5_path,
                        uniprot_target, args.skip_bio, full_lengths)) as pool:
        for i, (info, feats) in enumerate(pool.imap_unordered(_full_worker, pairs, chunksize=3)):
            results.append(info)
            all_feats.append(feats)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(pairs)}] {time.time()-t0:.0f}s")

    # ── 主进程批量 RF 预测（用满所有 CPU 核心）──
    print(f"[RF] 主进程批量预测 {len(results)} 条 ...")
    t_rf = time.time()
    for name, mp in models.items():
        b = pickle.load(open(mp, 'rb'))
        model = b['model']
        imputer = b['imputer']
        feature_names = b['features']
        if hasattr(model, 'n_jobs'):
            model.n_jobs = -1  # 用满所有核心

        # 构建特征矩阵
        X = np.full((len(all_feats), len(feature_names)), np.nan, dtype=np.float64)
        for j, feats in enumerate(all_feats):
            if feats is None:
                continue
            for k, fname in enumerate(feature_names):
                X[j, k] = feats.get(fname, np.nan)

        # 排除特征为空的样本
        valid = ~np.isnan(X).all(axis=1)
        scores = np.full(len(all_feats), np.nan, dtype=np.float64)
        if valid.any():
            X_imp = imputer.transform(X[valid])
            scores[valid] = model.predict_proba(X_imp)[:, 1]

        for j, s in enumerate(scores):
            results[j][f'score_{name}'] = float(s) if not np.isnan(s) else np.nan

    print(f"[RF] 完成 ({time.time()-t_rf:.0f}s)")

    # ── 保存：RF 分数 + 全部特征合并成一个大表 ──
    big_rows = []
    for info, feats in zip(results, all_feats):
        r = dict(info)
        if feats:
            r.update(feats)
        big_rows.append(r)
    df_big = pd.DataFrame(big_rows)
    df_big.to_csv(args.output, sep='\t', index=False)
    print(f"\n完成: {len(df_big)} 条 → {args.output}  ({len(df_big.columns)} 列)")

    # ── 统计 ──
    if 'score_all_feat' in df_big.columns:
        print(f"  all_feat:     mean={df_big['score_all_feat'].mean():.4f}, "
              f"median={df_big['score_all_feat'].median():.4f}")
    if 'score_struct_only' in df_big.columns:
        print(f"  struct_only:  mean={df_big['score_struct_only'].mean():.4f}, "
              f"median={df_big['score_struct_only'].median():.4f}")

    # 关闭 T5
    if 't5' in db and hasattr(db['t5'], 'close'):
        db['t5'].close()


if __name__ == '__main__':
    main()
