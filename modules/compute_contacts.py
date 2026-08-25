#!/usr/bin/env python3
"""
从 ESMFold2 预测结果 (.npz) 计算 inter-chain contact positive (C+) 指标。

C+ 标准 (PMID 文献):
  1. 至少 1 对 inter-chain 重原子距离 < 5 Å
  2. 两残基 pLDDT 均 > 0.5
  3. 不存在任何重原子对距离 < 1 Å（clash 过滤）
  4. PAE(x,y) < 15 且 PAE(y,x) < 15

输出:
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
    """从 npz 文件计算 C+ 指标。返回 dict 或 None。"""
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

    # ── 按链分组原子 ──
    a_mask = asym_id[atom2res] == 0
    b_mask = asym_id[atom2res] == 1
    if not a_mask.any() or not b_mask.any():
        return None

    coords_A = coords[a_mask]
    coords_B = coords[b_mask]
    res_A = atom2res[a_mask].astype(int)
    res_B = atom2res[b_mask].astype(int)

    # ── KDTree 查找所有 5 Å 以内的跨链原子对 ──
    tree_A = cKDTree(coords_A)
    pairs = tree_A.query_ball_tree(cKDTree(coords_B), r=DIST_CONTACT)
    # pairs[a_idx] = list of b_idx within distance r

    # ── 聚合到残基对 ──
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

    # ── 筛选 C+ ──
    c_plus = 0
    c_clash = 0 
    for (ri, rj), _ in res_pair_dist.items():
        # clash 过滤
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

    # ── 链 pLDDT ──
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

    # ── 加载 PKL ──
    print("加载 domain_pairs.pkl ...")
    with open(pkl_path, 'rb') as f:
        raw_pairs = pickle.load(f)
    print(f"  共 {len(raw_pairs)} 条记录")

    # ── 构建 dimer_id 列表 ──
    dimer_ids = []
    for entry in raw_pairs:
        dimer_id = (f"{entry['uniprot_A']}_{entry['dom_A_start']}_{entry['dom_A_end']}__"
                    f"{entry['uniprot_B']}_{entry['dom_B_start']}_{entry['dom_B_end']}")
        dimer_ids.append(dimer_id)

    # unique dimer 映射
    unique_ids, first_idx = [], {}
    for i, did in enumerate(dimer_ids):
        if did not in first_idx:
            first_idx[did] = i
            unique_ids.append(did)

    print(f"  unique dimer: {len(unique_ids)}")

    # ── 并行计算 ──
    n_workers = min(cpu_count(), 32)
    print(f"  使用 {n_workers} 个 worker 并行计算 ...")

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

    # ── 保存 pkl ──
    pkl_out = os.path.join(out_dir, 'contact_positive_results.pkl')
    with open(pkl_out, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── 输出 TSV ──
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
                    

    # ── 摘要 ──
    elapsed = time.time() - t0
    n_ok = len(results)
    n_total = len(unique_ids)
    print(f"\n{'='*50}")
    print(f"  完成: {elapsed:.1f}s")
    print(f"  Computed: {n_ok}/{n_total}")
    print(f"  Contact positive (C+ >= 5): {n_contact_pos} "
          f"({100*n_contact_pos/n_total:.1f}%)")
    print(f"  pkl saved: {pkl_out}")
    print(f"  tsv saved: {tsv_out}")


if __name__ == '__main__':
    main()
