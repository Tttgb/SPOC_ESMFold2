#!/usr/bin/env python3
"""检查 data/spoc/ 生物数据库是否齐全（不执行任何下载）。

用法:
    python scripts/check_spoc_db.py
    CLASSIFIER_PACKAGE=/path/to/classifier_package python scripts/check_spoc_db.py
"""
import os
import sys

REPO = os.environ.get(
    'CLASSIFIER_PACKAGE',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPOC = os.path.join(REPO, 'data', 'spoc')

# 推理所需的 spoc 数据库（相对 data/spoc/ 的路径 -> 说明）
REQUIRED = {
    'AlphaMissence/AlphaMissense_aa_substitutions.tsv': 'AlphaMissense 全蛋白组替换评分',
    'CoexpressDB/':                                  '人源共表达数据库（目录）',
    'DepMap/CRISPRGeneEffect.csv':                   'DepMap CRISPR 基因效应',
    'ProtT5_embedding/per-protein.h5':               'ProtT5 蛋白嵌入',
    'biogrid/BIOGRID-ALL-5.0.258.tab3.txt':          'BioGRID 相互作用',
    'biogrid/biogrid_ORCS/protein_hit_screens.pkl':  'CRISPR ORCS 筛选',
}


def main():
    print(f'仓库根: {REPO}')
    print(f'数据目录: {SPOC}')
    print('=' * 60)
    if not os.path.isdir(SPOC):
        print(f'[✗] data/spoc/ 不存在。请下载各生物数据库后按 README“数据说明”放置。')
        sys.exit(1)

    missing = []
    for rel, desc in REQUIRED.items():
        p = os.path.join(SPOC, rel)
        ok = os.path.isfile(p) if not rel.endswith('/') else os.path.isdir(p)
        tag = 'OK ' if ok else 'MISSING'
        print(f'[{tag}] {rel:<45s} # {desc}')
        if not ok:
            missing.append(rel)

    print('=' * 60)
    if missing:
        print(f'缺失 {len(missing)} 项，对应生物特征将缺失（RF imputer 填充 NaN）。')
        print('可用 --skip_bio 完全跳过生物特征。')
        sys.exit(1)
    print('数据库齐全，可运行完整推理。')


if __name__ == '__main__':
    main()
