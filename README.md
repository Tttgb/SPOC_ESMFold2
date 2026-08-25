# DDI-Classifier 打包版（GitHub 仓库）

基于 **ESMFold2 预测结构 + 公开生物数据库** 的蛋白 domain–domain 相互作用（DDI）
二分类器，附批量推理脚本与论文全量图的复现 notebook。

- **模型**：Random Forest，三套——
  - `all-feature RF`（39 结构 + 14 生物特征）
  - `structure-only RF`（49 结构特征）
  - `degree-match RF`（纯相互作用网络度特征，作数据泄露检验基线）
- **推理**：`batch_inference.py`，对给定 dimer 结构（CIF/NPZ）输出 RF 分数 + 全特征表
- **复现**：`notebooks/visualization.ipynb`，复现论文中 ROC / AUPR / FDR-recall /
  置换重要性 / degree 泄露基线 / ARF6 案例等全部图

---

## 目录结构

```
classifier_package/
├── README.md
├── requirements.txt
├── batch_inference.py          # 批量推理主脚本（独立运行）
├── modules/                    # 特征计算模块
│   ├── structure_feature_export.py   # 结构特征（pLDDT/PAE/接触/ipSAE 等）
│   ├── compute_contacts.py           # 正接触 C+ 计算
│   └── biology_feature_export.py     # 生物特征（BioGRID/Coexpr/CRISPR/DepMap/T5/AM）
├── notebooks/
│   └── visualization.ipynb     # 论文图复现 notebook（完整版）
├── data/
│   ├── models/                 # 推理用模型 pkl（入库）
│   ├── deeploc_output/         # DeepLoc/ID 映射缓存（入库，AlphaMissense 缓存除外）
│   ├── fasta/                  # 人类蛋白质组参考序列（入库）
│   ├── spoc/                   # 11GB 生物数据库（不入库，见 README“数据说明”）
│   └── analysis/               # notebook 配套数据（入库，自包含）
│       ├── {random,PDB_decoy,XL_MS,PDB_contact,homodimer_random,XL_MS_random}/
│       │                       各来源 *_features_biology.tsv
│       ├── dataset/            # train/test.tsv、三套模型 pkl、置换重要性 csv
│       └── example/ARF6_0/     # ARF6 案例（CIF/NPZ/推理输出）
├── test_input/                 # 推理示例输入（3 对 dimer）
├── test_out.tsv                # 推理示例输出（78 列特征 + RF 分数）
└── scripts/
    └── check_spoc_db.py        # 检查 data/spoc 数据库完整性
```

---

## 环境与安装

```bash
# Python 3.9+
pip install -r requirements.txt
```

依赖：`numpy pandas scipy scikit-learn biopython h5py tqdm requests
matplotlib seaborn py3Dmol ipykernel`

> 推理仅需 `numpy/pandas/scipy/scikit-learn/biopython/h5py/tqdm/requests`；
> `matplotlib/seaborn/py3Dmol/ipykernel` 用于 notebook 绘图复现。

---

## 一、模型推理（batch_inference.py）

对一批 dimer（每个含 `.cif` + `.npz`，来自 ESMFold2 多聚体预测）批量打分，
输出每对的 RF 概率 + 全部 78 列特征。

```bash
# 基本用法
python batch_inference.py \
    --input_dir <dimer目录> \
    --target_uniprot <目标蛋白UniProt> \
    --output <out.tsv>

# 示例：test_input/ 下 3 对 ARF6 候选 → 与仓库内 test_out.tsv 一致的输出
python batch_inference.py --input_dir test_input --target_uniprot P62330 --output ARF6_0.tsv

# 可选：只算结构特征（跳过生物特征数据库加载）
python batch_inference.py --input_dir test_input --target_uniprot P62330 --output ARF6_0.tsv --skip_bio
```

参数：

| 参数 | 说明 |
|---|---|
| `--input_dir` | 必需。包含 dimer 的 `.cif` + `.npz` 的目录 |
| `--target_uniprot` | 必需。目标蛋白 UniProt ID（如 ARF6 → `P62330`） |
| `--output` | 输出 tsv 路径，默认 `batch_results.tsv` |
| `--skip_bio` | 跳过生物特征（仅用 structure 模型） |

**输入格式**：ESMFold2 多聚体预测输出，每个 dimer 两个文件
`<dimer_id>.cif`（模型结构）与 `<dimer_id>.npz`（含 `plddt`/`pae`/`iptm` 等）。
若 NPZ 缺少 `sample_atom_coords`/`input_asym_id` 等键，脚本会自动从 CIF 适配生成
`<dimer_id>_ad.npz`。

**输出**：`gene / uniprot_A / uniprot_B / target / fname / n_c+ / score_all_feat /
score_struct_only` + 39 结构特征（pLDDT、PAE、接触、化学、pDockQ、ipTM、ipSAE 19 项…）
+ 14 生物特征（BioGRID、共表达、CRISPR、DepMap、ProtT5、AlphaMissense）。

> 首次运行会加载 `data/spoc/` 生物数据库（约 5–10 分钟），并自动生成
> `data/deeploc_output/cache_alphamissense.pkl` 缓存（首次约 3 分钟，之后秒级）。

---

## 二、论文图复现（notebooks/visualization.ipynb）

单个完整 notebook 复现论文中与**特征分析 / 模型评估**相关的全部图：

| 章节 | 内容 | 输出图 |
|---|---|---|
| 路径配置 + 全流程追踪（可选） | 自动指向仓库内 `data/analysis/`；追踪表需原始中间数据，缺失自动跳过 | 表格 |
| 特征加载 + train/test 划分 | 6 来源特征合并、train/test 正负样本统计 | 表格 |
| 所有特征 ROC | 高亮 Top3/Bottom3/ipTM | `fig_roc_all_features.pdf` |
| 结构关键分数 ROC | test set，AUC<0.5 自动反转 | `fig_roc_struct_key_scores.pdf` |
| 结构 vs 生物特征 ROC | 结构（蓝）vs 生物（红）对比 | `fig_roc_struct_vs_bio.pdf` |
| 三模型 AUPR | Precision-Recall，all/struct/degree-match | `fig_aupr_test_3models.pdf` |
| FDR=5% 时 Recall | 1:1 抽样，均值曲线求交 | `fig_fdr5_recall_threshold.pdf` |
| 多分数 × 多比例 Recall/FDR | 1:1~1:128 子集 | `fig_fdr5_recall_ratio_*.pdf` |
| 1:128 子集 Recall 柱状图 | FDR=5%/10%，带误差棒 | `fig_recall_fdr*_bars_*.pdf` |
| 固定阈值法 Recall 散点 | 1:128 子集 × 100 次抽样 | `fig_recall_fixedthr_*.pdf` |
| RF all vs struct 散点 | 两模型预测概率对比 | `fig_rf_all_vs_struct_scatter.pdf` |
| 置换重要性 | Gini + test AUC drop | `fig_perm_{gini,aucdrop}_*.pdf` |
| ARF6 案例 | PAE/pLDDT、3D 结构、RF score vs ipTM/ipSAE | `fig_pae_plddt_structure.pdf` 等 |

**运行方式**：

```bash
cd notebooks
jupyter notebook            # 或 VS Code 打开 notebooks/visualization.ipynb
```

- 从上到下依次运行所有 cell；
- 首个 cell 会自动把 `REPO` 指向仓库根（当前工作目录的上一级），`BASE` 指向
  `data/analysis/`；也可设置环境变量 `CLASSIFIER_PACKAGE` 覆盖仓库根；
- 图统一输出到仓库根 `figures/` 目录（自动创建）。

> **全流程追踪表（C01）** 依赖原始 pipeline 中间数据（`split_domian` /
> `dataset_domain` / `contact_positive` / `homology_reduce_1st,2nd` 等，体积大，
> 未随仓库发布）。数据缺失时自动跳过该表，**不影响后续所有绘图**。本机若想复现：
> `export ESMFOLD_RAW_DATA=<原始 ESMFOLD_filter 目录>` 后重跑首个 cell。

---

## 三、数据说明

### 随仓库发布（可直接使用）

| 路径 | 说明 |
|---|---|
| `data/models/` | 推理用模型：`rf_all_feat_model.pkl`、`rf_struct_feat_model.pkl` |
| `data/fasta/` | `human_proteomes_reviewed.fasta`（人类蛋白质组参考） |
| `data/deeploc_output/` | `cache_id_mapping.pkl`、`cache_deeploc.pkl`（ID 映射 / DeepLoc 缓存） |
| `data/analysis/` | notebook 复现配套数据（6 来源特征、train/test、三套模型、置换重要性 csv、ARF6 案例） |
| `test_input/`、`test_out.tsv` | 推理示例输入与输出 |

### 需自行获取：`data/spoc/`（约 11 GB）

`data/spoc/` 存放生物特征所需的外部数据库（**未随仓库发布**，已在 `.gitignore`
中排除）。请下载后按如下结构放置：

```
data/spoc/
├── AlphaMissence/AlphaMissense_aa_substitutions.tsv      # AlphaMissense 全蛋白组
├── CoexpressDB/                                           # 人源共表达数据库
├── DepMap/CRISPRGeneEffect.csv                            # DepMap CRISPR 基因效应
├── ProtT5_embedding/per-protein.h5                        # ProtT5 蛋白嵌入
├── biogrid/BIOGRID-ALL-5.0.258.tab3.txt                   # BioGRID 相互作用
└── biogrid/biogrid_ORCS/protein_hit_screens.pkl           # CRISPR ORCS 筛选
```

运行 `python scripts/check_spoc_db.py` 可检查上述文件是否齐全、缺失项会逐个列出。

> 若部分数据库缺失，推理仍可运行（缺失数据库对应的特征列会缺失，由 RF 的
> imputer 填充 NaN；`--skip_bio` 可完全跳过生物特征）。

### 运行时自动生成（无需入库）

- `data/deeploc_output/cache_alphamissense.pkl`（约 132 MB，从 `data/spoc/AlphaMissence`
  生成，首次约 3 分钟，已 `.gitignore`）
- `figures/`（notebook 图输出，已 `.gitignore`）

---

## 引用 / License

本仓库由 `classifier_package` 整理发布。若用于论文，请引用
**DDI-Classifier（domain–domain interaction classifier）** 相关工作；License 见
`LICENSE`（默认 MIT，可自行替换）。

如有问题请开 issue。
