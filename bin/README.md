# `bin/` —— 项目归档区

创建时间：2026-09-21
最近整理：2026-09-22（移入 `_ab*`、`_baseline_*`、`_dimcheck`、`_vis3f`、`_gui_dataset`、`_kclass_check`、`_relabel_check`）；
        2026-09-22 晚（S1/S3a/S3b/S4 落地后追加 `_s1_check`、`_scheme_ab`、`_s1_thresholds`、`_s3a_rich`、
        `_ab13_*`、`_s3b_replay`、`_stack_dim`）

根目录只保留**当前仍在用**的脚本与文档。一次性脚本、被取代的实现、专项诊断出图、
以及临时运行产物，统一移到这里，让根目录一眼能看出哪些是流水线的一部分。

> 这里的东西**不是删除**，只是搬家：需要时可以直接拿回去用（见每项的“如何恢复”）。
> 归档目录仍被 `.gitignore` 忽略，所以快照不会进入版本库。

## 归档清单

| 条目 | 原位置 | 为什么归档 | 如何恢复 |
|---|---|---|---|
| `after_split_data/` | 根目录 | `reference_code/pipeline.py` 的输出（`train` / `validation` / `test` 的 `arrays/*.npy` + `split_manifest.json` + `provenance/`）。**仅** `verify_dyn_envelope.py` 当作算法参照读它，其余流水线一律走 `raw_data/`；`load_split()` 也读不了它（它把验证集叫 `validation/` 而代码找 `val/`）。 | 已把 `verify_dyn_envelope.py` 的 `REFERENCE_DIR` 指向 `bin/after_split_data`；换位置时设环境变量 `DYN_REFERENCE_DIR` 即可 |
| `reference_code/` | 根目录 | 参考实现的原始工作目录（`pipeline.py` 的 `segment()` 等），是本仓库算法的源头。**没有任何 Python 代码 import 它**，只在注释/文档里作为算法出处被引用。 | 直接读即可；单独跑某脚本时需自己装 `scipy`（仓库主流程只用 NumPy） |
| `_recon.ps1` | 根目录 | 2026-09 迁移 Miniconda 时的一次性**环境侦查**脚本（统计目录体积、剩余磁盘、`.condarc` / `pip.ini` 位置）。与 EMD/MLP 流水线无关，迁移已完成。 | `powershell -File bin\_recon.ps1`（只读，不修改任何东西） |
| `visualize_dyn_envelope_cases.py` | 根目录 | `dyn_envelope` 移植期的**专项诊断出图**：只画 `N35_P1_01` / `N2_P2_01` / `N5_P7_01` 三个“单通道包络不过阈 → 退化为联合 RMS 规则”的特例。GUI 不调用它，`README.md` / `PROJECT_SUMMARY.md` 的脚本清单里也没有它，结论已经写进正文。 | `python bin\visualize_dyn_envelope_cases.py`（在项目根目录执行，输出 `visualizations/dyn_envelope_cases.png`；需要 `matplotlib`） |
| `backup/` | 根目录 | 历史快照总目录，含 `0_no_gui`、`1_preprocess_change`、`2_no_custom_range`、`3_no_envelope`、`4_stale_dyn_preport`、`5_experiments_2026-09-17`。现有代码都不引用它们，只作对照与追溯。 | 需要对照时把对应子目录拷回根目录并改名，或直接读其中的 `config.json` / `metrics.json` |
| `_recon_check/` | 根目录 | 2026-09-17 的临时 `--output-dir` 冒烟输出（2 组实验）。 | 用相同命令行重跑即可，不需要恢复 |
| `_smoke_all/` | 根目录 | 2026-09-17 的全矩阵冒烟输出（16 组 `channels_3` 短跑）。**早于双塔 MLP 默认改写**，指标不可直接引用。 | 同上；正式结果请写到 `experiments/` |
| `_verify_fix/` | 根目录 | 2026-09-17 校验 `dyn_envelope` 修复时的临时输出（3 组）。 | 同上 |
| `_ab2/` | 根目录 | 2026-09-22 特征集 A/B 扫描的**原始输出**：`legacy_none` 与 `compact_core` 两组定位特征 × seed 42–46，共 10 组实验（`_ab_run.ps1` 的产物）。结论已写进 `PROJECT_SUMMARY.md` §8.4，目录本身只是留档。 | `bin\_ab_run.ps1` 重跑（在项目根目录执行） |
| `_ab_legacy/` | 根目录 | 同一轮 A/B 扫描里 legacy 侧的早期输出（只有 `form_top3_mean__regions_dyn_envelope__channels_1` 一组），已被 `_ab2/` 取代。 | 不需要恢复，看 `_ab2/` |
| `_ab_run.ps1` | 根目录 | 驱动上述扫描的 PowerShell 脚本：按 `legacy_none` / `compact_core` × seeds 42–46 逐组调用 `run_emd_experiments.py`。**是文件不是目录**，所以仍被 git 跟踪。 | `powershell -File bin\_ab_run.ps1`（必须在项目根目录执行，脚本内路径都是相对的） |
| `_baseline_full/` | 根目录 | 浅层基线的**全量**输出（`baseline_report.txt` / `baseline_results.json`），对应 `PROJECT_SUMMARY.md` §10 记录的结论。 | 用文档里的 `run_baseline_models.py` 命令重跑 |
| `_baseline_smoke/` | 根目录 | 同上的短跑冒烟版，只用来验证 CLI 与输出格式，指标不可引用。 | 同上 |
| `_dimcheck/` | 根目录 | 特征维度自检的临时输出（`channels_3` 一组），用于核对 `PROJECT_SUMMARY.md` §8.3 的维度表。 | 重跑对应命令即可 |
| `_vis3f/` | 根目录 | 可视化回归对照输出（`emd/`、`pre/` 各 2 张图 + `selection_manifest.json`），留给 `bin/_gui_dataset/regress_vis.py` 做快照比对。 | `python bin\_gui_dataset\regress_vis.py compare ...` |
| `_gui_dataset/` | 根目录 | 2026-09-22 多数据集选择器的**无头冒烟脚本**：`smoke_dataset_gui.py`（GUI 列结构）、`smoke_dataset_switch.py`（数据层 `discover_datasets` / `ExperimentCatalog`）、`smoke_kclass_figures.py`（k 分类图表可见性），外加可视化回归工具 `regress_vis.py` 与两份快照 JSON。 | `python bin\_gui_dataset\smoke_dataset_switch.py`（脚本内 `APP_DIR` 已随搬家改成 `parents[2]`，可直接跑） |
| `_kclass_check/` | 根目录 | k 分类重构（`label_scheme` / `classification_metrics`）的校验脚本与回归输出：`check_metrics.py`（纯计算自检）、`check_binary_regression.py` ＋ `bin_regression/`（与既有二分类实验逐位对照）、`cls2_run/` `cls3_run/`（2/3 类实跑）、`baseline_regression/`、`cm_regression/`、`vis_regression/`、`json_deep_diff.py`。 | `python bin\_kclass_check\check_metrics.py`（脚本内 `ROOT` 已改成 `parents[2]`） |
| `_relabel_check/` | 根目录 | 按厚度重标注的校验数据集 `cls3/`（3 类、约 96 MB，与 `raw_data_relabeled/cls3_thr0.8_1.2` 同族）＋一次性还原脚本 `restore_raw_data.py`。 | 数据集可直接用（`--data-dir bin/_relabel_check/cls3`）；`restore_raw_data.py` 是历史记录：它引用的 `keep/manifest.csv` 已不存在，`cls3/README.txt` 里的路径也是搬家前的 |
| `_scheme_ab/`、`_s1_check/` | 根目录 | `compare_label_schemes.py` **定稿前**的两次中间配对报告：前者只按 label scheme 分组，后者按 `scheme@scaler[cols]` 交叉。| 直接用定稿后的 `compare_label_schemes.py` 重跑（命令见 `PROJECT_SUMMARY.md` §11.1） |
| `_s1_thresholds/` | 根目录 | **S1 正式结果**：4 个阈值 × `lda,prior` × 5×5 折的配对报告（`comparison_report.txt` / `comparison_results.json`），外加地板值/边界样本运算脚本 `_boundary.py`。`PROJECT_SUMMARY.md` §11.1 的两张表出自这里。 | 用 §11.1 的命令重跑；`_boundary.py` 和 `_s3a_rich/_layout.py`、`_s3b_replay/_compare.py` 一样是**一次性的辅助脚本，不参与归档流程** |
| `_s3a_rich/` | 根目录 | S3a（`rich` 特征族）的**特征提取产物**（`form_top3_mean__regions_dyn_envelope__channels_1__cls2_thr1.3`）＋ `_paired` / `_paired2` 两份配对报告＋列布局核对脚本 `_layout.py`。§11.2 的表格出自 `_paired2`。 | 不加 `--feature-set` 相关开关重跑即可；`--experiment` 直接指向这个目录可复用特征 |
| `_ab13_keep/`、`_ab13_ratio/`、`_ab13_atten/` | 根目录 | 1.3 mm 阈值下 S3b（`--branch-ratio`）的 **A/B 三件套**：`_ab13_keep` 是 16 列基线提取，`_ab13_ratio` / `_ab13_atten` 是 `ratio` / `attenuation` 臂。后两者各带配对报告；`_ab13_ratio/_paired_seed7/` 是把 fold 种子换成 7 的**独立重复**——§11.3 里唯一那个 +1.3 pt 正结果就在这里翻转成 −1.7 pt。 | `compare_label_schemes.py --experiment bin/_ab13_keep/...` 重跑；换种子用 `--seed 7 --output-dir .../_paired_seed7` |
| `_s3b_replay/` | 根目录 | S3b 验证时**手工重放旧实验产物**用的目录（含 `form_top3_mean__regions_dyn_envelope__channels_1`）＋ `_compare.py` 逐位比对脚本。`verify_branch_ratio.py` / `verify_channel_contrast.py` 的夹具与它同源。 | 跑 `python verify_branch_ratio.py`（会自己重建同样的重放目录） |
| `_stack_dim/` | 根目录 | **全选项叠加**（`rich` + `channels_3` + `contrast normalized` + `branch-ratio attenuation`）的维度核对输出，用来确认 §11.5 的 `feature_dim = 123` / `[41, 41, 41]` / `fan_in = 192`。 | 用 §11.5 的命令重跑 |
| `__pycache__/` | 根目录 | Python 字节码缓存，其中 `dynamic_split*.pyc`、`_grad_check*.pyc` 对应的**源文件已从仓库删除**，只会造成误导。 | 可随时删除，Python 会自动重建（根目录会再生成一个，属正常现象） |

## 约定

1. **判断标准**：能不能被 GUI 或 `README.md` / `PROJECT_SUMMARY.md` 里的推荐命令直接用到？
   不能，就进 `bin/`。
2. **`verify_*.py` 不归档**：`verify_dyn_envelope.py`、`verify_mlp_gradients.py`、
   `verify_channel_aggregation.py`、`verify_channel_contrast.py`、`verify_branch_ratio.py`、
   `verify_emd_features.py`、`verify_scalers.py` 是七份**可重跑的回归自检**，两份文档都明确引用了它们，继续留在根目录。
   同理，`compare_label_schemes.py` 是 S1 配对评价协议的正式工具（§11.1），也不归档。
   > `verify_dyn_envelope.py` 虽然留在根目录，但它读的参考数据在 `bin/after_split_data/`，
   > 所以搬动 `bin/after_split_data/` 时需同步改 `REFERENCE_DIR` 或设 `DYN_REFERENCE_DIR`。
   > 依赖方向是“根目录脚本 → bin 数据”，`bin/` 里的东西从不反向 import 根目录。
3. **`_recon.ps1`、`visualize_dyn_envelope_cases.py` 仍被 git 跟踪**（`.gitignore` 的 `_*/`、
   `__pycache__/`、`backup/` 规则只匹配**目录**，或在忽略目录内部生效），所以历史没有丢；
   而 `bin/backup/`、`bin/_*/`、`bin/__pycache__/`、`bin/after_split_data/`、`bin/reference_code/` 依旧被忽略。
4. 归档时用 `git mv`（保留历史）而不是删除再新建；被 `.gitignore` 忽略的目录（如
   `after_split_data/`、`reference_code/`）用普通移动即可。
5. ⚠️ `bin/reference_code/` 里 `pipeline.py` 开头按相对位置找 `vendor`：
   原来 `parents[1]` = `骨分层/`，现在 = 项目根，所以那个可选路径失效（`if vendor.exists()` 保护，
   不会报错）。需要单独重跑参考脚本时，手工把 `scipy` 等依赖装上即可。
6. **搬家后要检查 `parents[N]`**：脚本里凡是 `Path(__file__).resolve().parents[1]` 来找项目根的，
   下沉一层后都得多走一级。2026-09-22 这次已把以下脚本改成 `parents[2]` 并验证仍可运行，
   路径拼接里也补上了 `bin/`：
   - `bin/_gui_dataset/smoke_dataset_gui.py`、`smoke_dataset_switch.py`、`smoke_kclass_figures.py`（`APP_DIR`）
   - `bin/_kclass_check/check_metrics.py`（`ROOT`）、`check_binary_regression.py`（`ROOT` 与 `NEW`）
   - `bin/_relabel_check/restore_raw_data.py`（`ROOT` 与 `MANIFEST`）

   > 仍然失效的两处只影响“当时怎么跑的”这类历史信息，不影响代码：
   > `bin/_relabel_check/restore_raw_data.py` 依赖的 `keep/manifest.csv` 已经不存在，
   > `bin/_relabel_check/cls3/README.txt` 与 `bin/_kclass_check/**/summary.json`、
   > `baseline_results.json` 里记录的 `_xxx_check/...` 路径是搬家前的位置。

## 当前根目录剩下的临时目录

无。`_*/` 规则只匹配目录，所以根目录下已不再有临时目录；剩下四个目录都是流水线的一部分：
`bin/`（归档区）、`experiments/`（实验输出）、`visualizations/`（出图）、
`raw_data/` ＋ `raw_data_relabeled/`（数据）。根目录里由 Python 自动重建的 `__pycache__/` 属正常现象，可随时删除
