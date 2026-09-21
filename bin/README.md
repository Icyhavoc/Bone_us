# `bin/` —— 项目归档区

创建时间：2026-09-21

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
| `__pycache__/` | 根目录 | Python 字节码缓存，其中 `dynamic_split*.pyc`、`_grad_check*.pyc` 对应的**源文件已从仓库删除**，只会造成误导。 | 可随时删除，Python 会自动重建 |

## 约定

1. **判断标准**：能不能被 GUI 或 `README.md` / `PROJECT_SUMMARY.md` 里的推荐命令直接用到？
   不能，就进 `bin/`。
2. **`verify_*.py` 不归档**：`verify_dyn_envelope.py`、`verify_mlp_gradients.py`、
   `verify_channel_aggregation.py` 是三份**可重跑的回归自检**，两份文档都明确引用了它们，继续留在根目录。
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
