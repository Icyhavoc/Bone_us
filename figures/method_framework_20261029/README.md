# 整体方法框架图

代码依据：`Icyhavoc/Bone_us`，已 fetch 的 `origin/main` 提交 `f2eebeeafe707ff5009674ab81ea19a96a6e9f83`，2026-10-29 核对。

配置：`--forms max1 --channel-mode 4 --region-set dyn_envelope_full --model-variant cross_vote`。用户名称 `full_dynamic_envelope` 对应代码 `dyn_envelope_full`，不是另一个实现。

本图应用 Imbad0202/academic-research-skills 中 academic-paper/agents/visualization_agent.md 的代码绘图流程：根据结构关系绘制概念图、可区分的颜色与文本标签、提供可复现代码、PNG/SVG/PDF、图注及 LaTeX 引用。未调用文生图 API。图为方法结构示意，不含实验结果或实测波形。

安装结果：通过官方 skill-installer 的 `--method git --path academic-paper` 成功安装至 `C:/Users/27871/.codex/skills/academic-paper`，来源仓库提交 `b8d050f24f2e1c465c4410eace8634ea3af15930`。本次只安装所需 academic-paper 技能，不代表整个仓库的五个技能全部安装。全库克隆与 ZIP 下载因停滞/超时未完成，已停止；本次使用的原文另保存在 skill_sources 中。

视觉检查：已检查 PNG，完成两轮连线/注释排版调整；六路、尺寸、选帧规则和投票公式均与代码核对。统计图坐标、误差条等检查项不适用于本概念框架图。图面采用英文；图注与复现说明见本文件。

## 核对记录

- `emd_pipeline.py::detect_dynamic_span`：完整双通道帧堆栈先用于定位，定位器逐通道使用自身 Top3 帧构造信号，Hilbert 包络与平滑后确定主体窗；不是先 max1 再定位。
- `channel4_overlap_order`：各自主体窗内每帧最大绝对幅值选 Top10，帧号取交集；max1 在交集中按 Beam1 峰值选第一帧，同幅度按帧号稳定排序。
- `prepare_frame_input`：无交集时回退旧双通道相似度排序。在 Beam1 Top10 中，使用同一绝对主体窗，±5 点位移的最大绝对去均值相关系数 × RMS 最小/最大比。不是当前 channel3 的独立选帧。
- `prepare_dynamic_envelope_branches`：各通道主体窗默认 170 点；尾窗从主体末尾连续取 600 点，不足右侧补零；全段为 896 点（默认深度 0–5 mm）。
- `resample_signals` 与 `tukey_window`：not-a-knot 三次样条到 512 点，再乘对称 Tukey 窗 α=0.3。包络用于边界定位，模型输入仍保留波形正负号。
- `vit_pipeline.py::prepare_vit_sample`：分支顺序为主体 ch1/ch2、尾窗 ch1/ch2、全段 ch1/ch2；输入 [N,6,1,512]。
- `vit_model.py::WaveformEncoder`：Conv1D patch embedding、固定正弦位置编码、Transformer、均值池化、LayerNorm。
- `RegionChannelCrossVoteViT`：六路独立编码器与分类头；各头为 Linear→GELU→Dropout→Linear→softmax。类别概率等权相加后 argmax；训练对平均概率取正确类负对数。二分类也不做 reward 或阈值搜索。

## 复现与使用

执行 `python draw_framework.py`，生成 `method_framework.png`（4800×3000，300 DPI）、`method_framework.svg` 和 `method_framework.pdf`。此为适合讨论、演示的详细总览；缩至论文双栏宽度前建议精简文字，避免字号过小。

**Figure 1.** *Overall framework for ultrasound bone classification.*

Note. A shared frame is selected by the channel4 Top10-intersection rule with the legacy similarity fallback. Each beam contributes a dynamic main window, a continuous padded tail and a full-signal branch. Signed signals are resampled and windowed before six independent 1D ViT classifiers. Equal-weight class-probability summation determines the predicted class. Window lengths and preprocessing parameters shown are code defaults; class definitions depend on the selected dataset.

```latex
% Requires \usepackage{graphicx}
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{method_framework.pdf}
  \caption{Overall framework for ultrasound bone classification using shared-frame selection, dynamic signal regions, and six independent 1D ViT classifiers with probability-based cross-voting.}
  \label{fig:method-framework}
\end{figure*}
```

图文追溯：源数据为上述 Git 提交的代码；变换脚本为同目录 draw_framework.py；图支持的方法说明为本文件核对记录；局限是示意图未展示全部定位阈值、人工校正表或实验超参数，不表达分类性能。
