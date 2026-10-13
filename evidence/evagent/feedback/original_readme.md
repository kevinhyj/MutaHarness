# Protein RNA PPI 评测图表 v2

核查日期：2026-10-11（八张图均为Arial；三张多轮图的横纵轴刻度为25 pt，本次仅精简PPI多轮图的legend文字，保留全部七项，去掉重复“+ RF”、Gemini版本及“10% reference”说明；Protein／RNA和另五张不变，数据、配色、坐标及误差范围不变，Protein多轮仍为共同47项）。这里只保留 **8张当前图**：few-shot三张、多轮迭代三张、少点→多点两张。监督基线统一选RF版；Gemini不接RF。PNG用于查看，可编辑SVG用于排版。

数据汇总表：[FEWSHOT_DATA_AND_PROTOCOLS.md](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/FEWSHOT_DATA_AND_PROTOCOLS.md)。以简表列出三体系的assay数、实际示例数、突变阶数和数据链接；配套逐assay清单保留完整路径及SHA。普通RNA独立cap32与历史热图共享K分开，PPI高阶热图标为未执行。本文继续作为唯一当前绘图入口。

| 评测 | Protein | RNA | PPI | 指标 |
| --- | ---: | ---: | ---: | --- |
| Few-shot，固定随机query，N=250 | 64个assay | 27个assay | 14个assay | 有符号Spearman ρ |
| 多轮迭代，五个不同1000池 | 共同47个 | 共同18个 | 共同37个 | 当批top10%命中率 |
| 少点→多点，精确测试4–12点，N=250 | 来源11个，按列取共同n | 来源13个，按列取共同n | — | 有符号Spearman ρ |

三类评测的候选、样本数和指标不同，不能混成一个结果。历史方案、箱图、SVR备版、累计图及排版预览不在本页展开；原文件保留在[旧索引](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/README.md)。

## Few-shot

测试集是在同一assay剩余池中随机抽取的250条记录，不限定单点或多点。图中`≤1／≤2`表示**已知示例的最大突变阶数**，不是1-shot／2-shot；每个assay各方法和两层共用同一query。

| 系统 | ≤1示例数 | ≤2累计示例数 |
| --- | ---: | ---: |
| Protein | 32 | 64 |
| RNA | 14–32 | 46–64 |
| PPI | 24–32 | 56–64 |

柱高为assay等权平均，误差线为±SEM。无纹／斜纹区分≤1／≤2；Gemini普通组与证据组分别显示。RNA证据组仅10/27项有天然MSA，不是27项MSA-only；`VenusREM*`是项目适配版，星号不是显著性。

### Protein

![Protein few-shot RF 64 assays](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/fewshot/protein/protein_bars.png)

[可编辑SVG](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/fewshot/protein/protein_bars.svg)

### RNA

![RNA few-shot RF 27 assays](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/fewshot/rna/rna_bars.png)

[可编辑SVG](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/fewshot/rna/rna_bars.svg)

### PPI

![PPI few-shot RF 14 assays](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/fewshot/ppi/ppi_bars.png)

[可编辑SVG](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/fewshot/ppi/ppi_bars.svg)

## 多轮迭代

先共享随机32条已知标签，**初始化不占五个池，也不画成第1轮**。之后每轮对一个不同的1000条候选池评分、选择32条，并获得这32条的实测反馈。各方法只累计自己的后续反馈：轮前32／64／96／128／160条，结束192条；本批seed=42。

Gemini使用CR25分块和支持集锚定尺度，加入可用MSA；Protein另加WT结构，RNA不加结构。专业表征和one-hot用RF。未知候选标签只进入独立评估器，不进入预测或训练。

纵轴是所选32条中达到**当前池前10%阈值**的比例，阈值包含边界并列。五池不同，不要求每轮单调上升。所有曲线在每张图内使用相同的完整五轮assay集合。

### Protein

![Protein multiround RF matched 47 assays with VenusREM](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/multiround_arial_ticks25_20261011_v3/protein/protein_rf_main.png)

[可编辑SVG](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/multiround_arial_ticks25_20261011_v3/protein/protein_rf_main.svg)

专业RF与Gemini均47/47完整五轮；MSH2第5轮chunk06已原参数重试成功，所有曲线共同取47。仅新增一个CR25请求，其余199块复用，未改提示／medium／seed42／候选池；完整1000排序选32与自身反馈验收通过。[补跑状态](/data/huangyanjie/EVAgent/experiments/multiround_disjoint5000_initial_random32_add32x5_seed42_20261008_v1/repairs/gemini_msh2_finalchunk_retry_20261010_v3/STATUS.json)。末轮Gemini39.16%、ESM650M33.05%、ESM3B31.91%、VenusREM31.72%、one-hot29.59%。旧共同46图保留，均值变化来自纳入MSH2，不是重跑其他assay。

### RNA

![RNA multiround RF matched 18 assays](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/multiround_arial_ticks25_20261011_v3/rna/rna_rf_main.png)

[可编辑SVG](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/multiround_arial_ticks25_20261011_v3/rna/rna_rf_main.svg)

共同18项；末轮Gemini58.33%、RNA-FM36.28%、Evo2 31.94%、one-hot37.50%。RiNALMo／RNAErnie等完整结果保留在数据中，本主图不另加曲线。

### PPI

![PPI multiround RF matched 37 assays with compact legend](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/ppi_multiround_compact_legend_20261011_v5/ppi/ppi37_rf.png)

[可编辑SVG](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/ppi_multiround_compact_legend_20261011_v5/ppi/ppi37_rf.svg)

BindingGym13＋AbAgym24，共37项，七种方法都完整五轮，每项等权；末轮Gemini35.22%、ESM3B＋RF23.48%、ESM650M＋RF22.80%、VenusREM＋RF23.99%、one-hot＋RF28.55%。VenusREM为项目独立链适配版（BindingGym统一13项结果＋AbAgym24），不是官方原生PPI复现。

ESM2 3B原生zero-shot采用独立链WT-marginal PLL，不使用support标签训练或排序。37项真实五轮均值为14.44／15.46／16.39／15.54／14.53%，图上仅画其等权平均 **15.27027027%** 的水平参考线，**无SEM阴影、无误差线**，不宣称逐轮实测相同。这里平均的是命中率，不是混合不同assay的原始fitness。完整逐轮统计及定义见[ZERO_SHOT_REFERENCE.json](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/ppi37_solid_rf_gold_zero_light_random_20261010_v3/cef3073d463e9862/ZERO_SHOT_REFERENCE.json)。两项历史Gemini输出处理和敏感性定义保留在[METHOD_NOTES.json](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/ppi37_solid_rf_gold_zero_light_random_20261010_v3/cef3073d463e9862/METHOD_NOTES.json)。

当前PPI样式：保留两列legend，只精简重复说明；标签为Gemini、ESM2 3B、ESM2 650M、VenusREM、One-hot、ESM2 3B zero-shot、Random。图例不重复“+ RF”，但五条学习曲线仍是原Gemini／四种RF方法，未换训练方法。曲线、字号、坐标、配色及真实SEM均不变。粉色圆点为Gemini，蓝色方点为ESM2 3B＋RF，奶黄色菱形为650M＋RF，青绿三角为VenusREM＋RF，灰色方点为one-hot＋RF；zero-shot是较深金黄虚线，Random是浅灰细虚线。旧版保留，不另加入八图入口。

三张图的学习曲线阴影均为**均值±SEM，SEM=assay间样本标准差/√n**，不是95%CI，也不是多seed重复区间。Protein／PPI阴影alpha=0.20，RNA=0.45；透明度不同不改变区间宽度。所有zero-shot与Random均无SEM阴影或误差线：三系统zero线是各自同集合五轮平均参考，非逐轮实测常数；RNA展示的zero由保存结果选择，不能当独立验证。PPI Random画名义10%参考，Protein／RNA保留实际随机均值。

## 少点到多点

这里只保留Gemini的两张累计阶数热图。横轴为测试样本**恰好4–12点突变**，每配置N=250；纵轴≤1／≤2／≤3表示累计已知示例。Protein示例32→64→96；RNA按历史共享K→2K→3K，K=14–32。因此同时增加了示例数量和阶数，不能解释为只改变阶数的因果对照，也不是多轮迭代。

每列上方n是三层共同完成的assay数。格内为有符号Spearman均值，缺失为灰色`—`；色条0–0.5只控制颜色，负数和完整数值仍保留在数据中。

### Protein

![Protein low to high mutation order heatmap](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/low_to_high/protein/protein_heatmap_max3.png)

[可编辑SVG](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/low_to_high/protein/protein_heatmap_max3.svg)

### RNA

![RNA low to high mutation order heatmap](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/low_to_high/rna/rna_heatmap_max3.png)

[可编辑SVG](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/low_to_high/rna/rna_heatmap_max3.svg)

## 绘图规范

下面是当前八图的实际参数，供后续沿用；字体统一Arial，三张多轮图已针对拼版放大文字，画布大小不变。Protein、RNA、PPI分别出图，不把不同队列或指标拼成同一曲线。

### 尺寸、文字和坐标

| 图形 | 画布宽×高（英寸） | 字号（pt） | 当前纵轴 |
| --- | --- | --- | --- |
| Few-shot柱图 | 6.6×6.0 | 标题23、轴标签22、刻度20、图例20 | Protein／PPI：0–0.8；RNA：0–0.6，Spearman ρ |
| 五轮曲线 | 8.1×6.25，宽／高约1.3 | 标题26、轴标签25、刻度25；Protein／RNA图例18，PPI精简图例17.5 | Protein／PPI：0–50%；RNA：0–100% |
| 少点→多点热图 | 6.8×5.4 | 标题23、轴标签22、刻度及n为20、格内数字12.5 | ≤1／≤2／≤3；色条0–0.5 |

字体统一`Arial`，包括标题、坐标轴、刻度、图例、热图数字、n和色条；白底，标题左对齐。使用本机真实Arial字体文件，未用替代字体；编辑SVG的电脑也需安装Arial，否则查看软件可能替换字体。当前[字体配置](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/FONT_AND_STYLE.json)记录字体路径与SHA。柱图／曲线隐藏上、右边框；曲线只加淡横向网格（alpha=0.20、线宽0.6）。标题写系统和实际共同`n`，不把配置数当assay数。纵轴不额外写`Mean`；RF及锚定策略在正文交代，PPI多轮legend只保留简短方法名与必要的zero-shot区分。

柱图横轴模型名旋转60°、右对齐；双柱宽0.32、中心偏移±0.18。五轮横轴只标1–5，不新增“随机初始化”点。三张多轮图例均保留两列，PPI仅精简标签，未移动坐标区或曲线；参数见[多轮大字版配置](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/multiround_arial_ticks25_20261011_v3/FONT_AND_STYLE.json)与[PPI精简图例配置](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/ppi_multiround_compact_legend_20261011_v5/FONT_AND_STYLE.json)。上述轴范围仅是当前显示设置：新图若均值±SEM越界或出现负均值，应另存版本并扩轴，不能裁掉误差或抹去负值。

### 配色和图例

| 方法类别 | 柱／阴影填色 | Protein／RNA曲线色 |
| --- | --- | --- |
| Gemini | 粉色`#F5B5BF` | `#D9788D` |
| 序列语言模型（ESM2、RNA-FM、Evo2等） | 奶黄`#FAEEDA` | `#B79B63` |
| VenusREM／结构类 | 青绿`#A8DCD6` | `#43968D` |
| One-hot | 浅灰`#E2E3E5` | `#74777F` |

同色方法用不同标记或线型区分，不只依赖颜色。当前PPI学习曲线均为实线：ESM2 3B蓝色`#547FAC`（阴影`#BCD0E4`）、650M奶黄`#B79B63`（阴影`#FAEEDA`）、VenusREM青绿`#43968D`，one-hot阴影为`#CDD0D6`；zero-shot深金黄`#9C762E`虚线，Random浅灰`#B8B2AA`虚线。Protein／RNA的Random仍为`#8C8179`，未同步改色；不要误称三系统现图配色完全相同。热图使用浅至深蓝灰，缺失填`#EEEEEE`；完整色阶和柱图布局见[STYLE.json](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/organized_20261008/data/STYLE.json)。

柱图无纹／`///`区分≤1／≤2，图例放画布右上、两列、无框。三张曲线图例均放坐标区左上、两列、无框，PPI只精简冗余文字，不删除方法项；学习曲线线宽2.5、点大小6。Protein／RNA zero-shot线宽1.7；PPI zero-shot为2.6、Random为1.8；两种参考线均不加点或SEM带。图例避开数据，不把图例外移到大块留白中；具体方法的线型与标记沿用下方对应代码。

### 误差、热图和导出

- 柱图误差线为各柱自身±SEM，线宽0.8、端帽2；曲线阴影同样为±SEM。按共同assay等权计算，`SEM = std(ddof=1) / sqrt(n)`，不是95%CI。只能调阴影透明度，不能乘缩小系数；当前Protein／PPI=0.20、RNA=0.45。Zero-shot／Random不画误差带，参考线口径见上文。
- 热图格内保留三位小数和符号（如`0.239`、`-0.010`），缺失写`—`，不补零；上方n逐列显示三层共同完成数量。色条横放底部、刻度0／0.25／0.5；深浅格自动使用对比清晰的白／黑字。颜色范围之外的数值仍按原值显示，不裁改数据。
- 每图同时输出300 dpi PNG和可编辑SVG，`svg.fonttype='none'`；保留文字和矢量曲线，不把截图嵌入SVG。柱图／热图导出使用`bbox_inches='tight', pad_inches=0.04`，因此最终边界尺寸可能不同于画布；曲线使用上述固定画布和内部留白。
- 重绘只读取保存的逐assay指标及集合文件，缺失不填零；先核对各方法相同assay／query、完整轮数、均值及SEM，再检查图例遮挡、字被裁切和轴截断。新图另存目录，保存样式、数据、代码版本及SHA，更新本页链接和八图索引；不覆盖历史资产，也不因改图启动模型任务。

## 数据和代码

三张few-shot柱图及两张热图沿用[Arial初版脚本](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/source/render_v2_arial.py)；Protein／RNA多轮图使用[多轮字号放大脚本](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/multiround_arial_ticks25_20261011_v3/source/enlarge_multiround_text.py)，PPI当前版使用[精简PPI图例脚本](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/ppi_multiround_compact_legend_20261011_v5/source/compact_ppi_legend.py)。脚本都只读取保存的指标，保留配色、画布、均值和SEM；初版脚本生成的多轮小字图不再是当前版。下表保留原数据和历史代码来源；具体当前代码／图路径以八图索引为准。原始STYLE文件保留原样，当前字体字号以本页配置为准。复现Protein／RNA及历史完整标签PPI三图：

```bash
/data/huangyanjie/EVAgent/code/.venv/bin/python \
  /data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/multiround_arial_ticks25_20261011_v3/source/enlarge_multiround_text.py \
  --output-dir /tmp/evagent_multiround_large_text_new
```

输出目录必须尚不存在。脚本默认读取旁边冻结的修改前索引，不随当前入口变化而换队列；无模型推理或重新训练。此次多轮目录含三组PNG／SVG、作图数据副本、源码、字体配置和SHA清单；另五张图及旧版均保留。新索引仍为八行，只替换三张多轮的入口。作图数据是保存的指标，不是向预测器提供的query标签；原始预测、反馈、请求和验收仍在各实验目录。

复现当前精简legend的PPI（仅输出一组PNG／SVG）：

```bash
/data/huangyanjie/EVAgent/code/.venv/bin/python \
  /data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/ppi_multiround_compact_legend_20261011_v5/source/compact_ppi_legend.py \
  --output-dir /tmp/evagent_ppi_compact_legend_new
```

仅PPI多轮入口更新，其他七行及旧文件不变。[精简图例核查](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/ppi_multiround_compact_legend_20261011_v5/VALIDATION.json)确认全部七方法legend保留、图例在坐标区内，SVG曲线／SEM带几何、其他文字、Arial和数据副本保持一致。

| 内容 | 数据和结果 | 原始绘图来源（历史代码） |
| --- | --- | --- |
| 三张few-shot柱图 | [逐assay PLOT_DATA.csv](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/organized_20261008/data/PLOT_DATA.csv)／[集合与实际shot](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/organized_20261008/data/COHORTS.json) | [organize_figures.py](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/organized_20261008/organize_figures.py) |
| 两张少点→多点热图 | [54格全精度数据与名单](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/organized_20261008/data/HEATMAP_PLOT_DATA.csv) | 同上 |
| Protein多轮 | [逐assay逐轮](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/protein47_msh2_complete_20261010_v7/0e8010abc8dd00f9/data/PER_ASSAY_ROUNDS.csv)／[五轮表](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/protein47_msh2_complete_20261010_v7/0e8010abc8dd00f9/MODEL_BY_ROUND.csv) | [plot_protein47_msh2_complete_20261010.py](/data/huangyanjie/EVAgent/code/tools/figures/plot_protein47_msh2_complete_20261010.py) |
| RNA多轮 | [逐assay逐轮](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/rna18_selected_zero_20261009_v4/c00f486d4c0c8751/data/PER_ASSAY_ROUNDS.csv)／[五轮表](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/rna18_selected_zero_20261009_v4/c00f486d4c0c8751/MODEL_BY_ROUND.csv) | [plot_rna18_selected_zero_20261009.py](/data/huangyanjie/EVAgent/code/tools/figures/plot_rna18_selected_zero_20261009.py) |
| PPI多轮 | [逐assay逐轮（1295行）](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/ppi37_solid_rf_gold_zero_light_random_20261010_v3/cef3073d463e9862/PER_ASSAY_ROUNDS.csv)／[五轮表](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/ppi37_solid_rf_gold_zero_light_random_20261010_v3/cef3073d463e9862/MODEL_BY_ROUND.csv) | [restyle_ppi37_reference_lines_20261010.py](/data/huangyanjie/EVAgent/code/tools/figures/restyle_ppi37_reference_lines_20261010.py)（仅改样式；原汇总代码保存于新目录source） |

多轮原始运行：[Protein／RNA实验](/data/huangyanjie/EVAgent/experiments/multiround_disjoint5000_initial_random32_add32x5_seed42_20261008_v1/)；PPI分别为[BindingGym13](/data/huangyanjie/EVAgent/experiments/ppi_disjoint5000_initial32_batch1000_seed42_20261008_v1/)和[AbAgym24](/data/huangyanjie/EVAgent/experiments/ppi_disjoint5032_abagym24_seed42_20261009_v1/)，不跨assay混合标签。少点→多点的版本区别和构造入口见[实验指南](/data/huangyanjie/EVAgent/reports/single_to_multi_experiment_guide_20261002/README.md)，热图对应其中的累计版本C，不是固定0／8／16／32条单点示例的版本A。

完整八图路径与SHA见[FIGURE_INDEX_V2.csv](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/FIGURE_INDEX_V2.csv)。[初版Arial导出检查](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/arial_20261011_v1/VALIDATION.json)记录八图字体统一；[多轮字号更新检查](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/multiround_arial_ticks25_20261011_v3/VALIDATION.json)确认三图字体、文字边界、数据副本及旧图SHA，原SVG与新版的曲线／误差带路径一致，另外五张保持不变；三张放大版已目检。历史[初版索引核查](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/V2_VALIDATION_20261009.json)和[绘图规范补充核查](/data/huangyanjie/EVAgent/reports/fewshot_figure_refresh_20261001/V2_STYLE_VALIDATION_20261009.json)仍保留。后续只更新本页对应图、数量及索引；改图另存新目录，不覆盖冻结图，不为当前入口追加逐次进度日志。复现必须输出到全新目录，不能直接执行可能写回原快照的默认命令；模型重载沿用原环境，普通看图不需要GPU或权重。
