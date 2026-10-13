# EVAgent结果图

`evagent_overview.png/svg` 用于主README的紧凑概览，读取服务器原均值CSV，保留蛋白10个方法、RNA9个方法及全部负分。横轴统一为−0.03至0.62，包含零点；条形高度0.40，颜色沿用原图。每个面板的最高均值加粗，没有添加置信区间。

`original_protein.png/svg`、`original_rna.png/svg`、`protein_feedback.png/svg` 是按字节复制的服务器当前原图。输入条件、共同队列与反馈协议见 [结果说明](../../docs/evagent.md)，源文件SHA见 [导入清单](../../evidence/evagent/manifest.json)。

```bash
python -m pip install -r requirements-figures.txt
python -m scripts.plot_evagent_results
```

脚本先核对来源哈希、逐实验均值和反馈SEM，再排版已有指标；不运行模型或预测程序。输出哈希与字体保存在 `plot_manifest.json`。
