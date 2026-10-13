# 来源与提交范围

本仓库采用此前已完成的实验结果与原生程序，整理自本地 `/Users/hyj/Desktop/Kimi`。原始实验数据来自服务器 `a100_enzyme2` 的 `/data/huangyanjie/EVAgent`。原实验目录和旧提交包保留。

| 内容 | 原始来源 | 仓库位置 |
|---|---|---|
| Protein程序集成，10个assay | `submission/ProgramPortfolio_Option2_v4` | `inputs/`、`programs/`、`reference_predictions/` |
| 程序集成指标、原调用、容量失败与交付恢复 | 同包内 `reports/` 与 `experiments/portfolio_protein01/02` | `evidence/scientific_program_ensemble/` |
| 零样本首批4个 / 16条件 | `explorations/protein_zero_native01` | `evidence/protein_zero_native01/` |
| 零样本新增6个 / 24条件 | `explorations/protein_zero_native02` | `evidence/protein_zero_native02/` |
| 原始MSA与结构分析工具 | `explorations/protein_zero_native02` | `tools/` |
| 可运行工具输入示例 | 首批AMFR联合条件的原始公共输入与WT坐标 | `examples/zero_shot/amfr/` |

`artifacts_manifest.json` 保留10个protein条件的输入与源码哈希。程序集成的输入、科研源码和原预测按字节复制；工程入口经过重构，排名融合统一在 `runtime/ranking.py`，执行仍使用原输入与冻结程序，并要求输出与原预测完全一致。原Skill归档为 `evidence/scientific_program_ensemble/original_skill.md`；仓库Skill增加了已有结果优先的说明。

事件日志以gzip压缩保存，内容对应原调用。容量失败、未知用量和completion-only恢复没有归并为原运行全部成功。零样本归档包括全部40条件的最终预测、原生最终响应、调用记录、生成程序和报告。

为了保持提交精简，未复制全部大型中间缓存、完整MSA/PDB重复副本、RNA实验与此前无关探索。主报告的统计来自原有指标文件；本仓库提供程序集成的完整冷重放材料，以及结构/MSA工具示例。零样本完整原始证据仍在源实验目录；不宣称这个精简包包含零样本全实验的所有复跑输入。

本次打包阶段只做复制、排版、哈希核对、存档指标算术复核和结果页面修正，未调用模型或执行预测程序。可复核状态见 [packaging_verification.json](../evidence/packaging_verification.json)，包含页面的70个分数和1860个候选数据核对。浏览器中逐一检查了20个任务的详情同步、实验切换记忆、键盘选择和桌面/小屏布局。模型请求标识为 `gpt-6.1-sol`；原生JSONL没有单独返回可独立核验的后端模型标识。

工程重构另通过11个排名与输入边界单元测试、Ruff静态/格式检查和页面回归检查。重构前后的报告、页面数据和图表使用的70个数值一致；冻结程序未重新执行，因此这些检查不等于重新复现实验。

公开数据主要来自 [ProteinGym](https://github.com/OATML-Markslab/ProteinGym)，部分复合物任务来自原EVAgent公共导出。源码与输入中的原始assay名称、端点、采样和来源信息均保留。公共数据可能被模型预训练见过；运行时隔离避免提供隐藏标签，不等于训练集无泄漏。
