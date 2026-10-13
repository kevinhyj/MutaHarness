# GFP案例：从96条反馈到186个候选的排序

GFP是绿色荧光蛋白。本例用它的公开荧光测量，检验Codex能否从有限反馈预测未测变体，以及Scientific Program Ensemble能否改善一次建模选择。

## 1. 原始任务与输入

原任务的中文转述：

> 根据已测实验反馈，为现有未测变体预测荧光排序。使用参考序列、实验目标和已测结果，自由建模与验证，交付可重复执行的预测程序。每个候选都需要一个有限分数，分数越高越优先。

[原始prompt](../evidence/scientific_program_ensemble/portfolio_protein01/calls/gfp_s2501/candidate_00/native_00/prompt.txt) · [完整case.json](../inputs/gfp_s2501.json)

| 输入 | 实际内容 |
|---|---|
| 实验 | `GFP_AEQVI_Sarkisyan_2016`；FACS荧光读出，DMS_score越高越好 |
| WT | 238 aa参考序列 |
| 已测反馈 | 96条：32个初始单点变体＋64条独立抽样的历史反馈 |
| 未测候选 | 186个；每条包含ID、突变、序列，没有fitness字段 |
| 可用证据 | 序列、实验上下文和反馈；本例没有MSA或结构坐标 |

从真实输入中摘取两条记录：

```json
{
  "support": [{"id": "S001", "mutation": "V112E", "fitness": 1.343387412865}],
  "query": [{"id": "M002", "mutation": "T38S:E142V:Y151F:S205F", "mutation_order": 4}]
}
```

完整文件还包含参考序列与每条变体的序列。这项任务的难点是：单点变化与组合变化共同出现，少量反馈允许多个合理模型，荧光读出又可能有压缩和饱和。原生Codex可以分析这些问题，并用本地科学计算工具检验模型。

## 2. 原生Codex如何完成任务？

原生对照是第一个独立会话交付的**程序A**。它从参考序列解析突变，构造理化变化、位点和突变负担特征，用已测反馈比较模型，再交付`proposal.py:propose(case, assignment)`。

其最终方案包括核回归、非线性核和全局非线性响应模型；每次执行从传入的support重新拟合，为全部query输出排序分数。一次程序内部也可以包含集成，Codex没有被限制成简单公式。

[程序A](../programs/gfp_s2501/candidate_00/proposal.py) · [原决策说明](../programs/gfp_s2501/candidate_00/decision_note.txt) · [186条原预测](../reference_predictions/gfp_s2501/candidate_00.json)

## 3. Harness增加了什么？

同一份输入交给三个独立科学会话，每个会话自由选择模型。Harness负责统一执行检查、候选覆盖检查和最终排名融合。

| 科学方案 | 最终程序中的主要方法 | 实际交付 |
|---|---|---|
| A：非线性响应 | 核回归、平滑非线性核、sigmoid全局响应 | [程序](../programs/gfp_s2501/candidate_00/proposal.py) · [说明](../programs/gfp_s2501/candidate_00/decision_note.txt) |
| B：树模型与响应模型 | 两个ExtraTrees回归器＋正则化非线性响应 | [程序](../programs/gfp_s2501/candidate_01/proposal.py) · [说明](../programs/gfp_s2501/candidate_01/decision_note.txt) |
| C：组合特征与位点反馈 | 树模型、辅助Ridge及单点反馈的位点校正 | [程序](../programs/gfp_s2501/candidate_02/proposal.py) · [说明](../programs/gfp_s2501/candidate_02/decision_note.txt) |

三份程序都从当前已测反馈重新拟合。原验收记录中，三者均完成、重复执行相同、覆盖全部186个候选。排名融合采用**并列平均秩百分位＋三份等权平均**，分数尺度无需相同，成员与权重不按测试结果挑选。

[A验收](../evidence/scientific_program_ensemble/portfolio_protein01/calls/gfp_s2501/candidate_00/manifest.json) · [B验收](../evidence/scientific_program_ensemble/portfolio_protein01/calls/gfp_s2501/candidate_01/manifest.json) · [C验收](../evidence/scientific_program_ensemble/portfolio_protein01/calls/gfp_s2501/candidate_02/manifest.json) · [融合实现](../runtime/ranking.py)

## 4. 输出发生了怎样的变化？

以下从保存的三份预测计算，排名1表示最优先。同分时按候选ID排序。

| 候选 | 突变 | 原生单程序A的排名 | 三程序集成排名 |
|---|---|---:|---:|
| M051 | H139R:H148L | 61 | 30 |
| M119 | S72G:I152V | 1 | 5 |
| M002 | T38S:E142V:Y151F:S205F | 99 | 111 |

融合利用各方案的共同判断与分歧，改变候选优先级。保存预测的完整输出可以直接复查：

```bash
python3 harness.py ensemble --out outputs/gfp_ensemble
```

查看`ranking.csv`中的三份分数、融合排名和`rank_disagreement_sd`，以及`manifest.json`中的来源哈希。此命令复查已有预测的融合；冻结程序重新拟合的命令见[使用说明](getting-started.md)。

## 5. 实测：区分生成预算与融合收益

测试模型请求为`gpt-6.1-sol/high`，Fast关闭。三个会话使用相同输入。下表将用于选程序的**已测反馈CV**与用于报告结果的**186个候选测试**分开：

| 方案 | 已测反馈CV Spearman | 未测候选测试Spearman |
|---|---:|---:|
| 程序A：固定的原生单次对照 | 0.6927 | 0.5303 |
| 程序B：CV选择的同预算单程序 | **0.7040** | 0.5433 |
| 程序C | 0.7031 | **0.5952** |
| 三程序集成 | — | **0.5819** |

**相对单次原生Codex，提升0.0515；相对同预算CV选单程序，提升0.0386。** 第二个对照使用完全相同的三份程序，区别在于选择一个还是融合三个。

测试结果显示C最高，但根据已测反馈的CV会选择B。这正是该评估要检验的决策：生成结束、测试结果未知时，如何使用已经获得的三个科学方案。融合提高了这次可实施选择的排序质量。

本次主指标是Spearman；选样诊断也保留在原记录中：前20重合率为A **0.30**、CV选择B **0.40**、C **0.30**、融合 **0.35**。整体排序与小预算候选选择分别评价。

[原指标记录](../evidence/scientific_program_ensemble/pooled_protein_metrics.json) · [十项任务的完整结果](results.md) · [Scientific Program Ensemble Skill](../.agents/skills/scientific-program-ensemble/SKILL.md)
