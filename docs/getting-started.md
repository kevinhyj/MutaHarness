# 跑通两个 Harness

下载仓库后，在根目录执行。两个默认示例只需Python 3.11+标准库，无需API key或安装科学计算依赖。

```bash
python3 harness.py ensemble --out outputs/gfp_ensemble
python3 harness.py evidence --out outputs/amfr_evidence
```

输出目录必须是新目录；输入检查失败时不会生成结果目录。已有预测融合与证据分析都不会启动模型或重新拟合程序。

## 1. Scientific Program Ensemble

默认读入GFP的96条已测反馈、186个候选及三份已有预测。融合前核对候选ID、分数完整性和有限数值，使用并列平均秩与等权融合。

| 输出 | 内容 |
|---|---|
| `ensemble.json` | 每个候选的融合分数，越高越优先 |
| `ranking.csv` | 降序排名、三份原分数、排名分歧 |
| `manifest.json` | 输入/输出SHA-256、样本数和执行操作 |

接入自己的数据：

```bash
python3 harness.py ensemble --case my_case.json --predictions program_a.json program_b.json program_c.json --out outputs/my_ensemble
```

融合入口所需的最小格式如下。这是格式示意，实际任务可直接使用[完整GFP输入](../inputs/gfp_s2501.json)。

```json
{"case_id": "my_assay", "query": [{"id": "variant-A"}, {"id": "variant-B"}]}
```

每份预测提供同一批候选的分数：

```json
{"scores": {"variant-A": 0.2, "variant-B": 0.8}}
```

三份预测必须覆盖完全相同的ID，分数统一为越高越优先；分数尺度可以不同。`support`可包含已测反馈，其ID必须与`query`分离，`query`不含`fitness`。分歧是检查信号，不是概率置信度。

新的科研任务由[程序集成Skill](../.agents/skills/scientific-program-ensemble/SKILL.md)指导生成三个方案，再将完整预测交给这个入口。已有冻结程序的重新拟合仍使用`demo.py`，依赖见`requirements.txt`；它与默认的预测融合分开。

## 2. Structure + MSA

默认使用AMFR的250个候选、同源序列位点摘要及WT结构摘要。先检查序列与位点映射，再输出两类科学证据。

| 输出 | 内容 |
|---|---|
| `msa.json` | 替换频率、log-odds、深度、缺失与覆盖率 |
| `structure.json` | WT暴露、局部接触、位点置信度和可用距离 |
| `manifest.json` | 与融合入口相同的来源与输出记录 |

接入自己的数据：

```bash
python3 harness.py evidence --case my_case.json --out outputs/my_evidence
```

以[AMFR输入](../examples/zero_shot/amfr/case.json)为完整模板。必需字段：

| 字段 | 要求 |
|---|---|
| `domain` | `protein`或`ppi` |
| `reference_sequence` | 非空WT序列 |
| `query` | 唯一ID与候选序列；PPI使用`mutated_chains` |
| `reference_chains` | PPI的链名→WT序列映射 |
| `evidence.format` | `protein_site_table_v1` |
| `evidence.sites` | 1-based位点、WT、链、MSA频率及可用WT结构摘要 |
| `evidence.source_metadata` | 证据来源与映射记录 |

零样本入口要求没有已测support。它读取已经对齐的证据摘要；原始FASTA/MSA/PDB需要先完成映射与摘要准备。缺失证据保留为null，不补成观测值。[结构＋MSA Skill](../.agents/skills/structure-msa-fitness/SKILL.md)说明如何进一步使用原始PDB及按实验目标组织预测。

## 验证

```bash
python3 -m unittest discover -s tests -v
python3 -m scripts.audit_existing
```

测试覆盖两个命令在仓库外工作目录执行、输出来源哈希、自定义输入ID对齐、已有目录保护、标签误入和证据错误时的失败行为。所有示例使用已有材料。
