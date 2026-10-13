#!/usr/bin/env python3
"""Present historical metrics and predictions. Never invoke an LLM or a predictor."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text())
LABELS={'amfr':'AMFR','his7':'HIS7','yap_peptide':'YAP–肽','bh3_mcl1':'BH3–Mcl-1',
        'ube4b':'UBE4B','cbx4':'CBX4','gfp':'GFP','pabp':'PABP','kras_abundance':'KRAS丰度','kras_raf1':'KRAS–RAF1',
        'gcn4':'GCN4','grb2':'GRB2','yap1':'YAP1','phot':'PHOT','ras_abundance':'KRAS丰度','ynzc':'YNZC'}


def best_cells(values):
    """Emphasize the actual row maximum, including tied controls."""
    best=max(values)
    return [f'**{v:.4f}**' if abs(v-best)<1e-12 else f'{v:.4f}' for v in values]


def main():
    pooled=read(ROOT/'evidence/scientific_program_ensemble/pooled_protein_metrics.json')
    summaries=[read(ROOT/'evidence'/name/'SUMMARY.json') for name in ['protein_zero_native01','protein_zero_native02']]
    zero_rows=sum([s['rows'] for s in summaries],[])
    cases=list(dict.fromkeys(r['case_id'] for r in zero_rows))
    get=lambda case,arm:next(r for r in zero_rows if r['case_id']==case and r['arm']==arm)
    means={arm:statistics.mean(get(case,arm)['spearman'] for case in cases) for arm in ['basic','msa_tools','structure_tools','joint_tools']}
    lines=['# 两项改进的已有实验结果','',
           '所有数字整理自此前完成的本地原生Codex实验，模型请求为 `gpt-6.1-sol/high`，Fast关闭。这里核对存档指标与哈希，没有重新跑模型或预测程序。','',
           'Spearman（ρ）表示预测排序与实验排序的一致程度，越高越好。以下为小规模任务队列的等权汇总，不是ProteinGym完整榜单分数。','',
           '![两项改进的总体结果](../assets/results/results_overview.png)','',
           '柱高为10个任务的平均值。纵轴分别截断为0.480–0.610和0.575–0.630，图中标有断轴。下表每行最高分加粗，包括对照胜出或并列的情况。','',
           '## 1. Structure + MSA：零样本','',
           '10个protein/PPI assay，每个250个固定候选，0条目标实验反馈。基础原生Codex仍能使用本地科学工具并编写预测程序；改进组增加对应的结构/MSA原始文件、位点证据和分析工具。','',
           '| assay | 基础Codex | 仅MSA | 仅结构 | 联合 | 联合−基础 |','|---|---:|---:|---:|---:|---:|']
    for case in cases:
        a=[get(case,arm)['spearman'] for arm in means]
        lines.append(f"| {LABELS[case]} | "+' | '.join(best_cells(a))+f" | {a[3]-a[0]:+.4f} |")
    lines.append('| **平均** | '+' | '.join(f'**{v:.4f}**' for v in means.values())+f" | **{means['joint_tools']-means['basic']:+.4f}** |")
    wins=sum(get(c,'joint_tools')['spearman']>get(c,'basic')['spearman'] for c in cases)
    lines += ['',f'联合条件 **{wins}/10** 提升。新增6个的单独结果是 **0.5233 → 0.5901**；首批4个单独结果是 **0.4778 → 0.5487**。保留分批口径，合并10个是次要描述性汇总。','',
              '**排序相关性与挑出最优候选是不同指标。** 10个联合条件的平均Top-25重合率为 **0.244**，基础为 **0.272**；整体相关性提高，没有带来这一小预算选样指标的同步提高。','',
              '两批40/40条件完成。新增6个的joint/basic已知input+output token比约 **2.40×**；预处理未计入原生会话耗时，缓存输入已含在输入量里，token量不等于美元账单。每条件一次有效科学会话，没有多次生成稳定性结论。KRAS两个端点共享蛋白，原报告保留蛋白聚类敏感性。','',
              '原始摘要：[首批](../evidence/protein_zero_native01/SUMMARY.json)、[新增6个](../evidence/protein_zero_native02/SUMMARY.json)；最终分析、输入/输出绑定与transport审计同目录归档。','',
              '## 2. Scientific Program Ensemble：从实验反馈中学习','',
              '每个assay使用同一份96条已测反馈（32个初始单点 + 64个随机历史反馈），对剩余186个候选排序。三份程序由三个独立原生Codex科学会话生成，各自可以建模和验证，最终等权平均排名。','',
              '| assay | 单次Codex | 同预算CV选一个 | 三程序集成 | 集成−CV单程序 |','|---|---:|---:|---:|---:|']
    for r in pooled['rows']:
        values=[r['candidate_rho'][0],r['cv_selected_single_rho'],r['ensemble_rho']]
        lines.append(f"| {LABELS[r['case_id']]} | "+' | '.join(best_cells(values))+f" | {r['delta_vs_cv_single']:+.4f} |")
    single=statistics.mean(r['candidate_rho'][0] for r in pooled['rows']);cv=statistics.mean(r['cv_selected_single_rho'] for r in pooled['rows']);en=statistics.mean(r['ensemble_rho'] for r in pooled['rows'])
    lines.append(f'| **平均** | **{single:.4f}** | **{cv:.4f}** | **{en:.4f}** | **{en-cv:+.4f}** |')
    lines += ['',
              '- 相对单次Codex：**8/10提升，平均 +0.0221**；生成预算为3个科学会话对1个，不能把收益全归于聚合。',
              '- 相对同预算CV单程序：**8/10提升，平均 +0.0250**；描述性assay bootstrap区间 **[+0.0058, +0.0445]**。',
              '- 追加6个单独分析：质量恢复后 **5/6提升，平均 +0.0332**；首批4个平均 **+0.0127**，其原主要门槛未通过。',
              '- PABP下降 **0.0193**，PHOT下降 **0.0184**。AMFR与YNZC来自同一研究，原指标保留按研究分组的敏感性分析。','',
              '**交付与质量分开记录：** 首批12/12科学会话完成；追加批原18次为17完成、1容量失败。随后另做1次只完成交付验收的调用，恢复同一份哈希不变的程序，没有新增科学建模。追加批恢复后质量条件满足；原运行验收仍未通过，未知用量仍保留。共30次原科学会话 + 1次交付恢复会话，不能写成原30次均成功。','',
              '原始指标：[10个汇总](../evidence/scientific_program_ensemble/pooled_protein_metrics.json)、[首批4个](../evidence/scientific_program_ensemble/protein_metrics.json)、[追加6个恢复后](../evidence/scientific_program_ensemble/protein_additional6_metrics_recovered.json)、[原尝试](../evidence/scientific_program_ensemble/protein_additional6_original_attempts_metrics.json)。每个原生会话可能包含多个底层模型请求。','',
              '![各任务的配对提升与下降](../assets/results/results_by_assay.png)','',
              '粉红表示提升，蓝色表示下降；两图横轴范围一致。右图比较同预算CV单程序与排名融合。[矢量图、源数据与绘图脚本](../assets/results/README.md)。','',
              '## 3. 如何解读与复核','',
              '两项干预分别验证：SPE实验没有结构/MSA输入，零样本四组没有实验support。两套队列与候选不同，不是同一次累计优化，也不应将两项增益相加。','',
              '反馈CV用于选择同预算单程序，但架构已在全部support上形成，因此不是完全嵌套验证。测试结果没有用于挑成员或调融合权重。一次反馈划分、一次会话/条件与公开benchmark预训练暴露仍限制结论。','',
              '既有32-shot结构/MSA探索也提示：单点反馈的收益未可靠迁移到多点组合，部分条件失败、延时恢复后下降。它没有被包装成全面的few-shot正向结论。本提交以已完成的程序集成实验展示反馈学习，以零样本实验展示结构/MSA证据收益。','',
              '这些是真实公开实验数据上的离线回放。没有新湿实验、主动实验选样、药效验证或Claude/Cradle内部模型实测；原生Codex是实际对照。结构/MSA工具和原始证据同时变化，尚未分离每个组件的贡献，也未证明LLM直接理解三维坐标。','',
              '`scripts/audit_existing.py` 核对30份程序、10份输入的原哈希、存档指标算术，以及页面的70个分数和1860个候选的排名融合与分歧。它不执行预测程序、不打开测试标签、不调用模型。原实验的独立Spearman复核与标签隔离记录已归档；完整材料范围见[来源说明](provenance.md)。','']
    (ROOT/'docs/results.md').write_text('\n'.join(lines))
    # Offline viewer uses ONLY the existing predictions. Rank fusion is stored-output arithmetic.
    def ranks(values):
        out=[0.]*len(values);order=sorted(range(len(values)),key=lambda i:values[i]);i=0
        while i<len(order):
            j=i+1
            while j<len(order) and values[order[j]]==values[order[i]]:j+=1
            for k in order[i:j]:out[k]=(i+j)/2/len(values)
            i=j
        return out
    viewer={'model':'gpt-6.1-sol / high',
            'zero':{'means':means,'feedback_count':0,'query_count':250,
                    'top25_means':{arm:statistics.mean(get(c,arm)['top25_overlap'] for c in cases) for arm in means},'cases':[]},
            'ensemble':{'means':{'direct':single,'cv':cv,'ensemble':en},'feedback_count':96,'query_count':186,'cases':[]}}
    for case in cases:
        campaign='protein_zero_native01' if any(r['case_id']==case for r in summaries[0]['rows']) else 'protein_zero_native02'
        viewer['zero']['cases'].append({'id':case,'label':LABELS[case],'source':f'evidence/{campaign}/SUMMARY.json',
                                       'scores':{arm:get(case,arm)['spearman'] for arm in means}})
    manifest=read(ROOT/'artifacts_manifest.json')
    for r in pooled['rows']:
        condition=next(k for k,v in manifest['conditions'].items() if v['case_id']==r['case_id'])
        case=read(ROOT/'inputs'/f'{condition}.json');ids=[v['id'] for v in case['query']]
        predictions=[read(ROOT/'reference_predictions'/condition/f'candidate_{k:02}.json')['scores'] for k in range(3)]
        votes=[ranks([p[i] for i in ids]) for p in predictions]
        candidates=[{'id':v['id'],'mutation':v['mutation'],'score':2*statistics.mean(v[j] for v in votes)-1,
                     'program_scores':[p[v['id']] for p in predictions],
                     'disagreement':statistics.pstdev(v[j] for v in votes)} for j,v in enumerate(case['query'])]
        viewer['ensemble']['cases'].append({'id':condition,'label':LABELS[r['case_id']],
                    'source':'evidence/scientific_program_ensemble/pooled_protein_metrics.json',
                    'scores':{'direct':r['candidate_rho'][0],'cv':r['cv_selected_single_rho'],'ensemble':r['ensemble_rho']},
                    'cv_selected':r['cv_selected_candidate'],'candidates':candidates})
    (ROOT/'demo').mkdir(exist_ok=True)
    (ROOT/'demo/data.js').write_text('window.MUTA_DATA = '+json.dumps(viewer,ensure_ascii=False)+';\n')
    print('Rendered existing results only; no LLM call, predictor execution or new experiment.')


if __name__=='__main__':main()
