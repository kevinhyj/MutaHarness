"""Load historical metrics once for the report, viewer and figures."""

from pathlib import Path
from statistics import mean

from runtime.artifacts import ROOT, read_json
from runtime.ranking import query_ids, rank_ensemble

ZERO_SOURCES = [
    "evidence/protein_zero_native01/SUMMARY.json",
    "evidence/protein_zero_native02/SUMMARY.json",
]
ENSEMBLE_SOURCE = "evidence/scientific_program_ensemble/pooled_protein_metrics.json"
ZERO_METHODS = ("basic", "msa_tools", "structure_tools", "joint_tools")
LABELS = {
    "amfr": "AMFR",
    "his7": "HIS7",
    "yap_peptide": "YAP–肽",
    "bh3_mcl1": "BH3–Mcl-1",
    "ube4b": "UBE4B",
    "cbx4": "CBX4",
    "gfp": "GFP",
    "pabp": "PABP",
    "kras_abundance": "KRAS丰度",
    "kras_raf1": "KRAS–RAF1",
    "gcn4": "GCN4",
    "grb2": "GRB2",
    "yap1": "YAP1",
    "phot": "PHOT",
    "ras_abundance": "KRAS丰度",
    "ynzc": "YNZC",
}


def aggregate(cases: list[dict], feedback: int, queries: int) -> dict:
    methods = cases[0]["scores"]
    return {
        "means": {method: mean(case["scores"][method] for case in cases) for method in methods},
        "feedback_count": feedback,
        "query_count": queries,
        "cases": cases,
    }


def load_metrics(root: Path = ROOT) -> dict:
    zero_cases = []
    zero_rows = []
    for source in ZERO_SOURCES:
        rows = read_json(root / source)["rows"]
        zero_rows.extend(rows)
        scores_by_case = {}
        for row in rows:
            scores_by_case.setdefault(row["case_id"], {})[row["arm"]] = row["spearman"]
        for case_id, scores in scores_by_case.items():
            zero_cases.append(
                {
                    "id": case_id,
                    "label": LABELS[case_id],
                    "source": source,
                    "scores": {method: scores[method] for method in ZERO_METHODS},
                }
            )
    zero = aggregate(zero_cases, feedback=0, queries=250)
    zero["top25_means"] = {
        arm: mean(row["top25_overlap"] for row in zero_rows if row["arm"] == arm)
        for arm in zero["means"]
    }
    conditions = read_json(root / "artifacts_manifest.json")["conditions"]
    condition_ids = {details["case_id"]: id_ for id_, details in conditions.items()}
    ensemble_cases = [
        {
            "id": condition_ids[row["case_id"]],
            "label": LABELS[row["case_id"]],
            "source": ENSEMBLE_SOURCE,
            "scores": {
                "direct": row["candidate_rho"][0],
                "cv": row["cv_selected_single_rho"],
                "ensemble": row["ensemble_rho"],
            },
            "cv_selected": row["cv_selected_candidate"],
        }
        for row in read_json(root / ENSEMBLE_SOURCE)["rows"]
    ]
    return {
        "model": "gpt-6.1-sol / high",
        "zero": zero,
        "ensemble": aggregate(ensemble_cases, feedback=96, queries=186),
    }


def load_viewer(root: Path = ROOT) -> dict:
    data = load_metrics(root)
    for case in data["ensemble"]["cases"]:
        input_case = read_json(root / "inputs" / f"{case['id']}.json")
        predictions = [
            read_json(root / "reference_predictions" / case["id"] / f"candidate_{i:02}.json")[
                "scores"
            ]
            for i in range(3)
        ]
        mutations = {query["id"]: query["mutation"] for query in input_case["query"]}
        case["candidates"] = [
            {**candidate, "mutation": mutations[candidate["id"]]}
            for candidate in rank_ensemble(query_ids(input_case), predictions)
        ]
    return data
