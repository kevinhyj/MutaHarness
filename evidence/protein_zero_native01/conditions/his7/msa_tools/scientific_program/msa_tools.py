"""Fixed, label-free MSA scores for canonical protein_site_table_v1 cases.

CLI: python msa_tools.py --case case.json --output msa_scores.json
API: score_case(case, verified_rsa=None) returns the same JSON-compatible sidecar.

Scores have a fitness-compatible sign: positive favors ALT over WT. The two
scorers are sum(log(f_ALT / f_WT)) and sum(logit(f_ALT) - logit(f_WT)).
Existing summary frequencies are used without extra smoothing, renormalization,
position weights, or fitting. Missing/undefined terms remain None. Observed-term
partial sums are explicitly separate from complete candidate scores.

This is an independent-site model. It has no epistasis or cross-chain coupling.
RSA scaling is optional and requires a separately verified, sequence-bound RSA
artifact. Absolute SASA in the canonical case is never interpreted as RSA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCORE_FORMAT = "protein_msa_fixed_scores_v1"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
METHODS = ("sum_log_frequency_ratio", "sum_true_log_odds_ratio")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _references(case: Mapping[str, Any]) -> dict[str | None, str]:
    reference = case.get("reference_sequence")
    if not isinstance(reference, str) or not reference:
        raise ValueError("A nonempty reference_sequence is required")
    chains = case.get("reference_chains")
    if chains is None:
        return {None: reference}
    if not isinstance(chains, dict) or not chains:
        raise ValueError("reference_chains must be a nonempty chain-to-sequence object")
    if any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in chains.items()):
        raise ValueError("Invalid reference_chains")
    return dict(chains)


def _profile_neff(case: Mapping[str, Any]) -> float | None:
    """Only copy an explicitly reported Neff; unique/raw depth is not Neff."""
    metadata = case["evidence"].get("source_metadata", [])
    items = metadata if isinstance(metadata, list) else [metadata]
    values = []
    for item in items:
        if not isinstance(item, dict):
            continue
        provenance = item.get("provenance", {})
        method = provenance.get("MSA_method", {}) if isinstance(provenance, dict) else {}
        value = _number(method.get("neff")) if isinstance(method, dict) else None
        if value is not None:
            values.append(value)
    if len(set(values)) > 1:
        raise ValueError("Conflicting explicitly reported MSA Neff values")
    return values[0] if values else None


def _site_index(case: Mapping[str, Any], references: Mapping[str | None, str]):
    evidence = case.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("format") != "protein_site_table_v1":
        raise ValueError("Expected evidence.format == protein_site_table_v1")
    rows = evidence.get("sites")
    if not isinstance(rows, list):
        raise ValueError("evidence.sites must be a list")
    index = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every site row must be an object")
        chain, position = row.get("chain"), row.get("position")
        if chain not in references:
            raise ValueError(f"Unknown chain in site table: {chain!r}")
        if isinstance(position, bool) or not isinstance(position, int) or not 1 <= position <= len(references[chain]):
            raise ValueError(f"Invalid site position: {position!r}")
        if row.get("WT") != references[chain][position - 1]:
            raise ValueError(f"WT mismatch at {chain!r}:{position}")
        key = (chain, position)
        if key in index:
            raise ValueError(f"Duplicate site row: {key}")
        index[key] = row
    return index


def _site_diagnostic(row: Mapping[str, Any], profile_neff: float | None) -> dict[str, Any]:
    msa = row.get("msa")
    msa = msa if isinstance(msa, dict) else {}
    depth = _number(msa.get("non_gap_depth"))
    kind = msa.get("depth_kind")
    frequencies = msa.get("frequencies")
    gap_raw = _number(msa.get("gap_fraction"))
    gap = gap_raw
    explicit_neff = _number(msa.get("neff", msa.get("profile_neff")))
    flags = []
    if not isinstance(frequencies, dict) or not frequencies:
        flags.append("frequencies_missing")
    if depth is None:
        flags.append("non_gap_depth_missing")
    elif depth <= 0:
        flags.append("no_observed_homologs")
    if gap is None:
        flags.append("gap_fraction_missing")
    elif not 0 <= gap <= 1:
        if -1e-12 <= gap <= 1 + 1e-12:
            gap = min(1.0, max(0.0, gap))
            flags.append("gap_fraction_roundoff_clamped")
        else:
            flags.append("gap_fraction_invalid")
    if msa.get("homolog_support") == "none" and "no_observed_homologs" not in flags:
        flags.append("no_observed_homologs")
    return {
        "chain": row.get("chain"), "position": row["position"], "WT": row["WT"],
        "non_gap_depth": depth, "depth_kind": kind,
        "effective_non_gap_depth": depth if kind == "effective_weighted" else None,
        "profile_neff": explicit_neff if explicit_neff is not None else (profile_neff if row.get("chain") is None else None),
        "gap_fraction": gap, "gap_fraction_raw": gap_raw, "homolog_support": msa.get("homolog_support"),
        "profile_scope": msa.get("profile_scope"), "missing_flags": flags,
    }


def _candidate_substitutions(candidate: Mapping[str, Any], references: Mapping[str | None, str]):
    if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
        raise ValueError("Every query candidate needs a string id")
    if None in references:
        sequences = {None: candidate.get("sequence")}
    else:
        sequences = candidate.get("mutated_chains")
        if not isinstance(sequences, dict) or set(sequences) != set(references):
            raise ValueError(f"Candidate {candidate['id']} requires every reference chain in mutated_chains")
    changes = []
    for chain, reference in references.items():
        sequence = sequences.get(chain)
        if not isinstance(sequence, str) or len(sequence) != len(reference):
            raise ValueError(f"Candidate {candidate['id']} has an unsupported indel/missing sequence in {chain!r}")
        for position, (wt, alt) in enumerate(zip(reference, sequence), 1):
            if wt != alt:
                changes.append({"chain": chain, "position": position, "WT": wt, "ALT": alt})
    order = candidate.get("mutation_order")
    if order is not None and order != len(changes):
        raise ValueError(f"Candidate {candidate['id']} mutation_order does not match its sequence")
    return changes


def _verified_rsa_index(artifact: Mapping[str, Any] | None, references: Mapping[str | None, str]):
    """Accept fraction-unit RSA with explicit verification and sequence hashes."""
    if artifact is None:
        return {}
    if not isinstance(artifact, dict) or artifact.get("format") != "verified_rsa_v1":
        raise ValueError("RSA must be a verified_rsa_v1 artifact")
    if artifact.get("verified") is not True or artifact.get("units") != "fraction":
        raise ValueError("RSA artifact needs verified=true and units=fraction")
    hashes = artifact.get("reference_sequences_sha256")
    expected = {"__single__" if c is None else c: _sha256(s) for c, s in references.items()}
    if hashes != expected:
        raise ValueError("RSA artifact reference sequence hashes do not match the case")
    if not artifact.get("provenance"):
        raise ValueError("Verified RSA requires calculation provenance")
    rows = artifact.get("sites")
    if not isinstance(rows, list):
        raise ValueError("RSA sites must be a list")
    index = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("RSA site rows must be objects")
        chain, position = row.get("chain"), row.get("position")
        if chain not in references or isinstance(position, bool) or not isinstance(position, int) or not 1 <= position <= len(references[chain]):
            raise ValueError("Invalid verified RSA site coordinate")
        if row.get("WT") != references[chain][position - 1]:
            raise ValueError("Verified RSA WT mismatch")
        key = (chain, position)
        if key in index:
            raise ValueError("Duplicate verified RSA site")
        value = _number(row.get("rsa"))
        if row.get("verified") is True and value is not None and 0 <= value <= 1:
            index[key] = value
    return index


def _aggregate(terms: list[dict[str, Any]], field: str) -> dict[str, Any]:
    observed = [term[field] for term in terms if term.get(field) is not None]
    count, total = len(observed), len(terms)
    return {
        "score": math.fsum(observed) if count == total else None,
        "observed_term_partial_sum": math.fsum(observed) if observed or total == 0 else None,
        "observed_terms": count, "required_terms": total,
        "coverage": count / total if total else 1.0,
        "complete": count == total,
    }


def score_case(case: Mapping[str, Any], *, verified_rsa: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compute an ordered, independent-site sidecar without using assay labels."""
    if case.get("domain") not in {"protein", "ppi"}:
        raise ValueError("Only protein and protein-protein interaction cases are supported")
    references = _references(case)
    sites = _site_index(case, references)
    neff = _profile_neff(case)
    rsa = _verified_rsa_index(verified_rsa, references)
    query = case.get("query")
    if not isinstance(query, list):
        raise ValueError("case.query must be an ordered candidate list")
    if len({q.get("id") for q in query if isinstance(q, dict)}) != len(query):
        raise ValueError("Candidate ids must be unique")
    diagnostics = [_site_diagnostic(row, neff) for row in sites.values()]
    diagnostic_index = {(row["chain"], row["position"]): row for row in diagnostics}
    candidates = []
    for candidate in query:
        changes = _candidate_substitutions(candidate, references)
        terms = []
        for change in changes:
            key = (change["chain"], change["position"])
            term = dict(change)
            term.update({name: None for name in METHODS})
            term.update({"rsa_scaled_log_frequency_ratio": None, "rsa_scaled_true_log_odds_ratio": None,
                         "verified_rsa_fraction": rsa.get(key), "missing_flags": [], "rsa_missing_flags": []})
            term.update({name: None for name in ("frequency_WT", "frequency_ALT", "non_gap_depth", "depth_kind",
                                                "effective_non_gap_depth", "profile_neff", "gap_fraction",
                                                "homolog_support", "profile_scope")})
            row = sites.get(key)
            if row is None:
                term["missing_flags"].append("site_missing")
            else:
                diagnostic = diagnostic_index[key]
                term.update({k: v for k, v in diagnostic.items() if k not in {"missing_flags", "chain", "position", "WT"}})
                term["missing_flags"].extend(diagnostic["missing_flags"])
                msa = row.get("msa")
                frequencies = msa.get("frequencies", {}) if isinstance(msa, dict) else {}
                frequencies = frequencies if isinstance(frequencies, dict) else {}
                f_wt, f_alt = _number(frequencies.get(change["WT"])), _number(frequencies.get(change["ALT"]))
                term.update({"frequency_WT": f_wt, "frequency_ALT": f_alt})
                for label, value in (("WT", f_wt), ("ALT", f_alt)):
                    if value is None:
                        term["missing_flags"].append(f"frequency_{label}_missing")
                    elif not 0 < value <= 1:
                        term["missing_flags"].append(f"frequency_{label}_undefined_for_log_ratio")
                    if value is not None and not 0 < value < 1:
                        term["missing_flags"].append(f"frequency_{label}_undefined_for_log_odds")
                if change["WT"] not in AA or change["ALT"] not in AA:
                    term["missing_flags"].append("nonstandard_substitution")
                unsupported = {"no_observed_homologs", "nonstandard_substitution", "gap_fraction_invalid"}
                if f_wt is not None and f_alt is not None and 0 < f_wt <= 1 and 0 < f_alt <= 1 and not unsupported.intersection(term["missing_flags"]):
                    lr = math.log(f_alt) - math.log(f_wt)
                    term[METHODS[0]] = lr
                    if key in rsa:
                        term["rsa_scaled_log_frequency_ratio"] = (1 - rsa[key]) * lr
                    if f_wt < 1 and f_alt < 1:
                        lor = lr + math.log1p(-f_wt) - math.log1p(-f_alt)
                        term[METHODS[1]] = lor
                        if key in rsa:
                            term["rsa_scaled_true_log_odds_ratio"] = (1 - rsa[key]) * lor
            if key not in rsa:
                term["rsa_missing_flags"].append("verified_rsa_missing")
            terms.append(term)
        result = {"id": candidate["id"], "mutation": candidate.get("mutation"),
                  "mutation_order": len(changes), "substitution_terms": terms,
                  "scores": {name: _aggregate(terms, name) for name in METHODS},
                  "rsa_scaled_scores": {name: _aggregate(terms, field) for name, field in
                                        zip(METHODS, ("rsa_scaled_log_frequency_ratio", "rsa_scaled_true_log_odds_ratio"))}}
        candidates.append(result)
    return {
        "format": SCORE_FORMAT, "case_id": case.get("case_id"),
        "candidate_order": [row["id"] for row in candidates],
        "reference_sequences_sha256": {"__single__" if c is None else c: _sha256(s) for c, s in references.items()},
        "score_direction": "higher means ALT is more evolutionarily compatible than WT",
        "frequency_policy": "Use supplied frequencies exactly; no new pseudocount, renormalization or site weights",
        "epistasis": False, "cross_chain_coupling": False,
        "verified_rsa_supplied": verified_rsa is not None,
        "profile_neff": neff, "site_diagnostics": diagnostics,
        "scientific_boundaries": [
            "Independent-site additive mutation scores; no combination epistasis.",
            "Unpaired chain MSAs do not provide cross-chain coevolution.",
            "Non-gap unique/raw row counts are not effective weighted Neff.",
            "Summary frequencies may include upstream pseudocounts and only queried amino acids.",
            "A missing or undefined term makes the complete candidate score null.",
            "Observed-term partial sums are incomplete evidence, not complete candidate scores.",
            "No assay labels or historical predictions are consulted.",
            "Optional RSA scaling uses separately verified relative solvent accessibility only.",
        ],
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verified-rsa", type=Path, help="Optional verified_rsa_v1 JSON; never a raw SASA table")
    args = parser.parse_args()
    if args.output.resolve() == args.case.resolve() or (args.verified_rsa and args.output.resolve() == args.verified_rsa.resolve()):
        parser.error("Output must not overwrite an input artifact")
    case = json.loads(args.case.read_text(encoding="utf-8"))
    rsa = json.loads(args.verified_rsa.read_text(encoding="utf-8")) if args.verified_rsa else None
    sidecar = score_case(case, verified_rsa=rsa)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    coverage = {name: {"complete_candidates": sum(c["scores"][name]["complete"] for c in sidecar["candidates"]),
                       "total_candidates": len(sidecar["candidates"])} for name in METHODS}
    print(json.dumps({"case_id": sidecar["case_id"], "output": str(args.output.resolve()), "coverage": coverage}))


if __name__ == "__main__":
    main()
