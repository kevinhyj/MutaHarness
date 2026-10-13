"""Local WT structure descriptors for public protein/ppi canonical cases.

No fitness score, learned prior, network call, or mutant coordinate inference is
performed. ``describe_case(case)`` consumes only reference sequences, query
variants, and the ``evidence.sites[*].wt_structure`` summary plus its provenance.

API::

    packet = describe_case(case)
    one = describe_candidate(case, case["query"][0])
    exposure = rsa_sites(case)

CLI (stdout unless --out is given)::

    python structure_tools.py --case case.json --out structure_descriptors.json
    python structure_tools.py --case case.json --candidate-id M001
    python structure_tools.py --case case.json --rsa-only

RSA here is a *derived normalization of supplied WT SASA*, not an independent
recalculation from raw coordinates. The Tien/Wilke theoretical ALLOWED reference
areas were derived with a different ASA calculation from this case's supplied
Shrake-Rupley summaries. Retain values above 1 to expose algorithm, terminal, or
geometry differences. Never interpret this as mutant RSA or a fitness effect.

Nearest-neighbor lists have at most three nonlocal intrachain CA neighbors.
Their absence is unknown pair geometry, not evidence that a pair is distant.
The separate CA-neighbor count can be complete according to source metadata;
it does not make that three-neighbor list a complete contact graph.

Summary-only mode does not reconstruct a full graph. The separate
``raw_structure_evidence(case, pdb_path)`` API / ``--raw-pdb`` CLI mode builds
complete cutoff graphs over observed WT atoms and recomputes SASA using NumPy.
Neither mode produces DSSP classes, mutant clashes, energies, or epistasis.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any


# Tien et al. (2013), theoretical maximum ASA in ALLOWED Ramachandran region,
# Table 1. Matches the "Wilke" table in Biopython Bio.Data.PDBData.
MAXASA_A2 = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}
RSA_PROVENANCE = {
    "normalization": "Tien_2013_theoretical_ALLOWED_Wilke",
    "citation": "Tien MZ et al. PLoS ONE 8(11):e80635 (2013), Table 1",
    "doi": "https://doi.org/10.1371/journal.pone.0080635",
    "implementation_reference": "https://raw.githubusercontent.com/biopython/biopython/master/Bio/Data/PDBData.py",
    "formula": "supplied WT absolute_SASA_A2 / MAXASA_A2[WT]",
    "maximum_areas_A2": MAXASA_A2,
    "clipped": False,
    "raw_coordinate_recomputed": False,
    "compatibility_caveat": "Supplied isolated-chain Shrake-Rupley ASA and reference ASA need not use identical radii/algorithms; RSA is a derived exposure descriptor.",
}
FORBIDDEN_FIELDS = {
    "support", "support_ids", "fitness", "label", "labels", "dms_score",
    "ground_truth", "support_calibration", "normalized_score", "anchor",
    "predictions", "oldpredictions", "old_predictions",
}
_AA = frozenset(MAXASA_A2)
_MUTATION = re.compile(r"(?:(?P<chain>[A-Za-z0-9_.-]+):)?(?P<wt>[ACDEFGHIKLMNPQRSTVWY])(?P<pos>[1-9][0-9]*)(?P<alt>[ACDEFGHIKLMNPQRSTVWY])")


def _reject_measured_fields(value: Any) -> None:
    """Inspect field names only; refuse labeled/nonpublic case payloads."""
    if isinstance(value, Mapping):
        bad = FORBIDDEN_FIELDS.intersection(str(k).lower() for k in value)
        if bad:
            raise ValueError("Public zero-shot case required; forbidden fields: " + ", ".join(sorted(bad)))
        for child in value.values():
            _reject_measured_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_measured_fields(child)


def _number(value: Any, name: str, minimum: float | None = None,
            maximum: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is nonfinite or outside its allowed range")
    return result


def _count(value: Any, name: str) -> int | None:
    number = _number(value, name, 0)
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _sequences(case: Mapping[str, Any]) -> tuple[str | None, dict[str | None, str]]:
    default_chain = case.get("context", {}).get("variant_chain")
    reference = case.get("reference_sequence")
    if not isinstance(reference, str) or not reference:
        raise ValueError("Missing nonempty reference_sequence")
    chains = dict(case.get("reference_chains", {}))
    if default_chain in chains and chains[default_chain] != reference:
        raise ValueError("reference_sequence disagrees with variant reference chain")
    chains.setdefault(default_chain, reference)
    if any(not isinstance(sequence, str) for sequence in chains.values()):
        raise ValueError("Reference chain sequences must be strings")
    return default_chain, chains


def _structure_sources(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Copy structure metadata only, excluding MSA priors or assay data."""
    sources = []
    for item in case.get("evidence", {}).get("source_metadata", []):
        provenance = item.get("provenance", {})
        source = {k: provenance[k] for k in (
            "WT_structure", "structure_features_sha256",
            "structure_limitations", "structure_measurement_definitions",
        ) if k in provenance}
        if "structure" in item:
            source["structure"] = item["structure"]
        if source and source not in sources:
            sources.append(source)
    return sources


def _af2_assertion(sources: list[dict[str, Any]], chain: str | None) -> dict[str, Any] | None:
    for item in sources:
        wt = item.get("WT_structure", {})
        origin = str(wt.get("origin_asserted_by_caller", "")).lower()
        if origin in {"alphafold", "alphafold2", "af2"} and (chain is None or wt.get("chain_id") == chain):
            return {"source": "WT_structure.origin_asserted_by_caller",
                    "value": wt["origin_asserted_by_caller"],
                    "chain_id": wt.get("chain_id"),
                    "pdb_sha256": wt.get("pdb_sha256"),
                    "independently_verified_origin": False}
    return None


def _prepare(case: Mapping[str, Any]) -> dict[str, Any]:
    _reject_measured_fields(case)
    if case.get("domain") not in {"protein", "ppi"}:
        raise ValueError("Only protein/ppi cases are accepted")
    if case.get("evidence", {}).get("format") != "protein_site_table_v1":
        raise ValueError("Expected canonical protein_site_table_v1 evidence")
    default_chain, sequences = _sequences(case)
    sites = {}
    for site in case["evidence"].get("sites", []):
        chain = site.get("chain", default_chain)
        position = _count(site.get("position"), "site position")
        if position is None or position < 1 or chain not in sequences or position > len(sequences[chain]):
            raise ValueError("Invalid site reference coordinate")
        if site.get("WT") != sequences[chain][position - 1]:
            raise ValueError(f"Structure WT mismatch at {chain}:{position}")
        key = (chain, position)
        if key in sites:
            raise ValueError(f"Duplicate canonical site: {key}")
        structure = site.get("wt_structure", {})
        if not isinstance(structure, Mapping):
            raise ValueError("wt_structure must be an object")
        sites[key] = structure
    return {"default_chain": default_chain, "sequences": sequences,
            "sites": sites, "sources": _structure_sources(case)}


def _parse_changes(state: Mapping[str, Any], query: Mapping[str, Any]) -> list[dict[str, Any]]:
    notation = query.get("mutation")
    if not isinstance(notation, str):
        raise ValueError("Each candidate must supply mutation notation")
    # PPI uses chain:AApositionAA separated by semicolons. Protein uses colons.
    whole = _MUTATION.fullmatch(notation)
    one_substitution = whole is not None and (
        whole["chain"] is None or whole["chain"] in state["sequences"])
    tokens = notation.split(";") if ";" in notation else (
        [notation] if one_substitution else notation.split(":"))
    if notation in {"", "WT"}:
        tokens = []
    changes, seen = [], set()
    expected = {chain: list(seq) for chain, seq in state["sequences"].items()}
    for token in tokens:
        match = _MUTATION.fullmatch(token.strip())
        if match is None:
            raise ValueError(f"Unsupported substitution notation: {token}")
        chain = match["chain"] or state["default_chain"]
        position, wt, alt = int(match["pos"]), match["wt"], match["alt"]
        key = (chain, position)
        sequence = state["sequences"].get(chain)
        if sequence is None or position > len(sequence) or sequence[position - 1] != wt:
            raise ValueError(f"Mutation does not match reference: {token}")
        if key in seen:
            raise ValueError(f"Repeated mutation position: {key}")
        seen.add(key)
        expected[chain][position - 1] = alt
        changes.append({"chain": chain, "position": position, "WT": wt, "ALT": alt,
                        "identity_substitution": wt == alt})
    if query.get("sequence") is not None and query["sequence"] != "".join(expected[state["default_chain"]]):
        raise ValueError("Candidate sequence does not match its mutation notation")
    for chain, sequence in query.get("mutated_chains", {}).items():
        if chain not in expected or sequence != "".join(expected[chain]):
            raise ValueError("Candidate mutated_chains disagree with mutation notation")
    return changes


def _neighbors(state: Mapping[str, Any], chain: str | None, position: int,
               structure: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    rows = structure.get("nearest_nonlocal_position_WT_distanceA")
    if rows is None:
        return None
    if not isinstance(rows, list) or len(rows) > 3:
        raise ValueError("Expected at most three supplied nonlocal CA neighbors")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("Invalid nearest-neighbor summary row")
        neighbor = _count(row[0], "neighbor position")
        if neighbor is None or neighbor < 1 or neighbor > len(state["sequences"][chain]) or abs(neighbor - position) <= 2:
            raise ValueError("Invalid nonlocal neighbor coordinate")
        if neighbor in seen or row[1] != state["sequences"][chain][neighbor - 1]:
            raise ValueError("Duplicate neighbor or neighbor WT mismatch")
        seen.add(neighbor)
        result.append({"chain": chain, "position": neighbor, "WT": row[1],
                       "CA_distance_A": _number(row[2], "CA distance", 0)})
    return result


def _site_descriptor(state: Mapping[str, Any], change: Mapping[str, Any],
                     confidence_threshold: float) -> dict[str, Any]:
    chain, position, wt = change["chain"], change["position"], change["WT"]
    structure = state["sites"].get((chain, position), {})
    asa = _number(structure.get("absolute_SASA_A2"), "absolute_SASA_A2", 0)
    rsa = None if asa is None else asa / MAXASA_A2[wt]
    phi = _number(structure.get("phi_deg"), "phi_deg", -180, 180)
    psi = _number(structure.get("psi_deg"), "psi_deg", -180, 180)
    origin = _af2_assertion(state["sources"], chain)
    plddt = _number(structure.get("pLDDT"), "pLDDT", 0, 100) if origin else None
    neighbors = _neighbors(state, chain, position, structure)
    interface_distance = _number(structure.get("nearest_other_chain_CA_distance_A"), "interface CA distance", 0)
    observed = {
        "absolute_SASA_A2": asa, "phi_deg": phi, "psi_deg": psi,
        "pLDDT": plddt,
        "intrachain_CA_neighbors_within8A": _count(structure.get("CA_neighbors_within8A"), "CA neighbors"),
        "CA_count_complete_per_source": structure.get("contacts_complete"),
        "nearest_nonlocal_CA_subset": neighbors,
        "nearest_other_chain": structure.get("nearest_other_chain"),
        "nearest_other_chain_CA_distance_A": interface_distance,
        "other_chain_CA_within8A": _count(structure.get("other_chain_CA_within8A"), "other chain CA count"),
        "pdb_residue_number": structure.get("pdb_residue_number"),
        "pdb_insertion_code": structure.get("pdb_insertion_code"),
    }
    complete = observed["CA_count_complete_per_source"]
    if complete is not None and not isinstance(complete, bool):
        raise ValueError("contacts_complete must be boolean or null")
    return {
        **change,
        "structure_summary_present": bool(structure),
        "wt_observations": observed,
        "derived_descriptors": {
            "wt_rsa": rsa, "wt_rsa_exceeds_one": None if rsa is None else rsa > 1,
            "rsa_reference_maximum_A2": MAXASA_A2[wt],
            "phi_sin": None if phi is None else math.sin(math.radians(phi)),
            "phi_cos": None if phi is None else math.cos(math.radians(phi)),
            "psi_sin": None if psi is None else math.sin(math.radians(psi)),
            "psi_cos": None if psi is None else math.cos(math.radians(psi)),
            "plddt_at_least_threshold": None if plddt is None else plddt >= confidence_threshold,
            "cross_chain_CA_within8A": None if interface_distance is None else interface_distance <= 8,
        },
        "confidence_provenance": origin,
        "confidence_threshold": confidence_threshold,
        "nearest_subset_is_complete_graph": False,
        "raw_coordinate_recomputed": False,
    }


def _pair_descriptor(left: Mapping[str, Any], right: Mapping[str, Any],
                     cutoff_A: float) -> dict[str, Any]:
    same_chain = left["chain"] == right["chain"]
    separation = abs(left["position"] - right["position"]) if same_chain else None
    observations = []
    if same_chain:
        for source, target in ((left, right), (right, left)):
            for neighbor in source["wt_observations"]["nearest_nonlocal_CA_subset"] or []:
                if neighbor["position"] == target["position"] and neighbor["CA_distance_A"] is not None:
                    observations.append({"reported_from_position": source["position"],
                                         "CA_distance_A": neighbor["CA_distance_A"]})
    distances = [item["CA_distance_A"] for item in observations]
    # Two summaries can differ by rounding; retain both rather than invent precision.
    consistent = None if len(distances) < 2 else max(distances) - min(distances) <= 0.03
    distance = distances[0] if distances and consistent is not False else None
    if distance is not None:
        status = "observed_in_truncated_nearest_CA_subset"
    elif consistent is False:
        status = "conflicting_supplied_distances"
    elif not same_chain:
        status = "unknown_cross_chain_pair_geometry"
    elif separation is not None and separation <= 2:
        status = "unknown_sequence_local_pair_excluded_by_summary"
    else:
        status = "unknown_not_observed_in_available_subset"
    return {
        "left": {k: left[k] for k in ("chain", "position", "WT", "ALT")},
        "right": {k: right[k] for k in ("chain", "position", "WT", "ALT")},
        "same_chain": same_chain, "sequence_separation": separation,
        "wt_CA_distance_A": distance, "CA_cutoff_A": cutoff_A,
        "within_CA_cutoff": None if distance is None else distance <= cutoff_A,
        "distance_observations": observations,
        "reciprocal_distance_consistent": consistent,
        "status": status,
        "interaction_or_epistasis_inferred": False,
    }


def _describe_candidate(state: Mapping[str, Any], query: Mapping[str, Any],
                        confidence_threshold: float, pair_cutoff_A: float) -> dict[str, Any]:
    sites = [_site_descriptor(state, change, confidence_threshold)
             for change in _parse_changes(state, query)]
    pairs = [_pair_descriptor(a, b, pair_cutoff_A) for a, b in itertools.combinations(sites, 2)]
    return {
        "id": query.get("id"), "mutation": query.get("mutation"),
        "mutation_count": len(sites), "sites": sites, "mutated_site_pairs": pairs,
        "coverage": {
            "sites_with_structure_summary": sum(site["structure_summary_present"] for site in sites),
            "sites_with_derived_WT_RSA": sum(site["derived_descriptors"]["wt_rsa"] is not None for site in sites),
            "sites_with_source_asserted_AF2_pLDDT": sum(site["wt_observations"]["pLDDT"] is not None for site in sites),
            "total_mutated_pairs": len(pairs),
            "pairs_with_observed_WT_CA_distance": sum(pair["wt_CA_distance_A"] is not None for pair in pairs),
            "pairs_with_unknown_WT_CA_distance": sum(pair["wt_CA_distance_A"] is None for pair in pairs),
        },
    }


def _parameters(confidence_threshold: float, pair_cutoff_A: float) -> tuple[float, float]:
    confidence = _number(confidence_threshold, "confidence_threshold", 0, 100)
    cutoff = _number(pair_cutoff_A, "pair_cutoff_A", 0)
    if confidence is None or cutoff is None:
        raise ValueError("Thresholds cannot be null")
    return confidence, cutoff


def describe_candidate(case: Mapping[str, Any], query: Mapping[str, Any], *,
                       confidence_threshold: float = 70.0, pair_cutoff_A: float = 8.0) -> dict[str, Any]:
    """Validate substitutions and return nullable static WT descriptors."""
    _reject_measured_fields(query)
    confidence, cutoff = _parameters(confidence_threshold, pair_cutoff_A)
    return _describe_candidate(_prepare(case), query, confidence, cutoff)


def describe_case(case: Mapping[str, Any], candidates: list[Mapping[str, Any]] | None = None, *,
                  confidence_threshold: float = 70.0, pair_cutoff_A: float = 8.0) -> dict[str, Any]:
    """Describe all public query variants; no ranking or score is imposed."""
    state = _prepare(case)
    confidence, cutoff = _parameters(confidence_threshold, pair_cutoff_A)
    selected = case.get("query", []) if candidates is None else candidates
    _reject_measured_fields(selected)
    return {
        "format": "protein_wt_structure_descriptors_v1", "case_id": case.get("case_id"),
        "coordinate_system": "1-based reference sequence, chain identifiers kept separate",
        "reference_sequence_sha256": hashlib.sha256(case["reference_sequence"].encode()).hexdigest(),
        "source_metadata": state["sources"], "rsa_provenance": RSA_PROVENANCE,
        "definitions": {
            "intrachain_CA_count": "Observed intrachain CA distance <=8 A, excluding |i-j|<=2; source-reported count completeness is separate from graph completeness.",
            "nearest_subset": "Up to three observed nonlocal intrachain CA neighbors without a distance cutoff; absent edges are unknown.",
            "interface": "Source-supplied cross-chain CA geometry; no heavy-atom interactions or validated biological assembly inferred.",
            "confidence": "pLDDT interpreted only where source explicitly asserts AlphaFold origin; threshold is a reliability descriptor, never a fitness score.",
            "torsions": "Supplied signed WT backbone torsions; sine/cosine encoding is arithmetic, no DSSP class inferred.",
        },
        "limitations": [
            "All geometry is static WT summary data, with no mutant coordinates, clash, energy, or epistasis prediction.",
            "Missing observations stay null; no imputation to zero or mean.",
            "Top-three neighbors are not a full contact graph; unobserved mutant pairs remain unknown.",
            "RSA is normalization of supplied isolated-chain absolute SASA, not raw-coordinate verification or mutant SASA.",
            "No PAE is supplied, so long-range relative placement confidence is unavailable.",
            "PPI CA proximity alone does not establish chemistry, binding energy, or biological assembly validity.",
        ],
        "candidates": [_describe_candidate(state, query, confidence, cutoff) for query in selected],
    }


def rsa_sites(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return derived exposure by WT site, for a caller's optional modeling use."""
    state = _prepare(case)
    rows = []
    for (chain, position), structure in state["sites"].items():
        wt = state["sequences"][chain][position - 1]
        asa = _number(structure.get("absolute_SASA_A2"), "absolute_SASA_A2", 0)
        rows.append({"chain": chain, "position": position, "WT": wt,
                     "rsa": None if asa is None else asa / MAXASA_A2[wt],
                     "absolute_SASA_A2": asa, "reference_maximum_A2": MAXASA_A2[wt],
                     "source": "summary_absolute_sasa_normalization",
                     "raw_coordinate_recomputed": False})
    return {"format": "derived_wt_rsa_v1", "units": "fraction",
            "case_id": case.get("case_id"),
            "reference_sequence_sha256": hashlib.sha256(case["reference_sequence"].encode()).hexdigest(),
            "sites": sorted(rows, key=lambda row: (str(row["chain"]), row["position"])),
            "provenance": RSA_PROVENANCE, "structure_sources": state["sources"]}


_THREE_TO_ONE = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))
_THREE_TO_ONE.update({"MSE": "M", "UNK": "X"})
_SIDECHAIN_ATOMS = {
    "A": "CB", "R": "CB CG CD NE CZ NH1 NH2", "N": "CB CG OD1 ND2",
    "D": "CB CG OD1 OD2", "C": "CB SG", "Q": "CB CG CD OE1 NE2",
    "E": "CB CG CD OE1 OE2", "G": "", "H": "CB CG ND1 CD2 CE1 NE2",
    "I": "CB CG1 CG2 CD1", "L": "CB CG CD1 CD2", "K": "CB CG CD CE NZ",
    "M": "CB CG SD CE", "F": "CB CG CD1 CD2 CE1 CE2 CZ", "P": "CB CG CD",
    "S": "CB OG", "T": "CB OG1 CG2", "W": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "Y": "CB CG CD1 CD2 CE1 CE2 CZ OH", "V": "CB CG1 CG2",
}
# Standard vdW radii (A), same values as Biopython's documented SASA defaults.
# Unknown elements are not assigned an invented radius: their SASA is unavailable.
_RADII_A = {"H": 1.2, "HE": 1.4, "C": 1.7, "N": 1.55, "O": 1.52,
            "F": 1.47, "NA": 2.27, "MG": 1.73, "P": 1.8, "S": 1.8,
            "CL": 1.75, "K": 2.75, "CA": 2.31, "NI": 1.63, "CU": 1.4,
            "ZN": 1.39, "SE": 1.9, "BR": 1.85, "CD": 1.58, "I": 1.98, "HG": 1.55}


def _read_pdb(path: Path, *, include_zero_occupancy: bool = False) -> list[dict[str, Any]]:
    """Read first-model supplied atoms; select one residue conformer."""
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        record = line[:6].strip()
        if record == "ENDMDL":
            break
        if record not in {"ATOM", "HETATM"}:
            continue
        resname = line[17:20].strip().upper()
        if resname in {"HOH", "WAT", "DOD"}:
            continue
        chain, number, insertion = line[21:22].strip(), int(line[22:26]), line[26:27].strip()
        atom_name, altloc = line[12:16].strip(), line[16:17].strip()
        protein = resname in _THREE_TO_ONE
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            cleaned = re.sub(r"^[0-9]+", "", atom_name)
            element = cleaned[:1].upper() if protein else cleaned[:2].upper()
            if element not in _RADII_A and len(element) > 1:
                element = element[:1]
        occupancy = float(line[54:60].strip() or "1")
        if occupancy <= 0 and not include_zero_occupancy:
            continue
        coord = [float(line[i:i + 8]) for i in (30, 38, 46)]
        if any(not math.isfinite(x) for x in coord):
            raise ValueError("Nonfinite PDB coordinates")
        bfactor = float(line[60:66].strip() or "0")
        key = (chain, number, insertion, resname)
        residue = grouped.setdefault(key, {"pdb_chain": chain, "pdb_residue_number": number,
            "pdb_insertion_code": insertion, "residue_name": resname,
            "WT": _THREE_TO_ONE.get(resname), "protein": protein, "choices": []})
        residue["choices"].append({"name": atom_name, "element": element, "coord": coord,
                                   "occupancy": occupancy, "altloc": altloc, "bfactor": bfactor})
    result = []
    for residue in grouped.values():
        choices = residue.pop("choices")
        occupancies: dict[str, float] = {}
        for atom in choices:
            if atom["altloc"]:
                occupancies[atom["altloc"]] = occupancies.get(atom["altloc"], 0) + atom["occupancy"]
        conformer = min(occupancies, key=lambda alt: (-occupancies[alt], alt != "A", alt)) if occupancies else None
        selected = {}
        for atom in choices:
            if atom["altloc"] and atom["altloc"] != conformer:
                continue
            name = atom["name"]
            if name in selected and atom["occupancy"] <= selected[name]["occupancy"]:
                continue
            selected[name] = atom
        residue["atoms"] = selected
        residue["selected_altloc"] = conformer
        residue["available_altlocs"] = sorted(occupancies)
        if selected:
            result.append(residue)
    if not result:
        raise ValueError("PDB contains no included atoms in first model")
    return result


def _map_raw_residues(state: Mapping[str, Any], residues: list[dict[str, Any]],
                      explicit_map: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_chain: dict[str, list[dict[str, Any]]] = {}
    for residue in residues:
        if residue["protein"]:
            by_chain.setdefault(residue["pdb_chain"], []).append(residue)
    mappings, used = [], set()
    for chain, reference in state["sequences"].items():
        if chain is None:
            asserted = _af2_assertion(state["sources"], None)
            raw_chain = asserted.get("chain_id") if asserted else None
            if raw_chain is None:
                matches = [key for key, values in by_chain.items()
                           if "".join(r["WT"] for r in values) == reference]
                if len(matches) != 1:
                    raise ValueError("Cannot uniquely choose raw PDB chain for reference")
                raw_chain = matches[0]
        else:
            raw_chain = chain
        observed = by_chain.get(raw_chain, [])
        if not observed:
            raise ValueError(f"Missing expected PDB chain: {raw_chain}")
        observed_sequence = "".join(r["WT"] for r in observed)
        chain_map = None if explicit_map is None else explicit_map.get(chain, explicit_map.get(str(chain)))
        if chain_map is not None:
            lookup = {(r["pdb_residue_number"], r["pdb_insertion_code"]): r for r in observed}
            pairs = []
            for pos_text, target in chain_map.items():
                position = int(pos_text)
                raw_key = (int(target["pdb_residue_number"]), target.get("pdb_insertion_code", ""))
                if raw_key not in lookup:
                    raise ValueError("Explicit mapping names an unobserved residue")
                pairs.append((position, lookup[raw_key]))
            method = "explicit_reference_position_to_author_residue_validated_by_WT"
        elif observed_sequence == reference:
            pairs = list(enumerate(observed, 1))
            method = "exact_whole_chain_sequence"
        elif observed_sequence == reference.replace("X", ""):
            positions = [i for i, aa in enumerate(reference, 1) if aa != "X"]
            pairs = list(zip(positions, observed))
            method = "exact_reference_sequence_after_removing_explicit_unknown_X_slots"
        elif reference.count(observed_sequence) == 1:
            start = reference.index(observed_sequence) + 1
            pairs = list(zip(range(start, start + len(observed)), observed))
            method = "unique_exact_contiguous_reference_subsequence"
        else:
            raise ValueError(f"Raw PDB chain {raw_chain} does not uniquely match reference construct; provide explicit residue_map")
        positions_used = set()
        for position, residue in pairs:
            if position < 1 or position > len(reference) or position in positions_used or reference[position - 1] != residue["WT"]:
                raise ValueError("Raw PDB mapping WT identity/range/uniqueness mismatch")
            positions_used.add(position)
            residue["chain"], residue["position"] = chain, position
            if id(residue) in used:
                raise ValueError("One observed residue maps to multiple reference residues")
            used.add(id(residue))
        mappings.append({"chain": chain, "pdb_chain": raw_chain, "method": method,
                         "reference_length": len(reference), "mapped_observed_residues": len(pairs),
                         "unobserved_positions": [i for i in range(1, len(reference) + 1) if i not in positions_used]})
    # Validate all canonical author-number anchors already supplied for PPI sites.
    for residue in residues:
        if "position" not in residue:
            continue
        summary = state["sites"].get((residue["chain"], residue["position"]), {})
        if summary.get("pdb_residue_number") is not None and (
            summary["pdb_residue_number"] != residue["pdb_residue_number"] or
            summary.get("pdb_insertion_code", "") != residue["pdb_insertion_code"]):
            raise ValueError("Raw mapping conflicts with supplied canonical PDB anchor")
    selected_chains = {mapping["pdb_chain"] for mapping in mappings}
    selected = [r for r in residues if id(r) in used or (not r["protein"] and r["pdb_chain"] in selected_chains)]
    return selected, mappings


def _raw_node(residue: Mapping[str, Any]) -> dict[str, Any]:
    return {"chain": residue.get("chain"), "position": residue.get("position"),
            "pdb_chain": residue["pdb_chain"], "pdb_residue_number": residue["pdb_residue_number"],
            "pdb_insertion_code": residue["pdb_insertion_code"],
            "residue_name": residue["residue_name"], "WT": residue["WT"], "protein": residue["protein"]}


def _sasa_arrays(np: Any, atoms: list[dict[str, Any]], points: int, probe: float) -> list[float] | None:
    if any(atom["element"] not in _RADII_A for atom in atoms):
        return None
    coords = np.asarray([a["coord"] for a in atoms], dtype=float)
    radii = np.asarray([_RADII_A[a["element"]] + probe for a in atoms], dtype=float)
    z = 1 - (np.arange(points, dtype=float) + .5) * (2 / points)
    angles = np.arange(points, dtype=float) * (math.pi * (3 - math.sqrt(5)))
    radial = np.sqrt(1 - z * z)
    unit = np.column_stack((np.cos(angles) * radial, np.sin(angles) * radial, z))
    values = []
    for i, (coord, radius) in enumerate(zip(coords, radii)):
        center_dist_sq = np.sum((coords - coord) ** 2, axis=1)
        nearby = np.flatnonzero((center_dist_sq < (radii + radius) ** 2) & (np.arange(len(atoms)) != i))
        exposed = np.ones(points, dtype=bool)
        surface = coord + unit * radius
        for j in nearby:
            exposed &= np.sum((surface - coords[j]) ** 2, axis=1) > radii[j] ** 2
            if not np.any(exposed):
                break
        values.append(float(np.sum(exposed) * 4 * math.pi * radius * radius / points))
    return values


def raw_structure_evidence(case: Mapping[str, Any], pdb_path: str | Path, *,
                           residue_map: Mapping[str, Any] | None = None,
                           heavy_atom_cutoff_A: float = 5.0, CA_cutoff_A: float = 8.0,
                           sasa_n_points: int = 100, probe_radius_A: float = 1.4) -> dict[str, Any]:
    """Recompute public WT geometry, complete observed-atom graphs and SASA.

    Mapping is exact full-chain, removal of explicit unknown-X slots, or unique
    exact contiguous subsequence, with canonical PDB numbering anchors checked.
    ``residue_map`` optionally supplies {chain: {reference_position: {
    'pdb_residue_number': number, 'pdb_insertion_code': code}}}; every WT identity
    is checked. SHA must match a declared source hash when one is present.

    No raw atom is substituted, repacked or scored for mutant fitness. Complete
    graphs cover *observed* selected WT atoms only; missing atoms remain missing.
    """
    state = _prepare(case)
    path = Path(pdb_path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = {source.get("WT_structure", {}).get("pdb_sha256") or
                source.get("structure", {}).get("sha256") for source in state["sources"]}
    expected.discard(None)
    if expected and sha not in expected:
        raise ValueError("Raw PDB SHA256 does not match declared public WT source")
    cutoff = _number(heavy_atom_cutoff_A, "heavy_atom_cutoff_A", 0)
    ca_cutoff = _number(CA_cutoff_A, "CA_cutoff_A", 0)
    probe = _number(probe_radius_A, "probe_radius_A", 0)
    points = _count(sasa_n_points, "sasa_n_points")
    if cutoff is None or cutoff <= 0 or ca_cutoff is None or ca_cutoff <= 0 or probe is None or probe <= 0 or points is None or points < 1:
        raise ValueError("Cutoffs, probe radius and SASA point count must be positive")
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("NumPy is required for raw coordinate calculations; summary-only APIs use standard Python") from exc
    # BindingGym's supplied _hm files use 0.00 for every occupancy field. Treat
    # that file-wide placeholder as modeled-coordinate metadata only when the
    # canonical provenance identifies a source-supplied reference complex.
    atom_lines = [line for line in path.read_text().splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    all_zero_occupancy = bool(atom_lines) and all(float(line[54:60].strip() or "1") == 0 for line in atom_lines)
    source_supplied_complex = any(source.get("structure", {}).get("kind") == "source-supplied multi-chain reference coordinates" for source in state["sources"])
    include_zero = all_zero_occupancy and source_supplied_complex
    residues, mappings = _map_raw_residues(state, _read_pdb(path, include_zero_occupancy=include_zero), residue_map)
    atoms, owner = [], []
    for index, residue in enumerate(residues):
        for atom in residue["atoms"].values():
            if atom["element"] in {"H", "D"}:
                continue
            atoms.append(atom)
            owner.append(index)
    coords = np.asarray([atom["coord"] for atom in atoms], dtype=float)
    edges: dict[tuple[int, int], dict[str, Any]] = {}
    for i, coord in enumerate(coords):
        distance_sq = np.sum((coords[i + 1:] - coord) ** 2, axis=1)
        for offset in np.flatnonzero(distance_sq <= cutoff ** 2):
            j = i + 1 + int(offset)
            if owner[i] == owner[j]:
                continue
            key = tuple(sorted((owner[i], owner[j])))
            distance = math.sqrt(float(distance_sq[offset]))
            edge = edges.setdefault(key, {"minimum_heavy_atom_distance_A": distance,
                                         "observed_atom_pairs_within_cutoff": 0,
                                         "closest_atom_pair": [atoms[i]["name"], atoms[j]["name"]]})
            edge["observed_atom_pairs_within_cutoff"] += 1
            if distance < edge["minimum_heavy_atom_distance_A"]:
                edge["minimum_heavy_atom_distance_A"] = distance
                edge["closest_atom_pair"] = [atoms[i]["name"], atoms[j]["name"]]
    heavy_graph = [{"left": _raw_node(residues[i]), "right": _raw_node(residues[j]),
                    **edge} for (i, j), edge in sorted(edges.items())]
    ca_graph = []
    for i, left in enumerate(residues):
        if not left["protein"] or "CA" not in left["atoms"]:
            continue
        for right in residues[i + 1:]:
            if not right["protein"] or "CA" not in right["atoms"]:
                continue
            distance = math.dist(left["atoms"]["CA"]["coord"], right["atoms"]["CA"]["coord"])
            if distance <= ca_cutoff:
                ca_graph.append({"left": _raw_node(left), "right": _raw_node(right), "CA_distance_A": distance})
    # Retain explicitly supplied hydrogen atoms for ASA, matching the old
    # Shrake-Rupley summary's all-atom definition, while graphs use heavy atoms.
    sasa_atoms, sasa_owner = [], []
    for index, residue in enumerate(residues):
        for atom in residue["atoms"].values():
            sasa_atoms.append(atom)
            sasa_owner.append(index)
    complex_sasa = _sasa_arrays(np, sasa_atoms, points, probe)
    isolated_sasa: dict[int, float | None] = {}
    for raw_chain in {r["pdb_chain"] for r in residues if r["protein"]}:
        indices = [i for i, residue_index in enumerate(sasa_owner)
                   if residues[residue_index]["protein"] and residues[residue_index]["pdb_chain"] == raw_chain]
        values = _sasa_arrays(np, [sasa_atoms[i] for i in indices], points, probe)
        isolated_sasa.update({i: None if values is None else values[k] for k, i in enumerate(indices)})
    atom_indices: dict[int, list[int]] = {}
    for i, residue_index in enumerate(sasa_owner):
        atom_indices.setdefault(residue_index, []).append(i)
    raw_sites, lookup = [], {}
    for chain, reference in state["sequences"].items():
        for position, wt in enumerate(reference, 1):
            matches = [(i, r) for i, r in enumerate(residues) if r.get("chain") == chain and r.get("position") == position]
            site = {"chain": chain, "position": position, "WT": wt, "coordinate_present": bool(matches),
                    "pdb_chain": None, "pdb_residue_number": None, "pdb_insertion_code": None,
                    "CA_coordinate_A": None, "backbone_complete": None, "sidechain_complete": None,
                    "missing_standard_heavy_atoms": None, "selected_altloc": None,
                    "source_asserted_AF2_pLDDT": None,
                    "isolated_chain_observed_atom_SASA_A2": None, "complex_observed_atom_SASA_A2": None,
                    "observed_atom_SASA_buried_by_context_A2": None,
                    "isolated_chain_wt_RSA": None, "complex_wt_RSA": None,
                    "heavy_atom_contacts": None, "cross_chain_heavy_atom_contacts": None}
            if matches:
                index, residue = matches[0]
                present = set(residue["atoms"])
                expected_atoms = set("N CA C O".split()) | set(_SIDECHAIN_ATOMS.get(wt, "").split())
                missing = sorted(expected_atoms - present) if wt in _AA else None
                ids = atom_indices[index]
                iso_values = [isolated_sasa.get(i) for i in ids]
                iso = sum(iso_values) if all(v is not None for v in iso_values) else None
                comp = None if complex_sasa is None else sum(complex_sasa[i] for i in ids)
                reliable_sasa_identity = wt in _AA and not missing
                origin = _af2_assertion(state["sources"], chain)
                contacts = []
                for (a, b), edge in edges.items():
                    if index not in (a, b):
                        continue
                    other = residues[b if a == index else a]
                    contacts.append({"neighbor": _raw_node(other), **edge})
                site.update({"pdb_chain": residue["pdb_chain"], "pdb_residue_number": residue["pdb_residue_number"],
                    "pdb_insertion_code": residue["pdb_insertion_code"],
                    "CA_coordinate_A": residue["atoms"].get("CA", {}).get("coord"),
                    "backbone_complete": set("N CA C O".split()).issubset(present),
                    "sidechain_complete": None if wt not in _AA else set(_SIDECHAIN_ATOMS[wt].split()).issubset(present),
                    "missing_standard_heavy_atoms": missing, "selected_altloc": residue["selected_altloc"],
                    "source_asserted_AF2_pLDDT": _number(residue["atoms"]["CA"]["bfactor"], "raw pLDDT", 0, 100) if origin and "CA" in present else None,
                    "isolated_chain_observed_atom_SASA_A2": iso, "complex_observed_atom_SASA_A2": comp,
                    "observed_atom_SASA_buried_by_context_A2": None if iso is None or comp is None else iso - comp,
                    "isolated_chain_wt_RSA": iso / MAXASA_A2[wt] if iso is not None and reliable_sasa_identity else None,
                    "complex_wt_RSA": comp / MAXASA_A2[wt] if comp is not None and reliable_sasa_identity else None,
                    "heavy_atom_contacts": contacts,
                    "cross_chain_heavy_atom_contacts": [c for c in contacts if c["neighbor"]["pdb_chain"] != residue["pdb_chain"]]})
            raw_sites.append(site)
            lookup[(chain, position)] = site
    variants = []
    for query in case.get("query", []):
        changes = _parse_changes(state, query)
        pairs = []
        for left, right in itertools.combinations(changes, 2):
            a, b = lookup[(left["chain"], left["position"])], lookup[(right["chain"], right["position"])]
            distance = None if a["CA_coordinate_A"] is None or b["CA_coordinate_A"] is None else math.dist(a["CA_coordinate_A"], b["CA_coordinate_A"])
            pairs.append({"left": left, "right": right, "wt_CA_distance_A": distance,
                          "within_CA_cutoff": None if distance is None else distance <= ca_cutoff,
                          "status": "observed_raw_WT_CA_distance" if distance is not None else "unknown_missing_CA_coordinate"})
        variants.append({"id": query.get("id"), "mutations": changes, "mutated_site_pairs": pairs})
    return {
        "format": "protein_raw_WT_structure_evidence_v1", "case_id": case.get("case_id"),
        "reference_sequence_sha256": hashlib.sha256(case["reference_sequence"].encode()).hexdigest(),
        "raw_source": {"path": str(path.resolve()), "sha256": sha,
                       "matches_declared_source_sha256": None if not expected else True,
                       "model_selection": "first", "altloc_policy": "residue conformer with greatest summed occupancy, ties prefer A then lexical; shared blank-altloc atoms retained",
                       "zero_occupancy_atoms_included": include_zero,
                       "occupancy_interpretation": "file-wide zero placeholder in source-supplied modeled complex" if include_zero else "zero-occupancy atoms omitted",
                       "hydrogen_atoms_in_contact_graph": False, "explicit_hydrogen_atoms_in_SASA": True,
                       "waters_included": False},
        "mapping": mappings, "sites": raw_sites, "candidates": variants,
        "heavy_atom_residue_contact_graph": heavy_graph, "CA_residue_contact_graph": ca_graph,
        "graph_scope": {"complete_for_selected_observed_atoms_at_cutoff": True,
                        "all_physical_atoms_present": False, "sequence_local_pairs_included": True,
                        "heavy_atom_cutoff_A": cutoff, "CA_cutoff_A": ca_cutoff,
                        "heavy_atom_count": len(atoms), "residue_count_with_coordinates": len(residues),
                        "graph_direction": "undirected; each residue pair once"},
        "sasa_provenance": {"algorithm": "Shrake-Rupley, deterministic golden-spiral sphere sampling",
            "probe_radius_A": probe, "n_points_per_atom": points, "radii_A": _RADII_A,
            "radii_reference": "https://raw.githubusercontent.com/biopython/biopython/master/Bio/PDB/SASA.py",
            "rsa_normalization": {**RSA_PROVENANCE,
                "formula": "recomputed WT SASA / MAXASA_A2[WT]; separate isolated-chain and complex scopes",
                "raw_coordinate_recomputed": True},
            "raw_coordinate_recomputed": True, "independently_reproduced_with_external_library": False,
            "isolated_scope": "all explicitly supplied selected protein atoms of one physical PDB chain, including supplied hydrogens; no other chains or nonprotein context",
            "complex_scope": "all selected protein plus nonwater nonprotein atoms, including explicitly supplied hydrogens",
            "missing_heavy_atoms_policy": "observed-atom ASA remains reported; standard WT RSA null if standard heavy atoms are missing or WT is X",
            "unknown_element_policy": "SASA null for affected calculation; no invented radii"},
        "limitations": ["Static WT coordinates; no mutant repacking, clash, energy or epistasis calculation.",
            "Cutoff graphs are complete over observed selected atoms only; missing residues/atoms are not imputed.",
            "Raw complexes are source-supplied; biological assembly validity is not independently established.",
            "No PAE is supplied; per-residue pLDDT does not establish confidence in long-range placement.",
            "SASA uses a finite sampling grid; normalized RSA is retained above 1 and is not calibrated fitness."]}


# Stable name for the native01 preparation interface.
raw_coordinate_evidence = raw_structure_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, type=Path, help="Public canonical protein case.json")
    parser.add_argument("--out", type=Path, help="Write JSON; default stdout")
    parser.add_argument("--candidate-id", action="append", help="Limit to IDs; may be repeated")
    parser.add_argument("--rsa-only", action="store_true", help="Return site-wise derived WT RSA only")
    parser.add_argument("--raw-pdb", type=Path, help="Separate raw WT structure augmentation mode (requires NumPy)")
    parser.add_argument("--heavy-atom-cutoff-A", type=float, default=5.0)
    parser.add_argument("--sasa-n-points", type=int, default=100)
    parser.add_argument("--confidence-threshold", type=float, default=70.0)
    parser.add_argument("--pair-cutoff-A", type=float, default=8.0)
    args = parser.parse_args(argv)
    try:
        case = json.loads(args.case.read_text())
        if args.rsa_only and (args.candidate_id or args.raw_pdb):
            raise ValueError("--rsa-only cannot be combined with candidate selection or raw mode")
        if args.raw_pdb:
            if args.candidate_id:
                raise ValueError("Raw mode exports full WT evidence; --candidate-id is summary-only")
            packet = raw_structure_evidence(case, args.raw_pdb, heavy_atom_cutoff_A=args.heavy_atom_cutoff_A,
                CA_cutoff_A=args.pair_cutoff_A, sasa_n_points=args.sasa_n_points)
        elif args.rsa_only:
            packet = rsa_sites(case)
        else:
            candidates = None
            if args.candidate_id:
                requested = set(args.candidate_id)
                candidates = [query for query in case.get("query", []) if query.get("id") in requested]
                absent = requested.difference(query["id"] for query in candidates)
                if absent:
                    raise ValueError("Unknown candidate IDs: " + ", ".join(sorted(absent)))
            packet = describe_case(case, candidates, confidence_threshold=args.confidence_threshold,
                                   pair_cutoff_A=args.pair_cutoff_A)
        output = json.dumps(packet, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
