"""Compact reproducible evidence audit, without outcome labels."""
import json
import math
from pathlib import Path
import numpy as np
import structure_tools
from predict import load_and_validate


def dihedral(points):
    p0, p1, p2, p3 = np.asarray(points, dtype=float)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - b0.dot(b1) * b1
    w = b2 - b2.dot(b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v).dot(w), v.dot(w))))


def coordinates(case):
    residues = structure_tools._read_pdb(Path('raw_evidence/6VJJ.pdb'), include_zero_occupancy=True)
    selected, mappings = structure_tools._map_raw_residues(structure_tools._prepare(case), residues, None)
    return {(r['chain'], r['position']): r for r in selected if 'position' in r}, mappings


def torsions(index, chain, position):
    def at(p, atom):
        return index[(chain, p)]['atoms'][atom]['coord']
    try:
        phi = dihedral([at(position - 1, 'C'), at(position, 'N'),
                        at(position, 'CA'), at(position, 'C')])
    except KeyError:
        phi = None
    try:
        psi = dihedral([at(position, 'N'), at(position, 'CA'),
                        at(position, 'C'), at(position + 1, 'N')])
    except KeyError:
        psi = None
    return phi, psi


def main():
    case, manifest, msa, structure = load_and_validate()
    index, mappings = coordinates(case)
    assert mappings == structure['raw_coordinate_evidence']['mapping']
    raw = {(r['chain'],r['position']): r for r in structure['raw_coordinate_evidence']['sites']}
    report = {'mapping': mappings, 'MSA_complete_candidates': sum(
        c['scores']['sum_log_frequency_ratio']['complete'] for c in msa['candidates']),
        'sites': []}
    for (chain, pos), site in raw.items():
        if chain != 'A':
            continue
        assert site['WT'] == case['reference_chains'][chain][pos - 1]
        assert np.allclose(site['CA_coordinate_A'], index[(chain, pos)]['atoms']['CA']['coord'])
        phi, psi = torsions(index, chain, pos)
        report['sites'].append({'chain': chain, 'position': pos, 'WT': site['WT'],
                                'phi': phi, 'psi': psi,
                                'rsa': site['isolated_chain_wt_RSA'],
                                'raf_burial_A2': site['observed_atom_SASA_buried_by_context_A2']})
    Path('evidence_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Mapping and CA coordinates reproduced; MSA complete:',report['MSA_complete_candidates'])
    mutations = {(t['position'], t['ALT']): t['WT'] for c in msa['candidates'] for t in c['substitution_terms']}
    print('Pro/Gly/positive-phi substitutions (WT geometry, no relaxed mutant prediction):')
    for (p,a),w in sorted(mutations.items()):
        phi, psi = torsions(index,'A',p)
        if a == 'P' or w == 'P' or w == 'G' and phi > 0:
            print(f'{w}{p}{a}: phi={phi:.1f}, psi={psi:.1f}')
    print('Interface closest charged functional atoms:')
    terminal = {'D': ['OD1','OD2'], 'E': ['OE1','OE2'], 'R': ['NE','NH1','NH2'], 'K': ['NZ']}
    for (ch,p),r in index.items():
        if ch != 'A' or r['WT'] not in terminal:
            continue
        best=[]
        for (pch,pp),pr in index.items():
            if pch != 'B' or pr['WT'] not in terminal:
                continue
            for an in terminal[r['WT']]:
                for pn in terminal[pr['WT']]:
                    d=np.linalg.norm(np.array(r['atoms'][an]['coord'])-pr['atoms'][pn]['coord'])
                    if d < 4.5: best.append((round(float(d),2),an,pr['WT']+str(pp),pn))
        if best: print(r['WT']+str(p), sorted(best))


if __name__ == '__main__':
    main()
