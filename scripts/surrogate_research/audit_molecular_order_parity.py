"""Audit original versus canonical-SMILES molecular features for a search batch."""
import csv
import json
from pathlib import Path

import numpy as np
from tackai.fusion.mol_encoder import MolEncoder
from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.xgb_ensemble import XGBStackedEnsemble
from tackai.fusion.molecular_spec import MolecularFeatureEncoder

snapshot = Path('models/research_v1')
roles = np.load(snapshot / 'partition.npy', allow_pickle=False)
rows, seen = [], set()
with (snapshot / 'rows.csv').open(newline='') as stream:
    for i, row in enumerate(csv.DictReader(stream)):
        if roles[i] == 'search' and row['canonical_smiles'] not in seen:
            rows.append((i, row['smiles'], row['canonical_smiles']))
            seen.add(row['canonical_smiles'])
            if len(rows) == 128:
                break
molecular = MolEncoder(use_cache=False, n_workers=1)
fp, descriptors, ok = molecular.featurize([s for _, original, canonical in rows for s in (original, canonical)])
assert ok.all()
feature_diffs = []
for j, (row_id, original, canonical) in enumerate(rows):
    offset = 2 * j
    cols = np.flatnonzero(descriptors[offset] != descriptors[offset + 1])
    if len(cols) or not np.array_equal(fp[offset], fp[offset + 1]):
        feature_diffs.append({'row': row_id, 'input_differs': original != canonical,
                              'fingerprint_diff': int(np.count_nonzero(fp[offset] != fp[offset + 1])),
                              'descriptors': [{'name': molecular.descriptors[k], 'original': float(descriptors[offset,k]),
                                               'canonical': float(descriptors[offset+1,k])} for k in cols]})
X = np.load(snapshot / 'baseline.npy', mmap_mode='r', allow_pickle=False)
original_matrix = np.concatenate([fp[::2], descriptors[::2]], axis=1)
close = np.isclose(original_matrix, X[[i for i, _, _ in rows], :original_matrix.shape[1]], rtol=1e-6, atol=1e-6)
mismatches = [{'row': rows[i][0], 'column': int(k),
               'name': molecular.descriptors[k - fp.shape[1]] if k >= fp.shape[1] else 'fingerprint',
               'live': float(original_matrix[i,k]), 'snapshot': float(X[rows[i][0],k])}
              for i, k in np.argwhere(~close)]
if mismatches:
    baseline = XGBStackedEnsemble.from_pretrained('models/ensembles/autotpd4_pdc50_refit')
    artifact = CandidateArtifact.from_pretrained('models/research_precal/pdc50_r2_v2')
    candidate_matrix = np.load(snapshot / 'morgan_r2_2048_bit.npy', mmap_mode='r', allow_pickle=False)
    for record in mismatches:
        row_id = record['row']
        j = next(j for j, row in enumerate(rows) if row[0] == row_id)
        original_smiles = rows[j][1]
        candidate_encoder = MolecularFeatureEncoder(artifact.molecular_spec, use_cache=False, n_workers=1)
        cfp, cdesc, cok = candidate_encoder.featurize([original_smiles])
        assert cok.all()
        baseline_frozen = np.array(X[row_id:row_id + 1])
        baseline_live = baseline_frozen.copy()
        baseline_live[:, :original_matrix.shape[1]] = original_matrix[j:j + 1]
        candidate_frozen = np.array(candidate_matrix[row_id:row_id + 1])
        candidate_live = candidate_frozen.copy()
        candidate_live[:, :cfp.shape[1] + cdesc.shape[1]] = np.concatenate([cfp, cdesc], axis=1)
        record['reference_frozen_mean'] = float(baseline.predict(baseline_frozen)['mean'][0])
        record['reference_live_mean'] = float(baseline.predict(baseline_live)['mean'][0])
        record['candidate_frozen_mean'] = float(artifact.predict(candidate_frozen)['mean'][0])
        record['candidate_live_mean'] = float(artifact.predict(candidate_live)['mean'][0])
print(json.dumps({'batch':len(rows), 'source_vs_canonical_differences':feature_diffs,
                  'source_vs_snapshot_mismatches':mismatches}, indent=2))
