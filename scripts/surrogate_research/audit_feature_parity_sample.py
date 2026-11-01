"""Audit sampled search molecules against frozen features, with prediction impact.

No labels are read; original frozen inputs and candidate artifacts remain untouched.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.molecular_spec import MolecularFeatureEncoder, MolecularFeatureSpec
from tackai.fusion.research import sha256
from tackai.fusion.xgb_ensemble import XGBStackedEnsemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20261101)
    args = parser.parse_args()
    if args.count < 1 or args.output.exists():
        raise ValueError('Need positive count and a fresh output path')
    snapshot = args.snapshot
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    candidate = CandidateArtifact.from_pretrained(args.candidate)
    if (candidate.recipe.get('task') != 'pdc50'
            or candidate.recipe.get('fit_stage') != 'pre_calibration'
            or candidate.context_contract.get('snapshot_manifest_sha256') != sha256(snapshot / 'manifest.json')):
        raise ValueError('Expected matching trusted search-only pDC50 bundle')
    feature = candidate.recipe['feature']
    for name in ('rows.csv', 'partition.npy', 'baseline.npy', f'{feature}.npy'):
        if sha256(snapshot / name) != manifest['artifact_hashes'][name]:
            raise ValueError(f'Snapshot hash mismatch: {name}')
    roles = np.load(snapshot / 'partition.npy', allow_pickle=False)
    table = pd.read_csv(snapshot / 'rows.csv', usecols=['smiles', 'canonical_smiles'])
    if len(table) != len(roles):
        raise ValueError('Input rows do not match partitions')
    search = table.loc[roles == 'search'].drop_duplicates('canonical_smiles', keep='first')
    if len(search) < args.count:
        raise ValueError('Not enough unique search molecules')
    rng = np.random.default_rng(args.seed)
    sample = search.iloc[np.sort(rng.choice(len(search), size=args.count, replace=False))]
    sample_rows = sample.index.to_numpy()
    smiles = sample['smiles'].tolist()
    ref_meta = manifest['features']['baseline']
    ref_spec = MolecularFeatureSpec(radius=ref_meta['radius'], size=ref_meta['fp_size'],
                                    count=ref_meta['counts'], descriptor_names=tuple(ref_meta['descriptors']))
    reference = XGBStackedEnsemble.from_pretrained(args.reference)
    for model in [*reference.models, *candidate.models]:
        if hasattr(model, 'set_params') and 'n_jobs' in model.get_params():
            model.set_params(n_jobs=1)
    result = {}
    for name, predictor, spec, feature_name in (
        ('reference', reference, ref_spec, 'baseline'),
        ('candidate', candidate, candidate.molecular_spec, feature),
    ):
        molecular = MolecularFeatureEncoder(spec, use_cache=False, n_workers=1)
        fp, desc, ok = molecular.featurize(smiles)
        molecular.close()
        if not ok.all():
            raise ValueError(f'Invalid search molecule for {name}')
        live = np.concatenate([fp, desc], axis=1)
        X = np.load(snapshot / f'{feature_name}.npy', mmap_mode='r', allow_pickle=False)
        frozen = np.asarray(X[sample_rows], dtype=np.float32)
        frozen_mol = frozen[:, :live.shape[1]]
        mismatches = ~np.isclose(live, frozen_mol, rtol=1e-6, atol=1e-6)
        mismatch_rows = np.flatnonzero(mismatches.any(axis=1))
        live_X = frozen.copy()
        live_X[:, :live.shape[1]] = live
        frozen_pred = predictor.predict(frozen)['mean']
        live_pred = predictor.predict(live_X)['mean']
        delta = np.asarray(live_pred - frozen_pred, dtype=float)
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite prediction difference')
        cols = np.flatnonzero(mismatches.any(axis=0))
        summary = []
        for col in cols:
            label = 'fingerprint' if col < fp.shape[1] else molecular.descriptors[col - fp.shape[1]]
            summary.append({'column': int(col), 'name': label, 'rows': int(mismatches[:, col].sum()),
                            'max_absolute_difference': float(np.max(np.abs(live[:, col] - frozen_mol[:, col])))})
        result[name] = {'feature_mismatch_molecules': int(len(mismatch_rows)), 'columns': summary,
                        'feature_mismatch_row_ids': sample_rows[mismatch_rows].tolist(),
                        'prediction_changed_molecules': int(np.count_nonzero(np.abs(delta) > 1e-8)),
                        'max_absolute_prediction_delta': float(np.max(np.abs(delta))),
                        'prediction_change_examples': [{'row': int(sample_rows[i]), 'delta': float(delta[i])}
                            for i in np.flatnonzero(np.abs(delta) > 1e-8)[:20]]}
    output = {'format': 'tack-search-feature-parity-sample/v1', 'sample_count': args.count,
              'selection': 'uniform random over unique canonical search molecules, first occurrence',
              'population_count': len(search), 'seed': args.seed, 'sample_row_ids': sample_rows.tolist(),
              'snapshot_manifest_sha256': sha256(snapshot / 'manifest.json'),
              'candidate_model_sha256': sha256(args.candidate / 'candidate.joblib'),
              'reference_files_sha256': {p.name: sha256(p) for p in sorted(args.reference.iterdir())
                                         if p.suffix in {'.json', '.ubj'}},
              'script_sha256': sha256(__file__), 'rtol': 1e-6, 'atol': 1e-6, 'results': result,
              'note': 'Search-only sample, no labels; candidate is not final, reference training context unverified.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in result}, indent=2))


if __name__ == '__main__':
    main()
