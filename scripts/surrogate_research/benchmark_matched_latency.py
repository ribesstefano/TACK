"""Measure original and research pDC50 ensembles on identical search molecules and context."""
import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.context import ContextEncoder
from tackai.fusion.data import FusionData
from tackai.fusion.molecular_spec import MolecularFeatureEncoder, MolecularFeatureSpec
from tackai.fusion.research import sha256
from tackai.fusion.xgb_ensemble import XGBStackedEnsemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.batch < 1 or args.repeats < 1:
        raise ValueError('Need fresh output, positive batch and repeats')
    manifest_path = args.snapshot / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    artifact = CandidateArtifact.from_pretrained(args.candidate)
    if artifact.task != 'regression' or artifact.recipe.get('task') != 'pdc50':
        raise ValueError('Need a trusted pDC50 candidate')
    if artifact.context_contract.get('snapshot_manifest_sha256') != sha256(manifest_path):
        raise ValueError('Candidate and snapshot differ')
    candidate_name = artifact.recipe['feature']
    for name in ('rows.csv', 'partition.npy', 'baseline.npy', f'{candidate_name}.npy'):
        if sha256(args.snapshot / name) != manifest['artifact_hashes'][name]:
            raise ValueError(f'Snapshot hash mismatch: {name}')
    roles = np.load(args.snapshot / 'partition.npy', allow_pickle=False)
    rows, smiles, seen = [], [], set()
    with (args.snapshot / 'rows.csv').open(newline='') as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            value = row['canonical_smiles']
            if roles[i] == 'search' and value not in seen:
                rows.append(i)
                smiles.append(value)
                seen.add(value)
                if len(rows) == args.batch:
                    break
    if len(smiles) != args.batch:
        raise ValueError('Insufficient distinct search molecules')
    encoder = ContextEncoder(cache_dir=args.snapshot / 'context', protein_space='combined')
    if encoder.fingerprint() != artifact.context_contract.get('context_fingerprint'):
        raise ValueError('Context fingerprint mismatch')
    reference_metadata = manifest['features']['baseline']
    spec = MolecularFeatureSpec(radius=reference_metadata['radius'], size=reference_metadata['fp_size'],
                                count=reference_metadata['counts'],
                                descriptor_names=tuple(reference_metadata['descriptors']))
    start = time.perf_counter()
    reference = XGBStackedEnsemble.from_pretrained(args.reference)
    reference_load = time.perf_counter() - start
    if reference.task != 'regression':
        raise ValueError('Reference must be regression')
    for model in [*reference.models, *artifact.models]:
        if hasattr(model, 'set_params') and 'n_jobs' in model.get_params():
            model.set_params(n_jobs=1)
    pipelines = {}
    for name, predictor, feature, molecular_spec in (
            ('reference', reference, 'baseline', spec),
            ('candidate', artifact, candidate_name, artifact.molecular_spec)):
        molecular = MolecularFeatureEncoder(molecular_spec, use_cache=True, n_workers=1)
        data = FusionData(encoder=encoder, featurizer=molecular)
        if data.dims != manifest['features'][feature]['dims']:
            raise ValueError(f'{name} feature dimensions do not match snapshot')
        X = np.load(args.snapshot / f'{feature}.npy', mmap_mode='r', allow_pickle=False)
        width = data.dims['fingerprint'] + data.dims['descriptors']
        if any(getattr(m, 'n_features_in_', X.shape[1]) != X.shape[1] for m in predictor.models):
            raise ValueError(f'{name} model dimensions do not match')
        pipelines[name] = {'predictor': predictor, 'molecular': molecular, 'data': data,
                           'X': X, 'width': width, 'context': np.asarray(X[rows[0], width:]).reshape(1, -1)}
    if not np.array_equal(pipelines['reference']['context'], pipelines['candidate']['context']):
        raise ValueError('Snapshot contexts differ between molecular feature matrices')

    def score(name, check_parity=False):
        item = pipelines[name]
        start = time.perf_counter()
        fp, desc, valid = item['molecular'].featurize(smiles)
        after_encoding = time.perf_counter()
        matrix = item['data'].assemble_features(item['context'], fp, desc)
        result = item['predictor'].predict(matrix)
        end = time.perf_counter()
        if not valid.all() or len(result['mean']) != args.batch or any(not np.isfinite(v).all() for v in result.values()):
            raise ValueError(f'{name} returned invalid output')
        if check_parity:
            np.testing.assert_allclose(matrix[:, :item['width']], item['X'][rows, :item['width']], rtol=1e-6, atol=1e-6)
        return {'encode_ms': (after_encoding - start) * 1000, 'predict_ms': (end - after_encoding) * 1000,
                'total_ms': (end - start) * 1000}

    cold = {name: score(name, check_parity=True) for name in ('reference', 'candidate')}
    warm = {name: [] for name in cold}
    for i in range(args.repeats):
        for name in (('reference', 'candidate') if i % 2 == 0 else ('candidate', 'reference')):
            warm[name].append(score(name))
    for item in pipelines.values():
        item['molecular'].close()
    def summarize(samples):
        return {key: {'median': float(np.median([x[key] for x in samples])),
                      'p95': float(np.quantile([x[key] for x in samples], .95))} for key in samples[0]}
    files = sorted(p for p in args.reference.iterdir() if p.suffix in {'.json', '.ubj'} and not p.name.endswith('.params.json'))
    output = {'snapshot_sha256': sha256(manifest_path), 'context_fingerprint': encoder.fingerprint(),
              'candidate': str(args.candidate), 'candidate_sha256': sha256(args.candidate / 'candidate.joblib'),
              'reference': str(args.reference), 'reference_files_sha256': {p.name: sha256(p) for p in files},
              'reference_load_seconds': reference_load, 'batch': args.batch, 'repeats': args.repeats,
              'first_search_row': rows[0], 'selected_rows': rows, 'cold': cold,
              'warm': {name: summarize(samples) for name, samples in warm.items()},
              'max_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'note': ('Local one-process CPU microbenchmark; candidate is pre-calibration and lacks final intervals. '
                       'Reference context training fingerprint is not recorded; matching snapshot input contract and '
                       'model widths alone do not prove historical training-context parity. Excludes HTTP and RL.')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
