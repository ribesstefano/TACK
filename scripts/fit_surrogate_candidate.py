"""Fit a manually selected candidate from frozen search/stacking/calibration groups.

The JSON recipe pins every member and parameter. Base members train only on
search groups, with shared group folds independent of endpoint labels. Stacking
labels are read only for optional convex weights and regression residual scale;
calibration labels are read only for conformal or temperature calibration.
Confirmation rows are never used for fitting or evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from tackai.fusion.candidate_artifact import CandidateArtifact
from tackai.fusion.data import make_splits
from tackai.fusion.molecular_spec import MolecularFeatureSpec
from tackai.fusion.research import read_npz_rows as _read_npz_rows, sha256

try:
    from .benchmark_surrogates import fit_learner
except ImportError:
    from benchmark_surrogates import fit_learner

RECIPE_SCHEMA_VERSION = 1
FAMILIES = {"xgb", "current_xgb", "extratrees", "catboost"}
ROLES = ("search", "stacking", "calibration", "confirmation")


def _require_int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_recipe(recipe):
    """Validate an explicit, fully pinned candidate recipe without tuning it."""
    expected = {"schema_version", "snapshot_hash", "task", "feature", "label_policy", "cv",
                "members", "stacking", "calibration"}
    if not isinstance(recipe, dict) or set(recipe) != expected:
        raise ValueError(f"recipe fields must be exactly {sorted(expected)}")
    if recipe["schema_version"] != RECIPE_SCHEMA_VERSION:
        raise ValueError("unsupported recipe schema version")
    snapshot_hash = recipe["snapshot_hash"]
    if (not isinstance(snapshot_hash, str) or len(snapshot_hash) != 64
            or any(char not in "0123456789abcdef" for char in snapshot_hash)):
        raise ValueError("snapshot_hash must be a lowercase SHA256 digest")
    if recipe["task"] not in {"dmax", "pdc50", "activity"}:
        raise ValueError("task must be dmax, pdc50, or activity")
    if recipe["label_policy"] not in {"exact", "compat"}:
        raise ValueError("label_policy must be exact or compat")
    if not isinstance(recipe["feature"], str) or not recipe["feature"]:
        raise ValueError("feature must name a frozen snapshot matrix")

    cv = recipe["cv"]
    if not isinstance(cv, dict) or set(cv) != {"repeats", "folds", "seeds"}:
        raise ValueError("cv must contain exactly repeats, folds, and seeds")
    repeats = _require_int(cv["repeats"], "cv.repeats", 1)
    folds = _require_int(cv["folds"], "cv.folds", 2)
    if not isinstance(cv["seeds"], list) or len(cv["seeds"]) != repeats:
        raise ValueError("cv.seeds must contain exactly one seed per repeat")
    for i, seed in enumerate(cv["seeds"]):
        _require_int(seed, f"cv.seeds[{i}]")

    members = recipe["members"]
    if not isinstance(members, list) or len(members) != repeats * folds:
        raise ValueError("members must specify exactly one model for every repeat/fold")
    coordinates = set()
    for i, member in enumerate(members):
        if not isinstance(member, dict) or set(member) != {
                "family", "repeat", "fold", "seed", "params"}:
            raise ValueError(f"members[{i}] must specify family/repeat/fold/seed/params")
        if member["family"] not in FAMILIES:
            raise ValueError(f"members[{i}].family must be one of {sorted(FAMILIES)}")
        repeat = _require_int(member["repeat"], f"members[{i}].repeat")
        fold = _require_int(member["fold"], f"members[{i}].fold")
        _require_int(member["seed"], f"members[{i}].seed")
        if repeat >= repeats or fold >= folds:
            raise ValueError(f"members[{i}] references a fold outside cv")
        if (repeat, fold) in coordinates:
            raise ValueError("each shared repeat/fold must have exactly one member")
        coordinates.add((repeat, fold))
        if not isinstance(member["params"], dict):
            raise ValueError(f"members[{i}].params must be an object")
        reserved = {"n_jobs", "thread_count", "random_state", "random_seed"} & set(member["params"])
        if reserved:
            raise ValueError(f"members[{i}].params overrides controlled resources or seed: {sorted(reserved)}")
        _validate_nonnegative_params(member["params"], f"members[{i}].params")
    if coordinates != {(r, f) for r in range(repeats) for f in range(folds)}:
        raise ValueError("recipe must cover every shared repeat/fold")

    stacking = recipe["stacking"]
    if (not isinstance(stacking, dict)
            or set(stacking) != {"weights", "l2"}
            or stacking["weights"] not in {"uniform", "fit"}):
        raise ValueError("stacking must specify weights='uniform'|'fit' and l2")
    l2 = stacking["l2"]
    if isinstance(l2, bool) or not isinstance(l2, (int, float)) or not np.isfinite(l2) or l2 < 0:
        raise ValueError("stacking.l2 must be a finite nonnegative number")
    calibration = recipe["calibration"]
    if not isinstance(calibration, dict) or set(calibration) not in ({"alpha"}, {"alpha", "unit"}):
        raise ValueError("calibration must contain alpha and optional unit")
    alpha = calibration["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("calibration.alpha must lie strictly between 0 and 1")
    unit = calibration.get("unit", "row")
    if unit not in {"row", "group_max"} or (recipe["task"] == "activity" and unit != "row"):
        raise ValueError("group_max calibration is available for regression tasks only")
    return recipe


def _validate_nonnegative_params(params, name):
    """Reject negative numerical hyperparameters before estimator construction."""
    for key, value in params.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if not np.isfinite(value):
                raise ValueError(f"{name}.{key} must be finite")
            if value < 0 and not (key in {"n_jobs", "thread_count"} and value == -1):
                raise ValueError(f"{name}.{key} must not be negative")


def read_npz_rows(path, key, indices):
    """Compatibility wrapper for the shared role-limited NPZ reader."""
    return _read_npz_rows(path, key, indices)


def _estimator(member, task, jobs=1):
    """Construct the pinned benchmark-family estimator with explicit defaults."""
    classification = task == "activity"
    family = member["family"]
    seed = member["seed"]
    if family == "extratrees":
        cls = ExtraTreesClassifier if classification else ExtraTreesRegressor
        model = cls(n_estimators=300, min_samples_leaf=3, max_features=.7,
                    n_jobs=jobs, random_state=seed)
    elif family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if classification else CatBoostRegressor
        model = cls(iterations=500, depth=6, learning_rate=.05, thread_count=jobs,
                    random_seed=seed, verbose=False, allow_writing_files=False)
    else:
        import xgboost as xgb
        cls = xgb.XGBClassifier if classification else xgb.XGBRegressor
        options = dict(n_estimators=400, max_depth=4, learning_rate=.05,
                       subsample=.85, colsample_bytree=.8, min_child_weight=5.,
                       reg_lambda=10., tree_method="hist", n_jobs=jobs,
                       random_state=seed)
        if classification:
            options.update(objective="binary:logistic", eval_metric="logloss")
        else:
            options["objective"] = "reg:squarederror"
        model = cls(**options)
    # CatBoost get_params() only returns parameters already set, not every
    # accepted constructor option such as l2_leaf_reg from the tuned search.
    accepted = set(model.get_params(deep=True)) | set(inspect.signature(type(model).__init__).parameters)
    parameters = {key: value for key, value in member["params"].items()
                  if key != "alpha_w" or family != "current_xgb"}
    unknown = set(parameters) - accepted
    if unknown:
        raise ValueError(f"unknown {family} parameter(s): {sorted(unknown)}")
    model.set_params(**parameters)
    return model


def _fit_convex_weights(predictions, y, task, l2):
    """Fit a regularized simplex pool using stacking rows only."""
    n_models = predictions.shape[1]
    uniform = np.full(n_models, 1 / n_models)

    def objective(weights):
        pooled = predictions @ weights
        if task == "activity":
            pooled = np.clip(pooled, 1e-7, 1 - 1e-7)
            loss = -np.mean(y * np.log(pooled) + (1 - y) * np.log1p(-pooled))
        else:
            loss = np.mean((y - pooled) ** 2)
        return float(loss + l2 * np.sum((weights - uniform) ** 2))

    result = minimize(objective, uniform, method="SLSQP",
                      bounds=[(0., 1.)] * n_models,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                      options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"stacking weight fit failed: {result.message}")
    weights = np.maximum(result.x, 0.)
    return weights / weights.sum()


def _task_targets(labels_path, policy, task, role_indices):
    """Read only task-label offsets for the explicitly authorized row indices."""
    label_key = f"{policy}_{task}"
    result = {}
    for role, indices in role_indices.items():
        result[role] = read_npz_rows(labels_path, label_key, indices)
    return result


def _load_frozen_inputs(snapshot, recipe):
    """Verify frozen file hashes, feature dimensions and task-independent roles."""
    snapshot = Path(snapshot)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "tack-surrogate-research/v1":
        raise ValueError("unsupported research snapshot format")
    feature_name = recipe["feature"]
    if feature_name not in manifest.get("features", {}):
        raise ValueError(f"feature {feature_name!r} is absent from the frozen snapshot")
    feature_path = snapshot / f"{feature_name}.npy"
    labels_path = snapshot / "labels.npz"
    groups_path = snapshot / "groups.npy"
    roles_path = snapshot / "partition.npy"
    required = [feature_path, labels_path, groups_path, roles_path]
    for path in required:
        expected_hash = manifest["artifact_hashes"].get(path.name)
        if not expected_hash or sha256(path) != expected_hash:
            raise ValueError(f"frozen snapshot hash mismatch: {path.name}")

    X = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    groups = np.load(groups_path, allow_pickle=False)
    roles = np.load(roles_path, allow_pickle=False)
    if X.ndim != 2 or len(groups) != X.shape[0] or len(roles) != X.shape[0]:
        raise ValueError("snapshot features, groups and partitions are not aligned")
    if not np.all(np.isfinite(X)):
        raise ValueError("snapshot feature matrix contains non-finite values")
    if not set(np.unique(roles)).issubset(set(ROLES)):
        raise ValueError("snapshot contains an unknown partition role")
    for group in np.unique(groups):
        group_roles = set(roles[groups == group])
        if len(group_roles) != 1:
            raise ValueError(f"group {group!r} overlaps frozen partitions")
    dimensions = manifest["features"][feature_name]["dims"]
    if X.shape[1] != sum(dimensions.values()):
        raise ValueError("feature matrix width differs from snapshot dimension manifest")
    feature_meta = manifest["features"][feature_name]
    from rdkit import rdBase
    if manifest["versions"]["rdkit"] != importlib.metadata.version("rdkit"):
        raise ValueError("RDKit distribution differs from the frozen feature-generation version")
    molecular_spec = MolecularFeatureSpec(
        radius=feature_meta["radius"], size=feature_meta["fp_size"],
        count=feature_meta["counts"], descriptor_names=tuple(feature_meta["descriptors"]),
        rdkit_version=rdBase.rdkitVersion)
    return snapshot, manifest, X, groups, roles, labels_path, molecular_spec


def _fit_one(snapshot, recipe, output, *, pre_calibration=False):
    recipe = validate_recipe(recipe)
    if sha256(Path(snapshot) / "manifest.json") != recipe["snapshot_hash"]:
        raise ValueError("Recipe was frozen for a different snapshot")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    (snapshot, snapshot_manifest, X, groups, roles, labels_path,
     molecular_spec) = _load_frozen_inputs(snapshot, recipe)

    role_indices = {role: np.flatnonzero(roles == role) for role in ROLES}
    role_groups = {role: set(groups[idx].tolist()) for role, idx in role_indices.items()}
    for i, first in enumerate(ROLES):
        for second in ROLES[i + 1:]:
            if role_groups[first] & role_groups[second]:
                raise ValueError(f"molecular groups overlap {first} and {second}")
    role_counts = {role: {"rows": int(len(role_indices[role])),
                          "groups": int(len(role_groups[role]))} for role in ROLES}
    search_idx = role_indices["search"]
    if len(search_idx) < recipe["cv"]["folds"]:
        raise ValueError("search partition is too small for requested shared folds")

    # The shared fold assignment is independent of task labels. Apply label masks
    # only after every group's search fold has been determined.
    folds = make_splits(np.zeros(len(search_idx)), groups[search_idx],
                        recipe["cv"]["repeats"], recipe["cv"]["folds"],
                        seeds=recipe["cv"]["seeds"])
    y_search = _task_targets(labels_path, recipe["label_policy"], recipe["task"],
                             {"search": search_idx})["search"]
    finite_search = np.isfinite(y_search)
    model_specs = {(member["repeat"], member["fold"]): member
                   for member in recipe["members"]}
    models, fitted_members = [], []
    classification = recipe["task"] == "activity"
    for repeat, repeat_folds in enumerate(folds):
        for fold, (train_local, held_local) in enumerate(repeat_folds):
            member = model_specs[(repeat, fold)]
            train_mask = np.zeros(len(search_idx), dtype=bool)
            train_mask[train_local] = True
            train_mask &= finite_search
            train_rows = search_idx[train_mask]
            if not len(train_rows):
                raise ValueError(f"no labeled search rows for repeat {repeat}, fold {fold}")
            groups_train = set(groups[train_rows].tolist())
            held_groups = set(groups[search_idx[held_local]].tolist())
            if groups_train & held_groups:
                raise AssertionError("shared search fold split a molecular group")
            y_train = y_search[train_mask]
            if classification and set(np.unique(y_train)) != {0.0, 1.0}:
                raise ValueError(f"activity fold {repeat}/{fold} lacks both training classes")
            estimator = _estimator(member, recipe["task"])
            fit_learner(estimator, member["family"], X[train_rows], y_train,
                        member["params"], classification)
            models.append(estimator)
            fitted_members.append({
                **member,
                "training_rows": int(len(train_rows)),
                "training_groups": int(len(groups_train)),
                "training_rows_sha256": hashlib.sha256(train_rows.tobytes()).hexdigest(),
            })

    candidate_recipe = {**recipe, "members": fitted_members}
    candidate_recipe["fit_stage"] = "pre_calibration" if pre_calibration else "calibrated"
    candidate_recipe["snapshot_manifest_sha256"] = sha256(snapshot / "manifest.json")
    candidate_recipe["recipe_sha256"] = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    candidate = CandidateArtifact(models, "classification" if classification else "regression",
                                  molecular_spec, recipe=candidate_recipe)

    package_versions = {"scikit-learn": sklearn_version, "numpy": np.__version__}
    families = sorted({member["family"] for member in recipe["members"]})
    package_for_family = {"xgb": "xgboost", "current_xgb": "xgboost", "catboost": "catboost"}
    for family in families:
        if family in package_for_family:
            package_versions[package_for_family[family]] = importlib.metadata.version(
                package_for_family[family])
    code_files = [Path(__file__), Path("scripts/benchmark_surrogates.py"),
                  Path("tackai/fusion/candidate_artifact.py"),
                  Path("tackai/fusion/molecular_spec.py"), Path("tackai/fusion/research.py")]
    context_contract = {
        "context_fingerprint": snapshot_manifest["context_fingerprint"],
        "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
        "snapshot_artifact_hashes": {
            path.name: snapshot_manifest["artifact_hashes"][path.name]
            for path in (snapshot / f"{recipe['feature']}.npy", snapshot / "labels.npz",
                         snapshot / "groups.npy", snapshot / "partition.npy")},
        "feature_name": recipe["feature"],
        "feature_dims": snapshot_manifest["features"][recipe["feature"]]["dims"],
        "feature_width": int(X.shape[1]),
        "molecular_feature_spec": molecular_spec.to_dict(),
        "snapshot_versions": snapshot_manifest["versions"],
        "runtime_model_package_versions": package_versions,
        "code_sha256": {str(path): sha256(path) for path in code_files},
        "role_counts": role_counts,
        "task_label_counts": {"search": int(np.isfinite(y_search).sum()),
                               "stacking": None, "calibration": None},
    }
    if pre_calibration:
        candidate.context_contract = context_contract
        candidate.save_pretrained(output)
        return output

    stack_rows = role_indices["stacking"]
    if not classification or recipe["stacking"]["weights"] == "fit":
        y_stacking = _task_targets(labels_path, recipe["label_policy"], recipe["task"],
                                   {"stacking": stack_rows})["stacking"]
        stack_mask = np.isfinite(y_stacking)
        stack_rows = stack_rows[stack_mask]
        y_stacking = y_stacking[stack_mask]
        if not len(stack_rows):
            raise ValueError("no labeled stacking rows for requested post-fit step")
        if classification and recipe["stacking"]["weights"] == "fit":
            if set(np.unique(y_stacking)) != {0.0, 1.0}:
                raise ValueError("stacking weight fit requires both activity classes")
            stack_predictions = candidate._predictions(X[stack_rows])
            candidate.weights_ = _fit_convex_weights(
                stack_predictions, y_stacking, recipe["task"], recipe["stacking"]["l2"])
        elif recipe["stacking"]["weights"] == "fit":
            stack_predictions = candidate._predictions(X[stack_rows])
            candidate.weights_ = _fit_convex_weights(
                stack_predictions, y_stacking, recipe["task"], recipe["stacking"]["l2"])
        if recipe["task"] != "activity":
            candidate.fit_residual_scale(X[stack_rows], y_stacking)

    calibration_rows = role_indices["calibration"]
    y_calibration = _task_targets(labels_path, recipe["label_policy"], recipe["task"],
                                  {"calibration": calibration_rows})["calibration"]
    calibration_mask = np.isfinite(y_calibration)
    calibration_rows = calibration_rows[calibration_mask]
    y_calibration = y_calibration[calibration_mask]
    if classification and set(np.unique(y_calibration)) != {0.0, 1.0}:
        raise ValueError("activity calibration requires both labels")
    unit = recipe["calibration"].get("unit", "row")
    candidate.calibrate(X[calibration_rows], y_calibration, alpha=recipe["calibration"]["alpha"],
                        unit=unit, groups=groups[calibration_rows] if unit == "group_max" else None)

    role_label_counts = {"search": int(np.isfinite(y_search).sum()),
                         "stacking": int(len(stack_rows)) if not classification
                         or recipe["stacking"]["weights"] == "fit" else None,
                         "calibration": int(len(calibration_rows))}
    context_contract["task_label_counts"] = role_label_counts
    candidate.context_contract = context_contract
    candidate.save_pretrained(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-calibration", action="store_true",
                        help="Save a search-only model bundle without reading stacking/calibration labels")
    args = parser.parse_args(argv)
    recipe = json.loads(args.recipe.read_text())
    output = _fit_one(args.snapshot, recipe, args.output,
                      pre_calibration=args.pre_calibration)
    print(json.dumps({"artifact": str(output), "fit_stage":
                      "pre_calibration" if args.pre_calibration else "calibrated",
                      "confirmation_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
