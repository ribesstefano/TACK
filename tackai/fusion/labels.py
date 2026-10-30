"""Evidence-preserving endpoint bounds for research training.

The published compatibility targets in ``data.make_targets`` remain unchanged.
This module explicitly distinguishes exact, approximate, censored and unknown
measurements. Unknown operators are never silently interpreted as equality.
"""
import numpy as np
import pandas as pd


def endpoint_bounds(values, operators, *, dc50=False, range_min=None, range_max=None):
    """Return transformed lower/upper bounds and an exact-observation mask.

    DC50 input is nM and reverses bounds under pDC50 transformation. Dmax input
    is percent degraded. Open bounds use the nearest representable float so that
    strict activity thresholds retain their mathematical meaning.
    """
    values = np.asarray(values, float)
    lo, hi = np.full(len(values), np.nan), np.full(len(values), np.nan)
    exact = np.zeros(len(values), bool)
    minimum, maximum = (0., np.inf) if dc50 else (0., 100.)
    for i, (value, op) in enumerate(zip(values, operators)):
        op = "" if pd.isna(op) else str(op).strip().lower()
        if op in {"range", "between", "-"}:
            if range_min is None or range_max is None:
                continue
            a, b = float(range_min[i]), float(range_max[i])
            if not (np.isfinite(a) and np.isfinite(b) and minimum <= a <= b <= maximum):
                continue
            lo[i], hi[i] = a, b
        elif not np.isfinite(value) or not minimum <= value <= maximum or (dc50 and value <= 0):
            continue
        elif op in {"", "=", "=="}:
            lo[i] = hi[i] = value
            exact[i] = True
        elif op in {">", ">=", "≥"}:
            lo[i], hi[i] = np.nextafter(value, np.inf) if op == ">" else value, maximum
        elif op in {"<", "<=", "≤"}:
            lo[i], hi[i] = minimum, np.nextafter(value, -np.inf) if op == "<" else value
        # Approximate and unknown operators intentionally remain unknown.
    invalid = lo > hi
    lo[invalid] = hi[invalid] = np.nan
    exact[invalid] = False
    if dc50:
        with np.errstate(divide="ignore", invalid="ignore"):
            lo, hi = 9. - np.log10(hi), 9. - np.log10(lo)
        # log10 can round adjacent raw values back to an exact threshold. Enforce
        # strictness in transformed space after applying the decreasing map.
        for i, op in enumerate(operators):
            if np.isnan(lo[i]) or np.isnan(hi[i]):
                continue
            if str(op).strip() == ">":
                hi[i] = np.nextafter(9. - np.log10(values[i]), -np.inf)
            elif str(op).strip() == "<":
                lo[i] = np.nextafter(9. - np.log10(values[i]), np.inf)
    else:
        lo, hi = lo / 100., hi / 100.
    return lo, hi, exact


def research_targets(table, policy="exact"):
    """Build point labels and censor bounds without guessing unknown activity.

    ``exact`` uses only exact endpoints; ``censored`` also returns informative
    intervals and activity labels proven by their bounds. Regression point labels
    are always NaN for censored observations; callers must use an interval loss.
    Activity requires compatible known endpoint times when both endpoints are
    needed. A single known failing endpoint can still establish inactivity.
    """
    if policy not in {"exact", "censored"}:
        raise ValueError("policy must be exact or censored")
    bounds, labels = {}, {}
    for task, val, prefix in [("dmax", "dmax_raw_pct", "Dmax"),
                              ("pdc50", "dc50_nM", "DC50")]:
        lo, hi, exact = endpoint_bounds(
            table[val], table[prefix + "_operator"], dc50=task == "pdc50",
            range_min=table[prefix + "_range_min"].to_numpy(),
            range_max=table[prefix + "_range_max"].to_numpy())
        labels[task] = np.where(exact, lo, np.nan)
        if policy == "exact":
            lo, hi = np.where(exact, lo, np.nan), np.where(exact, hi, np.nan)
        bounds[task] = np.column_stack([lo, hi])
    d, p = bounds["dmax"], bounds["pdc50"]
    activity = np.full(len(table), np.nan)
    activity[(d[:, 1] <= .8) | (p[:, 1] <= 6.)] = 0.
    times = table[["Dmax_h", "DC50_h"]].to_numpy(float)
    same_time = np.isfinite(times).all(axis=1) & np.isclose(times[:, 0], times[:, 1])
    activity[(d[:, 0] > .8) & (p[:, 0] > 6.) & same_time] = 1.
    labels["activity"] = activity
    return labels, bounds
