"""Masked supervision, bounded outputs and portable neural artifact checks."""
import numpy as np
import torch

from tackai.fusion.neural import NeuralPredictor, TASK_ORDER, masked_loss


def test_masked_labels_contribute_no_gradient():
    prediction = torch.tensor([[.5, 0., 0.], [.5, 0., 0.]], requires_grad=True)
    targets = torch.tensor([[.8, float("nan"), 1.], [float("nan"), 2., float("nan")]])
    loss = masked_loss(prediction, targets, torch.ones(2))
    loss.backward()
    assert prediction.grad[0, 1] == 0
    assert prediction.grad[1, 0] == prediction.grad[1, 2] == 0
    assert torch.isfinite(prediction.grad).all()


def test_all_missing_loss_is_differentiable_zero():
    prediction = torch.ones((2, 3), requires_grad=True)
    loss = masked_loss(prediction, torch.full((2, 3), float("nan")), torch.ones(2))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(prediction.grad, torch.zeros((2, 3)))


def test_interval_loss_only_pushes_predictions_outside_evidence_bounds():
    prediction = torch.tensor([[.8, 5., 0.]], requires_grad=True)
    targets = torch.full((1, 3), float("nan"))
    bounds = torch.tensor([[[.7, 1.], [-float("inf"), 4.]]])
    loss = masked_loss(prediction, targets, torch.ones(2), bounds=bounds)
    loss.backward()
    assert prediction.grad[0, 0] == 0  # Dmax is already consistent with its lower bound.
    assert prediction.grad[0, 1] > 0   # Descend toward the pDC50 upper bound.
    assert prediction.grad[0, 2] == 0


def test_interval_exact_measurements_match_point_loss():
    prediction = torch.tensor([[.8, 5., 0.]])
    targets = torch.tensor([[.5, 4., 1.]])
    bounds = torch.stack([targets[:, :2], targets[:, :2]], dim=2)
    point = masked_loss(prediction, targets, torch.ones(2))
    interval = masked_loss(prediction, targets, torch.ones(2), bounds=bounds)
    torch.testing.assert_close(point, interval)


def test_fit_normalization_never_uses_validation_and_roundtrips(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 12)).astype(np.float32)
    y = np.column_stack([rng.uniform(size=40), rng.normal(7, 1, 40), rng.integers(0, 2, 40)])
    y[::3, 1] = np.nan
    model = NeuralPredictor(molecular_width=8, width=8, dropout=0., seed=3).fit(
        X[:30], y[:30], epochs=3, validation=(X[30:] + 1000, y[30:]))
    np.testing.assert_allclose(model.mean_, X[:30].mean(axis=0, dtype=np.float64))
    before = model.predict(X[30:])
    assert np.all((before["dmax"] >= 0) & (before["dmax"] <= 1))
    assert np.all((before["activity"] >= 0) & (before["activity"] <= 1))
    model.save(tmp_path / "model")
    restored = NeuralPredictor.load(tmp_path / "model")
    after = restored.predict(X[30:])
    for task in TASK_ORDER:
        np.testing.assert_array_equal(before[task], after[task])
    assert all(len(x) == 0 for x in restored.predict(np.empty((0, 12))).values())
