import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chemprop")

from tackai.fusion.graph import GraphPredictor, _mol_graphs, masked_graph_loss  # noqa: E402


def test_chemprop_graphs_preserve_isomeric_stereochemistry():
    graphs, atom_dim, bond_dim = _mol_graphs(["N[C@@](C)(O)C(=O)O", "N[C@](C)(O)C(=O)O"])
    assert atom_dim > 0 and bond_dim > 0
    assert len(graphs) == 2
    assert not np.array_equal(graphs[0].V, graphs[1].V)


def test_missing_endpoint_labels_have_no_gradient():
    prediction = torch.tensor([[.4, .2, -.3], [.7, .1, .9]], requires_grad=True)
    target = torch.tensor([[float("nan"), float("nan"), 0.],
                           [float("nan"), float("nan"), 1.]])
    loss = masked_graph_loss(prediction, target, torch.ones(2))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(prediction.grad[:, :2], torch.zeros(2, 2))
    assert torch.isfinite(prediction.grad[:, 2]).all()


def test_standardized_pdc50_loss_is_not_scaled_twice():
    prediction = torch.tensor([[.2, 0., .1]], requires_grad=True)
    target = torch.tensor([[float("nan"), 1., float("nan")]])
    loss = masked_graph_loss(prediction, target, torch.tensor([2., 10.]))
    torch.testing.assert_close(loss, torch.tensor(1.))


def test_minibatch_loss_sums_to_full_masked_head_loss():
    prediction = torch.tensor([[.2, -.1, .4], [.4, .5, -.2], [.8, .3, .1]])
    target = torch.tensor([[.1, 0., float("nan")], [.3, float("nan"), 1.],
                           [float("nan"), .2, 0.]])
    scales = torch.tensor([.5, 1.])
    denominators = torch.isfinite(target).sum(dim=0).tolist()
    full = masked_graph_loss(prediction, target, scales)
    partial = sum(masked_graph_loss(prediction[rows], target[rows], scales,
                                    denominators=denominators)
                  for rows in (slice(0, 1), slice(1, 3)))
    torch.testing.assert_close(partial, full)


def test_tiny_fit_scores_new_molecules_and_roundtrips_safely(tmp_path):
    smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "COC", "CCCl"]
    side = np.arange(18, dtype=float).reshape(6, 3) / 10
    target = np.array([
        [.1, 5.1, 0.], [.4, 6.2, 0.], [.8, 7.3, 1.],
        [.3, 5.8, 0.], [.6, 6.5, 1.], [.2, 5.5, np.nan],
    ])
    model = GraphPredictor(side_dim=3, width=16, depth=2, batch_size=3, seed=7)
    model.fit(smiles[:4], side[:4], target[:4], epochs=2,
              validation=(smiles[4:], side[4:], target[4:]), patience=1)
    np.testing.assert_allclose(model.side_mean_, side[:4].mean(axis=0))
    query_smiles = ["[He]"]
    query_side = side[:1]
    predictions = model.predict(query_smiles, query_side)
    assert all(values.shape == (1,) and np.isfinite(values).all()
               for values in predictions.values())
    empty = model.predict([], np.empty((0, 3)))
    assert all(values.shape == (0,) for values in empty.values())
    destination = tmp_path / "model"
    model.save(destination, metadata={"test": True})
    restored = GraphPredictor.load(destination)
    restored_predictions = restored.predict(query_smiles, query_side)
    for task in predictions:
        np.testing.assert_allclose(predictions[task], restored_predictions[task], rtol=0, atol=1e-7)
    with pytest.raises(FileExistsError):
        model.save(destination)
