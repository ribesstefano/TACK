"""HTTP inference server for the fusion surrogates.

Exists so a client in a *different* Python environment can score molecules with a
`XGBStackedEnsemble` without installing `tackai` beside its own dependencies. The
motivating case is REINVENT4, which pins its own `torch`/`rdkit`: co-installing the
two is not merely untidy but unstable, because `torch` and `xgboost` each bundle a
`libomp` that segfaults when both run multi-threaded in one process (see the macOS
note in ``CLAUDE.md``). A separate process makes that impossible by construction.

The server trades a localhost round trip for keeping the ensemble and the molecular
featuriser warm. That is the whole point: a generative loop scores thousands of
batches, so paying rdkit import and ensemble deserialisation once rather than per
batch is what makes this viable at all. Scoring one batch of a few hundred molecules
costs barely more than scoring one molecule, so clients should send a population per
request, never a molecule per request.

`app.py` holds the FastAPI application and `schemas.py` the request/response models;
both import FastAPI and are therefore only reachable with the ``serve`` extra
installed. Nothing here is imported by ``tackai.serve``'s own package init, so
``import tackai.serve`` stays cheap and dependency-free.
"""
