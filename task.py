"""The training runs the autopilot drives, and why they are small.

WHY A TWO-SECOND RUN
--------------------
An agent that reasons about training curves needs to see many curves. A loop
that costs 20 minutes per iteration cannot be evaluated at all -- one comparison
against a baseline search is a day of compute, and nobody re-runs it, so the
numbers in the write-up are from a single sample of a stochastic process.

So the task is deliberately small: a synthetic tabular classification problem
and an MLP that trains in about a second on CPU. What that buys is **100
seeded head-to-head comparisons against random search**, which is the only way
to say anything true about whether the reasoning loop helps.

What it costs is stated in RESULTS.md rather than glossed: these are not the
dynamics of a large model. What generalises is the *diagnosis* -- overfitting,
underfitting, divergence, plateau are the same shapes at any scale -- not the
specific hyperparameters the agent lands on.

THE FAILURE MODES ARE PLANTED
-----------------------------
`make_task` can generate problems that are hard in specific, known ways:

    separable      linearly separable; almost anything works
    interacting    the label depends on feature INTERACTIONS, so a linear
                   model or a too-small hidden layer underfits by construction
    noisy          20% label noise, so a big model with no regularisation
                   overfits by construction
    imbalanced     9:1 classes, where accuracy is a misleading metric

The point is that the correct diagnosis is *known in advance*. `bench.py`
scores the agent on whether it reached a good model, and `test_autopilot.py`
scores it on whether it named the right failure mode -- which is the part that
a hyperparameter search cannot do at all.
"""

import math
import random

import torch
import torch.nn as nn

TASKS = ("separable", "interacting", "noisy", "imbalanced")


def make_task(kind="interacting", n=2000, n_features=12, seed=0):
    """Synthetic tabular binary classification with a known difficulty.

    Splits are fixed at 60/20/20 train/val/test and the test set is used
    exactly once, by `bench.py`, at the very end. The agent never sees it --
    an autopilot that tunes against the set it reports on is measuring its own
    ability to overfit a split.
    """
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, n_features, generator=generator)

    if kind == "separable":
        weights = torch.randn(n_features, generator=generator)
        logits = X @ weights
    elif kind == "interacting":
        # XOR-ish: the sign of a product. No linear boundary exists, and a
        # hidden layer that is too narrow cannot represent it.
        logits = (X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]
                  - 0.5 * X[:, 4] * X[:, 5])
    elif kind == "noisy":
        weights = torch.randn(n_features, generator=generator)
        logits = X @ weights
    elif kind == "imbalanced":
        weights = torch.randn(n_features, generator=generator)
        logits = X @ weights - 3.4       # pushes the positive rate to ~10%
    else:
        raise ValueError(f"unknown task {kind!r}; use one of {TASKS}")

    y = (logits > 0).long()
    if kind == "noisy":
        flip = torch.rand(n, generator=generator) < 0.20
        y = torch.where(flip, 1 - y, y)

    n_train, n_val = int(n * 0.6), int(n * 0.2)
    return {
        "kind": kind,
        "n_features": n_features,
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        "test": (X[n_train + n_val:], y[n_train + n_val:]),
        "positive_rate": round(y.float().mean().item(), 4),
    }


class MLP(nn.Module):
    def __init__(self, n_features, hidden=64, depth=2, dropout=0.0):
        super().__init__()
        layers, size = [], n_features
        for _ in range(depth):
            layers += [nn.Linear(size, hidden), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            size = hidden
        layers.append(nn.Linear(size, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# A plausible copied-from-a-tutorial starting point, on purpose. It is not
# terrible and it is not good: it underfits the interacting task badly and
# leaves headroom everywhere. Starting the agent from a well-tuned config would
# measure nothing.
DEFAULT_CONFIG = {
    "hidden": 16, "depth": 1, "dropout": 0.0, "lr": 3e-3,
    "weight_decay": 0.0, "epochs": 40, "batch_size": 64, "seed": 0,
}


def train_once(task, config, record_every=1):
    """Train one model. Returns the config, the curves, and the val metrics.

    The returned `history` is what the agent reasons over: per-epoch train and
    validation loss and accuracy. Nothing else is exposed -- deliberately, so
    the agent has to diagnose from curves the way a person reading a dashboard
    would, rather than from privileged information about the task.
    """
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    torch.manual_seed(cfg["seed"])

    Xtr, ytr = task["train"]
    Xva, yva = task["val"]
    model = MLP(task["n_features"], cfg["hidden"], cfg["depth"], cfg["dropout"])
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    history = []
    n = Xtr.shape[0]
    generator = torch.Generator().manual_seed(cfg["seed"])
    for epoch in range(cfg["epochs"]):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg["batch_size"]):
            index = order[start:start + cfg["batch_size"]]
            optimiser.zero_grad()
            loss = loss_fn(model(Xtr[index]), ytr[index])
            loss.backward()
            optimiser.step()

        if epoch % record_every and epoch != cfg["epochs"] - 1:
            continue
        model.eval()
        with torch.no_grad():
            train_loss, train_acc, _ = _evaluate(model, Xtr, ytr, loss_fn)
            val_loss, val_acc, val_balanced = _evaluate(model, Xva, yva, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss,
                        "val_acc": val_acc, "val_balanced_acc": val_balanced})

    final = history[-1]
    return {
        "config": cfg,
        "history": history,
        "val_loss": final["val_loss"],
        "val_acc": final["val_acc"],
        "val_balanced_acc": final["val_balanced_acc"],
        "best_val_acc": max(h["val_acc"] for h in history),
        "best_val_balanced_acc": max(h["val_balanced_acc"] for h in history),
        "params": sum(p.numel() for p in model.parameters()),
        "model": model,
    }


def _evaluate(model, X, y, loss_fn):
    """Loss, accuracy, and BALANCED accuracy.

    Balanced accuracy is carried alongside plain accuracy from the start
    because on the imbalanced task they disagree completely: a model that
    predicts the majority class for every row scores 0.88 accuracy and 0.50
    balanced accuracy. An autopilot optimising the first number would
    confidently converge on a model that has learned nothing, and every curve
    it looked at would be going the right way.
    """
    logits = model(X)
    loss = loss_fn(logits, y).item()
    predicted = logits.argmax(1)
    acc = (predicted == y).float().mean().item()
    recalls = []
    for label in (0, 1):
        mask = y == label
        if mask.any():
            recalls.append((predicted[mask] == label).float().mean().item())
    balanced = sum(recalls) / len(recalls) if recalls else 0.0
    return (round(loss, 5) if math.isfinite(loss) else float("inf"),
            round(acc, 5), round(balanced, 5))


def test_scores(model, task):
    """Scored ONCE, by bench.py, after the search is over."""
    X, y = task["test"]
    model.eval()
    with torch.no_grad():
        _, acc, balanced = _evaluate(model, X, y, nn.CrossEntropyLoss())
    return {"test_acc": acc, "test_balanced_acc": balanced}
