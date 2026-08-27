"""One test per real bug, plus the invariant.

    python test_autopilot.py

No pytest, no network, no downloaded dataset. Trains real models, so it takes
about 40 seconds -- every other suite in this set runs in one.

`verify_suite_is_not_decorative()` re-installs the pre-fix loop and asserts the
suite catches it.
"""

import sys

import autopilot as autopilot_mod
import task as task_mod
from autopilot import Autopilot, diagnose, propose, verdict_for

_results = []


def check(name):
    def wrap(fn):
        _results.append((name, fn))
        return fn
    return wrap


def curve(rows):
    """Build a history from (train_loss, val_loss) pairs, with sane extras."""
    out = []
    for epoch, (train, val) in enumerate(rows):
        out.append({"epoch": epoch, "train_loss": train, "val_loss": val,
                    "train_acc": 0.9, "val_acc": 0.9,
                    "val_balanced_acc": 0.9})
    return out


def run(kind, budget=8, seed=1, **kwargs):
    problem = task_mod.make_task(kind, seed=seed)
    agent = Autopilot(problem, budget=budget, **kwargs)
    agent.run()
    return agent, problem


# --------------------------------------------------------------------------
# THE INVARIANT
# --------------------------------------------------------------------------

@check("INVARIANT: every entry is a hypothesis with a recorded verdict")
def _():
    agent, _ = run("interacting", budget=6)
    assert agent.log
    for entry in agent.log:
        assert entry.hypothesis, f"step {entry.step} has no hypothesis"
        assert entry.verdict in ("confirmed", "refuted", "inconclusive"), \
            entry.verdict
        assert entry.diagnosis in autopilot_mod.DIAGNOSES, entry.diagnosis


@check("refuted hypotheses are recorded, not quietly dropped")
def _():
    # A log containing only confirmations is a log nobody is reading.
    verdicts = set()
    for kind in task_mod.TASKS:
        agent, _ = run(kind, budget=8)
        verdicts.update(entry.verdict for entry in agent.log)
    assert "refuted" in verdicts, verdicts
    assert "confirmed" in verdicts, verdicts


@check("a verdict is decided on the PREDICTION, not on accuracy")
def _():
    # A change that improves accuracy while failing to do what the hypothesis
    # claimed is a refuted hypothesis with a lucky outcome. Scoring it as a
    # confirmation is how a research log becomes a story.
    before = {"train_loss": 0.50, "val_loss": 0.60, "val_balanced_acc": 0.70}
    after = {"train_loss": 0.55, "val_loss": 0.40, "val_balanced_acc": 0.90}
    diagnosis = autopilot_mod.Diagnosis("underfitting", "certain")
    verdict, note = verdict_for(diagnosis, "train loss falls", before, after)
    assert verdict == "refuted", (verdict, note)     # accuracy rose; train loss did not
    verdict, _ = verdict_for(diagnosis, "val loss falls", before, after)
    assert verdict == "confirmed"


@check("each step changes exactly one thing")
def _():
    agent, _ = run("interacting", budget=8)
    for entry in agent.log:
        if not entry.change or entry.hypothesis.startswith("restart"):
            continue
        assert len(entry.change) <= 2, entry.change
        if len(entry.change) == 2:
            # The only permitted pair: a smaller step needs longer to run.
            assert set(entry.change) == {"lr", "epochs"}, entry.change


# --------------------------------------------------------------------------
# diagnosis
# --------------------------------------------------------------------------

@check("overfitting is read off the curve shape")
def _():
    history = curve([(0.9, 0.9), (0.6, 0.6), (0.4, 0.45), (0.25, 0.5),
                     (0.15, 0.6), (0.10, 0.7), (0.08, 0.8), (0.06, 0.9)])
    assert diagnose(history).name == "overfitting"


@check("underfitting is read off the curve shape")
def _():
    history = curve([(0.69,) * 2] * 10)
    assert diagnose(history).name == "underfitting"


@check("divergence is caught before anything else is read")
def _():
    assert diagnose(curve([(0.6, 0.6), (0.8, 0.9), (1.4, 1.6)])).name == "diverging"
    nan = curve([(0.6, 0.6), (0.5, 0.5)])
    nan[-1]["val_loss"] = float("inf")
    assert diagnose(nan).name == "diverging"


@check("majority-class collapse is caught, and loss curves cannot see it")
def _():
    # accuracy 0.88 and every loss curve going the right way, while the model
    # predicts one class for everything.
    history = curve([(0.5, 0.5), (0.35, 0.36), (0.3, 0.31), (0.28, 0.29)])
    for row in history:
        row["val_acc"], row["val_balanced_acc"] = 0.88, 0.50
    assert diagnose(history).name == "majority_class"
    healthy = [dict(row, val_balanced_acc=0.87) for row in history]
    assert diagnose(healthy).name != "majority_class"


@check("convergence is not confused with a plateau above the floor")
def _():
    converged = curve([(0.10, 0.11)] * 10)
    assert diagnose(converged).name == "converged"
    stuck = curve([(0.55, 0.56)] * 10)
    assert diagnose(stuck).name == "underfitting"


# --------------------------------------------------------------------------
# the three bugs
# --------------------------------------------------------------------------

@check("converged is a property of a run, not a reason to stop searching")
def _():
    # THE first bug: the agent stopped after one run on three of four tasks,
    # leaving 0.9355 on interacting where 0.9698 was reachable.
    agent, _ = run("separable", budget=6)
    assert len(agent.runs) > 1, \
        f"stopped after {len(agent.runs)} run(s); converged was treated as terminal"
    assert propose(autopilot_mod.Diagnosis("converged", "certain"),
                   dict(task_mod.DEFAULT_CONFIG), set())[0] is not None


@check("a noise floor is distinguished from underfitting, using TWO runs")
def _():
    # The second bug. On 20% label noise the Bayes error holds train loss near
    # 0.5 forever, so an absolute threshold says "underfitting" on every run.
    # No single curve can separate "too small" from "impossible"; two can.
    agent = Autopilot(task_mod.make_task("noisy", seed=1), budget=8)
    diagnosis = autopilot_mod.Diagnosis("underfitting", "certain")
    before = {"train_loss": 0.55, "val_loss": 0.56, "val_balanced_acc": 0.78}
    after = {"train_loss": 0.40, "val_loss": 0.57, "val_balanced_acc": 0.78}
    corrected = agent._reconsider(diagnosis, "confirmed", before, after)
    assert corrected.name == "noise_floor", corrected.name

    followed = {"train_loss": 0.40, "val_loss": 0.50, "val_balanced_acc": 0.80}
    still = agent._reconsider(diagnosis, "confirmed", before, followed)
    assert still.name == "underfitting", "val followed; this is ordinary progress"


@check("the noise floor is actually reached on the noisy task")
def _():
    agent, _ = run("noisy", budget=10)
    diagnoses = {entry.diagnosis for entry in agent.log}
    assert "noise_floor" in diagnoses, diagnoses
    assert agent.best["val_balanced_acc"] > 0.80, agent.best["val_balanced_acc"]


@check("the agent never returns budget unspent")
def _():
    # THE third bug, and the one that made it lose: it converged, ran out of
    # moves, and stopped with 4 of 10 runs unused -- 0.9056 against random
    # search's 0.9215.
    for kind in task_mod.TASKS:
        agent, _ = run(kind, budget=8)
        assert len(agent.runs) == 8, \
            f"{kind}: used {len(agent.runs)} of 8 runs"


@check("a restart lands somewhere untried")
def _():
    agent = Autopilot(task_mod.make_task("separable", seed=1), budget=6)
    agent.run()
    seen = [entry.change for entry in agent.log if entry.change]
    import json
    keys = [json.dumps(c, sort_keys=True) for c in seen]
    assert len(keys) == len(set(keys)), f"repeated a configuration: {keys}"


# --------------------------------------------------------------------------
# selection and honesty
# --------------------------------------------------------------------------

@check("selection is on balanced accuracy, which changes the answer")
def _():
    # On the imbalanced task, plain accuracy and balanced accuracy disagree
    # about which model is better -- and plain accuracy prefers the one that
    # has learned less.
    problem = task_mod.make_task("imbalanced", seed=1)
    assert problem["positive_rate"] < 0.2, problem["positive_rate"]
    result = task_mod.train_once(problem, {"epochs": 10, "hidden": 8})
    assert result["val_balanced_acc"] < result["val_acc"], \
        "this fixture no longer separates the two metrics"


@check("the test split is never touched during the search")
def _():
    problem = task_mod.make_task("interacting", seed=1)
    before = problem["test"][1].clone()
    agent = Autopilot(problem, budget=6)
    agent.run()
    assert (problem["test"][1] == before).all(), "the test labels moved"
    for entry in agent.log:
        assert "test" not in entry.hypothesis.lower()


@check("the research log is ascii and readable")
def _():
    agent, _ = run("interacting", budget=4)
    text = agent.research_log()
    assert text.isascii(), "non-ascii in a log that has to print on Windows"
    for expected in ("hypothesis", "change", "result", "verdict"):
        assert expected in text, expected


# --------------------------------------------------------------------------
# Phase 5: prove the suite fails on the code it was written against
# --------------------------------------------------------------------------

def _old_reconsider(self, diagnosis, verdict, previous, final):
    """No cross-run correction: underfitting forever on a noisy task."""
    return diagnosis


REGRESSIONS = {
    "_reconsider": (_old_reconsider,
                    ("a noise floor is distinguished from underfitting, using TWO runs",
                     "the noise floor is actually reached on the noisy task")),
}


def verify_suite_is_not_decorative():
    by_name = dict(_results)
    lines, total = [], 0
    for name, (old, targets) in REGRESSIONS.items():
        original = getattr(Autopilot, name)
        setattr(Autopilot, name, old)
        try:
            caught = 0
            for target in targets:
                try:
                    by_name[target]()
                except AssertionError:
                    caught += 1
        finally:
            setattr(Autopilot, name, original)
        total += caught
        lines.append(f"  pre-fix Autopilot.{name:<13} "
                     f"{caught}/{len(targets)} of its tests fail")
    return lines, total


def main():
    failed = 0
    for name, fn in _results:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}\n      {exc}")
    print(f"\n{len(_results) - failed}/{len(_results)} passed")

    print("\nagainst the pre-fix implementation:")
    lines, caught = verify_suite_is_not_decorative()
    print("\n".join(lines))
    if caught < len(REGRESSIONS):
        print("the suite does not detect the bug it was written for")
        failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
