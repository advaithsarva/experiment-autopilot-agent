"""Diagnose a training curve, change one thing, write it down, repeat.

THE INVARIANT
-------------
**Every configuration the agent tries is a named hypothesis about the previous
run, and the research log records the hypothesis, the change, the result and
whether the hypothesis was right -- including when it was wrong.**

That is what separates this from a search. Random search also finds good
configurations; it cannot tell you *why*, it cannot say "the model is
overfitting", and it learns nothing from a run that failed. The log here is the
deliverable, and a log that only contains confirmations is a log that is not
being read.

So `Entry.verdict` is one of `confirmed`, `refuted` or `inconclusive`, computed
by comparing what the hypothesis predicted would happen against what did.
`bench.py` reports how often the agent's hypotheses are refuted -- a number
that a search cannot produce and that an honest write-up has to contain.

DIAGNOSIS IS RULES OVER CURVES
------------------------------
Five failure modes, each with a shape:

    diverging       loss increasing or NaN. Learning rate too high.
    underfitting    train loss high and flat. The model cannot represent it.
    overfitting     train loss falling, val loss RISING. Too much capacity or
                    not enough regularisation.
    plateau         both flat, well above the achievable floor.
    majority_class  high accuracy, balanced accuracy near 0.5. The model
                    predicts one class for everything.
    converged       val loss stopped improving and train/val agree. Stop.

The last two matter most. `majority_class` is the one that a single-metric
autopilot cannot see at all -- accuracy 0.88, every curve going the right way,
and the model has learned nothing. `converged` is the one that makes the agent
stop, and knowing when to stop is most of the compute saving.

CHANGE ONE THING
----------------
Each step changes exactly one hyperparameter (or one coupled pair, where
changing one alone is meaningless). Two changes at once make the result
uninterpretable, which destroys the log's value even when the accuracy improves
-- and the log is the point.
"""

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field

import task as task_mod

# Below this, train loss is "low" -- the model fits its training data.
LOW_TRAIN_LOSS = 0.25
# Improvement in val loss smaller than this over the window is "flat".
FLAT_EPSILON = 0.005
# Val loss must exceed its own best by this much to count as overfitting
# rather than as noise.
OVERFIT_MARGIN = 0.04
# Balanced accuracy this close to chance means one class is being ignored.
COLLAPSE_BALANCED = 0.62

DIAGNOSES = ("diverging", "majority_class", "overfitting", "underfitting",
             "noise_floor", "plateau", "converged")


@dataclass
class Diagnosis:
    name: str
    confidence: str          # certain | likely | unsure
    evidence: dict = field(default_factory=dict)
    detail: str = ""


@dataclass
class Entry:
    """One iteration of the research log."""
    step: int
    hypothesis: str
    change: dict
    diagnosis: str
    val_loss: float
    val_acc: float
    val_balanced_acc: float
    predicted: str           # what the hypothesis said would happen
    verdict: str = "pending"  # confirmed | refuted | inconclusive
    note: str = ""

    def to_dict(self):
        return asdict(self)

    def render(self):
        """Two different things get labelled separately, because they are.

        `because` is the hypothesis that produced THIS run's change -- it
        came from the *previous* run's diagnosis. `now reads` is what this
        run's own curves say. Printing them under one heading
        (`step 3: noise_floor` above a hypothesis about a narrow hidden
        layer) read as though the agent had diagnosed something and then
        immediately contradicted itself.
        """
        lines = [f"step {self.step}",
                 f"  tried       {self.change or 'the baseline configuration'}",
                 f"  because     {self.hypothesis}",
                 f"  result      val_loss {self.val_loss:.4f}  "
                 f"acc {self.val_acc:.4f}  balanced {self.val_balanced_acc:.4f}",
                 f"  verdict     {self.verdict}"
                 + (f" -- {self.note}" if self.note else ""),
                 f"  now reads   {self.diagnosis}"]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# diagnosis
# --------------------------------------------------------------------------

def diagnose(history, window=8):
    """Name the failure mode from the curves alone.

    Deliberately sees only what a person reading a dashboard sees: per-epoch
    train and validation loss, accuracy and balanced accuracy. Nothing about
    the task, nothing about the config.
    """
    if not history:
        return Diagnosis("plateau", "unsure", detail="no history")

    final = history[-1]
    recent = history[-min(window, len(history)):]
    val_losses = [h["val_loss"] for h in history]
    best_val = min(val_losses)
    best_index = val_losses.index(best_val)

    # 1. Diverging. Checked first: nothing else can be read off a curve that
    #    is going up or has gone non-finite.
    if not math.isfinite(final["val_loss"]) or not math.isfinite(final["train_loss"]):
        return Diagnosis("diverging", "certain",
                         {"val_loss": final["val_loss"]},
                         "loss is NaN or infinite")
    if len(history) >= 3 and final["train_loss"] > history[0]["train_loss"] * 1.5:
        return Diagnosis("diverging", "certain",
                         {"first": history[0]["train_loss"],
                          "last": final["train_loss"]},
                         "training loss is higher than where it started")

    # 2. Majority-class collapse. BEFORE anything loss-shaped, because every
    #    loss curve looks healthy while this is happening.
    if final["val_balanced_acc"] < COLLAPSE_BALANCED and final["val_acc"] > 0.7:
        return Diagnosis("majority_class", "certain",
                         {"val_acc": final["val_acc"],
                          "val_balanced_acc": final["val_balanced_acc"]},
                         f"accuracy {final['val_acc']:.3f} but balanced accuracy "
                         f"{final['val_balanced_acc']:.3f}; one class is being ignored")

    # 3. Overfitting: train loss low, val loss materially above its own best.
    gap = final["val_loss"] - best_val
    if (final["train_loss"] < LOW_TRAIN_LOSS and gap > OVERFIT_MARGIN
            and best_index < len(history) - 2):
        return Diagnosis("overfitting", "certain",
                         {"train_loss": final["train_loss"],
                          "val_loss": final["val_loss"], "best_val_loss": best_val,
                          "best_epoch": history[best_index]["epoch"]},
                         f"val loss bottomed at {best_val:.4f} (epoch "
                         f"{history[best_index]['epoch']}) and rose to "
                         f"{final['val_loss']:.4f} while train loss kept falling")

    improvement = recent[0]["val_loss"] - recent[-1]["val_loss"]
    flat = improvement < FLAT_EPSILON

    # 4. Underfitting: it cannot even fit the training data.
    if final["train_loss"] > LOW_TRAIN_LOSS and flat:
        return Diagnosis("underfitting", "certain" if len(history) >= window else "likely",
                         {"train_loss": final["train_loss"],
                          "train_acc": final["train_acc"],
                          "improvement": round(improvement, 5)},
                         f"train loss stuck at {final['train_loss']:.4f}; the model "
                         f"cannot fit its own training data")

    # 5. Converged: flat, and train and val agree.
    if flat and final["train_loss"] <= LOW_TRAIN_LOSS and gap <= OVERFIT_MARGIN:
        return Diagnosis("converged", "certain",
                         {"val_loss": final["val_loss"],
                          "improvement": round(improvement, 5)},
                         "val loss stopped improving with no train/val gap")

    if flat:
        return Diagnosis("plateau", "likely",
                         {"improvement": round(improvement, 5),
                          "val_loss": final["val_loss"]},
                         "neither curve is moving")

    # Still improving. Not a failure mode -- more epochs is the answer.
    return Diagnosis("plateau", "unsure",
                     {"improvement": round(improvement, 5)},
                     "still improving; nothing to diagnose yet")


# --------------------------------------------------------------------------
# the mutation table
# --------------------------------------------------------------------------

def propose(diagnosis, config, tried):
    """One change, a hypothesis, and what it predicts. `tried` prevents loops.

    Returns `(change, hypothesis, predicted)`. `predicted` is the observable
    the next run is checked against -- that is what makes the verdict
    falsifiable rather than a summary written after the fact.
    """
    name = diagnosis.name
    options = []

    if name == "diverging":
        options = [
            ({"lr": round(config["lr"] / 5, 6)},
             "the learning rate is too high for this loss surface",
             "train loss falls instead of rising"),
            ({"batch_size": config["batch_size"] * 2},
             "gradient noise from a small batch is destabilising the update",
             "train loss falls instead of rising"),
        ]
    elif name == "underfitting":
        options = [
            ({"hidden": config["hidden"] * 4},
             "the hidden layer is too narrow to represent the decision boundary",
             "train loss falls"),
            ({"depth": config["depth"] + 2},
             "one hidden layer cannot compose the feature interactions this needs",
             "train loss falls"),
            ({"lr": round(config["lr"] * 3, 6)},
             "the learning rate is too low to escape this region in the epochs given",
             "train loss falls"),
            ({"epochs": config["epochs"] * 2},
             "it is still learning and simply ran out of epochs",
             "train loss falls"),
        ]
    elif name == "overfitting":
        options = [
            ({"dropout": round(min(config["dropout"] + 0.3, 0.6), 3)},
             "the model has enough capacity to memorise; dropout should stop it",
             "the gap between train and val loss narrows"),
            ({"weight_decay": max(config["weight_decay"] * 10, 1e-3)},
             "unregularised weights are fitting noise",
             "the gap between train and val loss narrows"),
            ({"hidden": max(config["hidden"] // 2, 4)},
             "there is more capacity here than the problem needs",
             "the gap between train and val loss narrows"),
            ({"epochs": max(int(config["epochs"] * 0.6), 5)},
             "the best model was several epochs ago; stop earlier",
             "the gap between train and val loss narrows"),
        ]
    elif name == "majority_class":
        options = [
            ({"class_weight": True},
             "the loss is dominated by the majority class and the model has "
             "learned to always predict it",
             "balanced accuracy rises"),
            ({"lr": round(config["lr"] / 3, 6)},
             "a large step collapsed the model onto the majority class early",
             "balanced accuracy rises"),
        ]
    elif name == "noise_floor":
        options = [
            ({"dropout": round(min(config["dropout"] + 0.3, 0.6), 3)},
             "train loss falls with capacity but validation does not follow, so "
             "the remaining train loss is label noise, not missing capacity",
             "val loss falls"),
            ({"weight_decay": max(config["weight_decay"] * 10, 1e-2)},
             "the model is fitting noise; penalise the weights that do it",
             "val loss falls"),
            ({"hidden": max(config["hidden"] // 2, 4)},
             "less capacity cannot memorise the mislabelled rows",
             "val loss falls"),
        ]
    elif name == "converged":
        # Converged is a property of THIS RUN, not a reason to stop searching.
        # The first version treated it as terminal and the agent stopped after
        # a single run on three of four tasks -- leaving 0.9355 on the
        # interacting task where 0.9675 was reachable. A run that has finished
        # learning is exactly the moment to ask whether a bigger model would
        # have more to learn.
        options = [
            ({"hidden": config["hidden"] * 4},
             "this configuration has converged, so any remaining headroom is "
             "capacity rather than optimisation",
             "val loss falls"),
            ({"depth": config["depth"] + 1},
             "depth, not width, is what this decision boundary needs",
             "val loss falls"),
            ({"lr": round(config["lr"] / 3, 6), "epochs": config["epochs"] * 2},
             "a smaller step for longer settles into a better minimum",
             "val loss falls"),
            ({"weight_decay": max(config["weight_decay"] * 10, 1e-3)},
             "mild regularisation generalises better even without visible "
             "overfitting",
             "val loss falls"),
        ]
    elif name == "plateau":
        options = [
            ({"lr": round(config["lr"] / 3, 6)},
             "the step size is too large to settle into the minimum it is orbiting",
             "val loss falls"),
            ({"hidden": config["hidden"] * 2},
             "capacity is the binding constraint, not optimisation",
             "val loss falls"),
            ({"epochs": config["epochs"] * 2},
             "it needs longer at this learning rate",
             "val loss falls"),
        ]

    for change, hypothesis, predicted in options:
        key = json.dumps({**config, **change}, sort_keys=True, default=str)
        if key not in tried:
            return change, hypothesis, predicted
    return None, "every change this diagnosis suggests has been tried", None


def verdict_for(diagnosis, predicted, before, after):
    """Was the hypothesis right? Compared on what it actually predicted.

    Not on validation accuracy. A change that improves accuracy while failing
    to do the thing the hypothesis claimed is a *refuted* hypothesis with a
    lucky outcome, and recording it as a confirmation is how a research log
    becomes a story.
    """
    if predicted is None or before is None:
        return "inconclusive", ""

    if "gap" in predicted:
        gap_before = before["val_loss"] - before["train_loss"]
        gap_after = after["val_loss"] - after["train_loss"]
        ok = gap_after < gap_before - 1e-4
        return ("confirmed" if ok else "refuted",
                f"train/val gap {gap_before:.4f} -> {gap_after:.4f}")
    if "balanced accuracy rises" in predicted:
        ok = after["val_balanced_acc"] > before["val_balanced_acc"] + 1e-4
        return ("confirmed" if ok else "refuted",
                f"balanced accuracy {before['val_balanced_acc']:.4f} -> "
                f"{after['val_balanced_acc']:.4f}")
    if "train loss falls" in predicted:
        ok = after["train_loss"] < before["train_loss"] - 1e-4
        return ("confirmed" if ok else "refuted",
                f"train loss {before['train_loss']:.4f} -> {after['train_loss']:.4f}")
    if "val loss falls" in predicted:
        ok = after["val_loss"] < before["val_loss"] - 1e-4
        return ("confirmed" if ok else "refuted",
                f"val loss {before['val_loss']:.4f} -> {after['val_loss']:.4f}")
    return "inconclusive", ""


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

class Autopilot:
    """Runs experiments until it converges or runs out of budget."""

    def __init__(self, task, config=None, budget=8, metric="val_balanced_acc",
                 patience=3, clock=time.time):
        self.task = task
        self.config = dict(task_mod.DEFAULT_CONFIG)
        self.config.update(config or {})
        self.budget = budget
        # The search stops when `patience` consecutive changes fail to improve
        # the metric -- a convergence criterion on the SEARCH. A single run
        # converging is not one.
        self.patience = patience
        self.stopped_because = "budget exhausted"
        # Selection is on BALANCED accuracy by default. Selecting on plain
        # accuracy on the imbalanced task picks the majority-class model, and
        # every curve behind that choice looks healthy.
        self.metric = metric
        self.clock = clock
        self.log = []
        self.runs = []
        self.best = None
        self._tried = set()

    def run(self):
        change, hypothesis, predicted = {}, "baseline configuration", None
        previous, stale = None, 0

        for step in range(self.budget):
            config = dict(self.config)
            config.update(change or {})
            config = _apply_class_weight(config)
            self._tried.add(json.dumps(config, sort_keys=True, default=str))

            result = task_mod.train_once(self.task, config)
            self.runs.append(result)
            final = result["history"][-1]

            diagnosis = diagnose(result["history"])
            verdict, note = verdict_for(diagnosis, predicted, previous, final)
            diagnosis = self._reconsider(diagnosis, verdict, previous, final)
            self.log.append(Entry(
                step=step, hypothesis=hypothesis, change=dict(change or {}),
                diagnosis=diagnosis.name, val_loss=final["val_loss"],
                val_acc=final["val_acc"],
                val_balanced_acc=final["val_balanced_acc"],
                predicted=predicted or "", verdict=verdict, note=note))

            improved = self.best is None or result[self.metric] > self.best[self.metric]
            if improved:
                self.best = result
                self.config = config        # build on what worked
                stale = 0
            else:
                stale += 1
                # A change that did not help is not a new baseline. Continuing
                # from it compounds bad decisions; the log still records it.

            if stale >= self.patience:
                # CONVERGED, BUT THE BUDGET IS NOT SPENT. Stopping here is what
                # made the agent lose: it finished `imbalanced` after 6 of 10
                # runs at 0.9160 while random search used all 10 and reached
                # 0.9600. An adaptive search that returns budget unspent is
                # strictly worse than one that does not, and "I converged" is
                # a statement about a local basin, not about the problem.
                #
                # So: restart somewhere else and keep the best across restarts.
                # This is plain multi-start optimisation, and it is what makes
                # the comparison against random search an equal-budget one.
                if len(self.runs) < self.budget:
                    change, hypothesis, predicted = self._restart()
                    stale, previous = 0, None
                    continue
                self._stop(f"{self.patience} consecutive changes without "
                           f"improving {self.metric}")
                break

            previous = final
            change, hypothesis, predicted = propose(
                diagnosis, self.config, self._tried)
            if change is None:
                # Same reasoning as the patience branch: running out of moves
                # for a diagnosis is not running out of problem. Restart while
                # budget remains rather than handing it back.
                if len(self.runs) < self.budget:
                    change, hypothesis, predicted = self._restart()
                    stale, previous = 0, None
                    continue
                self._stop("no untried change left for this diagnosis")
                break

        return self.best

    def _reconsider(self, diagnosis, verdict, previous, final):
        """Use the LAST run to correct the diagnosis of this one.

        THE BUG THIS EXISTS FOR
        -----------------------
        `underfitting` was decided by an absolute threshold: train loss above
        0.25 and flat. On a task with 20% label noise the Bayes error keeps
        train loss near 0.5 **forever**, so the agent diagnosed underfitting on
        every single run, added capacity four times, and stalled at 0.7875
        where regularisation reaches 0.8375. Every individual diagnosis was
        defensible from its own curve.

        No single curve can distinguish "too small to fit this" from "fitting
        this is impossible". Two runs can: if adding capacity **did** reduce
        train loss and validation did **not** follow, the remaining train loss
        is irreducible noise, and the answer is regularisation rather than more
        capacity. This is the one thing here a hyperparameter search cannot do
        at all -- it has no notion of a previous experiment.
        """
        if diagnosis.name != "underfitting" or previous is None:
            return diagnosis
        train_fell = final["train_loss"] < previous["train_loss"] - 1e-4
        val_followed = final["val_loss"] < previous["val_loss"] - 1e-4
        if verdict == "confirmed" and train_fell and not val_followed:
            return Diagnosis(
                "noise_floor", "likely",
                {"train_loss": final["train_loss"],
                 "previous_train_loss": previous["train_loss"],
                 "val_loss": final["val_loss"]},
                f"train loss fell {previous['train_loss']:.4f} -> "
                f"{final['train_loss']:.4f} but val loss did not; the remaining "
                f"train loss is label noise, not missing capacity")
        return diagnosis

    def _restart(self):
        """Jump to an unexplored configuration and carry on.

        Deliberately a *random* restart rather than a cleverer one. The point
        of the restart is to escape a basin the reasoning has exhausted, and a
        reasoned jump would land near where the reasoning already is -- which
        is the thing that failed. Random search is strong precisely because it
        does not correlate its samples, and this borrows that.
        """
        self._restarts = getattr(self, "_restarts", 0) + 1
        rng = random.Random(1000 + self._restarts)
        space = {
            "hidden": [8, 16, 32, 64, 128],
            "depth": [1, 2, 3],
            "dropout": [0.0, 0.1, 0.3, 0.5],
            "lr": [3e-4, 1e-3, 3e-3, 1e-2],
            "weight_decay": [0.0, 1e-3, 1e-2],
            "epochs": [20, 40, 80],
        }
        for _ in range(30):
            candidate = {k: rng.choice(v) for k, v in space.items()}
            key = json.dumps({**task_mod.DEFAULT_CONFIG, **candidate},
                             sort_keys=True, default=str)
            if key not in self._tried:
                self.config = dict(task_mod.DEFAULT_CONFIG)
                return (candidate,
                        f"restart {self._restarts}: this region is exhausted, "
                        f"so resample rather than return unspent budget",
                        None)
        self.config = dict(task_mod.DEFAULT_CONFIG)
        return {}, f"restart {self._restarts}: no untried configuration found", None

    def _stop(self, reason):
        entry = self.log[-1]
        entry.note = (entry.note + "; " if entry.note else "") + f"stopping: {reason}"
        self.stopped_because = reason

    # -- reporting -------------------------------------------------------

    def research_log(self):
        return "\n\n".join(entry.render() for entry in self.log)

    def summary(self):
        verdicts = {}
        for entry in self.log:
            verdicts[entry.verdict] = verdicts.get(entry.verdict, 0) + 1
        diagnoses = {}
        for entry in self.log:
            diagnoses[entry.diagnosis] = diagnoses.get(entry.diagnosis, 0) + 1
        return {
            "runs": len(self.runs),
            "budget": self.budget,
            "stopped_early": len(self.runs) < self.budget,
            "stopped_because": self.stopped_because,
            "restarts": getattr(self, "_restarts", 0),
            "best_val_acc": self.best["val_acc"] if self.best else None,
            "best_val_balanced_acc": (self.best["val_balanced_acc"]
                                      if self.best else None),
            "best_config": {k: v for k, v in self.best["config"].items()
                            if k != "seed"} if self.best else None,
            "diagnoses": diagnoses,
            "verdicts": verdicts,
        }

    def to_dict(self):
        return {"summary": self.summary(),
                "log": [entry.to_dict() for entry in self.log]}


def _apply_class_weight(config):
    """`class_weight` is a flag in the config; the trainer wants a real change.

    Implemented as a much lower learning rate plus more epochs, because the MLP
    trainer in `task.py` has no class-weighted loss and inventing one here
    would be a second implementation of the same thing. Named honestly so the
    log does not claim a mechanism the code does not have.
    """
    config = dict(config)
    if config.pop("class_weight", False):
        config["lr"] = round(config["lr"] / 4, 6)
        config["epochs"] = int(config["epochs"] * 1.5)
    return config
