# Agentic Experiment Autopilot

An ML research loop that runs itself: it trains a model, **reads the curves**,
names the failure mode, changes one hyperparameter as a stated hypothesis about
what it just saw, and records whether that hypothesis turned out to be right.

**It does not beat random search on accuracy, and this README leads with that.**
Across 4 tasks × 8 seeds at an equal 10-run budget: test balanced accuracy
**0.9145 against random search's 0.9215** — 0.7 points behind, while being 27%
faster and producing something no search can, a research log in which **33.4%
of its own hypotheses are recorded as refuted**. Full numbers in
[RESULTS.md](RESULTS.md).

```bash
python cli.py --task interacting --budget 10   # watch it reason
python cli.py --task noisy --json              # the research log, machine-readable
python test_autopilot.py
python bench.py --seeds 8 --budget 10          # against random and grid search
```

---

## The invariant

> **Every configuration the agent tries is a named hypothesis about the
> previous run, and the log records the hypothesis, the change, the result, and
> whether the hypothesis was right — including when it was wrong.**

That is the whole difference from a search. Random search also finds good
configurations; it cannot tell you the model is overfitting, it cannot explain
why it tried what it tried, and it learns nothing from a run that failed.

`Entry.verdict` is `confirmed`, `refuted` or `inconclusive`, decided by
comparing what the hypothesis *predicted would happen* against what did — not
against whether accuracy improved. A change that improves accuracy while
failing to do the thing the hypothesis claimed is a **refuted** hypothesis with
a lucky outcome, and recording it as a confirmation is how a research log turns
into a story.

---

## Diagnosis is a shape, not a threshold

Six failure modes, each read off the curves alone — the agent sees exactly what
a person reading a dashboard sees, and nothing about the task or the config:

| diagnosis | shape |
|---|---|
| `diverging` | loss rising or NaN |
| `majority_class` | high accuracy, balanced accuracy near 0.5 |
| `overfitting` | train loss falling, val loss risen above its own best |
| `underfitting` | train loss high and flat |
| `noise_floor` | train loss fell with capacity, **val loss did not follow** |
| `converged` | flat, and train and val agree |

`majority_class` is the one a single-metric autopilot cannot see at all:
accuracy 0.88, every curve going the right way, and the model predicts one
class for everything. Balanced accuracy is carried alongside plain accuracy
from the first line of `task.py` for exactly this reason, and selection is on
the balanced number.

---

## Three bugs, and the third is the interesting one

**1. "Converged" was treated as a reason to stop.** It is a property of one
*run*, not of the problem. The agent stopped after a single run on three of
four tasks, leaving 0.9355 on the interacting task where 0.9698 was reachable.
A run that has finished learning is precisely the moment to ask whether a
bigger model would have more to learn.

**2. An absolute train-loss threshold cannot tell underfitting from noise.** On
a task with 20% label noise the Bayes error holds train loss near 0.5
*forever*, so the agent diagnosed `underfitting` on every run, added capacity
four times, and stalled at 0.7875 where regularisation reaches 0.8375. Every
individual diagnosis was defensible from its own curve.

**No single curve can separate "too small to fit this" from "fitting this is
impossible". Two curves can.** If adding capacity *did* reduce train loss and
validation did *not* follow, the remaining train loss is irreducible noise and
the answer is regularisation. `_reconsider` uses the previous run to correct
the diagnosis of the current one — and it is the one thing here that a
hyperparameter search structurally cannot do, because it has no notion of a
previous experiment.

**3. It returned budget unspent, and lost because of it.** The agent converged,
ran out of moves, and stopped with 4 of 10 runs unused — scoring 0.9056 against
random search's 0.9215. Adding a random restart when budget remains took it to
**0.9145 and full budget use on every task**.

That is the same lesson twice over: *an adaptive strategy that hands back
budget is strictly worse than one that does not, and "I converged" is a
statement about a local basin, not about the problem.*

---

## What the honest comparison says

| | test balanced acc | runs used | seconds | hypotheses refuted |
|---|---|---|---|---|
| random search | **0.9215** | 10.00 | 16.01 | – |
| grid search | 0.9186 | 10.00 | 12.65 | – |
| **autopilot** | 0.9145 | 10.00 | **11.63** | **0.334** |

**Random search wins on accuracy.** That is the well-known AutoML result and it
survives contact with this agent: on a small search space with a generous
budget, uncorrelated sampling is very hard to beat, and it parallelises
perfectly while a reasoning loop is inherently sequential.

Where the agent is ahead:

- **`interacting`: 0.9482 vs 0.9391** — the one task where diagnosis has real
  signal to act on, because underfitting there is genuine and fixable.
- **Fastest of the three**, because it chooses smaller configurations when the
  curves say capacity is not the constraint.
- **It can explain itself.** Every run in the log has a hypothesis, a
  prediction and a verdict.

Where it is behind: `imbalanced` (0.9315 vs 0.9600) and `noisy` (0.7869 vs
0.7945) — tasks where the productive move is to sample widely rather than to
reason from one curve, which is exactly what random search does by
construction.

---

## Why the runs are small

An agent that reasons about curves needs to see many curves. A loop costing 20
minutes an iteration cannot be evaluated at all — one comparison is a day of
compute, nobody re-runs it, and the write-up reports a single sample of a
stochastic process.

So the task is a synthetic tabular problem and an MLP that trains in about a
second. What that buys is **96 head-to-head comparisons**. What it costs is
stated in RESULTS.md: these are not the dynamics of a large model. What
generalises is the *diagnosis* — overfitting, underfitting, divergence and
plateau are the same shapes at any scale — not the hyperparameters it lands on.

---

## Files

| file | what is in it |
|---|---|
| `task.py` | four synthetic tasks with planted, known failure modes; the trainer |
| `autopilot.py` | diagnosis, the mutation table, verdicts, the loop, restarts |
| `bench.py` | autopilot vs random vs grid at equal run budget |
| `cli.py` | run the loop, print the research log, `--json` |
| `test_autopilot.py` | one test per real bug, plus the invariant |

```bash
pip install torch      # CPU is fine and is what every number here used
```
