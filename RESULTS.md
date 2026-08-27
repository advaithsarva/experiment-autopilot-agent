# Results

Every number below has the command that produced it printed above it. Tasks
and runs are seeded and reproduce exactly.

Environment: Windows 11, Python 3.12.1, CPU-only PyTorch.

> **The headline is a loss, and it is at the top on purpose.** The reasoning
> loop does not beat random search on accuracy. §1 is that number. §2 onward is
> what it does provide, and §6 is what none of it has been tested on.

---

## 1. Against random search and grid search, at equal budget

```bash
python bench.py --seeds 8 --budget 10
```

4 tasks × 8 seeds = 32 head-to-head comparisons per method, 10 training runs
each. Selection on **validation** balanced accuracy; the number reported is
**test** balanced accuracy on a split no method ever saw.

| | test balanced acc | val balanced acc | worst case | runs used | seconds | hypotheses refuted |
|---|---|---|---|---|---|---|
| **random search** | **0.9215** | 0.9170 | 0.7675 | 10.00 | 16.01 | – |
| grid search | 0.9186 | 0.9201 | 0.7525 | 10.00 | 12.65 | – |
| autopilot | 0.9145 | 0.9186 | 0.7525 | 10.00 | **11.63** | **0.3344** |

**Random search wins, by 0.7 points.** That is the well-known AutoML result and
it survives contact with this agent: on a small search space with a generous
budget, uncorrelated sampling is extremely hard to beat, and it parallelises
perfectly while a reasoning loop is inherently sequential.

Per task:

| | separable | interacting | noisy | imbalanced |
|---|---|---|---|---|
| random | 0.9923 | 0.9391 | **0.7945** | **0.9600** |
| grid | 0.9913 | 0.9432 | 0.7918 | 0.9482 |
| **autopilot** | 0.9912 | **0.9482** | 0.7869 | 0.9315 |

The agent is ahead on **`interacting`** — the one task where the diagnosis has
real signal to act on, because the underfitting there is genuine and fixable by
exactly the change the rule proposes. It is behind on **`imbalanced`** and
**`noisy`**, which reward sampling widely over reasoning from one curve, and
that is what random search does by construction.

It is also the fastest of the three (11.63s against 16.01s), because when the
curves say capacity is not the constraint it stops buying capacity.

---

## 2. The bug that turned a near-tie into a loss

The first version of this table was worse, and for a reason worth recording:

| | test balanced acc | runs used |
|---|---|---|
| random search | 0.9215 | 10.00 |
| autopilot (**pre-fix**) | **0.9056** | **6.75** |
| autopilot (post-fix) | 0.9145 | 10.00 |

**The agent was handing back a third of its budget.** It converged, ran out of
moves for the current diagnosis, and stopped — on `imbalanced` it finished
after 6 of 10 runs at 0.9160 while random search used all 10 and reached
0.9600.

"I converged" is a statement about a local basin, not about the problem. The
fix is a **random restart** when budget remains, keeping the best across
restarts — plain multi-start optimisation, and what makes the comparison an
equal-budget one at all. Deliberately a *random* restart rather than a reasoned
one: the point is to escape a basin the reasoning has exhausted, and a reasoned
jump would land near where the reasoning already is.

**+0.9 points, and it closed most of the gap to random search.** Pinned by
`the agent never returns budget unspent`, which asserts all four tasks use the
full budget.

---

## 3. What no search can produce

```bash
python cli.py --task noisy --budget 8
```

```
step 1
  tried       {'hidden': 64}
  because     the hidden layer is too narrow to represent the decision boundary
  result      val_loss 0.5478  acc 0.7375  balanced 0.7373
  verdict     confirmed -- train loss 0.5050 -> 0.4561
  now reads   noise_floor

step 2
  tried       {'dropout': 0.3}
  because     train loss falls with capacity but validation does not follow, so
              the remaining train loss is label noise, not missing capacity
  result      val_loss 0.4994  acc 0.8175  balanced 0.8175
  verdict     confirmed -- val loss 0.5478 -> 0.4994
  now reads   underfitting
```

Read step 1 carefully: the change **worked on its own terms** -- train loss
fell from 0.5050 to 0.4561, exactly as the hypothesis predicted, so the verdict
is `confirmed` -- and validation got *worse*, 0.7875 to 0.7373. That is the
signature of a noise floor, and `now reads` records that the agent saw it. The
next step stops adding capacity and regularises instead.

**33.4% of the agent's own hypotheses are recorded as refuted.** That number
cannot be produced by a search — a search has no hypothesis to refute — and a
log in which everything is confirmed is a log nobody is reading.

The verdict is decided on **what the hypothesis predicted**, not on whether
accuracy improved. A change that raises accuracy while failing to do the thing
the hypothesis claimed is a *refuted* hypothesis with a lucky outcome, and
recording it as a confirmation is how a research log turns into a story. Pinned
by `a verdict is decided on the PREDICTION, not on accuracy`.

---

## 4. The diagnosis bug that needed two runs to see

```bash
python test_autopilot.py    # "a noise floor is distinguished from underfitting"
```

`underfitting` was decided by an absolute threshold: train loss above 0.25 and
flat. On the 20%-label-noise task the Bayes error holds train loss near 0.5
**forever**, so the agent diagnosed underfitting on every single run, added
capacity four times, and stalled:

| | val balanced acc |
|---|---|
| autopilot, pre-fix (capacity four times) | 0.7875 |
| autopilot, post-fix (regularisation) | **0.8249** |
| best configuration found by exhaustive grid | 0.8375 |

**No single curve can separate "too small to fit this" from "fitting this is
impossible."** Both look like a flat, high train loss. Two curves can: if
adding capacity *did* reduce train loss and validation did *not* follow, the
remaining train loss is irreducible noise, and the answer is regularisation
rather than more capacity.

`_reconsider` uses the previous run to correct the diagnosis of the current
one. It is the single thing in this project that a hyperparameter search
structurally cannot do, because a search has no notion of a previous
experiment — and it is worth **+3.7 points** on the task it applies to.

---

## 5. Balanced accuracy, carried from the first line

On the imbalanced task (12% positive rate) plain accuracy and balanced accuracy
disagree about which model is better, and **plain accuracy prefers the one that
has learned less** — a model predicting the majority class for every row scores
0.88 accuracy and 0.50 balanced accuracy, with every loss curve going the right
way the whole time.

So `task._evaluate` returns both from the start, selection defaults to the
balanced number, and `diagnose` checks `majority_class` **before** anything
loss-shaped:

```bash
python test_autopilot.py    # "majority-class collapse is caught, and loss curves cannot see it"
```

An autopilot optimising plain accuracy here converges confidently on a model
that has learned nothing, and every curve it looked at supports the decision.

---

## 6. Tests

```bash
python test_autopilot.py
```

```
17/17 passed

against the pre-fix implementation:
  pre-fix Autopilot._reconsider   2/2 of its tests fail
```

---

## 7. What has NOT been verified

- **This is a 12-feature synthetic tabular task and a small MLP.** These are not
  the dynamics of a large model. What generalises is the *diagnosis* —
  overfitting, underfitting, divergence and plateau are the same shapes at any
  scale — not the hyperparameters the agent lands on, and not the specific
  thresholds (`LOW_TRAIN_LOSS = 0.25` is calibrated to this loss scale and
  nothing else).
- **The search space is six knobs.** Random search is strongest exactly here;
  the case where reasoning should pay off is a large space with an expensive
  objective, and that case is untested because it is unaffordable to evaluate
  properly. Reporting a win on a space this small would be the dishonest
  version of this project.
- **`class_weight` is not a class-weighted loss.** The trainer has none, so
  `_apply_class_weight` implements it as a lower learning rate and more epochs.
  It is named honestly in the code so the log does not claim a mechanism the
  code does not have — but the hypothesis text is more confident than the
  implementation deserves.
- **No parallelism.** Random search is embarrassingly parallel and the agent is
  strictly sequential. At equal *wall clock on N machines* rather than equal
  run count, the gap would widen considerably in random search's favour.
- **Eight seeds per task.** The 0.7-point gap in §1 is not obviously outside
  seed noise, and no significance test was run. The honest reading is "the
  agent is not better", not "the agent is 0.7 points worse".
