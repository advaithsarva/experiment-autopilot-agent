# Results

Every number below comes from the command printed above it. Tasks and runs are seeded and reproduce exactly.

Environment: Windows 11, Python 3.12.1, CPU-only PyTorch.

> **The headline is a loss, and it is at the top on purpose.** The reasoning loop does not beat random search on accuracy. §1 gives that result. §2 onward shows what the agent does provide, and §6 states what has not been tested.

---

## 1. Against random search and grid search, at equal budget

```bash
python bench.py --seeds 8 --budget 10
```

4 tasks × 8 seeds = 32 head-to-head comparisons per method, with 10 training runs each. Selection is based on **validation** balanced accuracy; the number reported below is **test** balanced accuracy on a split no method ever saw.

|                   | test balanced acc | val balanced acc | worst case | runs used | seconds   | hypotheses refuted |
| ----------------- | ----------------- | ---------------- | ---------- | --------- | --------- | ------------------ |
| **random search** | **0.9215**        | 0.9170           | 0.7675     | 10.00     | 16.01     | –                  |
| grid search       | 0.9186            | 0.9201           | 0.7525     | 10.00     | 12.65     | –                  |
| autopilot         | 0.9145            | 0.9186           | 0.7525     | 10.00     | **11.63** | **0.3344**         |

**Random search wins by 0.7 points.** On a small search space with a generous budget, uncorrelated sampling is difficult to beat. It also parallelises naturally, while the reasoning loop is sequential.

Per task:

|               | separable | interacting | noisy      | imbalanced |
| ------------- | --------- | ----------- | ---------- | ---------- |
| random        | 0.9923    | 0.9391      | **0.7945** | **0.9600** |
| grid          | 0.9913    | 0.9432      | 0.7918     | 0.9482     |
| **autopilot** | 0.9912    | **0.9482**  | 0.7869     | 0.9315     |

The agent is ahead on **`interacting`**, the one task where the diagnosis has enough signal to act on: the underfitting is genuine, and the rule proposes the change that addresses it.

It is behind on **`imbalanced`** and **`noisy`**, where broad sampling is more useful than reasoning from a single curve. That is exactly what random search is designed to do.

The agent is also the fastest of the three: **11.63s vs 16.01s** for random search. When the curves indicate that capacity is not the constraint, it stops spending runs on capacity.

---

## 2. The bug that turned a near-tie into a loss

The first version of this table was worse:

|                         | test balanced acc | runs used |
| ----------------------- | ----------------- | --------- |
| random search           | 0.9215            | 10.00     |
| autopilot (**pre-fix**) | **0.9056**        | **6.75**  |
| autopilot (post-fix)    | 0.9145            | 10.00     |

**The agent was giving back a third of its budget.**

It would converge, exhaust the moves available for the current diagnosis, and stop. On `imbalanced`, it finished after 6 of 10 runs at 0.9160, while random search used all 10 and reached 0.9600.

"I converged" describes a local basin, not the problem as a whole.

The fix is a **random restart** when budget remains, keeping the best result across restarts. This is ordinary multi-start optimisation, and it makes the comparison genuinely equal-budget.

The restart is deliberately *random*. Once the reasoning has exhausted the current basin, another reasoned jump would tend to land near the same region. Randomness gives the search a chance to leave it.

**+0.9 points, and most of the gap to random search closes.**

The invariant is pinned by `the agent never returns budget unspent`, which requires all four tasks to use the full budget.

---

## 3. What no search can produce

```bash
python cli.py --task noisy --budget 8
```

```text
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

Step 1 is worth looking at closely. The change **worked on the specific prediction it made**: train loss fell from 0.5050 to 0.4561. The hypothesis was therefore confirmed.

Validation, however, got worse: 0.7875 → 0.7373.

That combination is what leads the agent to read the situation as a noise floor. The next step stops adding capacity and regularises instead.

**33.4% of the agent's hypotheses are recorded as refuted.**

A search can report which configurations performed better or worse, but it has no hypothesis to refute. The point of the log is that the agent makes an explicit prediction and then records whether the observed change matched it.

The verdict is based on **what the hypothesis predicted**, not simply on whether accuracy improved. If accuracy increases but the predicted mechanism does not occur, the hypothesis is still refuted. Otherwise the experiment log becomes a collection of post-hoc explanations.

Pinned by `a verdict is decided on the PREDICTION, not on accuracy`.

---

## 4. The diagnosis bug that needed two runs to see

```bash
python test_autopilot.py   # "a noise floor is distinguished from underfitting"
```

`underfitting` was originally decided using an absolute threshold: train loss above 0.25 and flat.

On the 20%-label-noise task, Bayes error keeps train loss near 0.5, so the agent diagnosed underfitting on every run, added capacity four times, and stalled:

|                                             | val balanced acc |
| ------------------------------------------- | ---------------- |
| autopilot, pre-fix (capacity four times)    | 0.7875           |
| autopilot, post-fix (regularisation)        | **0.8249**       |
| best configuration found by exhaustive grid | 0.8375           |

**No single curve can distinguish "too small to fit this" from "fitting this is impossible."**

Two curves can provide that evidence.

If adding capacity **does** reduce train loss but validation does **not** follow, the remaining train loss is more likely to be irreducible noise. The appropriate response is then regularisation rather than more capacity.

`_reconsider` uses the previous run to update the diagnosis of the current one. That is the main structural difference from a conventional hyperparameter search: the search evaluates configurations, while this loop treats the previous experiment as evidence for the next decision.

On the task where it applies, this is worth **+3.7 points**.

---

## 5. Balanced accuracy, carried from the first line

On the imbalanced task (12% positive rate), plain accuracy and balanced accuracy disagree about which model is better.

A model that predicts the majority class for every row gets **0.88 accuracy and 0.50 balanced accuracy**, even though its loss curves can look healthy throughout training.

So `task._evaluate` returns both metrics from the start. Selection defaults to balanced accuracy, and `diagnose` checks `majority_class` **before** looking at anything loss-shaped:

```bash
python test_autopilot.py   # "majority-class collapse is caught, and loss curves cannot see it"
```

An autopilot optimising plain accuracy can therefore converge on a model that has learned nothing useful, while every curve it inspected appears consistent with the decision.

---

## 6. Tests

```bash
python test_autopilot.py
```

```text
17/17 passed

against the pre-fix implementation:
  pre-fix Autopilot._reconsider   2/2 of its tests fail
```

---

## 7. What has NOT been verified

* **This is a 12-feature synthetic tabular task and a small MLP.** These are not the dynamics of a large model. What may generalise is the *diagnosis* — overfitting, underfitting, divergence and plateau have recognisable shapes at different scales. The hyperparameters the agent chooses do not automatically generalise, and neither do the specific thresholds (`LOW_TRAIN_LOSS = 0.25` is calibrated to this loss scale).

* **The search space is six knobs.** Random search is particularly strong in a space this small. The case where reasoning might have more room to help is a large search space with an expensive objective. That case has not been tested because evaluating it properly is currently unaffordable. Claiming a win here would be misleading.

* **`class_weight` is not a class-weighted loss.** The trainer has no class-weighted loss, so `_apply_class_weight` implements the intervention as a lower learning rate and more epochs. The code names this honestly so the log does not claim a mechanism the trainer does not implement. The hypothesis text, however, is more confident than the implementation warrants.

* **No parallelism.** Random search is embarrassingly parallel; the agent is strictly sequential. At equal wall-clock time on N machines rather than equal run count, random search would have a substantial advantage.

* **Eight seeds per task.** The 0.7-point gap in §1 is not clearly outside seed noise, and no significance test was run. The defensible conclusion is **"the agent is not better,"** not **"the agent is 0.7 points worse."**
