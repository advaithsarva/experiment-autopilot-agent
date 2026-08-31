# Agentic Experiment Autopilot

An ML research loop that runs experiments, reads the training curves, identifies a likely failure mode, changes one hyperparameter based on an explicit hypothesis, and then records whether the hypothesis was actually supported by the next run.

**It does not beat random search on accuracy, and this README says that upfront.**

Across 4 tasks × 8 seeds with the same 10-run budget, test balanced accuracy was **0.9145 for the autopilot versus 0.9215 for random search**. That puts it 0.7 points behind, although it was 27% faster. More importantly, it produces something a conventional search does not: a research log where **33.4% of its own hypotheses are recorded as refuted**.

Full results are in [RESULTS.md](RESULTS.md).

```bash
python cli.py --task interacting --budget 10    # watch it reason

python cli.py --task noisy --json                # machine-readable research log

python test_autopilot.py

python bench.py --seeds 8 --budget 10            # compare with random and grid search
```

---

## The invariant

> **Every configuration the agent tries is a named hypothesis about the previous run. The log records the hypothesis, the expected result, the change made, the actual result, and whether the hypothesis was right — including when it was wrong.**

That is the main distinction from a search procedure.

Random search can find good configurations, but it does not know whether the previous model was overfitting, why a particular configuration was selected, or what was learned from a failed experiment.

`Entry.verdict` can be `confirmed`, `refuted`, or `inconclusive`. The verdict comes from comparing the prediction made by the hypothesis with what actually happened, not from checking whether accuracy improved.

A configuration can improve accuracy and still produce a **refuted** hypothesis if it failed to produce the effect that was predicted. Treating every accuracy improvement as confirmation would turn an experiment log into a retrospective story rather than a record of what was actually learned.

---

## Diagnosis is based on curve shape

The agent identifies six failure modes from the training curves alone. It does not receive information about the task or configuration when making the diagnosis.

| Diagnosis        | Shape                                                                           |
| ---------------- | ------------------------------------------------------------------------------- |
| `diverging`      | Loss is rising or becomes NaN                                                   |
| `majority_class` | High accuracy while balanced accuracy stays near 0.5                            |
| `overfitting`    | Training loss keeps falling while validation loss rises above its previous best |
| `underfitting`   | Training loss remains high and mostly flat                                      |
| `noise_floor`    | Training loss falls as capacity increases, but validation loss does not follow  |
| `converged`      | Curves flatten and training and validation agree                                |

`majority_class` is particularly difficult for an autopilot that looks at only one metric. A model can report 0.88 accuracy while predicting the same class for almost every example.

For that reason, balanced accuracy is tracked alongside ordinary accuracy from the beginning of `task.py`, and configuration selection uses the balanced-accuracy score.

---

## Three bugs, and the third is the interesting one

### 1. `converged` was treated as a reason to stop

Convergence describes the result of **one run**, not the state of the entire problem.

The agent originally stopped after a single run on three of the four tasks. On `interacting`, it stopped at **0.9355**, even though **0.9698** was reachable.

Once a model has stopped improving, that can be the point where a larger model, different regularisation, or another configuration should be tested. Convergence does not by itself mean that the search is finished.

### 2. An absolute training-loss threshold cannot distinguish underfitting from noise

On the task with 20% label noise, Bayes error keeps the training loss around 0.5. The agent therefore classified every run as `underfitting`, increased model capacity four times, and stalled at **0.7875**. Regularisation instead reaches **0.8375**.

Each individual diagnosis looked reasonable when viewed only through its own curve.

The problem is that **one curve cannot distinguish "the model is too small" from "this loss cannot be reduced further."**

Two curves provide the missing information. If increasing capacity reduces training loss while validation loss does not improve, the remaining training loss is more likely to be irreducible noise. The productive move is then regularisation rather than additional capacity.

`_reconsider` uses the previous experiment to revise the interpretation of the current one. That is difficult for a conventional hyperparameter search to reproduce because the search does not maintain a hypothesis about what the previous experiment was supposed to demonstrate.

### 3. The agent left part of its budget unused

The original loop could converge, exhaust its available moves, and stop even when runs remained. In one benchmark it used only **6 of the available 10 runs**, reaching **0.9056** compared with **0.9215** for random search.

Adding a random restart whenever budget remained raised the score to **0.9145** and ensured that every task used its full budget.

The lesson is straightforward: an adaptive strategy should not give back unused experiments when those experiments could still provide information. Likewise, "I converged" describes a local training result, not necessarily the whole search problem.

---

## What the comparison actually shows

|               | Test balanced acc | Runs used |   Seconds | Hypotheses refuted |
| ------------- | ----------------: | --------: | --------: | -----------------: |
| Random search |        **0.9215** |     10.00 |     16.01 |                  – |
| Grid search   |            0.9186 |     10.00 |     12.65 |                  – |
| **Autopilot** |            0.9145 |     10.00 | **11.63** |          **0.334** |

**Random search wins on accuracy.**

That result is not hidden by the README. On a small search space with a reasonable budget, random sampling is difficult to beat. It also parallelises naturally, while a reasoning loop is sequential because each experiment depends on the previous one.

The autopilot does better in a few specific areas:

* **`interacting`: 0.9482 vs. 0.9391.** This is the task where the diagnosis provides useful information because the underfitting is genuine and can be addressed by changing capacity.

* **Fastest of the three.** The agent tends to select smaller configurations when the curves indicate that model capacity is not the limiting factor.

* **It records an explanation for every run.** Each experiment has a hypothesis, a prediction, and a verdict.

It performs worse on `imbalanced` (**0.9315 vs. 0.9600**) and `noisy` (**0.7869 vs. 0.7945**). These are cases where broad sampling is more useful than reasoning from a single previous experiment, which is exactly where random search has an advantage.

---

## Why the experiments are small

An agent that reasons about curves needs to see a large number of experiments. If each iteration takes 20 minutes, evaluating the approach becomes impractical: a single comparison takes hours of compute, repeated experiments become expensive, and the final result may represent only one sample of a stochastic process.

The benchmark therefore uses a synthetic tabular problem with an MLP that trains in roughly one second.

That makes **96 head-to-head comparisons** possible.

The tradeoff is important. These experiments do not represent the training dynamics of a large model, and that limitation is documented in `RESULTS.md`.

What should generalise is the **shape-based diagnosis**: overfitting, underfitting, divergence, and plateaus are observable training behaviours at different scales. The specific hyperparameters selected by this experiment should not be treated as generally optimal.

---

## Files

| File                | What is in it                                                          |
| ------------------- | ---------------------------------------------------------------------- |
| `task.py`           | Four synthetic tasks with known, planted failure modes and the trainer |
| `autopilot.py`      | Diagnosis, mutation table, verdicts, experiment loop, and restarts     |
| `bench.py`          | Autopilot vs. random vs. grid search at the same run budget            |
| `cli.py`            | Runs the loop, prints the research log, and supports `--json`          |
| `test_autopilot.py` | Tests for each real bug plus the main invariant                        |

```bash
pip install torch       # CPU is sufficient and was used for every reported result
```
