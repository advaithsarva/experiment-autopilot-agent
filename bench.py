"""The autopilot against random search and grid search, at the same budget.

    python bench.py
    python bench.py --seeds 12 --budget 10 --json
    python bench.py --task interacting

THE BASELINE THAT CAN WIN, AND OFTEN DOES
-----------------------------------------
Random search over a sensible range is the standard against which every AutoML
claim should be measured, and it is *strong*. It parallelises perfectly, it has
no failure modes, and on a small search space it is hard to beat. If the
reasoning loop cannot beat it on equal compute, the reasoning is decoration and
this file is where that gets said.

**Equal compute means equal number of training runs**, not equal wall clock.
The agent chooses its own epoch counts, so it can spend more or less time per
run; `seconds` is reported alongside so that trade is visible rather than
hidden inside the win.

WHAT IS SCORED, AND ON WHICH SPLIT
----------------------------------
Selection happens on **validation balanced accuracy**, for every method.
The number reported is **test** balanced accuracy, computed once, at the end,
on a split no method ever saw. Selecting and reporting on the same split
measures a method's ability to overfit a split, which random search is
extremely good at and which is not a virtue.

And a column no search can fill: `refuted`, the fraction of the agent's own
hypotheses that its next experiment disproved.
"""

import argparse
import json
import random
import statistics
import time

import autopilot as autopilot_mod
import task as task_mod

# The range random and grid search draw from. Deliberately the same knobs the
# agent can change, so the comparison is about the search strategy and not
# about who was given a better space.
SPACE = {
    "hidden": [8, 16, 32, 64, 128],
    "depth": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.3, 0.5],
    "lr": [3e-4, 1e-3, 3e-3, 1e-2],
    "weight_decay": [0.0, 1e-3, 1e-2],
    "epochs": [20, 40, 80],
}

METRIC = "val_balanced_acc"


def random_search(task, budget, seed):
    rng = random.Random(seed)
    best, runs = None, []
    for _ in range(budget):
        config = {k: rng.choice(v) for k, v in SPACE.items()}
        config["seed"] = 0
        result = task_mod.train_once(task, config)
        runs.append(result)
        if best is None or result[METRIC] > best[METRIC]:
            best = result
    return best, runs, {}


def grid_search(task, budget, seed):
    """A coarse grid, truncated at the budget. The honest cheap baseline.

    Ordered so the first `budget` entries are a spread rather than all the
    small models -- taking a grid in nested-loop order and cutting it short is
    a straw man, and the point is a baseline that can win.
    """
    import itertools
    keys = list(SPACE)
    combinations = list(itertools.product(*(SPACE[k] for k in keys)))
    rng = random.Random(0)          # fixed: the grid's order is not the variable
    rng.shuffle(combinations)
    best, runs = None, []
    for values in combinations[:budget]:
        config = dict(zip(keys, values))
        config["seed"] = 0
        result = task_mod.train_once(task, config)
        runs.append(result)
        if best is None or result[METRIC] > best[METRIC]:
            best = result
    return best, runs, {}


def agent_search(task, budget, seed):
    agent = autopilot_mod.Autopilot(task, budget=budget, metric=METRIC)
    best = agent.run()
    verdicts = agent.summary()["verdicts"]
    total = sum(verdicts.values()) or 1
    return best, agent.runs, {
        "refuted": verdicts.get("refuted", 0) / total,
        "confirmed": verdicts.get("confirmed", 0) / total,
        "diagnoses": agent.summary()["diagnoses"],
        "stopped_because": agent.summary()["stopped_because"],
    }


METHODS = {"random": random_search, "grid": grid_search, "agent": agent_search}


def run(kinds, seeds, budget):
    rows = {name: [] for name in METHODS}
    for kind in kinds:
        for seed in range(1, seeds + 1):
            problem = task_mod.make_task(kind, seed=seed)
            for name, fn in METHODS.items():
                start = time.perf_counter()
                best, runs, extra = fn(problem, budget, seed)
                elapsed = time.perf_counter() - start
                scores = task_mod.test_scores(best["model"], problem)
                rows[name].append({
                    "task": kind, "seed": seed,
                    "val": best[METRIC],
                    "test": scores["test_balanced_acc"],
                    "test_acc": scores["test_acc"],
                    "runs": len(runs), "seconds": elapsed, **extra})
    return rows


def aggregate(scores):
    out = {
        "test_balanced_acc": round(statistics.mean(
            [s["test"] for s in scores]), 4),
        "val_balanced_acc": round(statistics.mean([s["val"] for s in scores]), 4),
        "runs_used": round(statistics.mean([s["runs"] for s in scores]), 2),
        "seconds": round(statistics.mean([s["seconds"] for s in scores]), 2),
        "worst_test": round(min(s["test"] for s in scores), 4),
    }
    if "refuted" in scores[0]:
        out["hypotheses_refuted"] = round(statistics.mean(
            [s["refuted"] for s in scores]), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--task", choices=task_mod.TASKS,
                    help="one task instead of all four")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    kinds = [args.task] if args.task else list(task_mod.TASKS)
    rows = run(kinds, args.seeds, args.budget)

    report = {"tasks": kinds, "seeds": args.seeds, "budget": args.budget,
              "overall": {name: aggregate(scores) for name, scores in rows.items()},
              "by_task": {}}
    for kind in kinds:
        report["by_task"][kind] = {
            name: aggregate([s for s in scores if s["task"] == kind])
            for name, scores in rows.items()}

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"{len(kinds)} tasks x {args.seeds} seeds, budget {args.budget} "
          f"training runs per method\n")
    print(f"{'':<10}{'test bal':>10}{'val bal':>10}{'worst':>9}{'runs':>8}"
          f"{'sec':>8}{'refuted':>9}")
    for name in METHODS:
        row = report["overall"][name]
        refuted = row.get("hypotheses_refuted")
        print(f"{name:<10}{row['test_balanced_acc']:>10.4f}"
              f"{row['val_balanced_acc']:>10.4f}{row['worst_test']:>9.4f}"
              f"{row['runs_used']:>8.2f}{row['seconds']:>8.2f}"
              f"{('-' if refuted is None else f'{refuted:.4f}'):>9}")

    print("\ntest balanced accuracy by task")
    print(f"{'':<10}" + "".join(f"{k:>14}" for k in kinds))
    for name in METHODS:
        print(f"{name:<10}" + "".join(
            f"{report['by_task'][k][name]['test_balanced_acc']:>14.4f}"
            for k in kinds))
    print("\nruns actually used (the agent stops early; the searches do not)")
    for name in METHODS:
        print(f"{name:<10}" + "".join(
            f"{report['by_task'][k][name]['runs_used']:>14.2f}" for k in kinds))


if __name__ == "__main__":
    main()
