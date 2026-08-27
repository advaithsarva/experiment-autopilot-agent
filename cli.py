"""Run the autopilot on a task and print its research log.

    python cli.py                                  # default: interacting, budget 8
    python cli.py --task noisy --budget 12
    python cli.py --task imbalanced --json         # the log, machine-readable
    python cli.py --task separable --quiet         # summary only
    python cli.py --compare --budget 10            # against random search

THE LOG IS THE OUTPUT
---------------------
The best configuration is the least interesting thing this produces -- random
search finds one too, faster, and `bench.py` says so. What it cannot produce is
the middle column: a hypothesis about what the last run showed, a prediction
that could have been wrong, and a verdict recorded either way.

`--json` emits the whole log, so a downstream consumer gets the reasoning trace
rather than just the winning config. `verdict` is the field worth reading.
"""

import argparse
import json
import sys

import autopilot as autopilot_mod
import task as task_mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=task_mod.TASKS, default="interacting")
    ap.add_argument("--budget", type=int, default=8,
                    help="training runs the agent may spend")
    ap.add_argument("--patience", type=int, default=3,
                    help="non-improving steps before a restart")
    ap.add_argument("--seed", type=int, default=1, help="which task instance")
    ap.add_argument("--metric", default="val_balanced_acc",
                    choices=("val_balanced_acc", "val_acc"),
                    help="what to select on; balanced is the honest default")
    ap.add_argument("--compare", action="store_true",
                    help="also run random search at the same budget")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    problem = task_mod.make_task(args.task, seed=args.seed)
    agent = autopilot_mod.Autopilot(problem, budget=args.budget,
                                    patience=args.patience, metric=args.metric)
    agent.run()
    summary = agent.summary()

    if args.json:
        payload = agent.to_dict()
        payload["task"] = {"kind": args.task, "seed": args.seed,
                           "positive_rate": problem["positive_rate"]}
        payload["test"] = task_mod.test_scores(agent.best["model"], problem)
        if args.compare:
            import bench
            payload["random_search"] = _compare(bench, problem, args)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"task {args.task} (positive rate {problem['positive_rate']:.3f}), "
          f"budget {args.budget} runs, selecting on {args.metric}\n")
    if not args.quiet:
        print(agent.research_log())
        print()

    scores = task_mod.test_scores(agent.best["model"], problem)
    print(f"runs used       {summary['runs']} of {summary['budget']}"
          + (f", {summary['restarts']} restart(s)" if summary["restarts"] else ""))
    print(f"stopped because {summary['stopped_because']}")
    print(f"diagnoses       " + ", ".join(
        f"{k}={v}" for k, v in summary["diagnoses"].items()))
    print(f"verdicts        " + ", ".join(
        f"{k}={v}" for k, v in summary["verdicts"].items()))
    print(f"best val        acc {summary['best_val_acc']:.4f}  "
          f"balanced {summary['best_val_balanced_acc']:.4f}")
    print(f"HELD-OUT test   acc {scores['test_acc']:.4f}  "
          f"balanced {scores['test_balanced_acc']:.4f}")
    print(f"best config     " + ", ".join(
        f"{k}={v}" for k, v in summary["best_config"].items()))

    if args.compare:
        import bench
        other = _compare(bench, problem, args)
        print(f"\nrandom search   test balanced {other['test_balanced_acc']:.4f} "
              f"in {other['runs']} runs "
              f"({'ahead' if other['test_balanced_acc'] > scores['test_balanced_acc'] else 'behind'})")
    return 0


def _compare(bench, problem, args):
    best, runs, _ = bench.random_search(problem, args.budget, args.seed)
    scores = task_mod.test_scores(best["model"], problem)
    return {"runs": len(runs), **scores,
            "config": {k: v for k, v in best["config"].items() if k != "seed"}}


if __name__ == "__main__":
    sys.exit(main())
