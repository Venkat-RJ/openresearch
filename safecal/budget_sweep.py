"""Train-only hyperparameter selection for the Bayesian experiment budget.

A predeclared grid of max probe counts is selected using only training seeds;
then the chosen setting can be evaluated once on held-out and shifted tasks.
"""
from __future__ import annotations
import argparse
import json
import statistics
from time import perf_counter

from bayes_id import policy_bayes
from safecal import SafeCalEnv


def evaluate(seed_start: int, count: int, combined: bool, novel: bool,
             max_probes: int) -> dict:
    rewards, qualities, costs = [], [], []
    for seed in range(seed_start, seed_start + count):
        env = SafeCalEnv(seed, combined=combined, novel=novel)
        cal = policy_bayes(env, max_probes=max_probes)
        r = env.finish(cal)
        rewards.append(r["reward"])
        qualities.append(r["quality"])
        costs.append(r["test_cost"])
    return {"reward": statistics.fmean(rewards),
            "quality": statistics.fmean(qualities),
            "test_cost": statistics.fmean(costs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--json", default="budget_selection.json")
    parser.add_argument("--first-probes", type=int, default=0)
    parser.add_argument("--last-probes", type=int, default=4)
    args = parser.parse_args()
    start = perf_counter()
    results = {}
    for m in range(args.first_probes, args.last_probes + 1):
        single = evaluate(1000, args.tasks, False, False, m)
        compound = evaluate(50000, args.tasks, True, False, m)
        results[str(m)] = {"train_single": single, "train_compound": compound,
                           "macro_reward": (single["reward"]+compound["reward"])/2}
        print(f"max_probes={m} training macro_reward={results[str(m)]['macro_reward']:.5f} "
              f"single={single['reward']:.5f} compound={compound['reward']:.5f} "
              f"elapsed={perf_counter()-start:.1f}s", flush=True)
    best = max(range(args.first_probes, args.last_probes + 1),
               key=lambda m: (results[str(m)]["macro_reward"], -m))
    output = {"training_selection_only": True, "n_seeds_per_group": args.tasks,
              "single_seed_base": 1000, "compound_seed_base": 50000,
              "selected_in_partial_grid": best, "results": results,
              "selection_rule": "maximize equal-weight mean net reward on training single/compound seed groups"}
    with open(args.json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"BEST IN CURRENT GRID ONLY: max_probes={best}. Saved {args.json}")

if __name__ == "__main__":
    main()
