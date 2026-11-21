"""Deterministic CPU benchmark for SafeCal diagnostic policies."""
import argparse
import json
import statistics
import random
from collections import defaultdict
from safecal import SafeCalEnv, policy_calibration, plant_for_seed, oracle_quality
from bayes_id import policy_bayes

POLICIES = ("default", "random", "fixed", "greedy", "bayes", "bayes2")


def run(count: int = 60, split: str = "heldout", policies: tuple[str, ...] = POLICIES) -> dict:
    combined = split == "combined"
    novel = split == "shifted"
    seed_base = 900000 if split == "heldout" else 1200000 if combined else 1500000 if novel else 1000
    results = defaultdict(list)
    per_family = defaultdict(lambda: defaultdict(list))
    examples = []
    seed_rows = []
    for i in range(count):
        seed = seed_base + i
        plant, family = plant_for_seed(seed, combined=combined, novel=novel)
        results["oracle"].append(oracle_quality(plant))
        seed_row = {"seed": seed, "family": family, "oracle_quality": results["oracle"][-1], "policies": {}}
        for policy in policies:
            env = SafeCalEnv(seed, combined=combined, novel=novel)
            cal = (policy_bayes(env, max_probes=2) if policy == "bayes2"
                   else policy_bayes(env) if policy == "bayes"
                   else policy_calibration(env, policy))
            outcome = env.finish(cal)
            results[policy].append(outcome["reward"])
            results[policy + "_quality"].append(outcome["quality"])
            results[policy + "_cost"].append(outcome["test_cost"])
            per_family[family][policy].append(outcome["reward"])
            seed_row["policies"][policy] = {"reward": outcome["reward"], "quality": outcome["quality"], "cost": outcome["test_cost"], "calibration": vars(cal), "probes": [r["probe"] for r in env.records]}
            if i == 0 and policy in ("fixed", "greedy"):
                examples.append({"policy":policy, "probes": [r["probe"] for r in env.records],
                                 "calibration": vars(cal), "outcome": outcome})
        seed_rows.append(seed_row)
    summary = {p: round(statistics.fmean(scores), 5) for p, scores in results.items()}
    rng = random.Random(414)
    intervals = {}
    for policy in policies:
        if policy == "default":
            continue
        pairwise_diffs = [a-b for a, b in zip(results[policy], results["default"])]
        draws = [statistics.fmean(pairwise_diffs[rng.randrange(count)] for _ in range(count))
                 for _ in range(2000)]
        draws.sort()
        intervals[f"{policy}_minus_default"] = {"mean": round(statistics.fmean(pairwise_diffs), 5),
            "bootstrap_95": [round(draws[49], 5), round(draws[1949], 5)]}
    summary["oracle"] = round(summary["oracle"], 5)
    return {"split": split, "tasks": count, "policies": list(policies), "metric": "mean terminal reward (higher is better)",
            "summary": summary, "per_family_reward": {family: {name: round(statistics.fmean(arr), 5)
                    for name, arr in policies.items()} for family,policies in per_family.items()},
            "example_diagnostics": examples, "paired_differences": intervals, "seed_rows": seed_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--split", choices=["train", "heldout", "combined", "shifted"], default="heldout")
    parser.add_argument("--json", type=str, default="")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    args = parser.parse_args()
    report = run(args.tasks, args.split, tuple(dict.fromkeys(["default", *args.policies])))
    print(f"SafeCal v0 | {args.split} | {args.tasks} tasks | higher reward is better")
    print("policy        reward       quality      test_cost")
    for name in report["policies"]:
        s = report["summary"]
        print(f"{name:10}  {s[name]:>9.5f}    {s[name+'_quality']:>9.5f}    {s[name+'_cost']:>9.3f}")
    print(f"privileged oracle quality: {report['summary']['oracle']:.5f} (not a deployable baseline)")
    for key, value in report["paired_differences"].items():
        print(f"{key}: {value['mean']:+.5f}  bootstrap 95% [{value['bootstrap_95'][0]:+.5f}, {value['bootstrap_95'][1]:+.5f}]")
    if args.json:
        with open(args.json, "w", encoding="utf8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"Saved detailed results: {args.json}")

if __name__ == "__main__":
    main()
