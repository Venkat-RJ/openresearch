"""Fixed development acceptance gate, independent of which solver wins."""

GATE = dict(min_positive_pairs=8, min_positive_pairs_per_stratum=1,
            min_valid_per_solver=11, expected_cases=12, expected_executions=24,
            semantics="No unresolved validation or input discrepancies",
            selection="Winner counts do not affect acceptance",
            failure="Do not launch final study unchanged if any gate fails")
SOLVERS = ("frozen-v2", "egrasp")


def evaluate_gate(plan, results, canary_passed):
    expected = {(x["name"], s) for x in plan for s in SOLVERS}
    keys = [(r["instance"], r["solver"]) for r in results]
    integrity = (len(plan) == 12 and len(expected) == 24 and len(keys) == 24
                 and len(set(keys)) == 24 and set(keys) == expected)
    by_key = {(r["instance"], r["solver"]): r for r in results}
    valid_counts = {s: sum(r["solver"] == s and r["status"] == "VALID_IN_TIME"
                           for r in results) for s in SOLVERS}
    positives = 0
    strata = {}
    for entry in plan:
        stratum = f"tf{entry['tightness']}-{entry['weights']}"
        strata.setdefault(stratum, 0)
        pair = [by_key.get((entry["name"], s), {}) for s in SOLVERS]
        if all(r.get("status") == "VALID_IN_TIME" and isinstance(r.get("objective"), (int, float))
               and r["objective"] > 0 for r in pair):
            positives += 1
            strata[stratum] += 1
    resolved = all(r["status"] in {"VALID_IN_TIME", "SOLVER_FAILURE"} for r in results)
    checks = dict(timing=bool(canary_passed), integrity=integrity,
                  difficulty=positives >= 8,
                  coverage=len(strata) == 6 and all(n >= 1 for n in strata.values()),
                  reliability=all(n >= 11 for n in valid_counts.values()),
                  correctness=resolved)
    return dict(passed=all(checks.values()), checks=checks,
                positive_pairs=positives, positive_pairs_by_stratum=strata,
                valid_per_solver=valid_counts,
                interpretation="Development gate only, not a statistical reliability or superiority claim")
