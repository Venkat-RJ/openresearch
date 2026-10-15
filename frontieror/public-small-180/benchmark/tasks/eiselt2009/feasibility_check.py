#!/usr/bin/env python3
"""
Feasibility checker for the Competitive p-Hub Location Problem
with Attraction Functions.

Reference: Eiselt & Marianov, Computers & Operations Research 36 (2009) 3128-3135.

Original solution structure (per solution_schema.json):
  { "objective_value": <float>, "hub_locations": <list[int], length p> }

The original problem decision is "which p nodes to open as hubs". The
auxiliary variables w_{kl} (Eqs 4-5) are a McCormick-style linearization
of the bilinear term y_k * y_l and are not part of the original solution
structure: given hub_locations, w_{kl} is fully determined as
1[k in hubs] * 1[l in hubs]. Reconstructing w from hub_locations would
make Eqs (4), (5), and the w-portion of Eq (7) vacuously satisfied, so
those reformulation artifacts are not verified here.

The original problem's hard constraints reduce to:
  Constraint 3 (Eq 6): sum_k y_k = p
  Constraint 4 (Eq 7, y portion): y_k in {0, 1}, i.e. hub_locations is a
                                  set of distinct integers in [0, n-1].

  Constraint 5 (objective consistency -- Tier C defense): the reported
                objective_value must lie inside the feasible objective
                envelope [obj_min, obj_max] implied by hub_locations.
                See the envelope derivation below.

------------------------------------------------------------------------
Constraint 5 -- objective-consistency (envelope) check
------------------------------------------------------------------------
The eval pipeline trusts the program's self-reported objective_value.
LLM-evolved candidates can exploit this by fabricating an objective that
passes the constraint check but lies about the captured demand. This
check recomputes the objective range from hub_locations and rejects a
reported value outside it.

The objective (Eq 3) is
    z = sum_{i,j} h_ij * N_ij / (N_ij + C_ij)
where N_ij is the entering airline's total route utility and C_ij the
fixed competitor utility. N_ij has two parts:
  * 1-hub routes (k == l): ALWAYS active for every open hub -- there is
    no w-style toggle for them (the market-share constraint uses y_k
    directly). They contribute N1_ij.
  * 2-hub routes (k < l): active only when w_{kl} = 1, and w_{kl} is a
    free variable bounded by w_{kl} <= min(y_k, y_l). They contribute
    at most N2_ij (all hub pairs activated) and at least 0.

z is monotone increasing in N_ij, so for a fixed hub set the feasible
objective range of model P is exactly
    obj_min = sum_{i,j} h_ij * N1_ij / (N1_ij + C_ij)          (all w = 0)
    obj_max = sum_{i,j} h_ij * (N1_ij+N2_ij) / (N1_ij+N2_ij+C_ij)  (all w = 1)
Any feasible (y, w) of model P -- including Gurobi time-limit incumbents
that did not fully activate w -- yields an objective inside this exact
envelope. A reported value outside [obj_min, obj_max] (beyond a small
relative margin that absorbs benign formula-interpretation differences)
is provably inconsistent with the reported hub_locations.

The utility / travel-time / fare formulas below mirror gurobi_code.py
(the canonical reference implementation) exactly.
"""

import argparse
import json

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('hub_locations', 'objective_value')
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = ()
_FRONTIEROR_DERIVED_FIELD_RULES = ()
_FRONTIEROR_CONDITIONAL_FIELD_RULES = ()


def _frontieror_validate_solution_contract(solution):
    """Return a structured rejection when the public solution contract is absent."""
    violations = []
    if not isinstance(solution, dict):
        violations.append("solution must be a JSON object")
    else:
        missing = [
            field
            for field in _FRONTIEROR_REQUIRED_SOLUTION_FIELDS
            if field not in solution or solution[field] is None
        ]
        if missing:
            violations.append(
                "missing required solution field(s): " + ", ".join(missing)
            )

        for discriminator, expected_value, fields in _FRONTIEROR_CONDITIONAL_FIELD_RULES:
            if solution.get(discriminator) != expected_value:
                continue
            conditional_missing = [
                field
                for field in fields
                if field not in solution or solution[field] is None
            ]
            if conditional_missing:
                violations.append(
                    "missing conditionally required solution field(s): "
                    + ", ".join(conditional_missing)
                )

        empty = [
            field
            for field in _FRONTIEROR_NONEMPTY_SOLUTION_FIELDS
            if field in solution and solution[field] in (None, [], {})
        ]
        if empty:
            violations.append(
                "empty required decision witness field(s): " + ", ".join(empty)
            )

        for target, operation, source, source_key in _FRONTIEROR_DERIVED_FIELD_RULES:
            try:
                if operation == "assignment_key_set":
                    reported_set = set(solution[target])
                    expected_set = {entry[source_key] for entry in solution[source]}
                    if reported_set != expected_set:
                        violations.append(
                            f"{target}={sorted(reported_set)} does not match the "
                            f"decision-derived value {sorted(expected_set)}"
                        )
                    continue

                reported = float(solution[target])
                if operation == "length":
                    expected = float(len(solution[source]))
                elif operation == "mapping_value":
                    expected = float(solution[source][source_key])
                else:
                    raise ValueError("unknown derived-field operation")
            except (KeyError, TypeError, ValueError, OverflowError):
                violations.append(
                    f"cannot derive {target} from decision witness field {source}"
                )
                continue
            if (
                reported != reported
                or expected != expected
                or reported in (float("inf"), float("-inf"))
                or expected in (float("inf"), float("-inf"))
                or abs(reported - expected) > 1e-6
            ):
                violations.append(
                    f"{target}={reported} does not match the decision-derived "
                    f"value {expected}"
                )

        if "objective_value" in _FRONTIEROR_REQUIRED_SOLUTION_FIELDS:
            objective = solution.get("objective_value")
            if (
                isinstance(objective, bool)
                or not isinstance(objective, (int, float))
                or objective != objective
                or objective in (float("inf"), float("-inf"))
            ):
                violations.append("objective_value must be a finite JSON number")

    if not violations:
        return None
    return {
        "feasible": False,
        "violated_constraints": ["solution_schema"],
        "violations": violations,
        "violation_magnitudes": [],
    }


def _frontieror_write_unexpected_rejection(error):
    """Turn malformed candidate/checker exceptions into a normal rejection."""
    import json as _frontieror_json
    import sys as _frontieror_sys

    result_path = None
    for index, argument in enumerate(_frontieror_sys.argv):
        if argument == "--result_path" and index + 1 < len(_frontieror_sys.argv):
            result_path = _frontieror_sys.argv[index + 1]
            break
        if argument.startswith("--result_path="):
            result_path = argument.split("=", 1)[1]
            break
    if not result_path:
        return False
    try:
        with open(result_path, "w") as result_handle:
            _frontieror_json.dump(
                {
                    "feasible": False,
                    "violated_constraints": ["malformed_solution"],
                    "violations": [
                        "checker rejected malformed candidate: "
                        + type(error).__name__
                        + ": "
                        + str(error)[:500]
                    ],
                    "violation_magnitudes": [],
                },
                result_handle,
                indent=2,
            )
    except Exception:
        return False
    return True


TOL = 1e-5
EPS = 1e-5

# Constraint 5 (objective consistency).  ``w`` is intentionally not part of
# the published solution, so validate the reported objective against every
# discrete completion of those omitted hub-pair variables instead of accepting
# the entire continuous [min,max] interval.
OBJ_ABS_TOL = 1e-3
OBJ_REL_TOL = 1e-8


def objective_candidates(inst, hub_locations):
    """
    Recompute every objective attainable by a binary completion of the omitted
    two-hub activation variables for the given hub set.

    The dataset has at most five selected hubs (ten pair variables, 1024
    completions).  A defensive cap prevents adversarial inputs from turning
    this exact validation into an exponential resource attack.
    """
    n = inst['n_nodes']
    nodes = inst['nodes']
    dist = inst['distance_matrix']
    demand = inst['demand_matrix']
    params = inst['parameters']

    alpha = params['alpha']               # interhub fare discount factor
    gamma = params['gamma']               # time vs. cost trade-off weight
    beta = params['beta']                 # time decay exponent
    lam = params['lambda']                # cost decay exponent
    c_sc = params['cost_scalar']          # fare proportionality constant
    A_2 = params['attractiveness_2hub']
    A_1 = params['attractiveness_1hub']   # 25% higher than A_2 per paper

    comp = inst['competitor_hub_locations']

    hubs = sorted(set(int(h) for h in hub_locations))
    if not hubs:
        return None
    if any(h < 0 or h >= n for h in hubs):
        return None
    if any(int(m) < 0 or int(m) >= n for m in comp):
        return None

    def utility(t_val, c_val, A):
        """Gravity utility: A / (gamma*t^beta + (1-gamma)*c^lambda)  [Eq. (1)]"""
        denom = gamma * (t_val ** beta) + (1.0 - gamma) * (c_val ** lam)
        if denom <= 0.0:
            return 0.0
        return A / denom

    def t_2hub(i, k, l, j):
        return (dist[i][k] + dist[k][l] + dist[l][j]
                + nodes[i]['waiting_time']
                + nodes[k]['waiting_time']
                + nodes[l]['waiting_time'])

    def c_2hub(i, k, l, j):
        return c_sc * (dist[i][k] + alpha * dist[k][l] + dist[l][j])

    def t_1hub(i, k, j):
        return (dist[i][k] + dist[k][j]
                + nodes[i]['waiting_time']
                + nodes[k]['waiting_time'])

    def c_1hub(i, k, j):
        return c_sc * (dist[i][k] + dist[k][j])

    comp = [int(m) for m in comp]
    comp_pairs = [(m, q) for m in comp for q in comp if m <= q]

    selected_pairs = [
        (hubs[a], hubs[b])
        for a in range(len(hubs))
        for b in range(a + 1, len(hubs))
    ]
    if len(selected_pairs) > 20:
        return None
    totals = [0.0] * (1 << len(selected_pairs))
    for i in range(n):
        for j in range(n):
            h = demand[i][j]
            if h <= 0:
                continue
            # N1: 1-hub routes -- always active for every open hub.
            n1 = 0.0
            for k in hubs:
                n1 += utility(t_1hub(i, k, j), c_1hub(i, k, j), A_1)
            # N2 contribution of each omitted binary hub-pair variable.
            pair_utility = []
            for k, l in selected_pairs:
                u1 = utility(t_2hub(i, k, l, j), c_2hub(i, k, l, j), A_2)
                u2 = utility(t_2hub(i, l, k, j), c_2hub(i, l, k, j), A_2)
                pair_utility.append(max(u1, u2))
            # C: fixed competitor utility.
            c_ij = 0.0
            for (m, q) in comp_pairs:
                if m == q:
                    c_ij += utility(t_1hub(i, m, j), c_1hub(i, m, j), A_1)
                else:
                    u1 = utility(t_2hub(i, m, q, j), c_2hub(i, m, q, j), A_2)
                    u2 = utility(t_2hub(i, q, m, j), c_2hub(i, q, m, j), A_2)
                    c_ij += max(u1, u2)
            for mask in range(len(totals)):
                own_utility = n1 + sum(
                    pair_utility[pair_idx]
                    for pair_idx in range(len(pair_utility))
                    if mask & (1 << pair_idx)
                )
                denom = own_utility + c_ij
                if denom > 0.0:
                    totals[mask] += h * own_utility / denom
    return totals


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Competitive p-Hub Location"
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        inst = json.load(f)
    with open(args.solution_path) as f:
        sol = json.load(f)
        _frontieror_contract_result = _frontieror_validate_solution_contract(sol)
        if _frontieror_contract_result is not None:
            with open(args.result_path, "w") as _frontieror_result_handle:
                json.dump(
                    _frontieror_contract_result,
                    _frontieror_result_handle,
                    indent=2,
                )
            return

    n = inst['n_nodes']
    p = inst['p']
    hub_locations = sol.get('hub_locations', [])

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 3 (Eq 6): sum_k y_k = p
    # In original solution structure: |hub_locations| = p
    # ------------------------------------------------------------------
    lhs_val = float(len(hub_locations))
    rhs_val = float(p)
    excess = abs(lhs_val - rhs_val)
    if excess > TOL:
        violated_constraints.add(3)
        normalizer = max(abs(rhs_val), EPS)
        violations.append(
            f"Constraint 3 violated: |hub_locations|={int(lhs_val)} != p={int(rhs_val)}"
        )
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs_val,
            "rhs": rhs_val,
            "raw_excess": float(excess),
            "normalizer": float(normalizer),
            "ratio": float(excess / normalizer)
        })

    # ------------------------------------------------------------------
    # Constraint 4 (Eq 7, y portion + domain): hub_locations entries are
    # distinct integers in [0, n-1]. This is the original-structure
    # restatement of y_k in {0, 1}.
    # ------------------------------------------------------------------
    # Out-of-range entries.
    for h in hub_locations:
        if not isinstance(h, int) or h < 0 or h >= n:
            violated_constraints.add(4)
            violations.append(
                f"Constraint 4 violated: hub entry {h!r} not an integer in [0, {n - 1}]"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(h) if isinstance(h, (int, float)) else 0.0,
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            })

    # Duplicate hubs.
    if len(hub_locations) != len(set(hub_locations)):
        dup_count = len(hub_locations) - len(set(hub_locations))
        violated_constraints.add(4)
        violations.append(
            f"Constraint 4 violated: hub_locations has {dup_count} duplicate entries"
        )
        violation_magnitudes.append({
            "constraint": 4,
            "lhs": float(len(hub_locations)),
            "rhs": float(len(set(hub_locations))),
            "raw_excess": float(dup_count),
            "normalizer": max(float(len(set(hub_locations))), EPS),
            "ratio": dup_count / max(float(len(set(hub_locations))), EPS),
        })

    # ------------------------------------------------------------------
    # Constraint 5 (objective consistency): because the output omits the binary
    # pair activations w, existentially complete them and require the reported
    # value to match one of the finitely many attainable objectives.  Merely
    # checking the min/max envelope admitted values that no model solution can
    # produce.
    # ------------------------------------------------------------------
    reported_obj = sol.get('objective_value')
    reported = None
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None

    hubs_indexable = (
        len(hub_locations) > 0
        and all(isinstance(h, int) and 0 <= h < n for h in hub_locations)
    )

    if reported is not None and hubs_indexable:
        candidates = None
        try:
            candidates = objective_candidates(inst, hub_locations)
        except Exception:
            candidates = None

        if candidates:
            expected = min(candidates, key=lambda value: abs(reported - value))
            difference = abs(reported - expected)
            obj_tol = max(OBJ_ABS_TOL, OBJ_REL_TOL * max(abs(expected), 1.0))
            if difference > obj_tol:
                normalizer = max(abs(expected), EPS)
                violated_constraints.add(5)
                violations.append(
                    f"Constraint 5 violated: reported objective_value={reported} "
                    f"does not match any of the {len(candidates)} discrete "
                    f"hub-pair completions for hub_locations="
                    f"{sorted(hub_locations)}; nearest={expected}, "
                    f"difference={difference:.6g}, tolerance={obj_tol:.6g}"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": float(reported),
                    "rhs": float(expected),
                    "raw_excess": float(difference),
                    "normalizer": float(normalizer),
                    "ratio": float(difference / normalizer),
                })
        elif hubs_indexable:
            violated_constraints.add(5)
            violations.append(
                "Constraint 5 could not enumerate the omitted hub-pair "
                "variables within the checker resource limit"
            )
            violation_magnitudes.append({
                "constraint": 5, "lhs": 0.0, "rhs": 1.0,
                "raw_excess": 1.0, "normalizer": 1.0, "ratio": 1.0,
            })

    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if feasible else "INFEASIBLE"
    print(f"Result: {status}")
    if not feasible:
        print(f"Violated constraints: {sorted(violated_constraints)}")
        for v in violations:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
