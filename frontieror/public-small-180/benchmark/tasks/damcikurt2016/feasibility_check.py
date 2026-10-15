"""
Feasibility checker for Multistage Stochastic Self-Scheduling Unit Commitment (MSS).
Pan & Guan (2016), Operations Research 64(6):1482-1498.

Checks all hard constraints from the mathematical formulation.
Constraint numbering (top to bottom in formulation section):
  1: (1b) Minimum-up time
  2: (1c) Minimum-down time
  3: (1d) Startup link
  4: (1e) Shutdown variable definition (v_i = y_{i^-} - y_i + u_i)
  5: (1f) Generation bounds
  6: (1g) Ramp-up limit
  7: (1h) Ramp-down limit
  8: (1i) Binary/integrality of y, u (and implied v)
  9: (2)  Startup valid inequality (u_i <= min{y_i, 1 - y_{i^-}})
 10: Objective consistency -- reported objective_value must match the
     expected total profit recomputed from (x, y, u). Tier C anti-gaming
     defense against candidates that report a fabricated objective_value
     while keeping the decision variables technically feasible.

This is the side-by-side `_new` variant: it is byte-identical to
feasibility_check.py for constraints 1-9 and only ADDS constraint 10.
"""

import argparse
import json
import math
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('objective_value', 'u', 'x', 'y')
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


# Gurobi numerical noise on this model is ~1e-4 on large instances
# (e.g. large_41 OPT solution had drift ~1.21e-4 on ramp-up bounds and
# zero-bound generation constraints). 1e-5 was rejecting valid OPT
# solutions; 1e-3 leaves ample margin while still flagging genuine
# infeasibilities.
TOL = 1e-3
EPS = 1e-3


def build_tree_data(nodes_list):
    nodes = {}
    for n in nodes_list:
        nid = n["id"]
        nodes[nid] = {
            "time": n["time"],
            "parent": n["parent_id"],
            "children": n["children_ids"],
            "prob": n["probability"],
            "price": n["electricity_price_dollars_per_MWh"],
        }
    return nodes


def get_descendants(node_id, nodes):
    result = []
    stack = [node_id]
    while stack:
        cur = stack.pop()
        result.append(cur)
        stack.extend(nodes[cur]["children"])
    return result


def get_H_r(node_i, r, nodes):
    """H_r(i) = {k in descendants(i) : 0 <= t(k) - t(i) <= r - 1}"""
    t_i = nodes[node_i]["time"]
    desc = get_descendants(node_i, nodes)
    return [k for k in desc if 0 <= nodes[k]["time"] - t_i <= r - 1]


def record_violation(violations_list, magnitudes_list, constraint_idx, msg, lhs, rhs):
    violation_amount = abs(lhs - rhs)
    if constraint_idx == 8:
        # For integrality, measure distance from nearest integer
        # lhs is the value, rhs is the nearest integer
        violation_amount = abs(lhs - rhs)
    normalizer = max(abs(rhs), EPS)
    ratio = violation_amount / normalizer
    violations_list.append(msg)
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(ratio),
    })


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MSS Unit Commitment (Pan & Guan 2016)"
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--result_path", required=True)
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

    # Instance parameters
    gen = inst["generator"]
    C_lo = gen["C_lower_MW"]
    C_up = gen["C_upper_MW"]
    V_plus = gen["V_plus_MW_per_h"]
    V_minus = gen["V_minus_MW_per_h"]

    tree_cfg = inst["scenario_tree"]
    L = tree_cfg["L_min_up_time"]
    ell = tree_cfg["ell_min_down_time"]
    y0_init = tree_cfg.get("initial_generator_status_y0", 0)
    x0_init = tree_cfg.get("initial_generation_x0_MW", 0.0)

    nodes = build_tree_data(tree_cfg["nodes"])
    all_nodes = list(nodes.keys())
    non_root = [i for i in all_nodes if nodes[i]["parent"] is not None]
    root_id = [i for i in all_nodes if nodes[i]["parent"] is None][0]

    # Solution variables
    # The released certificate format is sparse.  Omitted entries take the
    # corresponding optional sparse_defaults value (zero when absent).  This
    # permits both all-off and stationary all-on policies to remain compact on
    # million-node trees without weakening any constraint check.
    sparse_defaults = sol.get("sparse_defaults", {})
    x_default = float(sparse_defaults.get("x", 0.0))
    y_default = float(sparse_defaults.get("y", 0.0))
    u_default = float(sparse_defaults.get("u", 0.0))
    if not all(math.isfinite(value) for value in (x_default, y_default, u_default)):
        raise ValueError("sparse defaults must be finite")
    x = defaultdict(
        lambda: x_default, {int(k): v for k, v in sol["x"].items()}
    )
    y = defaultdict(
        lambda: y_default, {int(k): v for k, v in sol["y"].items()}
    )
    u = defaultdict(
        lambda: u_default, {int(k): v for k, v in sol["u"].items()}
    )

    violated_constraints = set()
    violations = []
    magnitudes = []

    # ── Constraint 1: (1b) Minimum-up time ──
    # y_i - y_{i^-} <= y_k, for all i in V\{0}, for all k in H_L(i)
    for i in non_root:
        par = nodes[i]["parent"]
        H = get_H_r(i, L, nodes)
        for k in H:
            lhs_val = y[i] - y[par]
            rhs_val = y[k]
            excess = lhs_val - rhs_val
            if excess > TOL:
                violated_constraints.add(1)
                record_violation(violations, magnitudes, 1,
                    f"Min-up time violated: node i={i}, k={k}: "
                    f"y[{i}]-y[{par}]={lhs_val} > y[{k}]={rhs_val}",
                    lhs_val, rhs_val)

    # ── Constraint 2: (1c) Minimum-down time ──
    # y_{i^-} - y_i <= 1 - y_k, for all i in V\{0}, for all k in H_ell(i)
    for i in non_root:
        par = nodes[i]["parent"]
        H = get_H_r(i, ell, nodes)
        for k in H:
            lhs_val = y[par] - y[i]
            rhs_val = 1 - y[k]
            excess = lhs_val - rhs_val
            if excess > TOL:
                violated_constraints.add(2)
                record_violation(violations, magnitudes, 2,
                    f"Min-down time violated: node i={i}, k={k}: "
                    f"y[{par}]-y[{i}]={lhs_val} > 1-y[{k}]={rhs_val}",
                    lhs_val, rhs_val)

    # ── Constraint 3: (1d) Startup link ──
    # y_i - y_{i^-} <= u_i, for all i in V\{0}
    for i in non_root:
        par = nodes[i]["parent"]
        lhs_val = y[i] - y[par]
        rhs_val = u[i]
        excess = lhs_val - rhs_val
        if excess > TOL:
            violated_constraints.add(3)
            record_violation(violations, magnitudes, 3,
                f"Startup link violated: node i={i}: "
                f"y[{i}]-y[{par}]={lhs_val} > u[{i}]={rhs_val}",
                lhs_val, rhs_val)

    # ── Constraint 4: (1e) Shutdown variable definition ──
    # v_i = y_{i^-} - y_i + u_i, and v_i should be in {0,1}
    # Since v is eliminated, we check that the implied v_i is non-negative and binary
    for i in non_root:
        par = nodes[i]["parent"]
        v_i = y[par] - y[i] + u[i]
        # Check v_i >= 0 (equality constraint: v_i = y_{i^-} - y_i + u_i, must be valid)
        if v_i < -TOL:
            violated_constraints.add(4)
            record_violation(violations, magnitudes, 4,
                f"Shutdown var v[{i}] = y[{par}]-y[{i}]+u[{i}] = {v_i} < 0",
                v_i, 0.0)
        # Check v_i <= 1
        if v_i > 1 + TOL:
            violated_constraints.add(4)
            record_violation(violations, magnitudes, 4,
                f"Shutdown var v[{i}] = y[{par}]-y[{i}]+u[{i}] = {v_i} > 1",
                v_i, 1.0)
        # Check v_i is integer (close to 0 or 1)
        if abs(v_i - round(v_i)) > TOL:
            violated_constraints.add(4)
            record_violation(violations, magnitudes, 4,
                f"Shutdown var v[{i}] = {v_i} not binary",
                v_i, round(v_i))

    # ── Constraint 5: (1f) Generation bounds ──
    # C_lo * y_i <= x_i <= C_up * y_i, for all i in V
    for i in all_nodes:
        # Lower bound: x_i >= C_lo * y_i  =>  C_lo * y_i - x_i <= 0
        lb = C_lo * y[i]
        if lb - x[i] > TOL:
            violated_constraints.add(5)
            record_violation(violations, magnitudes, 5,
                f"Generation lower bound violated: node {i}: "
                f"x[{i}]={x[i]} < C_lo*y[{i}]={lb}",
                lb, x[i])
        # Upper bound: x_i <= C_up * y_i  =>  x_i - C_up * y_i <= 0
        ub = C_up * y[i]
        if x[i] - ub > TOL:
            violated_constraints.add(5)
            record_violation(violations, magnitudes, 5,
                f"Generation upper bound violated: node {i}: "
                f"x[{i}]={x[i]} > C_up*y[{i}]={ub}",
                x[i], ub)

    # ── Constraint 6: (1g) Ramp-up limit ──
    # x_i - x_{i^-} <= V^+ * y_{i^-} + C_up * (1 - y_{i^-}), for all i in V\{0}
    for i in non_root:
        par = nodes[i]["parent"]
        lhs_val = x[i] - x[par]
        rhs_val = V_plus * y[par] + C_up * (1 - y[par])
        excess = lhs_val - rhs_val
        if excess > TOL:
            violated_constraints.add(6)
            record_violation(violations, magnitudes, 6,
                f"Ramp-up violated: node {i}: "
                f"x[{i}]-x[{par}]={lhs_val} > V+*y[{par}]+C_up*(1-y[{par}])={rhs_val}",
                lhs_val, rhs_val)

    # ── Constraint 7: (1h) Ramp-down limit ──
    # x_{i^-} - x_i <= V^- * y_i + C_up * (1 - y_i), for all i in V\{0}
    for i in non_root:
        par = nodes[i]["parent"]
        lhs_val = x[par] - x[i]
        rhs_val = V_minus * y[i] + C_up * (1 - y[i])
        excess = lhs_val - rhs_val
        if excess > TOL:
            violated_constraints.add(7)
            record_violation(violations, magnitudes, 7,
                f"Ramp-down violated: node {i}: "
                f"x[{par}]-x[{i}]={lhs_val} > V-*y[{i}]+C_up*(1-y[{i}])={rhs_val}",
                lhs_val, rhs_val)

    # ── Constraint 8: (1i) Binary/integrality ──
    # y_i in {0,1} for all i; u_i in {0,1} for all i in V\{0}
    for i in all_nodes:
        if abs(y[i] - round(y[i])) > TOL or round(y[i]) not in (0, 1):
            violated_constraints.add(8)
            record_violation(violations, magnitudes, 8,
                f"y[{i}]={y[i]} not binary",
                y[i], round(y[i]))
    for i in non_root:
        if abs(u[i] - round(u[i])) > TOL or round(u[i]) not in (0, 1):
            violated_constraints.add(8)
            record_violation(violations, magnitudes, 8,
                f"u[{i}]={u[i]} not binary",
                u[i], round(u[i]))
    # x_i >= 0
    for i in all_nodes:
        if x[i] < -TOL:
            violated_constraints.add(8)
            record_violation(violations, magnitudes, 8,
                f"x[{i}]={x[i]} < 0 (non-negativity violated)",
                x[i], 0.0)

    # ── Constraint 9: (2) Startup valid inequality ──
    # u_i <= y_i and u_i <= 1 - y_{i^-}, for all i in V\{0}
    for i in non_root:
        par = nodes[i]["parent"]
        # u_i <= y_i
        excess1 = u[i] - y[i]
        if excess1 > TOL:
            violated_constraints.add(9)
            record_violation(violations, magnitudes, 9,
                f"Startup inequality violated: node {i}: "
                f"u[{i}]={u[i]} > y[{i}]={y[i]}",
                u[i], y[i])
        # u_i <= 1 - y_{i^-}
        rhs_val = 1 - y[par]
        excess2 = u[i] - rhs_val
        if excess2 > TOL:
            violated_constraints.add(9)
            record_violation(violations, magnitudes, 9,
                f"Startup inequality violated: node {i}: "
                f"u[{i}]={u[i]} > 1-y[{par}]={rhs_val}",
                u[i], rhs_val)

    # ── Constraint 10: Objective consistency (Tier C anti-gaming) ──
    # The full expected-profit objective (1a) is recomputed from the
    # solution variables (x, y, u). Every variable the objective depends
    # on is present in the solution schema, so a FULL recompute applies.
    #
    #   obj = sum_{i in V}      p_i * ( q_i*x_i - f(x_i) )
    #       - sum_{i in V\{0}}  p_i * ( U_bar*u_i + U_under*(y_{i^-} - y_i + u_i) )
    #
    # with the linear fuel-cost approximation f(x_i) = fuel_a*y_i + fuel_b*x_i
    # (the paper leaves the piecewise-linear form unspecified; the Gurobi
    # reference solver uses exactly this linear form).
    #
    # The root node has no startup variable u_0 and no shutdown term, so its
    # contribution is only p_0*(q_0*x_0 - f(x_0)).
    #
    # A candidate that reports a fabricated objective_value while keeping
    # (x, y, u) feasible passes constraints 1-9 but is rejected here.
    U_bar = gen["U_bar_startup_cost_dollars"]
    # Shutdown cost: the paper does not specify it separately; the instance
    # carries it under U_lower_shutdown_cost_dollars. Fall back to U_bar
    # (the symmetric default the Gurobi reference solver also applies).
    U_under = gen.get("U_lower_shutdown_cost_dollars", U_bar)
    fuel_a = gen["fuel_a_dollars_per_h"]
    fuel_b = gen["fuel_b_dollars_per_MWh"]

    reported_obj = sol.get("objective_value")
    try:
        reported = float(reported_obj)
    except (TypeError, ValueError):
        reported = None
    if reported is not None:
        true_obj = 0.0
        for i in all_nodes:
            p_i = nodes[i]["prob"]
            q_i = nodes[i]["price"]
            par = nodes[i]["parent"]
            true_obj += p_i * q_i * x[i] - p_i * fuel_b * x[i] - p_i * fuel_a * y[i]
            if par is not None:
                true_obj -= p_i * U_bar * u[i]
                true_obj -= p_i * U_under * (y[par] - y[i] + u[i])
        abs_diff = abs(reported - true_obj)
        # 0.1% relative tolerance with a 1e-3 absolute floor, mirroring TOL.
        tol = max(1e-3, 1e-3 * abs(true_obj))
        if abs_diff > tol:
            violated_constraints.add(10)
            record_violation(violations, magnitudes, 10,
                f"Objective consistency violated: reported objective_value="
                f"{reported} differs from recomputed expected profit="
                f"{true_obj} (|diff|={abs_diff:.6g}, tol={tol:.6g})",
                reported, true_obj)

    # Build result
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if feasible:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {sorted(violated_constraints)}")
        for v in violations[:10]:
            print(f"  - {v}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more violations")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
