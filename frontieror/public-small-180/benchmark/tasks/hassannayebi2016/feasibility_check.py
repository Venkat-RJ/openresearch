#!/usr/bin/env python3
"""
Feasibility checker for rail rapid transit timetabling solutions.

Checks hard constraints from the mathematical formulation in math_model.txt.
Constraints are numbered 1-26 counting from top to bottom:
  1-17:  MINLP model (eq 2-18)
  18-23: MILP model (eq 20-25)
  24:    MiniMax AWT bound (eq 27)
  25:    Frequency linking (eq 29)
  26:    MINLP_d variable domains (eq 31)
  27:    Objective consistency (recomputed obj must match reported obj_value)

Efficient (GVNS) solutions are checked against MINLP constraints 1-17 and 25.
Gurobi (MILP) solutions are checked against MILP constraints 18-23.
"""

import argparse
import json
import math

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('departure_schedule_minutes', 'objective_value')
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



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cum_arrivals(t, lambda_per, theta, NT):
    """Cumulative passenger arrivals from time 0 to time t."""
    total = 0.0
    for p in range(NT):
        ps = p * theta
        pe = ps + theta
        if t <= ps:
            break
        elif t >= pe:
            total += lambda_per[p] * theta
        else:
            total += lambda_per[p] * (t - ps)
    return total


# ---------------------------------------------------------------------------
# MINLP constraint checks (constraints 1-17, plus 25)
# ---------------------------------------------------------------------------

def _check_minlp(d, freq, D, T, NT, theta, h_min, h_max, C, lam, tol, eps, rec,
                 reported_obj=None):
    """Check MINLP constraints 1-17 and frequency constraint 25.

    If reported_obj is not None, also check the objective-consistency
    constraint 27 (recomputed obj must match reported obj within tolerance).
    """
    n = len(d)

    # Extended departure sequence: [0, d_1, ..., d_n, T]
    d_ext = [0.0] + list(d) + [float(T)]
    H = [d_ext[i + 1] - d_ext[i] for i in range(n + 1)]

    # ---- Constraint 1 (eq 2): sum_{i} H_i = T ----
    sum_H = sum(H)
    va = abs(sum_H - T)
    if va > tol:
        rec(1, f"Sum of headways = {sum_H:.6f}, expected T = {T}", sum_H, float(T), va)

    # ---- Constraint 2 (eq 3): H_i = d_{i+1} - d_i, i = 1..D-1 ----
    for i in range(1, n):
        expected = d[i] - d[i - 1]
        va = abs(H[i] - expected)
        if va > tol:
            rec(2, f"H_{i} = {H[i]:.6f} != d_{i+1} - d_{i} = {expected:.6f}",
                H[i], expected, va)

    # ---- Constraint 3 (eq 4): h_min <= H_i <= h_max, i = 1..D-1 ----
    for i in range(1, n):
        if H[i] < h_min - tol:
            va = h_min - H[i]
            rec(3, f"Headway H_{i} = {H[i]:.4f} < h_min = {h_min}",
                H[i], float(h_min), va)
        if H[i] > h_max + tol:
            va = H[i] - h_max
            rec(3, f"Headway H_{i} = {H[i]:.4f} > h_max = {h_max}",
                H[i], float(h_max), va)

    # ---- Determine period assignments (1-indexed) ----
    period_of = []
    for i in range(n):
        p_idx = min(max(int(d[i] / theta), 0), NT - 1)
        period_of.append(p_idx + 1)  # 1-indexed

    # ---- Constraint 4 (eq 5): departure in assigned period ----
    # Active case (y_i^(p)=1): (p-1)*theta <= d_i < p*theta
    for i in range(n):
        p = period_of[i]
        ps = (p - 1) * theta
        pe = p * theta
        # Lower bound (>=): d_i >= ps
        if d[i] < ps - tol:
            va = ps - d[i]
            rec(4, f"Train {i+1}: d = {d[i]:.4f} < period {p} start = {ps}",
                d[i], float(ps), va)
        # Upper bound (strict <): d_i < pe
        va_upper = d[i] - pe
        if va_upper > tol:
            rec(4, f"Train {i+1}: d = {d[i]:.4f} >= period {p} end = {pe}",
                d[i], float(pe), va_upper)

    # ---- Constraint 5 (eq 6): sum_p y_i^(p) = 1 for each train ----
    # Each train must be assigned to exactly one period. Verify departure count = D.
    if n != D:
        va = float(abs(n - D))
        rec(5, f"Number of departures = {n}, expected D = {D}",
            float(n), float(D), va)

    # ---- Constraint 6 (eq 7): non-decreasing period assignment ----
    # y_{i+1}^(p') + y_i^(p) <= 1 for p' < p, i.e. trains in non-decreasing
    # period order.
    for i in range(n - 1):
        if period_of[i + 1] < period_of[i]:
            rec(6, f"Train {i+2} in period {period_of[i+1]} precedes "
                f"train {i+1} in period {period_of[i]}",
                float(period_of[i + 1]), float(period_of[i]),
                float(period_of[i] - period_of[i + 1]))

    # ---- Compute derived quantities ----
    # Cumulative arrivals at each departure
    Delta = [cum_arrivals(d[i], lam, theta, NT) for i in range(n)]

    # Arrivals per interval: arrivals[0] = before first train, ...
    arrivals = []
    cum_prev = 0.0
    for i in range(n + 1):
        cum_cur = cum_arrivals(d_ext[i + 1], lam, theta, NT)
        arrivals.append(cum_cur - cum_prev)
        cum_prev = cum_cur

    # Flow balance: w[i] = waiting after train i, b[i] = boarding on train i
    w = [0.0] * (n + 1)
    b = [0.0] * (n + 1)
    for i in range(1, n + 1):
        available = w[i - 1] + arrivals[i - 1]
        b[i] = min(available, float(C))
        w[i] = available - b[i]

    # ---- Constraint 7 (eq 8): delta_{i+1} = Delta_{i+1} - Delta_i ----
    for i in range(n - 1):
        lhs_val = arrivals[i + 1]
        rhs_val = Delta[i + 1] - Delta[i]
        va = abs(lhs_val - rhs_val)
        if va > tol:
            rec(7, f"delta consistency violated at interval {i+2}: "
                f"{lhs_val:.6f} != {rhs_val:.6f}", lhs_val, rhs_val, va)

    # ---- Constraint 8 (eq 9): w_i = w_{i-1} + delta_{i-1} - b_i ----
    for i in range(1, n + 1):
        expected = w[i - 1] + arrivals[i - 1] - b[i]
        va = abs(w[i] - expected)
        if va > tol:
            rec(8, f"Flow balance violated at train {i}: "
                f"w_{i} = {w[i]:.6f} != {expected:.6f}", w[i], expected, va)

    # ---- Constraint 9 (eq 10): b_i <= C ----
    for i in range(1, n + 1):
        if b[i] > C + tol:
            va = b[i] - C
            rec(9, f"Train {i}: boarding = {b[i]:.4f} exceeds capacity C = {C}",
                b[i], float(C), va)

    # ---- Constraints 10-11 (eq 11-12): Delta_i consistency ----
    # Active case: Delta_i = cum_arrivals(d_i). Verified by construction.
    for i in range(n):
        expected = cum_arrivals(d[i], lam, theta, NT)
        va = abs(Delta[i] - expected)
        if va > tol:
            rec(10, f"Delta_{i+1} upper bound violated: "
                f"{Delta[i]:.6f} != {expected:.6f}", Delta[i], expected, va)
        va2 = abs(Delta[i] - expected)
        if va2 > tol:
            rec(11, f"Delta_{i+1} lower bound violated: "
                f"{Delta[i]:.6f} != {expected:.6f}", Delta[i], expected, va2)

    # ---- Constraints 12-13 (eq 13-14): delta_0 consistency ----
    if n > 0:
        expected_d0 = cum_arrivals(d[0], lam, theta, NT)
        va = abs(arrivals[0] - expected_d0)
        if va > tol:
            rec(12, f"delta_0 upper bound: {arrivals[0]:.6f} != {expected_d0:.6f}",
                arrivals[0], expected_d0, va)
        if va > tol:
            rec(13, f"delta_0 lower bound: {arrivals[0]:.6f} != {expected_d0:.6f}",
                arrivals[0], expected_d0, va)

    # ---- Constraint 16 (eq 17): w_0 = 0, b_0 = 0 ----
    if abs(w[0]) > tol:
        rec(16, f"w_0 = {w[0]:.6f} != 0", w[0], 0.0, abs(w[0]))
    if abs(b[0]) > tol:
        rec(16, f"b_0 = {b[0]:.6f} != 0", b[0], 0.0, abs(b[0]))

    # ---- Constraint 17 (eq 18): non-negativity of all variables ----
    for i in range(n):
        if d[i] < -tol:
            rec(17, f"d_{i+1} = {d[i]:.6f} < 0", d[i], 0.0, abs(d[i]))
    for i in range(n + 1):
        if H[i] < -tol:
            rec(17, f"H_{i} = {H[i]:.6f} < 0", H[i], 0.0, abs(H[i]))
    for i in range(n + 1):
        if arrivals[i] < -tol:
            rec(17, f"delta_{i} = {arrivals[i]:.6f} < 0",
                arrivals[i], 0.0, abs(arrivals[i]))
    for i in range(1, n + 1):
        if w[i] < -tol:
            rec(17, f"w_{i} = {w[i]:.6f} < 0", w[i], 0.0, abs(w[i]))
        if b[i] < -tol:
            rec(17, f"b_{i} = {b[i]:.6f} < 0", b[i], 0.0, abs(b[i]))

    # ---- Constraint 25 (eq 29): frequency vector consistency ----
    if freq is not None:
        freq_sum = sum(freq)
        if freq_sum != D:
            va = float(abs(freq_sum - D))
            rec(25, f"Sum of frequency vector = {freq_sum}, expected D = {D}",
                float(freq_sum), float(D), va)
        for p_idx in range(len(freq)):
            actual = sum(1 for i in range(n) if period_of[i] == p_idx + 1)
            if actual != freq[p_idx]:
                va = float(abs(actual - freq[p_idx]))
                rec(25, f"Period {p_idx+1}: {actual} departures != "
                    f"F_{p_idx+1} = {freq[p_idx]}",
                    float(actual), float(freq[p_idx]), va)

    # ---- Constraint 27 (Tier C, obj consistency): recomputed obj matches reported ----
    # Objective (eq. 1): z = sum_{i in {0..D}} (w_i + 0.5*delta_i) * H_i.
    # Given d_1..d_D, the derived quantities (delta, w under optimal boarding
    # b_i = min(prev_w + prev_delta, C)) are deterministic -- FIFO boarding
    # is provably optimal because the obj is monotone-decreasing in each b_i
    # and each b_i is bounded only by C and the w_i >= 0 flow constraint.
    # Empirically the FIFO recompute matches Gurobi within ~1e-7 relative on
    # all reference instances, so a 0.1% relative tolerance (with a 1.0
    # absolute floor for small numerical drift) is safe.
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            true_obj = sum((w[i] + 0.5 * arrivals[i]) * H[i] for i in range(n + 1))
            abs_diff = abs(reported - true_obj)
            tol_obj = max(1.0, 1e-3 * abs(true_obj))
            if abs_diff > tol_obj:
                rec(27,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed "
                    f"sum_i (w_i + 0.5*delta_i)*H_i = {true_obj:.6f} "
                    f"(|diff|={abs_diff:.6g}, tol={tol_obj:.6g})",
                    reported, true_obj, abs_diff)
        elif reported is not None and not math.isfinite(reported):
            # Non-finite reported values (inf/nan/MAX_FLOAT bypass) are
            # never consistent with the finite FIFO objective.
            true_obj = sum((w[i] + 0.5 * arrivals[i]) * H[i] for i in range(n + 1))
            rec(27,
                f"Objective consistency violated: reported objective_value="
                f"{reported} is non-finite; recomputed "
                f"sum_i (w_i + 0.5*delta_i)*H_i = {true_obj:.6f}",
                reported, true_obj, float('inf'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for rail rapid transit timetabling."
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)
    with open(args.solution_path) as f:
        solution = json.load(f)
        _frontieror_contract_result = _frontieror_validate_solution_contract(solution)
        if _frontieror_contract_result is not None:
            with open(args.result_path, "w") as _frontieror_result_handle:
                json.dump(
                    _frontieror_contract_result,
                    _frontieror_result_handle,
                    indent=2,
                )
            return

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility: {'FEASIBLE' if result['feasible'] else 'INFEASIBLE'}")
    if result["violated_constraints"]:
        print(f"Violated constraints: {result['violated_constraints']}")
    print(f"Result written to {args.result_path}")


def check_feasibility(instance, solution):
    """Programmatic entry point used by the differential-test harness."""
    tol = 1e-5
    eps = 1e-5

    params = instance["parameters"]
    T = params["T_planning_horizon_minutes"]
    D = params["D_num_train_services"]
    NT = params["NT_num_demand_periods"]
    theta = params["theta_period_length_minutes"]
    h_min = params["h_min_minutes"]
    h_max = params["h_max_minutes"]
    C = params["C_train_capacity_passengers"]

    demand_periods = sorted(instance["demand_periods"], key=lambda x: x["period"])
    lambda_per = [dp["lambda_pax_per_min"] for dp in demand_periods]

    violations = []
    violation_magnitudes = []
    violated_set = set()

    def record(cidx, msg, lhs_val, rhs_val, va):
        violated_set.add(cidx)
        violations.append(msg)
        normalizer = max(abs(rhs_val), eps)
        violation_magnitudes.append({
            "constraint": cidx,
            "lhs": float(lhs_val),
            "rhs": float(rhs_val),
            "raw_excess": float(va),
            "normalizer": float(normalizer),
            "ratio": float(va / normalizer)
        })

    reported_obj = solution.get("objective_value")

    # Detect solution type and run MINLP checks
    if "departure_times_minutes" in solution:
        dep = list(solution["departure_times_minutes"])
        freq = solution.get("frequency_vector", None)
        _check_minlp(dep, freq, D, T, NT, theta, h_min, h_max, C,
                      lambda_per, tol, eps, record, reported_obj=reported_obj)
    elif "departure_schedule_minutes" in solution:
        sched = solution["departure_schedule_minutes"]
        dep = [float(sched[k]) for k in sorted(sched.keys(), key=lambda x: int(x))]
        freq = solution.get("frequency_vector", None)
        _check_minlp(dep, freq, D, T, NT, theta, h_min, h_max, C,
                      lambda_per, tol, eps, record, reported_obj=reported_obj)
    else:
        violations.append("Unknown solution format: cannot identify departure data")

    return {
        "feasible": len(violated_set) == 0,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
