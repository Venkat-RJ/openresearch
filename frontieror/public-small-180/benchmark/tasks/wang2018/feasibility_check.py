"""
Feasibility checker for the Integrated Berth Allocation, QC Assignment,
and Yard Assignment Problem.

Reference:
    Wang, Zhen, Wang, Laporte (2018).
    Transportation Science 52(4):812-834.

Checks constraints (2)-(20) from the mathematical formulation.
Constraints (21)-(28) are variable domain constraints checked implicitly.
Constraints (29)-(33) are linearization constraints for the objective.
Constraint (34): objective-value consistency. The reported
``objective_value`` must match the value obtained by recomputing the
linearised objective from the variables present in the solution
(alpha, beta, berth, subblocks). This guards against score-gaming
exploits in which the program lies about how good its solution is.
"""

import argparse
import json
import math

TOL = 1e-5
EPS = 1e-5


# FRONTIEROR_RELEASE_CONTRACT_V2
def _validate_solution_contract(solution):
    violations = []
    if not isinstance(solution, dict):
        violations.append("solution must be a JSON object")
    else:
        for field in ("objective_value", "vessels", "berths", "qc_usage"):
            if field not in solution or solution[field] is None:
                violations.append(f"missing required solution field(s): {field}")
        objective = solution.get("objective_value")
        if objective is not None and (
            isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or not math.isfinite(objective)
        ):
            violations.append("objective_value must be a finite JSON number")
        if solution.get("vessels") is not None and not isinstance(solution["vessels"], list):
            violations.append("vessels must be a JSON array")
        if solution.get("berths") is not None and not isinstance(solution["berths"], list):
            violations.append("berths must be a JSON array")
        if solution.get("qc_usage") is not None and not isinstance(solution["qc_usage"], dict):
            violations.append("qc_usage must be a JSON object")

        for index, vessel in enumerate(solution.get("vessels", [])):
            if not isinstance(vessel, dict):
                violations.append(f"vessels[{index}] must be a JSON object")
                continue
            alternatives = (
                ("vessel_id",), ("alpha", "start_time"), ("beta", "end_time"),
                ("profile", "profile_idx"),
            )
            required = ("berth", "subblocks", "handling_time", "qc_per_timestep")
            missing = [field for field in required if field not in vessel]
            for names in alternatives:
                if not any(name in vessel for name in names):
                    missing.append("/".join(names))
            if missing:
                violations.append(
                    f"vessels[{index}] missing required field(s): " + ", ".join(missing)
                )

    if not violations:
        return None
    return {
        "feasible": False,
        "violated_constraints": ["solution_schema"],
        "violations": violations,
        "violation_magnitudes": [],
    }


def _write_unexpected_rejection(error):
    import sys
    result_path = None
    for index, argument in enumerate(sys.argv):
        if argument == "--result_path" and index + 1 < len(sys.argv):
            result_path = sys.argv[index + 1]
            break
        if argument.startswith("--result_path="):
            result_path = argument.split("=", 1)[1]
            break
    if not result_path:
        return False
    try:
        with open(result_path, "w") as handle:
            json.dump({
                "feasible": False,
                "violated_constraints": ["malformed_solution"],
                "violations": [
                    "checker rejected malformed candidate: "
                    + type(error).__name__ + ": " + str(error)[:500]
                ],
                "violation_magnitudes": [],
            }, handle, indent=2)
    except Exception:
        return False
    return True


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def record_violation(violations_list, magnitudes_list, constraint_idx, msg, lhs, rhs):
    """Record a constraint violation with normalized magnitude."""
    violation_amount = abs(lhs - rhs)  # will be refined per operator below
    # This function is called only when a violation is confirmed
    normalizer = max(abs(rhs), EPS)
    ratio = violation_amount / normalizer

    if constraint_idx not in [v["constraint"] for v in magnitudes_list]:
        pass  # allow multiple entries per constraint index
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(ratio),
    })
    violations_list.append(msg)


def check_leq(lhs, rhs):
    """Check lhs <= rhs. Returns (violated, violation_amount)."""
    excess = lhs - rhs
    return excess > TOL, max(0.0, excess)


def check_geq(lhs, rhs):
    """Check lhs >= rhs. Returns (violated, violation_amount)."""
    excess = rhs - lhs
    return excess > TOL, max(0.0, excess)


def check_eq(lhs, rhs):
    """Check lhs == rhs. Returns (violated, violation_amount)."""
    diff = abs(lhs - rhs)
    return diff > TOL, diff


def feasibility_check(instance, solution):
    """Check all hard constraints and return feasibility result."""
    contract_result = _validate_solution_contract(solution)
    if contract_result is not None:
        return contract_result

    violations = []
    magnitudes = []
    violated_set = set()

    # -------------------------------------------------------------------------
    # Extract instance data
    # -------------------------------------------------------------------------
    H = instance["H"]
    E = instance["E"]
    num_vessels = instance["num_vessels"]
    num_berths = instance["num_berths"]
    num_subblocks = instance["num_subblocks"]
    Q_cap = instance["QC_capacity_per_timestep"]  # length H+E, 0-indexed
    vessels_data = instance["vessels"]

    # Build vessel lookup by vessel_id
    vessel_by_id = {}
    for v in vessels_data:
        vessel_by_id[v["vessel_id"]] = v

    # -------------------------------------------------------------------------
    # Extract solution data
    # -------------------------------------------------------------------------
    sol_vessels = solution["vessels"]

    # Detect solution format (efficient vs gurobi)
    # Efficient format uses: start_time, end_time, qc_per_timestep (dict str->int)
    # Gurobi format uses: alpha, beta, qc_per_timestep (list), handling_time, profile

    # Normalize solution vessels into a common format
    assignments = []
    for sv in sol_vessels:
        vid = sv["vessel_id"]
        vdata = vessel_by_id[vid]

        # Determine start and end times
        if "alpha" in sv:
            alpha = sv["alpha"]
            beta = sv["beta"]
        else:
            alpha = sv["start_time"]
            beta = sv["end_time"]

        # Determine berth
        berth = sv["berth"]

        # Determine QC per timestep
        if isinstance(sv["qc_per_timestep"], dict):
            # Efficient format: {"1": 5, "2": 5, ...}
            qc_ts = {int(k): v for k, v in sv["qc_per_timestep"].items()}
        elif isinstance(sv["qc_per_timestep"], list):
            # Gurobi format: [5, 5, 4, 3] starting at alpha
            qc_ts = {}
            for m, qc in enumerate(sv["qc_per_timestep"]):
                qc_ts[alpha + m] = qc
        else:
            qc_ts = {}

        # Determine profile index and handling time
        if "profile_idx" in sv:
            p_idx = sv["profile_idx"]
        elif "profile" in sv:
            p_idx = sv["profile"]
        else:
            p_idx = None

        if "handling_time" in sv:
            h_p = sv["handling_time"]
        elif p_idx is not None and p_idx < len(vdata["qc_profiles"]):
            h_p = vdata["qc_profiles"][p_idx]["handling_time"]
        else:
            h_p = beta - alpha + 1

        # Determine subblocks
        subblocks = sv.get("subblocks", [])

        assignments.append({
            "vessel_id": vid,
            "alpha": alpha,
            "beta": beta,
            "berth": berth,
            "qc_per_timestep": qc_ts,
            "profile_idx": p_idx,
            "handling_time": h_p,
            "subblocks": subblocks,
            "vdata": vdata,
        })

    # -------------------------------------------------------------------------
    # Constraint (2): Each subblock reserved for at most one vessel
    # sum_i phi_{ik} <= 1 for all k in K
    # -------------------------------------------------------------------------
    subblock_usage = {}  # k -> list of vessel_ids
    for a in assignments:
        for k in a["subblocks"]:
            subblock_usage.setdefault(k, []).append(a["vessel_id"])

    for k, vids in subblock_usage.items():
        lhs = len(vids)
        rhs = 1
        violated, viol_amt = check_leq(lhs, rhs)
        if violated:
            violated_set.add(2)
            record_violation(violations, magnitudes, 2,
                             f"Subblock {k} reserved by {len(vids)} vessels: {vids}",
                             lhs, rhs)

    # -------------------------------------------------------------------------
    # Constraint (3): Exactly r_i subblocks reserved for vessel i
    # sum_k phi_{ik} = r_i for all i in V
    # -------------------------------------------------------------------------
    for a in assignments:
        lhs = len(a["subblocks"])
        rhs = a["vdata"]["reserved_subblocks"]
        violated, viol_amt = check_eq(lhs, rhs)
        if violated:
            violated_set.add(3)
            record_violation(violations, magnitudes, 3,
                             f"Vessel {a['vessel_id']}: has {lhs} subblocks, needs {rhs}",
                             lhs, rhs)

    # -------------------------------------------------------------------------
    # Constraint (4): Exactly one QC profile assigned to each vessel
    # sum_p gamma_{ip} = 1 for all i in V
    # We check that the profile_idx is valid for the vessel
    # -------------------------------------------------------------------------
    for a in assignments:
        num_profiles = len(a["vdata"]["qc_profiles"])
        p_idx = a["profile_idx"]
        if p_idx is None:
            # Profile not explicitly given; check if qc_per_timestep matches any profile
            matched = False
            for pi, prof in enumerate(a["vdata"]["qc_profiles"]):
                if prof["handling_time"] == a["handling_time"]:
                    expected_qc = {a["alpha"] + m: prof["qc_per_timestep"][m]
                                   for m in range(prof["handling_time"])}
                    if expected_qc == a["qc_per_timestep"]:
                        matched = True
                        a["profile_idx"] = pi
                        break
            if not matched:
                # Still may be valid if the algorithm used M6' (greedy QC assignment)
                # which doesn't correspond to a predefined profile.
                # In that case, we just check workload consistency.
                pass
        else:
            if p_idx < 0 or p_idx >= num_profiles:
                violated_set.add(4)
                lhs = 0
                rhs = 1
                record_violation(violations, magnitudes, 4,
                                 f"Vessel {a['vessel_id']}: profile index {p_idx} out of range [0, {num_profiles - 1}]",
                                 lhs, rhs)
            else:
                profile = a["vdata"]["qc_profiles"][p_idx]
                expected_h = profile["handling_time"]
                expected_qc = {
                    a["alpha"] + offset: qc
                    for offset, qc in enumerate(profile["qc_per_timestep"])
                }
                if a["handling_time"] != expected_h:
                    violated_set.add(4)
                    record_violation(
                        violations, magnitudes, 4,
                        f"Vessel {a['vessel_id']}: handling_time={a['handling_time']} "
                        f"does not match profile {p_idx} duration={expected_h}",
                        a["handling_time"], expected_h,
                    )
                if a["qc_per_timestep"] != expected_qc:
                    violated_set.add(4)
                    violations.append(
                        f"Vessel {a['vessel_id']}: qc_per_timestep does not match "
                        f"selected profile {p_idx}"
                    )

    # -------------------------------------------------------------------------
    # Constraint (5): Each vessel allocated to exactly one berth
    # sum_b omega_{ib} = 1 for all i in V
    # (Implicitly satisfied since each solution entry has one berth)
    # Check berth is in valid range
    # -------------------------------------------------------------------------
    for a in assignments:
        b = a["berth"]
        if b < 0 or b >= num_berths:
            violated_set.add(5)
            record_violation(violations, magnitudes, 5,
                             f"Vessel {a['vessel_id']}: berth {b} out of range [0, {num_berths - 1}]",
                             0, 1)

    # Also check that each vessel appears exactly once
    vessel_ids_in_sol = [a["vessel_id"] for a in assignments]
    for v in vessels_data:
        count = vessel_ids_in_sol.count(v["vessel_id"])
        if count != 1:
            lhs = count
            rhs = 1
            violated, viol_amt = check_eq(lhs, rhs)
            if violated:
                violated_set.add(5)
                record_violation(violations, magnitudes, 5,
                                 f"Vessel {v['vessel_id']}: appears {count} times in solution",
                                 lhs, rhs)

    # -------------------------------------------------------------------------
    # Constraint (6): Each vessel starts handling in exactly one time step
    # sum_{t in {1,...,H}} mu_{it} = 1
    # Check that alpha_i is in {1,...,H}
    # -------------------------------------------------------------------------
    for a in assignments:
        alpha = a["alpha"]
        if alpha < 1 or alpha > H:
            violated_set.add(6)
            lhs = alpha
            rhs_lo = 1
            rhs_hi = H
            if alpha < 1:
                record_violation(violations, magnitudes, 6,
                                 f"Vessel {a['vessel_id']}: start time {alpha} < 1",
                                 alpha, 1)
            else:
                record_violation(violations, magnitudes, 6,
                                 f"Vessel {a['vessel_id']}: start time {alpha} > H={H}",
                                 alpha, H)

    # -------------------------------------------------------------------------
    # Constraint (8): alpha_i + sum_p gamma_{ip} * h_{ip} - 1 = beta_i
    # i.e., beta_i = alpha_i + h_p - 1
    # -------------------------------------------------------------------------
    for a in assignments:
        lhs = a["alpha"] + a["handling_time"] - 1
        rhs = a["beta"]
        violated, viol_amt = check_eq(lhs, rhs)
        if violated:
            violated_set.add(8)
            record_violation(violations, magnitudes, 8,
                             f"Vessel {a['vessel_id']}: alpha({a['alpha']}) + h_p({a['handling_time']}) - 1 = {lhs} != beta({rhs})",
                             lhs, rhs)

    # -------------------------------------------------------------------------
    # Constraints (9), (10), (11): Sequencing at same berth
    # If vessels i and j are both at berth b, they cannot overlap in time.
    # Constraint (9): alpha_i + h_{ip} <= alpha_j + (1-delta_{ijb})*M
    # This means: if both at same berth, one must finish before the other starts.
    # We check: for all pairs at same berth, time intervals don't overlap.
    # -------------------------------------------------------------------------
    berth_assignments = {}  # berth -> list of assignments
    for a in assignments:
        berth_assignments.setdefault(a["berth"], []).append(a)

    for b, b_assignments in berth_assignments.items():
        for i in range(len(b_assignments)):
            for j in range(i + 1, len(b_assignments)):
                ai = b_assignments[i]
                aj = b_assignments[j]
                # Check if time intervals overlap: [alpha_i, beta_i] and [alpha_j, beta_j]
                overlap_start = max(ai["alpha"], aj["alpha"])
                overlap_end = min(ai["beta"], aj["beta"])
                if overlap_start <= overlap_end:
                    # Overlap exists - violated
                    # The violation is how much the first vessel extends into the second's time
                    # Using constraint (9): alpha_i + h_{ip} <= alpha_j (for the vessel that starts first)
                    if ai["alpha"] <= aj["alpha"]:
                        lhs = ai["alpha"] + ai["handling_time"]
                        rhs = aj["alpha"]
                    else:
                        lhs = aj["alpha"] + aj["handling_time"]
                        rhs = ai["alpha"]
                    violated, viol_amt = check_leq(lhs, rhs)
                    if violated:
                        violated_set.add(9)
                        record_violation(violations, magnitudes, 9,
                                         f"Vessels {ai['vessel_id']} and {aj['vessel_id']} overlap at berth {b}: "
                                         f"[{ai['alpha']},{ai['beta']}] and [{aj['alpha']},{aj['beta']}]",
                                         lhs, rhs)

    reported_berths = {}
    malformed_berths = []
    for berth_data in solution["berths"]:
        if not isinstance(berth_data, dict) or "berth" not in berth_data:
            malformed_berths.append(berth_data)
            continue
        berth_id = berth_data["berth"]
        if berth_id in reported_berths or not isinstance(berth_id, int):
            malformed_berths.append(berth_data)
            continue
        reported_berths[berth_id] = berth_data
    if malformed_berths or set(reported_berths) != set(range(num_berths)):
        violated_set.add(20)
        violations.append("berths must contain exactly one entry for each berth")
    for b, b_assignments in berth_assignments.items():
        berth_data = reported_berths.get(b)
        if berth_data is None:
            continue
        expected_rho = min(a["alpha"] for a in b_assignments)
        expected_varsigma = max(a["beta"] for a in b_assignments)
        if (
            berth_data.get("rho") != expected_rho
            or berth_data.get("varsigma") != expected_varsigma
        ):
            violated_set.add(20)
            violations.append(
                f"Berth {b}: reported rho/varsigma must equal actual "
                f"assignment extrema ({expected_rho}, {expected_varsigma})"
            )

    # -------------------------------------------------------------------------
    # Constraint (12): alpha_i >= a^f_i for all i in V
    # -------------------------------------------------------------------------
    for a in assignments:
        lhs = a["alpha"]
        rhs = a["vdata"]["feasible_time_window"][0]
        violated, viol_amt = check_geq(lhs, rhs)
        if violated:
            violated_set.add(12)
            record_violation(violations, magnitudes, 12,
                             f"Vessel {a['vessel_id']}: alpha={lhs} < a^f={rhs}",
                             lhs, rhs)

    # -------------------------------------------------------------------------
    # Constraint (13): beta_i <= b^f_i for all i in V
    # -------------------------------------------------------------------------
    for a in assignments:
        lhs = a["beta"]
        rhs = a["vdata"]["feasible_time_window"][1]
        violated, viol_amt = check_leq(lhs, rhs)
        if violated:
            violated_set.add(13)
            record_violation(violations, magnitudes, 13,
                             f"Vessel {a['vessel_id']}: beta={lhs} > b^f={rhs}",
                             lhs, rhs)

    # -------------------------------------------------------------------------
    # Constraints (16)-(17): QC capacity
    # First compute sigma_t for all t in T = {1,...,H+E}
    # sigma_t = number of QCs used at time step t
    # -------------------------------------------------------------------------
    sigma = {}
    for t in range(1, H + E + 1):
        sigma[t] = 0
    for a in assignments:
        for t, qc in a["qc_per_timestep"].items():
            if t in sigma:
                sigma[t] += qc

    reported_qc = {int(t): value for t, value in solution["qc_usage"].items()}
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in reported_qc.values()
    ):
        violated_set.add(15)
        violations.append("qc_usage values must be finite JSON numbers")
    else:
        qc_mismatches = [
            t for t in range(1, H + E + 1)
            if abs(float(reported_qc.get(t, 0)) - float(sigma[t])) > TOL
        ]
        invalid_times = [t for t in reported_qc if t < 1 or t > H + E]
        if qc_mismatches or invalid_times:
            violated_set.add(15)
            violations.append(
                "Reported qc_usage does not match profile-derived usage"
                f"; mismatched times={qc_mismatches[:20]}, invalid times={invalid_times[:20]}"
            )

    # Constraint (16): sigma_t <= Q_t for t in {E+1,...,H}
    for t in range(E + 1, H + 1):
        lhs = sigma[t]
        rhs = Q_cap[t - 1]  # Q_cap is 0-indexed
        violated, viol_amt = check_leq(lhs, rhs)
        if violated:
            violated_set.add(16)
            record_violation(violations, magnitudes, 16,
                             f"QC capacity at t={t}: sigma={lhs} > Q_t={rhs}",
                             lhs, rhs)

    # Constraint (17): sigma_t + sigma_{t+H} <= Q_t for t in {1,...,E}
    for t in range(1, E + 1):
        lhs = sigma.get(t, 0) + sigma.get(t + H, 0)
        rhs = Q_cap[t - 1]
        violated, viol_amt = check_leq(lhs, rhs)
        if violated:
            violated_set.add(17)
            record_violation(violations, magnitudes, 17,
                             f"QC periodicity at t={t}: sigma_t({sigma.get(t, 0)}) + sigma_{{t+H}}({sigma.get(t + H, 0)}) = {lhs} > Q_t={rhs}",
                             lhs, rhs)

    # -------------------------------------------------------------------------
    # Constraint (20): varsigma_b - rho_b <= H - 1 for all b in B
    # rho_b = min alpha_i for vessels at berth b
    # varsigma_b = max beta_i for vessels at berth b
    # -------------------------------------------------------------------------
    for b in range(num_berths):
        b_vessels = [a for a in assignments if a["berth"] == b]
        if not b_vessels:
            continue
        rho_b = min(a["alpha"] for a in b_vessels)
        varsigma_b = max(a["beta"] for a in b_vessels)
        lhs = varsigma_b - rho_b
        rhs = H - 1
        violated, viol_amt = check_leq(lhs, rhs)
        if violated:
            violated_set.add(20)
            record_violation(violations, magnitudes, 20,
                             f"Berth {b}: varsigma({varsigma_b}) - rho({rho_b}) = {lhs} > H-1={rhs}",
                             lhs, rhs)

    # -------------------------------------------------------------------------
    # Check subblock validity: subblocks should be in range [0, num_subblocks-1]
    # This relates to constraint (23): phi_{ik} in {0,1} / variable domains
    # -------------------------------------------------------------------------
    for a in assignments:
        for k in a["subblocks"]:
            if k < 0 or k >= num_subblocks:
                violated_set.add(23)
                record_violation(violations, magnitudes, 23,
                                 f"Vessel {a['vessel_id']}: subblock {k} out of range [0, {num_subblocks - 1}]",
                                 k, num_subblocks - 1)

    # -------------------------------------------------------------------------
    # Constraint (34): Objective-value consistency (linearized objective).
    #     reported objective_value must equal
    #         sum_i c^p_i * [(a^e_i - alpha_i)^+ + (beta_i - b^e_i)^+]
    #       + c^o * sum_i sum_{k in subblocks_i} D^L_{k, berth_i} * (l_i / r_i)
    #       + c^o * sum_i D^U_{berth_i} * u_i
    # Tier-C defense against score-gaming exploits. Every variable required
    # by the objective formula is present in the solution (alpha, beta,
    # berth, subblocks), so a full recompute is appropriate.
    # Skipped if instance fields or the reported objective are missing /
    # non-numeric (no fabrication path is exploitable in that case).
    # -------------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
    else:
        reported = None

    c_o = instance.get("penalty_cost_coefficient_co")
    D_L = instance.get("loading_route_lengths_D_L")
    D_U = instance.get("average_unloading_route_lengths_D_U")

    if reported is not None and c_o is not None and D_L is not None and D_U is not None:
        true_obj = 0.0
        recompute_ok = True
        for a in assignments:
            vdata = a["vdata"]
            berth = a["berth"]
            alpha = a["alpha"]
            beta = a["beta"]
            try:
                c_p_i = float(vdata["penalty_cost_coefficient"])
                a_e, b_e = vdata["expected_time_window"]
                l_i = float(vdata["loading_containers"])
                u_i = float(vdata["unloading_containers"])
                r_i = vdata["reserved_subblocks"]
                # Penalty for deviation from expected service window
                true_obj += c_p_i * (max(0, a_e - alpha) + max(0, beta - b_e))
                # Unloading transportation cost (independent of subblocks)
                if 0 <= berth < len(D_U):
                    true_obj += c_o * D_U[berth] * u_i
                else:
                    # Out-of-range berth; skip obj recompute (constraint 5 will fire)
                    recompute_ok = False
                    break
                # Loading transportation cost over reserved subblocks
                if r_i and r_i > 0:
                    for k in a["subblocks"]:
                        if 0 <= k < len(D_L) and 0 <= berth < len(D_L[k]):
                            true_obj += c_o * D_L[k][berth] * (l_i / r_i)
                        else:
                            recompute_ok = False
                            break
                    if not recompute_ok:
                        break
            except (KeyError, TypeError, ValueError, IndexError):
                recompute_ok = False
                break

        if recompute_ok:
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor.
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                violated_set.add(34)
                record_violation(
                    violations, magnitudes, 34,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed linearised objective="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={tol:.3g})",
                    reported, true_obj,
                )

    # -------------------------------------------------------------------------
    # Build result
    # -------------------------------------------------------------------------
    violated_constraints = sorted(violated_set)
    feasible = len(violated_constraints) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": magnitudes if not feasible else [],
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Integrated Berth Allocation, "
                    "QC Assignment, and Yard Assignment Problem")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = feasibility_check(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE - no constraint violations detected")
    else:
        print(f"INFEASIBLE - {len(result['violated_constraints'])} constraint(s) violated: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _unexpected_error:
        if not _write_unexpected_rejection(_unexpected_error):
            raise
