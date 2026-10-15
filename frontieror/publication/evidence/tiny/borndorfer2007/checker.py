#!/usr/bin/env python3
"""
Feasibility checker for the Line Planning Problem (LPP).

Paper: "A Column-Generation Approach to Line Planning in Public Transport"
Authors: Borndoerfer, Groetschel, Pfetsch (2007), Transportation Science 41(1), pp. 123-132.

Constraints (numbered top-to-bottom from the mathematical formulation):
  (1) y(P_st) = d_st                                for all (s,t) in D
  (2) y(P_a) - sum_{l: e(a) in l} kappa_l f_l <= 0  for all a in A
  (3) f(L_e) <= Lambda_e                             for all e in E
  (4) f_l <= F x_l                                   for all l in L
  (5) x_l in {0, 1}                                  for all l in L
  (6) f_l >= 0                                       for all l in L
  (7) y_p >= 0                                       for all p in P
  (8) [Tier C defense] objective_value reported by the program must be
      consistent with the variables in the solution. Specifically:
        obj_true = lambda * (sum_l C_l + sum_l f_l * sum_{e in l} c_e)
                   + (1 - lambda) * sum_a tau_a * y_a
      The first term (line costs) is exactly recomputable from active_lines.
      The second term (passenger travel time) is recomputed exactly when the
      solution provides either `arc_total_flow` or `active_passenger_paths`.
      When neither is provided (e.g. the Gurobi reference, which writes only
      the line plan), we bracket the travel time with
        lower = sum_{(s,t)} d_st * shortest_path_tt(s,t)  (full graph, no cap)
        upper = sum_a cap_a * tau_a                        (every arc full)
      and require the reported objective to lie in [obj_lb, obj_ub] within
      tolerance.

Since the solution files store only the line plan (active lines with edges and
frequencies) but not individual passenger path flows, constraints (1), (2), and (7)
are verified by solving a feasibility LP that checks whether a valid passenger
routing exists given the line capacities.
"""

import argparse
import heapq
import json
import sys
from collections import defaultdict

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = (
    'active_lines',
    'active_passenger_paths',
    'objective_value',
)
_FRONTIEROR_NONEMPTY_SOLUTION_FIELDS = (
    'active_lines',
    'active_passenger_paths',
)
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


# Numerical tolerances. Gurobi default FeasibilityTol/IntFeasTol = 1e-6,
# but accumulated FP error on multi-thousand-arc instances can drift to ~1e-5.
# 1e-4 absolute floor + 1e-5 relative gives ~10x slack over solver default --
# enough to absorb floating-point noise without masking real violations.
tol = 1e-4
eps = 1e-5
rel_tol = 1e-5


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def compute_violation(lhs, rhs, op):
    """
    Compute violation_amount for a constraint.
    op: 'le' (<=), 'ge' (>=), 'eq' (=)
    """
    if op == 'le':
        return max(0.0, lhs - rhs)
    elif op == 'ge':
        return max(0.0, rhs - lhs)
    elif op == 'eq':
        return abs(lhs - rhs)
    return 0.0


def record_violation(violations_list, magnitudes_list, constraint_idx, msg, lhs, rhs, op):
    """Check and record a violation if violation_amount > tol (absolute or relative)."""
    va = compute_violation(lhs, rhs, op)
    threshold = max(tol, rel_tol * max(abs(lhs), abs(rhs)))
    if va > threshold:
        violations_list.append((constraint_idx, msg))
        normalizer = max(abs(rhs), eps)
        magnitudes_list.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": va,
            "normalizer": normalizer,
            "ratio": va / normalizer,
        })
        return True
    return False


def _shortest_path_demand_sum(edges_data, od_pairs):
    """Lower bound on sum_p tau_p * y_p: each passenger takes the shortest
    s-t path on the full undirected graph (capacity ignored). Since the actual
    feasible set is a subset (passengers may only use arcs covered by active
    lines and subject to capacity), the LP-optimal travel time is at least
    this much."""
    adj = defaultdict(list)
    for e in edges_data:
        u, v = e["endpoints"]
        tt_e = float(e["traveling_time_seconds"])
        adj[u].append((v, tt_e))
        adj[v].append((u, tt_e))
    od_by_origin = defaultdict(list)
    for od in od_pairs:
        od_by_origin[od["origin"]].append((od["destination"], float(od["demand"])))
    total = 0.0
    for s, dest_demand in od_by_origin.items():
        dist = {s: 0.0}
        heap = [(0.0, s)]
        remaining = {d for d, _ in dest_demand}
        while heap and remaining:
            du, u = heapq.heappop(heap)
            if du > dist.get(u, float('inf')):
                continue
            if u in remaining:
                remaining.discard(u)
            for v, w in adj[u]:
                nd = du + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        for d, demand in dest_demand:
            sp = dist.get(d, None)
            if sp is None:
                # Unreachable -- contributes 0 to lower bound (cannot tighten).
                continue
            total += sp * demand
    return total


def _check_objective_consistency(
    violations, magnitudes,
    solution, params, mode_by_name, edge_by_id, lines,
    arc_travel_time, arc_cap, num_edges, od_pairs,
    edges_data,
    arc_total_flow_in, active_paths,
):
    """Constraint (8): reported objective_value must be consistent with the
    solution variables. See module docstring for the formula and the
    full-recompute vs bracketed cases."""
    reported_obj = solution.get("objective_value")
    if reported_obj is None:
        return
    try:
        reported = float(reported_obj)
    except (TypeError, ValueError):
        return
    if reported != reported:  # NaN
        return

    try:
        lam = float(params.get("lambda", 0.0))
    except (TypeError, ValueError):
        lam = 0.0

    # === Exact line cost from active_lines + mode/edge data ===
    line_cost_exact = 0.0
    for line in lines:
        mode = mode_by_name.get(line["mode_name"])
        if mode is None:
            continue
        try:
            C_l = float(mode.get("fixed_cost_per_line", 0.0))
        except (TypeError, ValueError):
            C_l = 0.0
        c_l_total = 0.0
        for eid in line["edges"]:
            edge = edge_by_id.get(eid)
            if edge is None:
                continue
            try:
                c_l_total += float(edge.get("operating_cost", 0.0))
            except (TypeError, ValueError):
                pass
        try:
            f_l = float(line["frequency"])
        except (TypeError, ValueError):
            f_l = 0.0
        line_cost_exact += C_l + c_l_total * f_l

    # === Travel time component ===
    tt_actual = None
    if arc_total_flow_in is not None:
        tt_actual = 0.0
        for aid, f in arc_total_flow_in.items():
            try:
                aid_norm = int(aid)
            except (ValueError, TypeError):
                aid_norm = aid
            try:
                tt_actual += float(f) * arc_travel_time.get(aid_norm, 0.0)
            except (TypeError, ValueError):
                pass
    elif active_paths is not None:
        tt_actual = 0.0
        for ap in active_paths:
            try:
                fl = float(ap.get("flow", 0.0))
            except (TypeError, ValueError):
                fl = 0.0
            path_tt = 0.0
            for aid in ap.get("arcs", []) or []:
                try:
                    aid_norm = int(aid)
                except (ValueError, TypeError):
                    aid_norm = aid
                path_tt += arc_travel_time.get(aid_norm, 0.0)
            tt_actual += fl * path_tt

    if tt_actual is not None:
        true_obj = lam * line_cost_exact + (1.0 - lam) * tt_actual
        record_violation(
            violations, magnitudes, 8,
            f"Objective consistency violated: reported objective_value={reported} "
            f"differs from recomputed lambda*line_cost + (1-lambda)*passenger_travel_time"
            f"={true_obj} (line_cost={line_cost_exact}, tt={tt_actual}, lambda={lam})",
            reported, true_obj, 'eq',
        )
        return

    # === Bracketed check (no flow data: e.g. Gurobi reference) ===
    tt_lb = _shortest_path_demand_sum(edges_data, od_pairs)
    tt_ub = 0.0
    num_arcs = 2 * num_edges
    for a in range(num_arcs):
        tt_ub += arc_cap.get(a, 0.0) * arc_travel_time.get(a, 0.0)

    obj_lb = lam * line_cost_exact + (1.0 - lam) * tt_lb
    obj_ub = lam * line_cost_exact + (1.0 - lam) * tt_ub

    record_violation(
        violations, magnitudes, 8,
        f"Objective below lower bound: reported objective_value={reported} < "
        f"lb={obj_lb} = lambda*line_cost + (1-lambda)*sum_(s,t) d_st*shortest_path_tt "
        f"(line_cost={line_cost_exact}, tt_lb={tt_lb}, lambda={lam})",
        reported, obj_lb, 'ge',
    )
    record_violation(
        violations, magnitudes, 8,
        f"Objective above upper bound: reported objective_value={reported} > "
        f"ub={obj_ub} = lambda*line_cost + (1-lambda)*sum_a (cap_a*tau_a) "
        f"(line_cost={line_cost_exact}, tt_ub={tt_ub}, lambda={lam})",
        reported, obj_ub, 'le',
    )


def check_feasibility(instance, solution):
    """Check all hard constraints of the LPP formulation."""
    violations = []       # list of (constraint_idx, message)
    magnitudes = []       # list of violation magnitude dicts

    # ---- Handle degenerate solutions ----
    if solution.get("objective_value") is None or solution.get("status") in ("no_solution", "infeasible"):
        # No actual solution to check -- return null feasibility
        return {
            "feasible": None,
            "violated_constraints": [],
            "violations": [f"No solution to check (status={solution.get('status', 'unknown')})"],
            "violation_magnitudes": [],
        }

    # ---- Extract instance data ----
    params = instance["global_parameters"]
    F = params["frequency_upper_bound_F"]
    edges_data = instance["network"]["edges"]
    num_nodes = instance["network"]["num_nodes"]
    num_edges = instance["network"]["num_edges"]
    od_pairs = instance["od_matrix"]
    modes = instance["modes"]

    # Build mode lookup by name
    mode_by_name = {}
    for m in modes:
        mode_by_name[m["name"]] = m

    # Edge data lookup
    edge_by_id = {}
    for e in edges_data:
        edge_by_id[e["id"]] = e

    # Edge capacity (Lambda_e)
    edge_capacity = {}
    for e in edges_data:
        edge_capacity[e["id"]] = e["edge_capacity"]

    # ---- Extract solution data ----
    active_lines = solution.get("active_lines", [])

    # Build line data: each line has edges, frequency, mode info
    lines = []
    for al in active_lines:
        mode_name = al["mode"]
        mode = mode_by_name[mode_name]
        lines.append({
            "edges": al["edges"],
            "frequency": al["frequency"],
            "mode_name": mode_name,
            "kappa": mode["vehicle_capacity"],
            "x": 1,  # line is active => x_l = 1
        })

    # =====================================================================
    # Constraint (5): x_l in {0, 1}
    # All active lines have x_l = 1 implicitly. Check frequency > 0 lines
    # have valid binary x. Since solution only lists active lines, x_l=1 for
    # all listed lines, which is trivially binary.
    # =====================================================================
    for i, line in enumerate(lines):
        x_val = line["x"]
        record_violation(violations, magnitudes, 5,
                         f"Line {i}: x_l = {x_val} is not in {{0, 1}}",
                         x_val, round(x_val), 'eq')

    # =====================================================================
    # Constraint (6): f_l >= 0  for all l in L
    # =====================================================================
    for i, line in enumerate(lines):
        f_l = line["frequency"]
        record_violation(violations, magnitudes, 6,
                         f"Line {i}: frequency f_l = {f_l} < 0",
                         -f_l, 0.0, 'le')  # -f_l <= 0 means f_l >= 0

    # =====================================================================
    # Constraint (4): f_l <= F * x_l  for all l in L
    # For active lines, x_l = 1, so f_l <= F.
    # =====================================================================
    for i, line in enumerate(lines):
        f_l = line["frequency"]
        x_l = line["x"]
        rhs = F * x_l
        record_violation(violations, magnitudes, 4,
                         f"Line {i}: frequency f_l = {f_l:.6f} > F * x_l = {rhs:.6f}",
                         f_l, rhs, 'le')

    # =====================================================================
    # Constraint (3): f(L_e) <= Lambda_e  for all e in E
    # Total frequency of all lines using edge e must not exceed edge capacity.
    # =====================================================================
    edge_freq_sum = defaultdict(float)
    for line in lines:
        for eid in line["edges"]:
            edge_freq_sum[eid] += line["frequency"]

    for eid in range(num_edges):
        Lambda_e = edge_capacity[eid]
        freq_sum = edge_freq_sum.get(eid, 0.0)
        record_violation(violations, magnitudes, 3,
                         f"Edge {eid}: total frequency {freq_sum:.6f} > edge capacity Lambda_e = {Lambda_e}",
                         freq_sum, Lambda_e, 'le')

    # =====================================================================
    # Constraints (1), (2), (7): Passenger flow feasibility
    #
    # (1) y(P_st) = d_st  for all (s,t) in D
    # (2) y(P_a) - sum_{l: e(a) in l} kappa_l f_l <= 0  for all a in A
    # (7) y_p >= 0
    #
    # Since the solution does not include individual path flows y_p, we check
    # whether a feasible passenger routing EXISTS by solving an LP:
    #   Find y >= 0 such that flow conservation (1) and capacity (2) hold.
    #
    # We formulate this as a multi-commodity flow on the directed arc graph.
    # Each OD pair (s,t) is a commodity with demand d_st.
    # Arc capacities come from the lines: cap_a = sum_{l: e(a) in l} kappa_l f_l.
    # =====================================================================

    # Build directed arcs from undirected edges
    # arc 2*eid: u->v, arc 2*eid+1: v->u
    arc_cap = defaultdict(float)  # arc_id -> capacity from lines
    for line in lines:
        kappa = line["kappa"]
        f_l = line["frequency"]
        for eid in line["edges"]:
            arc_cap[2 * eid] += kappa * f_l
            arc_cap[2 * eid + 1] += kappa * f_l

    # Build adjacency for directed arcs
    arc_adj = defaultdict(list)  # node -> list of (neighbor, arc_id)
    arc_travel_time = {}
    arc_endpoints = {}
    for e in edges_data:
        eid = e["id"]
        u, v = e["endpoints"]
        tt_e = e["traveling_time_seconds"]
        fwd = 2 * eid
        bwd = 2 * eid + 1
        arc_adj[u].append((v, fwd))
        arc_adj[v].append((u, bwd))
        arc_travel_time[fwd] = tt_e
        arc_travel_time[bwd] = tt_e
        arc_endpoints[fwd] = (u, v)
        arc_endpoints[bwd] = (v, u)

    # =====================================================================
    # Fast path A (preferred): solution provides verification aggregates
    # `od_served` and `arc_total_flow`. These are formulation-agnostic -- any
    # solver (paper's or LLM-generated, path-based or arc-based MCF) can
    # aggregate its own flows once at the end without committing to a
    # particular formulation. Checker uses these directly to verify (1)(2),
    # avoiding the LP slow path entirely.
    # =====================================================================
    od_served = solution.get("od_served", None)
    arc_total_flow_in = solution.get("arc_total_flow", None)
    active_paths = solution.get("active_passenger_paths", None)
    if (od_served is not None and arc_total_flow_in is not None
            and not active_paths):
        flow_per_od = defaultdict(float)
        for entry in od_served:
            flow_per_od[(entry["origin"], entry["destination"])] += entry["flow"]
        # arc_total_flow is dict; keys may be str or int depending on serializer
        flow_per_arc = {}
        for aid, f in arc_total_flow_in.items():
            try:
                aid_norm = int(aid)
            except (ValueError, TypeError):
                aid_norm = aid
            flow_per_arc[aid_norm] = float(f)

        for od in od_pairs:
            key = (od["origin"], od["destination"])
            d = od["demand"]
            served = flow_per_od.get(key, 0.0)
            record_violation(violations, magnitudes, 1,
                             f"OD ({od['origin']}, {od['destination']}): "
                             f"served {served:.6f} != demand {d}",
                             served, d, 'eq')

        for aid, total_flow in flow_per_arc.items():
            cap = arc_cap.get(aid, 0.0)
            record_violation(violations, magnitudes, 2,
                             f"Arc {aid}: total flow {total_flow:.6f} "
                             f"> capacity {cap:.6f}",
                             total_flow, cap, 'le')

        # Constraint (8): objective consistency.
        _check_objective_consistency(
            violations, magnitudes,
            solution, params, mode_by_name, edge_by_id, lines,
            arc_travel_time, arc_cap, num_edges, od_pairs,
            edges_data, arc_total_flow_in, active_paths,
        )

        return {
            "feasible": len(violations) == 0,
            "violated_constraints": violations,
            "violations": [v[1] for v in violations],
            "violation_magnitudes": magnitudes,
        }

    # =====================================================================
    # Fast path B (fallback): solution has `active_passenger_paths` (path-based
    # solvers). Aggregate per-OD and per-arc flows from path list, then verify
    # (1)(2). Same constraints, just different input format.
    # =====================================================================
    if active_paths:
        # Constraint (1): per-OD demand met by sum of path flows
        flow_per_od = defaultdict(float)
        # Constraint (2): per-arc total flow <= capacity from lines
        flow_per_arc = defaultdict(float)
        valid_od_keys = {
            (od["origin"], od["destination"]) for od in od_pairs
        }
        for ap in active_paths:
            key = (ap["origin"], ap["destination"])
            try:
                path_flow = float(ap["flow"])
            except (TypeError, ValueError):
                path_flow = -1.0
            arcs = ap.get("arcs", [])
            path_valid = key in valid_od_keys and path_flow >= -tol
            current = ap["origin"]
            if not isinstance(arcs, list) or (not arcs and key[0] != key[1]):
                path_valid = False
            else:
                for raw_aid in arcs:
                    try:
                        aid = int(raw_aid)
                    except (TypeError, ValueError):
                        path_valid = False
                        break
                    endpoints = arc_endpoints.get(aid)
                    if endpoints is None or endpoints[0] != current:
                        path_valid = False
                        break
                    current = endpoints[1]
                if current != ap["destination"]:
                    path_valid = False
            if not path_valid:
                record_violation(
                    violations, magnitudes, 7,
                    f"Passenger path {key} is not a valid directed path with "
                    f"nonnegative flow: arcs={arcs}, flow={ap.get('flow')}",
                    0.0, 1.0, 'eq',
                )
                continue
            flow_per_od[key] += path_flow
            for aid in arcs:
                flow_per_arc[int(aid)] += path_flow

        for od in od_pairs:
            key = (od["origin"], od["destination"])
            d = od["demand"]
            served = flow_per_od.get(key, 0.0)
            record_violation(violations, magnitudes, 1,
                             f"OD ({od['origin']}, {od['destination']}): "
                             f"sum y_p = {served:.6f} != demand {d}",
                             served, d, 'eq')

        for aid, total_flow in flow_per_arc.items():
            cap = arc_cap.get(aid, 0.0)
            if total_flow > cap + tol:
                record_violation(violations, magnitudes, 2,
                                 f"Arc {aid}: total flow {total_flow:.6f} "
                                 f"> capacity {cap:.6f}",
                                 total_flow, cap, 'le')

        # Constraint (8): objective consistency.
        _check_objective_consistency(
            violations, magnitudes,
            solution, params, mode_by_name, edge_by_id, lines,
            arc_travel_time, arc_cap, num_edges, od_pairs,
            edges_data, arc_total_flow_in, active_paths,
        )

        return {
            "feasible": len(violations) == 0,
            "violated_constraints": violations,
            "violations": [v[1] for v in violations],
            "violation_magnitudes": magnitudes,
        }

    # No complete passenger certificate.  The old fallback allocated
    # |OD|*|arcs| variables (over 21 million on the stored large instances),
    # could exhaust evaluator memory, and still used only a broad objective
    # envelope.  Fail closed and ask the solver to emit the already-supported
    # path or aggregate witness; patched gurobi_code.py emits both.
    record_violation(
        violations, magnitudes, 1,
        "Missing passenger-flow certificate: provide nonempty "
        "active_passenger_paths, or both od_served and arc_total_flow",
        0.0, 1.0, 'eq',
    )
    return {
        "feasible": False,
        "violated_constraints": sorted(set(c for c, _ in violations)),
        "violations": [message for _, message in violations],
        "violation_magnitudes": magnitudes,
    }

    # Legacy unreachable implementation retained below for provenance.
    # Slow path: solve multi-commodity flow LP via scipy linprog
    try:
        from scipy.optimize import linprog
        from scipy.sparse import lil_matrix
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    if _has_scipy and od_pairs:
        # Variables: y_{a,k} for each arc a and commodity k
        # where k indexes OD pairs
        num_od = len(od_pairs)
        num_arcs = 2 * num_edges

        # Variable index: k * num_arcs + a
        num_vars = num_od * num_arcs

        # Objective: minimize sum of flows (just find feasible)
        c_obj = [0.0] * num_vars

        # Equality constraints: flow conservation for each (commodity, node)
        # For commodity k with OD (s_k, t_k, d_k):
        #   sum_{a out of v} y_{a,k} - sum_{a into v} y_{a,k} = d_k if v=s_k
        #                                                       = -d_k if v=t_k
        #                                                       = 0 otherwise

        # Build incidence: for each node, list of (arc_id, +1 if outgoing, -1 if incoming)
        node_arcs = defaultdict(list)  # node -> list of (arc_id, sign)
        for e in edges_data:
            eid = e["id"]
            u, v = e["endpoints"]
            fwd = 2 * eid
            bwd = 2 * eid + 1
            node_arcs[u].append((fwd, +1))   # u -> v: outgoing from u
            node_arcs[v].append((fwd, -1))   # u -> v: incoming to v
            node_arcs[v].append((bwd, +1))   # v -> u: outgoing from v
            node_arcs[u].append((bwd, -1))   # v -> u: incoming to u

        # Equality constraints
        eq_rows = []
        eq_rhs = []
        for k, od in enumerate(od_pairs):
            s_k = od["origin"]
            t_k = od["destination"]
            d_k = od["demand"]
            for v in range(num_nodes):
                row = {}
                for arc_id, sign in node_arcs[v]:
                    col = k * num_arcs + arc_id
                    row[col] = row.get(col, 0.0) + sign
                if row:  # only add if node has arcs
                    eq_rows.append(row)
                    if v == s_k:
                        eq_rhs.append(float(d_k))
                    elif v == t_k:
                        eq_rhs.append(-float(d_k))
                    else:
                        eq_rhs.append(0.0)

        # Inequality constraints: sum_k y_{a,k} <= cap_a  for each arc a
        ineq_rows = []
        ineq_rhs = []
        for a in range(num_arcs):
            cap_a = arc_cap.get(a, 0.0)
            row = {}
            for k in range(num_od):
                col = k * num_arcs + a
                row[col] = 1.0
            if row:
                ineq_rows.append(row)
                ineq_rhs.append(cap_a)

        # Build sparse matrices
        n_eq = len(eq_rows)
        n_ineq = len(ineq_rows)

        A_eq = lil_matrix((n_eq, num_vars))
        b_eq = eq_rhs
        for i, row in enumerate(eq_rows):
            for col, val in row.items():
                A_eq[i, col] = val

        A_ub = lil_matrix((n_ineq, num_vars))
        b_ub = ineq_rhs
        for i, row in enumerate(ineq_rows):
            for col, val in row.items():
                A_ub[i, col] = val

        bounds = [(0.0, None)] * num_vars

        result = linprog(
            c_obj,
            A_ub=A_ub.tocsc(), b_ub=b_ub,
            A_eq=A_eq.tocsc(), b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'presolve': True, 'time_limit': 300},
        )

        if not result.success:
            # Passenger flow is infeasible -- determine which constraints are violated.
            # Check constraint (1): can demand be routed at all (ignoring capacity)?
            # Check constraint (2): is there enough capacity?

            def bfs_reachable(src, adj, n):
                visited = set()
                queue = [src]
                visited.add(src)
                while queue:
                    u = queue.pop(0)
                    for v, _ in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            queue.append(v)
                return visited

            # Re-solve with relaxed capacity to see if it's a capacity issue
            # vs connectivity issue
            # Try uncapacitated version (only flow conservation)
            result_uncp = linprog(
                c_obj,
                A_eq=A_eq.tocsc(), b_eq=b_eq,
                bounds=bounds,
                method='highs',
                options={'presolve': True, 'time_limit': 60},
            )

            if not result_uncp.success:
                # Even without capacity, flow conservation fails
                # This means the network is disconnected for some OD pair
                # or the arc graph doesn't allow routing
                for k, od in enumerate(od_pairs):
                    s_k = od["origin"]
                    t_k = od["destination"]
                    d_k = od["demand"]
                    # Check connectivity via arcs that have capacity > 0
                    # (i.e., covered by at least one line)
                    covered_adj = defaultdict(list)
                    for e in edges_data:
                        eid = e["id"]
                        u, v = e["endpoints"]
                        # Only include arcs with capacity (from active lines)
                        if arc_cap.get(2 * eid, 0.0) > 0:
                            covered_adj[u].append((v, 2 * eid))
                        if arc_cap.get(2 * eid + 1, 0.0) > 0:
                            covered_adj[v].append((u, 2 * eid + 1))

                    reachable = bfs_reachable(s_k, covered_adj, num_nodes)
                    if t_k not in reachable:
                        record_violation(violations, magnitudes, 1,
                                         f"OD ({s_k},{t_k}): demand {d_k} cannot be routed; "
                                         f"destination not reachable from origin via active lines",
                                         0.0, float(d_k), 'eq')
            else:
                # Uncapacitated is feasible but capacitated is not
                # => capacity constraint (2) is violated
                # Find which arcs are bottlenecks
                y_uncp = result_uncp.x
                for a in range(num_arcs):
                    cap_a = arc_cap.get(a, 0.0)
                    total_flow = sum(y_uncp[k * num_arcs + a] for k in range(num_od))
                    if total_flow > cap_a + tol:
                        eid = a // 2
                        e = edge_by_id[eid]
                        u, v = e["endpoints"]
                        direction = f"{u}->{v}" if a % 2 == 0 else f"{v}->{u}"
                        record_violation(violations, magnitudes, 2,
                                         f"Arc {a} ({direction}): passenger flow {total_flow:.4f} "
                                         f"exceeds line capacity {cap_a:.4f}",
                                         total_flow, cap_a, 'le')

                # Also record constraint (1) as violated since demand can't be met
                # with capacity limits
                for k, od in enumerate(od_pairs):
                    s_k = od["origin"]
                    t_k = od["destination"]
                    d_k = od["demand"]
                    # Check how much demand could actually be routed
                    # by solving max-flow per commodity (simplified: use the LP result)
                    record_violation(violations, magnitudes, 1,
                                     f"OD ({s_k},{t_k}): demand {d_k} may not be fully satisfiable "
                                     f"due to insufficient arc capacity from active lines",
                                     0.0, float(d_k), 'eq')
        # If result.success, constraints (1), (2), (7) are all satisfied
        # (y >= 0 is enforced by bounds, flow conservation by A_eq, capacity by A_ub)

    elif not _has_scipy and od_pairs:
        # Cannot check passenger flow constraints without scipy
        # Skip with a warning -- do not record as violation
        pass

    # =====================================================================
    # Constraint (8): objective consistency. See _check_objective_consistency.
    # =====================================================================
    _check_objective_consistency(
        violations, magnitudes,
        solution, params, mode_by_name, edge_by_id, lines,
        arc_travel_time, arc_cap, num_edges, od_pairs,
        edges_data, arc_total_flow_in, active_paths,
    )

    # =====================================================================
    # Compile results
    # =====================================================================
    violated_indices = sorted(set(c for c, _ in violations))
    violation_msgs = [msg for _, msg in violations]

    feasible = len(violated_indices) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": violation_msgs,
        "violation_magnitudes": magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Line Planning Problem (LPP)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)
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

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE: No constraint violations found.")
    else:
        print(f"INFEASIBLE: {len(result['violated_constraints'])} constraint(s) violated: "
              f"{result['violated_constraints']}")
        for msg in result["violations"]:
            print(f"  - {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as _frontieror_unexpected_error:
        if not _frontieror_write_unexpected_rejection(
            _frontieror_unexpected_error
        ):
            raise
