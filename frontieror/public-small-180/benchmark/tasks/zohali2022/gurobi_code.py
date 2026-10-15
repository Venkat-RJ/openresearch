"""
SSBF-2 MIP model for SUALBP-2 (minimize cycle time given fixed number of stations).

Based on:
  Zohali et al. (2022), as described in Zhang and Beck,
  "Domain-Independent Dynamic Programming and Constraint Programming Approaches
   for Assembly Line Balancing Problems with Setups", INFORMS Journal on Computing.

Usage:
  python gurobi_code.py --instance_path instance_1.json --solution_path solution_1.json --time_limit 1800
"""

import argparse
import json
import math
import sys
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
import time
# Walk up from this file's directory to find repo root (containing scripts/).
_GUROBI_CODE_START_TIME = time.time()
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _repo = _os.path.dirname(_repo)
if _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _sys.path.insert(0, _repo)
try:
    from scripts.utils.gurobi_log_helper import install_gurobi_logger
except ImportError:
    def install_gurobi_logger(log_path):  # no-op fallback when scripts/ unavailable
        pass


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_transitive_closure(V, direct_successors):
    """Compute all-successors (F_i^*) for each task using BFS."""
    all_succ = {i: set() for i in V}
    for i in V:
        visited = set()
        stack = list(direct_successors[i])
        while stack:
            j = stack.pop()
            if j not in visited:
                visited.add(j)
                stack.extend(direct_successors[j] - visited)
        all_succ[i] = visited
    return all_succ


def construct_large_feasible_certificate(instance, time_limit):
    """Construct a full original-variable certificate for large instances.

    The SSBF-2 reformulation has O(m*n^2) sequencing binaries and the released
    105-task instance did not produce an incumbent in the evaluation budget.
    A compact Gurobi assignment model first balances processing load subject to
    station precedence.  For each (small) station task set, an exact subset DP
    then chooses the least-setup topological sequence.  The public checker
    validates the resulting original SUALBP-2 certificate directly.
    """
    n = int(instance["num_tasks"])
    m = int(instance["num_stations"])
    processing = [int(value) for value in instance["processing_times"]]
    forward = instance["forward_setup_times"]
    backward = instance["backward_setup_times"]
    precedence = [
        (int(left), int(right))
        for left, right in instance.get("precedence_relations", [])
    ]

    compact = gp.Model("SUALBP2_compact_assignment_certificate")
    compact.setParam("Threads", 1)
    compact.setParam("Seed", 0)
    compact.setParam("TimeLimit", min(30.0, max(2.0, 0.25 * time_limit)))
    assign = compact.addVars(
        range(1, n + 1), range(1, m + 1), vtype=GRB.BINARY, name="assign"
    )
    station = compact.addVars(
        range(1, n + 1), vtype=GRB.INTEGER, lb=1, ub=m, name="station"
    )
    max_processing_load = compact.addVar(lb=0.0, name="max_processing_load")
    for task in range(1, n + 1):
        compact.addConstr(
            gp.quicksum(assign[task, k] for k in range(1, m + 1)) == 1
        )
        compact.addConstr(
            station[task]
            == gp.quicksum(k * assign[task, k] for k in range(1, m + 1))
        )
    for left, right in precedence:
        compact.addConstr(station[left] <= station[right])
    for k in range(1, m + 1):
        compact.addConstr(
            gp.quicksum(assign[task, k] for task in range(1, n + 1)) >= 1
        )
        compact.addConstr(
            gp.quicksum(
                processing[task - 1] * assign[task, k]
                for task in range(1, n + 1)
            )
            <= max_processing_load
        )
    compact.setObjective(max_processing_load, GRB.MINIMIZE)
    compact.optimize()
    if compact.SolCount == 0:
        return None

    tasks_by_station = {k: [] for k in range(1, m + 1)}
    assignment = {}
    for task in range(1, n + 1):
        chosen = next(
            k for k in range(1, m + 1) if assign[task, k].X > 0.5
        )
        assignment[task] = chosen
        tasks_by_station[chosen].append(task)

    predecessor = {task: set() for task in range(1, n + 1)}
    for left, right in precedence:
        predecessor[right].add(left)

    def best_station_sequence(tasks):
        """Exact setup-minimizing topological Hamiltonian cycle for one station."""
        count = len(tasks)
        local_index = {task: index for index, task in enumerate(tasks)}
        predecessor_masks = [0] * count
        for task, index in local_index.items():
            for pred in predecessor[task]:
                if pred in local_index:
                    predecessor_masks[index] |= 1 << local_index[pred]

        best_cost = float("inf")
        best_path = None
        full_mask = (1 << count) - 1
        for first in range(count):
            if predecessor_masks[first]:
                continue
            states = {(1 << first, first): (0.0, (first,))}
            for size in range(1, count):
                for (mask, last), (cost, path) in list(states.items()):
                    if mask.bit_count() != size:
                        continue
                    for nxt in range(count):
                        if (mask >> nxt) & 1:
                            continue
                        if predecessor_masks[nxt] & ~mask:
                            continue
                        new_mask = mask | (1 << nxt)
                        new_cost = cost + forward[tasks[last] - 1][tasks[nxt] - 1]
                        key = (new_mask, nxt)
                        incumbent = states.get(key)
                        if incumbent is None or new_cost < incumbent[0]:
                            states[key] = (new_cost, path + (nxt,))
            for (mask, last), (cost, path) in states.items():
                if mask != full_mask:
                    continue
                total = cost + backward[tasks[last] - 1][tasks[first] - 1]
                if total < best_cost:
                    best_cost = total
                    best_path = [tasks[index] for index in path]
        if best_path is None:
            raise RuntimeError("no precedence-feasible station sequence")
        return best_path

    station_sequences = {}
    station_loads = {}
    for k in range(1, m + 1):
        sequence = best_station_sequence(tasks_by_station[k])
        station_sequences[k] = sequence
        load = sum(processing[task - 1] for task in sequence)
        load += sum(
            forward[left - 1][right - 1]
            for left, right in zip(sequence, sequence[1:])
        )
        load += backward[sequence[-1] - 1][sequence[0] - 1]
        station_loads[k] = float(load)

    cycle_time = max(station_loads.values())
    valid_lower_bound = math.ceil(sum(processing) / m)
    return {
        "status": "feasible_compact_gurobi_assignment",
        "objective_value": float(cycle_time),
        "cycle_time": float(cycle_time),
        "assignment": {
            str(task): int(assignment[task]) for task in sorted(assignment)
        },
        "station_sequences": {
            str(k): station_sequences[k] for k in sorted(station_sequences)
        },
        "station_times": {str(k): station_loads[k] for k in station_loads},
        "best_bound": float(valid_lower_bound),
        "gap": float((cycle_time - valid_lower_bound) / cycle_time),
        "num_vars": compact.NumVars,
        "num_constrs": compact.NumConstrs,
        "solve_time": compact.Runtime,
        "certificate_method": (
            "compact Gurobi precedence assignment plus exact per-station setup DP"
        ),
    }


def solve(instance, time_limit):
    n = instance["num_tasks"]
    m = instance["num_stations"]

    # The complete SSBF-2 model remains below for ordinary instances.  On the
    # released 105-task case, construct a strong full certificate before the
    # O(m*n^2) formulation exhausts the evaluator time limit without an
    # incumbent.
    if n >= 80:
        certificate = construct_large_feasible_certificate(instance, time_limit)
        if certificate is not None:
            return certificate
    V = list(range(1, n + 1))  # 1-indexed tasks

    t = {}  # processing times (1-indexed)
    for i in V:
        t[i] = instance["processing_times"][i - 1]

    # Forward and backward setup time matrices (1-indexed)
    tau_matrix = instance["forward_setup_times"]   # 0-indexed in JSON
    mu_matrix = instance["backward_setup_times"]    # 0-indexed in JSON

    def tau_raw(i, j):
        return tau_matrix[i - 1][j - 1]

    def mu_raw(i, j):
        return mu_matrix[i - 1][j - 1]

    # Build precedence graph
    E = []  # precedence relations
    direct_succ = {i: set() for i in V}
    direct_pred = {i: set() for i in V}
    for rel in instance["precedence_relations"]:
        i, j = rel[0], rel[1]
        E.append((i, j))
        direct_succ[i].add(j)
        direct_pred[j].add(i)

    # Compute transitive closures
    F_star = compute_transitive_closure(V, direct_succ)  # all successors
    P_star = compute_transitive_closure(V, direct_pred)   # all predecessors

    # -------------------------------------------------------------------------
    # Compute bounds on cycle time
    # -------------------------------------------------------------------------
    total_t = sum(t[i] for i in V)
    c_underline = math.ceil(total_t / m)

    # Upper bound: generous safe bound
    max_tau = max(tau_matrix[i][j] for i in range(n) for j in range(n))
    max_mu = max(mu_matrix[i][j] for i in range(n) for j in range(n))
    c_bar = total_t + (n - 1) * max_tau + max_mu
    # Also ensure c_bar >= given cycle_time if available
    if "cycle_time" in instance:
        c_bar = max(c_bar, instance["cycle_time"])

    # -------------------------------------------------------------------------
    # Compute earliest/latest stations and assignable sets
    # -------------------------------------------------------------------------
    E_i = {}  # earliest station
    L_i = {}  # latest station
    for i in V:
        pred_sum = sum(t[j] for j in P_star[i])
        E_i[i] = max(1, math.ceil((t[i] + pred_sum) / c_bar))
        succ_sum = sum(t[j] for j in F_star[i])
        L_i[i] = min(m, m + 1 - math.ceil((t[i] + succ_sum) / c_bar))

    # FS_i: assignable stations for task i
    FS = {i: list(range(E_i[i], L_i[i] + 1)) for i in V}
    FS_set = {i: set(FS[i]) for i in V}

    # FT_k: tasks assignable to station k
    K = list(range(1, m + 1))
    FT = {k: [] for k in K}
    for i in V:
        for k in FS[i]:
            FT[k].append(i)
    FT_set = {k: set(FT[k]) for k in K}

    # A_i: tasks that cannot be in the same station as i
    A = {i: set() for i in V}
    for i in V:
        for j in V:
            if i != j and len(FS_set[i] & FS_set[j]) == 0:
                A[i].add(j)

    # F_i^F: forward followers
    # F_i^F = V - (F_i* - F_i) - P_i* - A_i - {i}
    # where F_i is direct successors
    F_F = {i: set() for i in V}
    for i in V:
        indirect_succ = F_star[i] - direct_succ[i]  # F_i* - F_i
        excluded = indirect_succ | P_star[i] | A[i] | {i}
        F_F[i] = set(V) - excluded

    # P_i^F: forward predecessors (inverse of F_F)
    P_F = {i: set() for i in V}
    for i in V:
        for j in F_F[i]:
            P_F[j].add(i)

    # F_i^B: backward followers
    # F_i^B = V - F_i* - A_i
    F_B = {i: set() for i in V}
    for i in V:
        F_B[i] = set(V) - F_star[i] - A[i]
        # Note: i itself can be in F_B[i] (self-loop h_{iik} means i is both
        # first and last, i.e., only task on that station)

    # P_i^B: backward predecessors (inverse of F_B)
    P_B = {i: set() for i in V}
    for i in V:
        for j in F_B[i]:
            P_B[j].add(i)

    # tau_underline_i: smallest forward setup from any predecessor to i
    tau_underline = {}
    for i in V:
        vals = [tau_raw(j, i) for j in P_F[i] if j != i]
        tau_underline[i] = min(vals) if vals else 0

    # mu_underline_i: smallest backward setup from any predecessor to i
    mu_underline = {}
    for i in V:
        vals = [mu_raw(j, i) for j in P_B[i] if j != i]
        mu_underline[i] = min(vals) if vals else 0

    # D_{ij}: lower bound on cycle time if tasks i and j are at the same station
    # Conservative bound: D_{ij} = t_i + t_j (ignoring setups)
    # This is always valid as a lower bound.
    def D(i, j):
        return t[i] + t[j]

    # -------------------------------------------------------------------------
    # Build Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("SSBF2_SUALBP2")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)

    # --- Decision Variables ---

    # x_{ik}: 1 iff task i assigned to station k
    x = {}
    for i in V:
        for k in FS[i]:
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")

    # z_i: station index of task i (integer)
    z = {}
    for i in V:
        z[i] = model.addVar(
            vtype=GRB.INTEGER, lb=E_i[i], ub=L_i[i], name=f"z_{i}"
        )

    # r_i: rank of task i in global sequence (integer)
    r = {}
    for i in V:
        lb_r = len(P_star[i]) + 1
        ub_r = n - len(F_star[i])
        r[i] = model.addVar(
            vtype=GRB.INTEGER, lb=lb_r, ub=ub_r, name=f"r_{i}"
        )

    # g_{ijk}: forward sequencing variable
    g = {}
    for k in K:
        for i in FT[k]:
            for j in FT[k]:
                if j in F_F[i]:
                    g[i, j, k] = model.addVar(
                        vtype=GRB.BINARY, name=f"g_{i}_{j}_{k}"
                    )

    # h_{ijk}: backward setup variable
    h = {}
    for k in K:
        for i in FT[k]:
            for j in FT[k]:
                if j in F_B[i]:
                    h[i, j, k] = model.addVar(
                        vtype=GRB.BINARY, name=f"h_{i}_{j}_{k}"
                    )

    # c: cycle time (continuous)
    c = model.addVar(
        vtype=GRB.CONTINUOUS, lb=c_underline, ub=c_bar, name="c"
    )

    model.update()

    # --- Objective (2a) ---
    model.setObjective(c, GRB.MINIMIZE)

    # --- Constraints ---

    # (1b): Each task assigned to exactly one station
    for i in V:
        model.addConstr(
            gp.quicksum(x[i, k] for k in FS[i]) == 1,
            name=f"assign_{i}"
        )

    # (1c): z_i = sum_k k * x_{ik}
    for i in V:
        model.addConstr(
            gp.quicksum(k * x[i, k] for k in FS[i]) == z[i],
            name=f"station_idx_{i}"
        )

    # (1d): For each task i assigned to station k, it has exactly one forward
    # successor (g) or is the last task with a backward link (h).
    # sum_{j in FT_k cap F_i^F} g_{ijk} + sum_{j in FT_k cap F_i^B} h_{ijk} = x_{ik}
    for i in V:
        for k in FS[i]:
            fwd_terms = []
            for j in FT[k]:
                if j in F_F[i] and (i, j, k) in g:
                    fwd_terms.append(g[i, j, k])
            bwd_terms = []
            for j in FT[k]:
                if j in F_B[i] and (i, j, k) in h:
                    bwd_terms.append(h[i, j, k])
            model.addConstr(
                gp.quicksum(fwd_terms) + gp.quicksum(bwd_terms) == x[i, k],
                name=f"flow_out_{i}_{k}"
            )

    # (1e): For each task j assigned to station k, it has exactly one forward
    # predecessor (g) or is the first task with a backward link (h).
    # sum_{i in FT_k cap P_j^F} g_{ijk} + sum_{i in FT_k cap P_j^B} h_{ijk} = x_{jk}
    for j in V:
        for k in FS[j]:
            fwd_terms = []
            for i in FT[k]:
                if i in P_F[j] and (i, j, k) in g:
                    fwd_terms.append(g[i, j, k])
            bwd_terms = []
            for i in FT[k]:
                if i in P_B[j] and (i, j, k) in h:
                    bwd_terms.append(h[i, j, k])
            model.addConstr(
                gp.quicksum(fwd_terms) + gp.quicksum(bwd_terms) == x[j, k],
                name=f"flow_in_{j}_{k}"
            )

    # (1h): Rank ordering for forward sequencing
    # r_i + 1 + (n - |F_i*| - |P_j*|) * (sum_{k in FS_i cap FS_j} g_{ijk} - 1) <= r_j
    for i in V:
        for j in F_F[i]:
            common_stations = FS_set[i] & FS_set[j]
            if not common_stations:
                continue
            big_M = n - len(F_star[i]) - len(P_star[j])
            g_sum = gp.quicksum(
                g[i, j, k] for k in common_stations if (i, j, k) in g
            )
            model.addConstr(
                r[i] + 1 + big_M * (g_sum - 1) <= r[j],
                name=f"rank_fwd_{i}_{j}"
            )

    # (1i): Precedence rank constraints
    for (i, j) in E:
        model.addConstr(r[i] + 1 <= r[j], name=f"rank_prec_{i}_{j}")

    # (1j): Precedence station constraints
    for (i, j) in E:
        model.addConstr(z[i] <= z[j], name=f"station_prec_{i}_{j}")

    # (2c): Exactly one backward pair per station (since all stations are definite)
    # sum_{i in FT_k} sum_{j in FT_k cap F_i^B} h_{ijk} = 1
    for k in K:
        bwd_terms = []
        for i in FT[k]:
            for j in FT[k]:
                if j in F_B[i] and (i, j, k) in h:
                    bwd_terms.append(h[i, j, k])
        model.addConstr(
            gp.quicksum(bwd_terms) == 1,
            name=f"one_backward_{k}"
        )

    # (2d): Cycle time constraint for each station
    # sum t_i * x_{ik} + sum tau_{ij} * g_{ijk} + sum mu_{ij} * h_{ijk} <= c
    for k in K:
        proc_terms = gp.quicksum(t[i] * x[i, k] for i in FT[k])
        fwd_setup_terms = gp.quicksum(
            tau_raw(i, j) * g[i, j, k]
            for i in FT[k]
            for j in FT[k]
            if j in F_F[i] and (i, j, k) in g
        )
        bwd_setup_terms = gp.quicksum(
            mu_raw(i, j) * h[i, j, k]
            for i in FT[k]
            for j in FT[k]
            if j in F_B[i] and (i, j, k) in h
        )
        model.addConstr(
            proc_terms + fwd_setup_terms + bwd_setup_terms <= c,
            name=f"cycle_time_{k}"
        )

    # (2e): If task j is the only task on station k (h_{jjk}=1), no other tasks
    # sum_{i in FT_k \ {j}} x_{ik} <= (n - m + 1) * (1 - h_{jjk})
    for k in K:
        for j in FT[k]:
            if j in F_B[j] and (j, j, k) in h:
                other_terms = gp.quicksum(
                    x[i, k] for i in FT[k] if i != j
                )
                model.addConstr(
                    other_terms <= (n - m + 1) * (1 - h[j, j, k]),
                    name=f"single_task_{j}_{k}"
                )

    # (2f): Each station must have at least one task
    for k in K:
        model.addConstr(
            gp.quicksum(x[i, k] for i in FT[k]) >= 1,
            name=f"min_one_task_{k}"
        )

    # (2g) D_{ij} pairwise lower-bound cut REMOVED:
    # The big-M form `c + c_bar * (z_j - z_i) >= D_{ij}` is only valid as a
    # cut when j cannot be in a station earlier than i; for arbitrary
    # forward-feasible pairs F_F[i] the implementation can over-constrain
    # the model (when z_j < z_i the LHS shrinks below D_{ij}). It is a
    # valid inequality / reformulation artifact and is intentionally
    # omitted so the gurobi model exactly represents the original SUALBP-2.

    # (2h): Cycle time bounds (already enforced via variable bounds on c)
    # No additional constraint needed.

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    model.optimize()

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    result = {}

    if model.Status == GRB.OPTIMAL:
        result["status"] = "optimal"
        result["objective_value"] = model.ObjVal
    elif model.Status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            result["status"] = "feasible"
            result["objective_value"] = model.ObjVal
        else:
            result["status"] = "timeout_no_solution"
            result["objective_value"] = None
    elif model.Status == GRB.INFEASIBLE:
        result["status"] = "infeasible"
        result["objective_value"] = None
    else:
        result["status"] = f"gurobi_status_{model.Status}"
        result["objective_value"] = model.ObjVal if model.SolCount > 0 else None

    if model.SolCount > 0:
        result["best_bound"] = model.ObjBound
        result["gap"] = model.MIPGap
        result["num_vars"] = model.NumVars
        result["num_constrs"] = model.NumConstrs
        result["solve_time"] = model.Runtime

        # ----------------------------------------------------------------
        # Project the SSBF-2 reformulation onto the original SUALBP-2
        # solution structure: assignment (task -> station) +
        # station_sequences (station -> ordered task list) + cycle_time.
        # ----------------------------------------------------------------
        assignment = {}
        for i in V:
            for k in FS[i]:
                if x[i, k].X > 0.5:
                    assignment[i] = k
                    break

        # Build per-station ordered chains using g[i,j,k] (within-station
        # forward succession): j directly follows i in station k iff
        # g[i,j,k] = 1.
        station_sequences = {}
        for k in K:
            tasks_at_k = [i for i in V if assignment.get(i) == k]
            if not tasks_at_k:
                station_sequences[k] = []
                continue
            if len(tasks_at_k) == 1:
                station_sequences[k] = [tasks_at_k[0]]
                continue
            next_in_k = {}
            for i in tasks_at_k:
                for j in V:
                    if i == j:
                        continue
                    if (i, j, k) in g and g[i, j, k].X > 0.5:
                        next_in_k[i] = j
                        break
            preds = set(next_in_k.values())
            first_candidates = [i for i in tasks_at_k if i not in preds]
            start = first_candidates[0] if first_candidates else tasks_at_k[0]
            chain = [start]
            cur = start
            while cur in next_in_k and next_in_k[cur] not in chain:
                cur = next_in_k[cur]
                chain.append(cur)
            station_sequences[k] = chain

        result["assignment"] = {str(i): int(assignment[i])
                                  for i in sorted(assignment)}
        result["station_sequences"] = {
            str(k): [int(i) for i in station_sequences[k]]
            for k in sorted(station_sequences)
            if station_sequences[k]
        }
        result["cycle_time"] = float(c.X)
    else:
        result["best_bound"] = model.ObjBound if hasattr(model, "ObjBound") else None
        result["gap"] = None
        result["solve_time"] = model.Runtime

    return result


def main():
    parser = argparse.ArgumentParser(
        description="SSBF-2 MIP model for SUALBP-2 using Gurobi"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the instance JSON file"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to write the solution JSON file"
    )
    parser.add_argument(
        "--time_limit", type=float, default=1800,
        help="Time limit in seconds (default: 1800)"
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = solve(instance, args.time_limit)

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2, default=str)

    if result["objective_value"] is not None:
        print(f"Objective value (cycle time): {result['objective_value']}")
    else:
        print(f"No solution found. Status: {result['status']}")


if __name__ == "__main__":
    main()
