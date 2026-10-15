#!/usr/bin/env python3
"""
Gurobi implementation of the Start-based MILP model for RACP
from Kreter et al. (2018), Section 4.1.1.

"Mixed-Integer Linear Programming and Constraint Programming Formulations
 for Solving Resource Availability Cost Problems"
European Journal of Operational Research.

Start-based formulation for RACP:
  Minimize  sum_{k in R} c_k * R_k
  subject to:
    (7)  sum_{t in W_i} x_{it} = 1                             for all i in V
    (8)  sum_{tau=ES_i}^{min(LS_i, t-p_i)} x_{i,tau}
         - sum_{tau=ES_j}^{t} x_{j,tau} >= 0                   for <i,j> in A, t in W_j
    (9)  sum_{i in V^r} r_{ik} *
         sum_{tau=max(ES_i, t-p_i+1)}^{min(LS_i, t)} x_{i,tau}
         <= R_k                                                 for k in R, t in T
    (10) x_{it} in {0,1}                                        for i in V, t in W_i
    (11) R_k >= 0                                                for k in R

For RACP/max, constraint (8) uses delta_{ij} instead of p_i.
"""

import argparse
import json
import sys
import time
_GUROBI_CODE_START_TIME = time.time()
from collections import deque
import os as _os, sys as _sys
# Walk up from this file's directory to find repo root (containing scripts/).
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


try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("Error: gurobipy is not installed. Please install Gurobi.", file=sys.stderr)
    sys.exit(1)


def load_instance(path):
    """Load a problem instance from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_es_ls(activities, arcs, arc_weights, d_bar):
    """
    Compute earliest start times (ES) and latest start times (LS)
    using forward/backward passes on the project network.

    For RACP: arcs are precedence constraints with weight delta_{ij} = p_i.
    For RACP/max: arcs have general weights delta_{ij}.

    Uses topological sort (Kahn's algorithm) for DAG processing.
    For networks with negative arc weights (max time lags creating cycles
    in the constraint graph), falls back to Bellman-Ford.
    """
    n_act = len(activities)
    act_ids = [a['id'] for a in activities]

    # Build adjacency lists
    successors = {i: [] for i in act_ids}
    predecessors = {i: [] for i in act_ids}
    for (i, j) in arcs:
        successors[i].append(j)
        predecessors[j].append(i)

    # Topological sort (Kahn's algorithm)
    in_degree = {i: len(predecessors[i]) for i in act_ids}
    queue = deque([i for i in act_ids if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(topo_order) != n_act:
        # Network has cycles (possible with max time lags) - use Bellman-Ford
        # **NOT SPECIFIED IN PAPER** for RACP instances, which are always acyclic.
        # Inferred assumption: Fall back to Bellman-Ford for general temporal constraints.
        ES = bellman_ford_es(act_ids, arcs, arc_weights)
    else:
        # Forward pass: compute ES (longest path from source)
        ES = {i: 0 for i in act_ids}
        for i in topo_order:
            for j in successors[i]:
                weight = arc_weights.get((i, j), 0)
                ES[j] = max(ES[j], ES[i] + weight)

    # Backward pass: compute LS
    # LS[i] = d_bar - longest_path_from_i_to_sink
    # Equivalently: LS[i] = min over successors j of (LS[j] - delta_{ij})
    LS = {i: d_bar for i in act_ids}
    if len(topo_order) == n_act:
        for i in reversed(topo_order):
            for j in successors[i]:
                weight = arc_weights.get((i, j), 0)
                LS[i] = min(LS[i], LS[j] - weight)
    else:
        LS = bellman_ford_ls(act_ids, arcs, arc_weights, d_bar)

    return ES, LS


def bellman_ford_es(act_ids, arcs, arc_weights):
    """Bellman-Ford algorithm to compute earliest start times (longest paths from source)."""
    ES = {i: 0 for i in act_ids}
    n = len(act_ids)
    for _ in range(n - 1):
        updated = False
        for (i, j) in arcs:
            w = arc_weights.get((i, j), 0)
            if ES[i] + w > ES[j]:
                ES[j] = ES[i] + w
                updated = True
        if not updated:
            break
    return ES


def bellman_ford_ls(act_ids, arcs, arc_weights, d_bar):
    """Bellman-Ford algorithm to compute latest start times (backward pass)."""
    LS = {i: d_bar for i in act_ids}
    n = len(act_ids)
    for _ in range(n - 1):
        updated = False
        for (i, j) in arcs:
            w = arc_weights.get((i, j), 0)
            if LS[j] - w < LS[i]:
                LS[i] = LS[j] - w
                updated = True
        if not updated:
            break
    return LS


def compute_peak_resource_usage(activities, schedule, n_resources, d_bar):
    """
    Compute the peak resource usage for each resource type given a schedule.
    Returns a list of peak usages [peak_k for k in range(n_resources)].
    """
    durations = {a['id']: a['duration'] for a in activities}
    requirements = {a['id']: a['resource_requirements'] for a in activities}

    peak = [0] * n_resources
    for t in range(d_bar):
        usage = [0] * n_resources
        for a in activities:
            i = a['id']
            p_i = durations[i]
            if p_i > 0 and schedule[i] <= t < schedule[i] + p_i:
                for k in range(n_resources):
                    usage[k] += requirements[i][k]
        for k in range(n_resources):
            peak[k] = max(peak[k], usage[k])
    return peak


def compute_schedule_cost(peak_usage, resource_costs):
    """Compute the total cost of providing resources given peak usage."""
    return sum(resource_costs[k] * peak_usage[k] for k in range(len(resource_costs)))


def solve_racp(instance, time_limit):
    """
    Solve the RACP (or RACP/max) using the start-based MILP formulation.
    Returns a dictionary with the solution.
    """
    start_time = time.time()

    activities = instance['activities']
    precedence_relations = instance['precedence_relations']
    resource_costs = instance['resource_costs']
    n_resources = instance['num_resource_types']
    problem_type = instance.get('problem_type', 'RACP')

    # Compute d_bar (maximum project duration)
    if 'max_project_duration' in instance:
        d_bar = instance['max_project_duration']
    else:
        d_bar = int(instance.get('deadline_factor', 1.0) *
                     instance.get('earliest_project_completion', 0))

    n_act = len(activities)
    durations = {a['id']: a['duration'] for a in activities}
    requirements = {a['id']: a['resource_requirements'] for a in activities}
    act_ids = [a['id'] for a in activities]

    # Build arc weights
    # For RACP: delta_{ij} = p_i (processing time of predecessor)
    # For RACP/max: delta_{ij} may differ and could be provided in instance
    arc_weights = {}
    if problem_type == 'RACP':
        for (i, j) in precedence_relations:
            arc_weights[(i, j)] = durations[i]
    elif problem_type == 'RACP/max':
        # Check if arc weights are provided in the instance
        if 'arc_weights' in instance:
            for idx, (i, j) in enumerate(precedence_relations):
                arc_weights[(i, j)] = instance['arc_weights'][idx]
        else:
            # **NOT SPECIFIED IN PAPER** for this instance format.
            # Inferred assumption: treat as RACP (delta_{ij} = p_i).
            for (i, j) in precedence_relations:
                arc_weights[(i, j)] = durations[i]
    else:
        # Default to RACP behavior
        for (i, j) in precedence_relations:
            arc_weights[(i, j)] = durations[i]

    # Preprocessing: compute ES and LS using Bellman-Ford (cf. Ahuja et al., 1993)
    ES, LS = compute_es_ls(activities, precedence_relations, arc_weights, d_bar)

    # Sanity check
    for i in act_ids:
        if ES[i] > LS[i]:
            return {
                'objective_value': None,
                'status': 'infeasible',
                'message': f'Activity {i} has ES={ES[i]} > LS={LS[i]}'
            }

    # Compute upper bound on objective value from ES-schedule and LS-schedule
    # (Section 6: "The upper bound on the objective value is calculated as the
    #  minimum of the objective values of the ES-schedule and LS-schedule.")
    es_schedule = {i: ES[i] for i in act_ids}
    ls_schedule = {i: LS[i] for i in act_ids}
    es_peak = compute_peak_resource_usage(activities, es_schedule, n_resources, d_bar)
    ls_peak = compute_peak_resource_usage(activities, ls_schedule, n_resources, d_bar)
    cost_es = compute_schedule_cost(es_peak, resource_costs)
    cost_ls = compute_schedule_cost(ls_peak, resource_costs)
    obj_ub = min(cost_es, cost_ls)

    if cost_es <= cost_ls:
        better_schedule = es_schedule
        better_peak = es_peak
    else:
        better_schedule = ls_schedule
        better_peak = ls_peak

    # Time periods: T = {0, 1, ..., d_bar}
    T = range(d_bar + 1)

    # Upper bounds on R_k: sum of all resource requirements (safe bound)
    R_ub = [0] * n_resources
    for k in range(n_resources):
        R_ub[k] = sum(requirements[i][k] for i in act_ids)

    # =========================================================================
    # Build Gurobi Model
    # =========================================================================
    model = gp.Model("RACP_start_based")
    model.setParam("Threads", 1)

    # Solver settings
    remaining_time = max(1, time_limit - int(time.time() - start_time))
    model.setParam("TimeLimit", remaining_time)
    model.setParam("OutputFlag", 1)

    # ----- Decision Variables -----

    # x[i,t] = 1 if activity i starts at time t, 0 otherwise (Eq. 10)
    x = {}
    for i in act_ids:
        for t in range(ES[i], LS[i] + 1):
            x[i, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    # R[k] = amount of resource k provided (Eq. 11)
    R = {}
    for k in range(n_resources):
        R[k] = model.addVar(lb=0.0, ub=R_ub[k], vtype=GRB.CONTINUOUS, name=f"R_{k}")

    model.update()

    # ----- Objective Function (Eq. 1/7-11 header) -----
    # Minimize sum_{k in R} c_k * R_k
    model.setObjective(
        gp.quicksum(resource_costs[k] * R[k] for k in range(n_resources)),
        GRB.MINIMIZE
    )

    # ----- Constraint (7): Assignment -----
    # Each activity starts at exactly one time in its feasible window
    for i in act_ids:
        model.addConstr(
            gp.quicksum(x[i, t] for t in range(ES[i], LS[i] + 1)) == 1,
            name=f"assign_{i}"
        )

    # ----- Constraint (8): Disaggregated Precedence/Temporal Constraints -----
    # For RACP: uses p_i; for RACP/max: uses delta_{ij}
    # For <i,j> in A, t in W_j:
    #   sum_{tau=ES_i}^{min(LS_i, t - delta_{ij})} x_{i,tau}
    #   - sum_{tau=ES_j}^{t} x_{j,tau} >= 0
    for (i, j) in precedence_relations:
        delta_ij = arc_weights[(i, j)]
        for t in range(ES[j], LS[j] + 1):
            lhs_ub = min(LS[i], t - delta_ij)
            if lhs_ub >= ES[i]:
                lhs = gp.quicksum(x[i, tau] for tau in range(ES[i], lhs_ub + 1))
            else:
                lhs = 0
            rhs = gp.quicksum(x[j, tau] for tau in range(ES[j], t + 1))
            model.addConstr(lhs - rhs >= 0, name=f"prec_{i}_{j}_{t}")

    # ----- Constraint (9): Resource Constraints -----
    # For each k in R, t in T:
    #   sum_{i in V^r} r_{ik} * sum_{tau=max(ES_i, t-p_i+1)}^{min(LS_i, t)} x_{i,tau} <= R_k
    for k in range(n_resources):
        for t in T:
            terms = []
            for i in act_ids:
                r_ik = requirements[i][k]
                p_i = durations[i]
                if r_ik == 0 or p_i == 0:
                    continue
                tau_lo = max(ES[i], t - p_i + 1)
                tau_hi = min(LS[i], t)
                if tau_lo <= tau_hi:
                    terms.append(
                        r_ik * gp.quicksum(
                            x[i, tau] for tau in range(tau_lo, tau_hi + 1)
                        )
                    )
            if terms:
                model.addConstr(
                    gp.quicksum(terms) <= R[k],
                    name=f"res_{k}_{t}"
                )

    # ----- Warm Start (MIP Start) -----
    # Provide the better of ES/LS schedule as initial solution
    for i in act_ids:
        for t in range(ES[i], LS[i] + 1):
            x[i, t].Start = 1.0 if t == better_schedule[i] else 0.0
    for k in range(n_resources):
        R[k].Start = better_peak[k]

    # =========================================================================
    # Optimize
    # =========================================================================
    model.optimize()

    # =========================================================================
    # Extract Solution
    # =========================================================================
    result = {}
    if model.SolCount > 0:
        result['objective_value'] = model.ObjVal

        # Extract start times
        start_times = {}
        for i in act_ids:
            for t in range(ES[i], LS[i] + 1):
                if x[i, t].X > 0.5:
                    start_times[i] = t
                    break
        result['start_times'] = start_times

        # Extract resource amounts
        resource_amounts = {}
        for k in range(n_resources):
            resource_amounts[k] = R[k].X
        result['resource_amounts'] = resource_amounts

        if model.Status == GRB.OPTIMAL:
            result['status'] = 'optimal'
        else:
            result['status'] = 'feasible'
    else:
        result['objective_value'] = None
        result['status'] = 'no_solution_found'

    return result


def main():
    parser = argparse.ArgumentParser(
        description="RACP Start-based MILP solver using Gurobi "
                    "(Kreter et al., 2018, Section 4.1)"
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help="Path to the JSON file containing the problem instance")
    parser.add_argument('--solution_path', type=str, required=True,
                        help="Path where the final solution JSON file will be written")
    parser.add_argument('--time_limit', type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = solve_racp(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    if result['objective_value'] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Status: {result['status']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
