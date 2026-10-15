#!/usr/bin/env python3
"""
BDDF (BDD-based Flow Formulation) for Pm||sum w_j T_j
Based on: Kowalczyk, Leus, Hojny, Ropke (2024)
"A flow-based formulation for parallel machine scheduling using decision diagrams"

Implements Formulation (6) - the paper's PRIMARY new formulation (BDDF).
Constructs a BDD representing all valid job sequences, then solves the
resulting MIP with Gurobi.
"""

import json
import argparse
import math
from collections import defaultdict, deque

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
    with open(path) as f:
        return json.load(f)


def compute_planning_horizon(jobs, m):
    """T = ceil((sum p_j - p_max) / m) + p_max  (Pessoa et al., 2010)."""
    P = sum(j['p'] for j in jobs)
    p_max = max(j['p'] for j in jobs)
    return math.ceil((P - p_max) / m) + p_max


def build_ordering_for_interval(jobs, e_left, e_right):
    """
    Build sigma_r for interval (e_left, e_right] following Baptiste & Sadykov (2009).

    Rules for short jobs i, j:
    - Long jobs (p_j >= interval length) come first
    - Late jobs (d_j <= e_left) before on-time jobs (d_j >= e_right)
    - Among on-time: LPT (p_i > p_j => i first)
    - Among late: WSPT (p_i/w_i < p_j/w_j => i first), ties by LPT

    Jobs where e_left < d_j < e_right are treated as late (conservative).
    ** INFERRED ASSUMPTION: ambiguous jobs (e_left < d_j < e_right) are treated
       as late, since they may be tardy for some completion times in the interval. **
    """
    interval_length = e_right - e_left

    long_jobs = []
    short_late = []
    short_ontime = []

    for j in jobs:
        if j['p'] >= interval_length:
            long_jobs.append(j)
        elif j['d'] >= e_right:
            short_ontime.append(j)
        else:
            # Late: d_j <= e_left, or ambiguous: e_left < d_j < e_right
            short_late.append(j)

    # Sort long jobs by LPT (descending processing time), ties by job id
    long_jobs.sort(key=lambda j: (-j['p'], j['id']))

    # Sort late short jobs by WSPT: p/w ascending, ties by p descending, then id
    short_late.sort(key=lambda j: (j['p'] / j['w'], -j['p'], j['id']))

    # Sort on-time short jobs by LPT: p descending, ties by id
    short_ontime.sort(key=lambda j: (-j['p'], j['id']))

    return long_jobs + short_late + short_ontime


def build_appropriate_partition(jobs, T):
    """
    Build an appropriate partition satisfying Theorem 1 (Baptiste & Sadykov, 2009).

    Start with due-date-based partition, then iteratively subdivide intervals
    that contain "special pairs" violating conditions (4) and (5).

    ** NOT SPECIFIED IN PAPER: The exact Clement (2015) procedure for minimum
       intervals is not reproduced. We use iterative subdivision instead. **
    """
    # Step (a): partition based on distinct due dates
    due_dates = sorted(set(j['d'] for j in jobs if j['d'] < T))
    endpoints = sorted(set([0] + due_dates + [T]))

    intervals = [(endpoints[i], endpoints[i + 1]) for i in range(len(endpoints) - 1)]
    orderings = [build_ordering_for_interval(jobs, e_l, e_r) for e_l, e_r in intervals]

    # Step (c)/(d): Check Theorem 1 and subdivide
    max_iterations = 200  # safety bound
    for _ in range(max_iterations):
        changed = False
        new_intervals = []
        new_orderings = []

        for (e_left, e_right), ordering in zip(intervals, orderings):
            split_point = None
            for i_pos in range(len(ordering)):
                for j_pos in range(i_pos + 1, len(ordering)):
                    ji = ordering[i_pos]
                    jj = ordering[j_pos]
                    # Condition (4): e_r <= e_{r-1} + p_j
                    cond4 = e_right <= e_left + jj['p']
                    # Condition (5): e_{r-1} >= d_i + ceil(w_j * p_i / w_i) - p_i
                    cond5 = e_left >= ji['d'] + math.ceil(jj['w'] * ji['p'] / ji['w']) - ji['p']
                    if not cond4 and not cond5:
                        # Split to make condition (4) hold for left sub-interval
                        candidate = e_left + jj['p']
                        if e_left < candidate < e_right:
                            split_point = candidate
                            break
                        # Fallback: midpoint
                        mid = (e_left + e_right) // 2
                        if e_left < mid < e_right:
                            split_point = mid
                            break
                        # If interval is length 1, can't split further
                        break
                if split_point is not None:
                    break

            if split_point is not None:
                changed = True
                new_intervals.append((e_left, split_point))
                new_orderings.append(build_ordering_for_interval(jobs, e_left, split_point))
                new_intervals.append((split_point, e_right))
                new_orderings.append(build_ordering_for_interval(jobs, split_point, e_right))
            else:
                new_intervals.append((e_left, e_right))
                new_orderings.append(ordering)

        intervals = new_intervals
        orderings = new_orderings
        if not changed:
            break

    return intervals, orderings


def build_bdd(jobs, intervals, orderings, T, m):
    """
    Build BDD following Algorithm 1 (Appendix A).

    Each node has configuration (global_pos, t).
    global_pos indexes into the global ordering of job representations.
    t is the cumulative processing time (= start time of the job at this node).

    Returns: root_id, nodes dict, high_edges, low_edges, edge_costs, edge_job_ids, incoming
    """
    n = len(jobs)
    q = len(intervals)
    job_by_id = {j['id']: j for j in jobs}

    # Build global ordering: all n jobs in interval 0 order, then interval 1, etc.
    # global_order[pos] = (job_id, interval_index)
    global_order = []
    for r in range(q):
        for j in orderings[r]:
            global_order.append((j['id'], r))

    nq = len(global_order)  # = n * q

    def minjob(current_pos, t):
        """Find minimum successor in global order where t + p_job falls in its interval."""
        for pos in range(current_pos + 1, nq):
            jid, r = global_order[pos]
            e_left, e_right = intervals[r]
            p_j = job_by_id[jid]['p']
            if e_left < t + p_j <= e_right:
                return pos
        return nq  # sentinel: no valid successor

    # Find root: first job representation whose processing time fits in its interval
    root_pos = None
    for pos in range(nq):
        jid, r = global_order[pos]
        e_left, e_right = intervals[r]
        p_j = job_by_id[jid]['p']
        if e_left < 0 + p_j <= e_right:
            root_pos = pos
            break

    if root_pos is None:
        # Fallback: use first position
        # ** INFERRED ASSUMPTION: if no job fits from t=0, use first position **
        root_pos = 0

    TERMINAL = (-1, -1)  # sentinel for terminal node

    # BDD construction via BFS with memoization
    # Nodes identified by (global_pos, t)
    nodes = {}  # node_key -> node_id (int)
    node_id_counter = [0]

    def get_node_id(key):
        if key not in nodes:
            nodes[key] = node_id_counter[0]
            node_id_counter[0] += 1
        return nodes[key]

    root_key = (root_pos, 0)
    root_id = get_node_id(root_key)
    terminal_id = get_node_id(TERMINAL)

    high_child = {}   # node_id -> child_node_id
    low_child = {}    # node_id -> child_node_id
    high_cost = {}    # node_id -> cost of high edge
    high_job = {}     # node_id -> job_id associated with high edge
    node_job = {}     # node_id -> job_id at this node
    node_start = {}   # node_id -> start time t at this node
    incoming = defaultdict(list)  # node_id -> list of (parent_id, edge_type)

    queue = deque([root_key])
    visited = set()

    while queue:
        key = queue.popleft()
        if key in visited or key == TERMINAL:
            continue
        visited.add(key)

        nid = get_node_id(key)
        pos, t = key
        jid, r = global_order[pos]
        p_j = job_by_id[jid]['p']

        node_job[nid] = jid
        node_start[nid] = t

        # High edge: schedule job j at time t
        t_high = t + p_j
        next_pos_high = minjob(pos, t_high)
        if next_pos_high >= nq:
            child_high_key = TERMINAL
        else:
            child_high_key = (next_pos_high, t_high)

        child_high_id = get_node_id(child_high_key)
        high_child[nid] = child_high_id
        high_cost[nid] = job_by_id[jid]['w'] * max(0, t + p_j - job_by_id[jid]['d'])
        high_job[nid] = jid
        incoming[child_high_id].append((nid, 'high'))

        if child_high_key != TERMINAL and child_high_key not in visited:
            queue.append(child_high_key)

        # Low edge: skip job j
        t_low = t
        next_pos_low = minjob(pos, t_low)
        if next_pos_low >= nq:
            child_low_key = TERMINAL
        else:
            child_low_key = (next_pos_low, t_low)

        child_low_id = get_node_id(child_low_key)
        low_child[nid] = child_low_id
        incoming[child_low_id].append((nid, 'low'))

        if child_low_key != TERMINAL and child_low_key not in visited:
            queue.append(child_low_key)

    return (root_id, terminal_id, nodes, high_child, low_child,
            high_cost, high_job, node_job, node_start, incoming)


def solve_bddf(instance, time_limit):
    """Solve Pm||sum w_j T_j using the BDDF formulation (Eq. 6) with Gurobi."""
    jobs = instance['jobs']
    m = instance['m']
    n = instance['n']
    T_val = instance.get('T', compute_planning_horizon(jobs, m))

    # Build appropriate partition
    intervals, orderings = build_appropriate_partition(jobs, T_val)

    # Build BDD
    (root_id, terminal_id, nodes, high_child, low_child,
     high_cost, high_job, node_job, node_start, incoming) = \
        build_bdd(jobs, intervals, orderings, T_val, m)

    print(f"BDD constructed: {len(nodes)} nodes")

    # All non-terminal, non-root internal nodes
    all_node_ids = set()
    for key, nid in nodes.items():
        if key != (-1, -1):  # not terminal
            all_node_ids.add(nid)

    # Create Gurobi model
    model = gp.Model("BDDF")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Variables for high edges (binary): x_h[nid]
    x_h = {}
    for nid in all_node_ids:
        if nid in high_child:
            x_h[nid] = model.addVar(
                vtype=GRB.BINARY,
                obj=high_cost.get(nid, 0),
                name=f"xh_{nid}"
            )

    # Variables for low edges (continuous >= 0): x_l[nid]
    x_l = {}
    for nid in all_node_ids:
        if nid in low_child:
            x_l[nid] = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                name=f"xl_{nid}"
            )

    model.update()

    # Constraint (6b): each job scheduled exactly once
    # sum_{e in A^1 : p_B(e) = j} x_e = 1 for all j in J
    job_high_edges = defaultdict(list)
    for nid in x_h:
        jid = high_job[nid]
        job_high_edges[jid].append(nid)

    for j in jobs:
        jid = j['id']
        if jid in job_high_edges and job_high_edges[jid]:
            model.addConstr(
                gp.quicksum(x_h[nid] for nid in job_high_edges[jid]) == 1,
                name=f"assign_{jid}"
            )
        else:
            print(f"WARNING: job {jid} has no high edges in BDD")

    # Constraint (6c): flow conservation for internal nodes (not root, not terminal)
    # x_{e^1_v} + x_{e^0_v} = sum_{e in delta^-(v)} x_e
    for nid in all_node_ids:
        if nid == root_id:
            continue

        # Outgoing flow
        out_vars = []
        if nid in x_h:
            out_vars.append(x_h[nid])
        if nid in x_l:
            out_vars.append(x_l[nid])

        # Incoming flow
        in_vars = []
        for parent_id, etype in incoming.get(nid, []):
            if etype == 'high' and parent_id in x_h:
                in_vars.append(x_h[parent_id])
            elif etype == 'low' and parent_id in x_l:
                in_vars.append(x_l[parent_id])

        if out_vars and in_vars:
            model.addConstr(
                gp.quicksum(out_vars) == gp.quicksum(in_vars),
                name=f"flow_{nid}"
            )

    # Constraint (6d): sum of incoming edges to terminal = m
    terminal_in = []
    for parent_id, etype in incoming.get(terminal_id, []):
        if etype == 'high' and parent_id in x_h:
            terminal_in.append(x_h[parent_id])
        elif etype == 'low' and parent_id in x_l:
            terminal_in.append(x_l[parent_id])

    if terminal_in:
        model.addConstr(
            gp.quicksum(terminal_in) == m,
            name="terminal_flow"
        )

    # Redundant constraint (7): root outgoing = m
    root_out = []
    if root_id in x_h:
        root_out.append(x_h[root_id])
    if root_id in x_l:
        root_out.append(x_l[root_id])
    if root_out:
        model.addConstr(
            gp.quicksum(root_out) == m,
            name="root_flow"
        )

    # Solve
    model.optimize()

    # Extract solution
    result = {"solver": "Gurobi_BDDF", "status": model.Status}

    if model.SolCount > 0:
        result['objective_value'] = model.ObjVal

        # Extract scheduled jobs and their start times
        schedule = {}
        for nid in x_h:
            if x_h[nid].X > 0.5:
                jid = high_job[nid]
                t = node_start[nid]
                schedule[jid] = t
        result['schedule'] = {str(k): v for k, v in schedule.items()}
    else:
        result['objective_value'] = None
        result['schedule'] = {}

    return result


def main():
    parser = argparse.ArgumentParser(
        description="BDDF formulation for Pm||sum w_j T_j (Kowalczyk et al., 2024)"
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path for output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum solver runtime in seconds')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = solve_bddf(instance, args.time_limit)

    # Write solution
    with open(args.solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result['objective_value'] is not None:
        print(f"Objective value: {result['objective_value']}")
    else:
        print("No feasible solution found within time limit.")


if __name__ == '__main__':
    main()
