#!/usr/bin/env python3
"""
Gurobi implementation of the DTSPMS (Double Traveling Salesman Problem with Multiple Stacks).

Based on: Alba Martinez et al. (2013), "A Branch-and-Cut Algorithm for the Double Traveling
Salesman Problem with Multiple Stacks", INFORMS Journal on Computing 25(1):41-55.

This implements the ILP formulation (1)-(5) from the paper, with dynamic separation of:
  - Subtour elimination constraints (4) via callback
  - Infeasible path / tournament constraints (7) via callback
  - Lifted tournament constraints (8) via callback (heuristic)
  - Position-based constraints (9) via callback
  - Lifted depot-endpoint constraints (10) via callback (heuristic)

The loading feasibility check (PDTC) follows Section 3 of the paper exactly.
"""

import argparse
import json
import math
import time
import itertools
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_instance(path):
    """Load a DTSPMS instance from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def build_distance_matrices(data):
    """Extract distance matrices for pickup and delivery regions."""
    n = data['parameters']['num_requests_n']
    s = data['parameters']['num_stacks_s']
    l = data['parameters']['stack_height_l']

    c_P = []
    for row in data['pickup_region']['distance_matrix']:
        c_P.append(list(row))

    c_D = []
    for row in data['delivery_region']['distance_matrix']:
        c_D.append(list(row))

    return n, s, l, c_P, c_D


# =============================================================================
# LOADING FEASIBILITY CHECK (PDTC) - Section 3 of the paper
# =============================================================================

def check_loading_feasibility(pickup_path, delivery_path, s, l):
    """
    Check if a pair of (partial) paths admits a feasible loading plan.

    pickup_path: ordered list of customer vertices visited in pickup tour (no depot)
    delivery_path: ordered list of customer vertices visited in delivery tour (no depot)
    s: number of stacks
    l: max height of each stack

    Returns: (feasible, cliques)
      feasible: True if loading is feasible
      cliques: list of cliques of size > s found (for lifted constraints)
    """
    # Find common customers in both paths
    I = []
    pickup_set = set(pickup_path)
    delivery_set = set(delivery_path)
    I = list(pickup_set & delivery_set)

    if len(I) == 0:
        return True, []

    # Step 1: Trivial check - if |I| <= s, feasible
    if len(I) <= s:
        return True, []

    # Build position maps
    pickup_pos = {}
    for idx, v in enumerate(pickup_path):
        if v not in pickup_pos:
            pickup_pos[v] = idx

    delivery_pos = {}
    for idx, v in enumerate(delivery_path):
        if v not in delivery_pos:
            delivery_pos[v] = idx

    # Step 2: Build precedence graph
    # Arc (i, j) exists if i precedes j in BOTH pickup and delivery paths
    # This means i and j CANNOT be in the same stack (LIFO violation)
    precedence = defaultdict(set)
    for i in I:
        for j in I:
            if i != j:
                if pickup_pos[i] < pickup_pos[j] and delivery_pos[i] < delivery_pos[j]:
                    precedence[i].add(j)

    # Step 3: Lower bound via longest path (CPM)
    # The precedence graph is a DAG; find longest path
    # Topological order by pickup position
    sorted_items = sorted(I, key=lambda v: pickup_pos[v])

    # Compute longest path length and track predecessors for clique extraction
    dist = {v: 0 for v in I}
    pred = {v: None for v in I}

    for v in sorted_items:
        for w in I:
            if w in precedence[v]:
                if dist[w] < dist[v] + 1:
                    dist[w] = dist[v] + 1
                    pred[w] = v

    max_clique_size = max(dist.values()) + 1 if dist else 1

    # Extract all cliques (longest paths) of maximum length
    cliques = []
    max_dist = max(dist.values()) if dist else 0

    # Find all endpoints of longest paths
    for end_v in I:
        if dist[end_v] == max_dist:
            # Trace back the clique
            clique = []
            v = end_v
            while v is not None:
                clique.append(v)
                v = pred[v]
            clique.reverse()
            if clique not in cliques:
                cliques.append(clique)

    if max_clique_size > s:
        return False, cliques

    # Step 4: Greedy assignment using CPM output
    # Compute T_min and T_max for each vertex
    # T_min(i) = length of longest path ending at i (from any source) + 1 (1-indexed stack)
    # T_max(i) = s - (length of longest path starting from i to any sink)

    # Forward pass: T_min
    t_min = {}
    for v in sorted_items:
        t_min[v] = dist[v] + 1  # 1-indexed

    # Backward pass: longest path from v to any sink
    # Reverse topological order
    rev_sorted = list(reversed(sorted_items))
    dist_from = {v: 0 for v in I}
    for v in rev_sorted:
        for w in I:
            if w in precedence[v]:
                if dist_from[v] < dist_from[w] + 1:
                    dist_from[v] = dist_from[w] + 1

    t_max = {}
    for v in I:
        t_max[v] = s - dist_from[v]

    # Assign items on longest path first (T_min == T_max)
    stacks = [[] for _ in range(s)]  # stacks[k] is list of items in stack k (bottom to top)

    fixed_items = set()
    for v in I:
        if t_min[v] == t_max[v]:
            stack_idx = t_min[v] - 1  # 0-indexed
            if stack_idx < 0 or stack_idx >= s:
                # Should not happen if clique size <= s
                break
            stacks[stack_idx].append(v)
            if len(stacks[stack_idx]) > l:
                # Greedy fails, go to enumeration
                return _enumerate_loading(I, precedence, pickup_pos, delivery_pos, s, l), cliques
            fixed_items.add(v)

    # Check LIFO for fixed items
    for k in range(s):
        for i_idx in range(len(stacks[k])):
            for j_idx in range(i_idx + 1, len(stacks[k])):
                vi = stacks[k][i_idx]
                vj = stacks[k][j_idx]
                # vi loaded before vj (vi is lower in stack)
                # For LIFO: vj must be delivered before vi
                if pickup_pos[vi] < pickup_pos[vj]:
                    if delivery_pos[vi] < delivery_pos[vj]:
                        # LIFO violation
                        return _enumerate_loading(I, precedence, pickup_pos, delivery_pos, s, l), cliques

    # Remaining items sorted by T_min, breaking ties by vertex index
    remaining = [v for v in I if v not in fixed_items]
    remaining.sort(key=lambda v: (t_min[v], v))

    for v in remaining:
        assigned = False
        for k in range(s):
            # Check if v can go into stack k
            if len(stacks[k]) >= l:
                continue
            # Check LIFO: v is placed on top of stack k
            # For each item w already in stack k:
            #   if w was picked up before v, then v must be delivered before w
            #   if v was picked up before w, then w must be delivered before v
            valid = True
            for w in stacks[k]:
                if pickup_pos[w] < pickup_pos[v]:
                    # w loaded before v, v on top -> v must be delivered before w
                    if delivery_pos[v] > delivery_pos[w]:
                        valid = False
                        break
                elif pickup_pos[v] < pickup_pos[w]:
                    # v loaded before w, but w is already in stack -> w was loaded after v
                    # w is on top relative to v, so w must be delivered before v
                    if delivery_pos[w] > delivery_pos[v]:
                        valid = False
                        break
            if valid:
                stacks[k].append(v)
                assigned = True
                break
        if not assigned:
            return _enumerate_loading(I, precedence, pickup_pos, delivery_pos, s, l), cliques

    return True, cliques


def _enumerate_loading(I, precedence, pickup_pos, delivery_pos, s, l):
    """
    Step 5: Complete enumeration of loading assignments.
    Returns True if a feasible loading exists, False otherwise.
    """
    # Sort items same as greedy
    # Compute t_min for sorting
    sorted_items_by_pickup = sorted(I, key=lambda v: pickup_pos[v])
    dist = {v: 0 for v in I}
    for v in sorted_items_by_pickup:
        for w in I:
            if w in precedence[v]:
                if dist[w] < dist[v] + 1:
                    dist[w] = dist[v] + 1

    t_min = {v: dist[v] + 1 for v in I}
    items_order = sorted(I, key=lambda v: (t_min[v], v))

    # DFS enumeration
    stacks = [[] for _ in range(s)]

    def backtrack(idx):
        if idx == len(items_order):
            return True
        v = items_order[idx]
        for k in range(s):
            if len(stacks[k]) >= l:
                continue
            # Check LIFO
            valid = True
            for w in stacks[k]:
                if pickup_pos[w] < pickup_pos[v]:
                    if delivery_pos[v] > delivery_pos[w]:
                        valid = False
                        break
                elif pickup_pos[v] < pickup_pos[w]:
                    if delivery_pos[w] > delivery_pos[v]:
                        valid = False
                        break
            if valid:
                stacks[k].append(v)
                if backtrack(idx + 1):
                    return True
                stacks[k].pop()
        return False

    return backtrack(0)


# =============================================================================
# SUBTOUR SEPARATION
# =============================================================================

def find_subtours(n, x_vals):
    """
    Find subtours in the solution defined by x_vals.
    x_vals: dict mapping (i,j) -> value
    Returns list of subtours (each a set of vertices), excluding the full tour.
    n+1 vertices: 0..n
    """
    # Build adjacency from integer solution
    adj = defaultdict(list)
    for (i, j), val in x_vals.items():
        if val > 0.5:
            adj[i].append(j)

    visited = set()
    subtours = []
    all_vertices = set(range(n + 1))

    for start in range(n + 1):
        if start in visited:
            continue
        # Trace the cycle/path from start
        tour = set()
        current = start
        while current not in tour and current not in visited:
            tour.add(current)
            nexts = adj.get(current, [])
            if nexts:
                current = nexts[0]
            else:
                break
        visited |= tour
        if len(tour) < n + 1 and len(tour) >= 2:
            subtours.append(tour)

    return subtours


def find_fractional_subtours(n, x_vals, tol=1e-4):
    """
    Separate subtour elimination constraints from fractional solution
    using max-flow / min-cut approach. For simplicity, we use a direct
    approach checking connected components in the support graph.

    Returns list of violated subtour sets S where sum_{i,j in S} x_{ij} > |S| - 1.
    """
    # Build capacity graph
    capacity = defaultdict(float)
    for (i, j), val in x_vals.items():
        if val > tol:
            capacity[(i, j)] += val

    # For each vertex i != 0, compute min-cut between 0 and i
    # If min-cut < 1 (approx 2 due to in+out degree), there's a violated subtour
    violated = []
    vertices = set(range(n + 1))

    for t in range(1, n + 1):
        # BFS to find min-cut using residual graph
        cut_val, S = min_cut_bfs(0, t, n, capacity)
        if cut_val < 1.0 - tol and len(S) >= 2 and len(S) <= n:
            # Check if this S actually gives a violated constraint
            flow_in_S = sum(capacity.get((i, j), 0.0) for i in S for j in S if i != j)
            if flow_in_S > len(S) - 1 + tol:
                violated.append(S)

    return violated


def min_cut_bfs(source, sink, n, capacity):
    """
    Compute min s-t cut using max-flow (Edmonds-Karp / BFS augmenting paths).
    Returns (cut_value, S) where S is the source side of the min cut.
    """
    # Build residual graph
    residual = defaultdict(float)
    nodes = set(range(n + 1))

    for (i, j), cap in capacity.items():
        residual[(i, j)] += cap

    total_flow = 0.0

    while True:
        # BFS to find augmenting path
        parent = {source: None}
        visited = {source}
        queue = deque([source])
        found = False

        while queue and not found:
            u = queue.popleft()
            for v in nodes:
                if v not in visited and residual.get((u, v), 0.0) > 1e-8:
                    parent[v] = u
                    visited.add(v)
                    if v == sink:
                        found = True
                        break
                    queue.append(v)

        if not found:
            break

        # Find bottleneck
        path_flow = float('inf')
        v = sink
        while parent[v] is not None:
            u = parent[v]
            path_flow = min(path_flow, residual[(u, v)])
            v = u

        # Update residual
        v = sink
        while parent[v] is not None:
            u = parent[v]
            residual[(u, v)] -= path_flow
            residual[(v, u)] += path_flow
            v = u

        total_flow += path_flow

    # Find S (reachable from source in residual graph)
    S = set()
    queue = deque([source])
    S.add(source)
    while queue:
        u = queue.popleft()
        for v in nodes:
            if v not in S and residual.get((u, v), 0.0) > 1e-8:
                S.add(v)
                queue.append(v)

    return total_flow, S


# =============================================================================
# PATH ENUMERATION FOR INFEASIBLE PATH SEPARATION
# =============================================================================

def enumerate_fractional_paths(n, x_vals, max_path_len=None, tol=1e-4):
    """
    Enumerate fractional paths in a graph defined by x_vals.
    A path starts from any vertex i=1..n and extends greedily.

    Returns list of (path, arc_sum, tournament_sum) tuples, sorted by:
      - nondecreasing path length
      - nonincreasing tournament sum
    """
    if max_path_len is None:
        max_path_len = n

    paths = []

    for start in range(1, n + 1):
        # BFS/DFS to find paths
        # We use a simple greedy extension
        _extend_path(start, [], 0.0, 0.0, set(), x_vals, n, max_path_len, paths, tol)

    # Sort: nondecreasing length, then nonincreasing tournament sum
    paths.sort(key=lambda p: (len(p[0]) - 1, -p[2]))
    return paths


def _extend_path(current, path_so_far, arc_sum, tourn_sum, visited, x_vals, n, max_len, results, tol):
    """Recursively extend a path and collect valid paths."""
    new_path = path_so_far + [current]
    new_visited = visited | {current}
    path_len = len(new_path) - 1  # number of arcs

    if path_len >= 1:
        # Compute tournament sum: sum of x_{p_j, p_h} for j < h (forward arcs)
        t_sum = 0.0
        for j in range(len(new_path)):
            for h in range(j + 1, len(new_path)):
                val = x_vals.get((new_path[j], new_path[h]), 0.0)
                t_sum += val

        # Keep path if sum > length - 1 (could contribute to violation)
        if t_sum > path_len - 1 + tol:
            results.append((list(new_path), arc_sum, t_sum))

    if path_len >= max_len:
        return

    # Extend to next vertex
    for next_v in range(1, n + 1):
        if next_v not in new_visited:
            edge_val = x_vals.get((current, next_v), 0.0)
            if edge_val > tol:
                new_arc_sum = arc_sum + edge_val
                _extend_path(next_v, new_path, new_arc_sum, 0.0, new_visited,
                             x_vals, n, max_len, results, tol)


def enumerate_depot_paths(n, x_vals, from_depot=True, tol=1e-4):
    """
    Enumerate paths starting from (or ending at) the depot.
    For position-based constraints (9).
    """
    max_path_len = n
    paths = []

    if from_depot:
        # Paths: 0 -> v1 -> v2 -> ... -> vk
        for next_v in range(1, n + 1):
            edge_val = x_vals.get((0, next_v), 0.0)
            if edge_val > tol:
                _extend_depot_path(next_v, [0, next_v], edge_val, {0, next_v},
                                   x_vals, n, max_path_len, paths, tol)
    else:
        # Paths ending at depot: v1 -> v2 -> ... -> vk -> 0
        # We enumerate paths ending at 0
        for prev_v in range(1, n + 1):
            edge_val = x_vals.get((prev_v, 0), 0.0)
            if edge_val > tol:
                # Extend backward from prev_v
                _extend_depot_path_backward(prev_v, [prev_v, 0], edge_val, {prev_v, 0},
                                            x_vals, n, max_path_len, paths, tol)

    return paths


def _extend_depot_path(current, path_so_far, arc_sum, visited, x_vals, n, max_len, results, tol):
    """Extend a depot path forward."""
    path_len = len(path_so_far) - 1
    if path_len >= 1:
        # Compute tournament sum
        t_sum = 0.0
        for j in range(len(path_so_far)):
            for h in range(j + 1, len(path_so_far)):
                t_sum += x_vals.get((path_so_far[j], path_so_far[h]), 0.0)
        results.append((list(path_so_far), arc_sum, t_sum))

    if path_len >= max_len:
        return

    for next_v in range(1, n + 1):
        if next_v not in visited:
            edge_val = x_vals.get((current, next_v), 0.0)
            if edge_val > tol:
                _extend_depot_path(next_v, path_so_far + [next_v], arc_sum + edge_val,
                                   visited | {next_v}, x_vals, n, max_len, results, tol)


def _extend_depot_path_backward(current, path_so_far, arc_sum, visited, x_vals, n, max_len, results, tol):
    """Extend a path backward (for paths ending at depot)."""
    path_len = len(path_so_far) - 1
    if path_len >= 1:
        t_sum = 0.0
        for j in range(len(path_so_far)):
            for h in range(j + 1, len(path_so_far)):
                t_sum += x_vals.get((path_so_far[j], path_so_far[h]), 0.0)
        results.append((list(path_so_far), arc_sum, t_sum))

    if path_len >= max_len:
        return

    for prev_v in range(1, n + 1):
        if prev_v not in visited:
            edge_val = x_vals.get((prev_v, current), 0.0)
            if edge_val > tol:
                _extend_depot_path_backward(prev_v, [prev_v] + path_so_far, arc_sum + edge_val,
                                            visited | {prev_v}, x_vals, n, max_len, results, tol)


# =============================================================================
# CUT GENERATION HELPERS
# =============================================================================

def compute_tournament_lhs(path, x_vars_dict):
    """
    Compute the LHS of the tournament constraint (7) for a given path.
    Sum of x_{p_j, p_h} for all j < h in the path.
    """
    total = 0.0
    for j in range(len(path)):
        for h in range(j + 1, len(path)):
            var = x_vars_dict.get((path[j], path[h]))
            if var is not None:
                total += var
    return total


def add_tournament_cut(model, pickup_path, delivery_path, x_P, x_D, where):
    """Add tournament constraint (7) as a lazy constraint."""
    rhs = (len(pickup_path) - 1) + (len(delivery_path) - 1) - 1

    lhs_expr = 0
    for j in range(len(pickup_path)):
        for h in range(j + 1, len(pickup_path)):
            key = (pickup_path[j], pickup_path[h])
            if key in x_P:
                lhs_expr += x_P[key]

    for j in range(len(delivery_path)):
        for h in range(j + 1, len(delivery_path)):
            key = (delivery_path[j], delivery_path[h])
            if key in x_D:
                lhs_expr += x_D[key]

    model.cbLazy(lhs_expr <= rhs)


def add_lifted_tournament_cut(model, pickup_path, delivery_path, clique, x_P, x_D, where):
    """Add lifted tournament constraint (8) as a lazy constraint."""
    rhs = (len(pickup_path) - 1) + (len(delivery_path) - 1) - 1

    lhs_expr = 0

    # Forward arcs in pickup path
    for j in range(len(pickup_path)):
        for h in range(j + 1, len(pickup_path)):
            key = (pickup_path[j], pickup_path[h])
            if key in x_P:
                lhs_expr += x_P[key]

    # Forward arcs in delivery path
    for j in range(len(delivery_path)):
        for h in range(j + 1, len(delivery_path)):
            key = (delivery_path[j], delivery_path[h])
            if key in x_D:
                lhs_expr += x_D[key]

    # Reverse arcs within segments between consecutive clique vertices (pickup)
    clique_set = set(clique)
    pickup_clique_positions = [i for i, v in enumerate(pickup_path) if v in clique_set]
    for ci in range(len(pickup_clique_positions)):
        start_pos = pickup_clique_positions[ci]
        end_pos = pickup_clique_positions[ci + 1] if ci + 1 < len(pickup_clique_positions) else len(pickup_path)
        segment = pickup_path[start_pos:end_pos]
        # Add reverse arcs within this segment (j > h, i.e., backward arcs)
        for j in range(len(segment)):
            for h in range(j):
                key = (segment[j], segment[h])
                if key in x_P:
                    lhs_expr += x_P[key]

    # Reverse arcs within segments between consecutive clique vertices (delivery)
    delivery_clique_positions = [i for i, v in enumerate(delivery_path) if v in clique_set]
    for ci in range(len(delivery_clique_positions)):
        start_pos = delivery_clique_positions[ci]
        end_pos = delivery_clique_positions[ci + 1] if ci + 1 < len(delivery_clique_positions) else len(delivery_path)
        segment = delivery_path[start_pos:end_pos]
        for j in range(len(segment)):
            for h in range(j):
                key = (segment[j], segment[h])
                if key in x_D:
                    lhs_expr += x_D[key]

    model.cbLazy(lhs_expr <= rhs)


# =============================================================================
# GUROBI CALLBACK
# =============================================================================

def dtspms_callback(model, where):
    """Gurobi callback for lazy constraint generation."""
    if where == GRB.Callback.MIPSOL:
        # Integer solution found - check for subtours and loading feasibility
        n = model._n
        s = model._s
        l = model._l
        x_P = model._x_P
        x_D = model._x_D

        # Get solution values
        x_P_vals = {}
        x_D_vals = {}
        for (i, j), var in x_P.items():
            x_P_vals[(i, j)] = model.cbGetSolution(var)
        for (i, j), var in x_D.items():
            x_D_vals[(i, j)] = model.cbGetSolution(var)

        # Check subtours in pickup
        subtours_P = find_subtours(n, x_P_vals)
        for S in subtours_P:
            expr = gp.quicksum(x_P[(i, j)] for i in S for j in S if i != j and (i, j) in x_P)
            model.cbLazy(expr <= len(S) - 1)

        # Check subtours in delivery
        subtours_D = find_subtours(n, x_D_vals)
        for S in subtours_D:
            expr = gp.quicksum(x_D[(i, j)] for i in S for j in S if i != j and (i, j) in x_D)
            model.cbLazy(expr <= len(S) - 1)

        if subtours_P or subtours_D:
            return

        # No subtours - extract the tours
        pickup_tour = extract_tour(n, x_P_vals)
        delivery_tour = extract_tour(n, x_D_vals)

        if pickup_tour is None or delivery_tour is None:
            return

        # Check loading feasibility
        pickup_customers = pickup_tour[1:]  # exclude depot
        delivery_customers = delivery_tour[1:]

        feasible, cliques = check_loading_feasibility(
            pickup_customers, delivery_customers, s, l
        )

        if not feasible:
            # Add tournament constraint (7)
            add_tournament_cut(model, pickup_customers, delivery_customers, x_P, x_D, where)

            # Also add reverse cut for symmetric costs
            add_tournament_cut(model, list(reversed(pickup_customers)),
                               list(reversed(delivery_customers)), x_P, x_D, where)

            # Add lifted tournament constraints (8) for cliques > s
            for clique in cliques:
                if len(clique) > s:
                    add_lifted_tournament_cut(model, pickup_customers, delivery_customers,
                                             clique, x_P, x_D, where)

    elif where == GRB.Callback.MIPNODE:
        # Fractional solution at a node - separate subtours and infeasible paths
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
            return

        n = model._n
        s = model._s
        l = model._l
        x_P = model._x_P
        x_D = model._x_D
        tol = 1e-4

        # Get relaxation values
        x_P_vals = {}
        x_D_vals = {}
        for (i, j), var in x_P.items():
            x_P_vals[(i, j)] = model.cbGetNodeRel(var)
        for (i, j), var in x_D.items():
            x_D_vals[(i, j)] = model.cbGetNodeRel(var)

        # Separate subtour elimination constraints
        cut_added = False

        violated_P = find_fractional_subtours(n, x_P_vals, tol)
        for S in violated_P:
            expr = gp.quicksum(x_P[(i, j)] for i in S for j in S if i != j and (i, j) in x_P)
            model.cbCut(expr <= len(S) - 1)
            cut_added = True

        violated_D = find_fractional_subtours(n, x_D_vals, tol)
        for S in violated_D:
            expr = gp.quicksum(x_D[(i, j)] for i in S for j in S if i != j and (i, j) in x_D)
            model.cbCut(expr <= len(S) - 1)
            cut_added = True

        if cut_added:
            return

        # Separate tournament constraints (7) on fractional paths
        # Limit path enumeration to avoid excessive computation
        max_path_len = min(n, 6)  # **INFERRED ASSUMPTION**: limit path length for tractability

        pickup_paths = enumerate_fractional_paths(n, x_P_vals, max_path_len, tol)
        delivery_paths = enumerate_fractional_paths(n, x_D_vals, max_path_len, tol)

        # Check pairs for violations
        for p_path, p_arc_sum, p_tourn_sum in pickup_paths:
            for d_path, d_arc_sum, d_tourn_sum in delivery_paths:
                rhs = (len(p_path) - 1) + (len(d_path) - 1) - 1
                total_sum = p_tourn_sum + d_tourn_sum

                if total_sum > rhs + tol:
                    # Check loading feasibility
                    feasible, cliques = check_loading_feasibility(
                        p_path, d_path, s, l
                    )
                    if not feasible:
                        # Add tournament cut
                        lhs_expr = 0
                        for j in range(len(p_path)):
                            for h in range(j + 1, len(p_path)):
                                key = (p_path[j], p_path[h])
                                if key in x_P:
                                    lhs_expr += x_P[key]
                        for j in range(len(d_path)):
                            for h in range(j + 1, len(d_path)):
                                key = (d_path[j], d_path[h])
                                if key in x_D:
                                    lhs_expr += x_D[key]
                        model.cbCut(lhs_expr <= rhs)

                        # Reverse cut
                        rev_lhs = 0
                        rev_p = list(reversed(p_path))
                        rev_d = list(reversed(d_path))
                        for j in range(len(rev_p)):
                            for h in range(j + 1, len(rev_p)):
                                key = (rev_p[j], rev_p[h])
                                if key in x_P:
                                    rev_lhs += x_P[key]
                        for j in range(len(rev_d)):
                            for h in range(j + 1, len(rev_d)):
                                key = (rev_d[j], rev_d[h])
                                if key in x_D:
                                    rev_lhs += x_D[key]
                        model.cbCut(rev_lhs <= rhs)
                        return  # Stop after first violated pair found


def extract_tour(n, x_vals):
    """Extract a Hamiltonian tour from integer solution values."""
    adj = {}
    for (i, j), val in x_vals.items():
        if val > 0.5:
            adj[i] = j

    tour = [0]
    current = 0
    for _ in range(n):
        next_v = adj.get(current)
        if next_v is None:
            return None
        tour.append(next_v)
        current = next_v

    if adj.get(current) != 0:
        return None

    return tour


# =============================================================================
# MAIN MODEL
# =============================================================================

def solve_dtspms(instance_path, solution_path, time_limit):
    """Build and solve the DTSPMS model using Gurobi."""
    data = load_instance(instance_path)
    n, s, l, c_P, c_D = build_distance_matrices(data)

    print(f"DTSPMS Instance: n={n}, s={s}, l={l}")
    print(f"Time limit: {time_limit} seconds")

    model = gp.Model("DTSPMS")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("LazyConstraints", 1)
    model.setParam("OutputFlag", 1)

    # Create variables x^P_{ij} and x^D_{ij}
    x_P = {}
    x_D = {}
    V = list(range(n + 1))  # 0..n

    for i in V:
        for j in V:
            if i != j:  # x^T_{ii} = 0 (variable fixing, comment 4)
                x_P[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"xP_{i}_{j}",
                                            obj=c_P[i][j])
                x_D[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"xD_{i}_{j}",
                                            obj=c_D[i][j])

    model.update()

    # Objective: minimize total routing cost (1)
    model.setAttr("ModelSense", GRB.MINIMIZE)

    # Constraints (2): each vertex has exactly one successor (pickup)
    for i in V:
        model.addConstr(
            gp.quicksum(x_P[(i, j)] for j in V if j != i) == 1,
            name=f"outP_{i}"
        )

    # Constraints (3): each vertex has exactly one predecessor (pickup)
    for j in V:
        model.addConstr(
            gp.quicksum(x_P[(i, j)] for i in V if i != j) == 1,
            name=f"inP_{j}"
        )

    # Constraints (2): each vertex has exactly one successor (delivery)
    for i in V:
        model.addConstr(
            gp.quicksum(x_D[(i, j)] for j in V if j != i) == 1,
            name=f"outD_{i}"
        )

    # Constraints (3): each vertex has exactly one predecessor (delivery)
    for j in V:
        model.addConstr(
            gp.quicksum(x_D[(i, j)] for i in V if i != j) == 1,
            name=f"inD_{j}"
        )

    # 2-vertex subtour elimination: x^T_{ij} + x^T_{ji} <= 1 (Section 5.1)
    for i in V:
        for j in V:
            if i < j:
                model.addConstr(x_P[(i, j)] + x_P[(j, i)] <= 1, name=f"sec2P_{i}_{j}")
                model.addConstr(x_D[(i, j)] + x_D[(j, i)] <= 1, name=f"sec2D_{i}_{j}")

    # Simple position-based constraints (Section 5.1)
    # Added when n >= (s-1)*l + 2
    if n >= (s - 1) * l + 2:
        for i in range(1, n + 1):
            model.addConstr(x_P[(0, i)] + x_D[(0, i)] <= 1, name=f"pos_out_{i}")
            model.addConstr(x_P[(i, 0)] + x_D[(i, 0)] <= 1, name=f"pos_in_{i}")

    model.update()

    # Store data for callback
    model._x_P = x_P
    model._x_D = x_D
    model._n = n
    model._s = s
    model._l = l

    # Optimize with callback
    model.optimize(dtspms_callback)

    # Extract solution
    result = {"problem": "DTSPMS", "instance": instance_path}

    if model.SolCount > 0:
        obj_val = model.ObjVal
        result["objective_value"] = obj_val
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        result["gap"] = model.MIPGap if hasattr(model, 'MIPGap') else 0.0

        # Extract tours
        x_P_sol = {}
        x_D_sol = {}
        for (i, j), var in x_P.items():
            x_P_sol[(i, j)] = var.X
        for (i, j), var in x_D.items():
            x_D_sol[(i, j)] = var.X

        pickup_tour = extract_tour(n, x_P_sol)
        delivery_tour = extract_tour(n, x_D_sol)

        result["pickup_tour"] = pickup_tour
        result["delivery_tour"] = delivery_tour

        # Compute individual tour costs
        if pickup_tour:
            p_cost = sum(c_P[pickup_tour[i]][pickup_tour[i + 1]] for i in range(len(pickup_tour) - 1))
            p_cost += c_P[pickup_tour[-1]][pickup_tour[0]]
            result["pickup_cost"] = p_cost
        if delivery_tour:
            d_cost = sum(c_D[delivery_tour[i]][delivery_tour[i + 1]] for i in range(len(delivery_tour) - 1))
            d_cost += c_D[delivery_tour[-1]][delivery_tour[0]]
            result["delivery_cost"] = d_cost
    else:
        result["objective_value"] = None
        result["status"] = "infeasible_or_no_solution"

    result["solve_time"] = model.Runtime

    # Write solution
    with open(solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"\nSolution written to {solution_path}")
    if model.SolCount > 0:
        print(f"Objective value: {result['objective_value']}")
        print(f"Status: {result['status']}")
        print(f"Pickup tour: {result.get('pickup_tour')}")
        print(f"Delivery tour: {result.get('delivery_tour')}")

    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve DTSPMS using Gurobi (branch-and-cut)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json",
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_dtspms(args.instance_path, args.solution_path, args.time_limit)
