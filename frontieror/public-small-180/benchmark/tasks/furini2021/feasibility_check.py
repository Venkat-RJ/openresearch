#!/usr/bin/env python3
"""
Feasibility checker for the Edge Interdiction Clique Problem (EICP)
from Furini et al. (2021).

Checks the hard constraints of the set-covering ILP formulation
(Formulation 3, equations 4a-4d) with clique-covering inequalities (5),
matching the model implemented in gurobi_code.py.

  Constraint 1 (W_E):  sum_{uv in E} w_uv <= k  (budget)
  Constraint 2 (W_E):  w_uv in {0,1}, uv in E  (edge validity / binary)
  Constraint 3 (4b):   sum_{ell in L} z_ell = 1  (exactly one level active)
  Constraint 4 (4c/5): clique-covering inequalities — verified via omega(G')
  Constraint 5 (4d):   z_ell in {0,1}  (binary z)
  Constraint 6 (obj):  objective consistency — reported objective_value
                       must equal the recomputed clique number omega(G')

The z_ell variables are reconstructed from the objective value: if theta is
the reported objective, then ell* = theta + 1, and z_{ell*} = 1.
Constraints 3 and 5 are satisfied by construction of this reconstruction.
Constraint 4 (4c/5) is verified by computing omega(G') and checking
omega(G') <= theta.

Constraint 6 is a Tier-C objective-recomputation defence against
score-gaming exploits: the EICP objective (1a / 4a) is theta, the clique
number of the interdicted graph G'.  It is fully determined by the
solution variables (interdicted_edges) and the instance, so we recompute
it exactly as omega(G') and require the reported objective_value to match.
Constraint 4 already enforces omega(G') <= theta (the interdiction must
actually achieve the claimed clique number, catching under-statement);
constraint 6 adds the complementary direction omega(G') >= theta, which
rejects solutions that over-state the objective.  Together they pin
theta = omega(G').  This is a *full* recompute: every variable the
objective depends on is present in the solution.
"""

import json
import argparse

# FRONTIEROR_RELEASE_CONTRACT_V2
_FRONTIEROR_REQUIRED_SOLUTION_FIELDS = ('interdicted_edges', 'objective_value')
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



# ============================================================================
# Max clique solver (branch-and-bound with greedy coloring bound)
# ============================================================================

def greedy_coloring(vertices_list, adj):
    """Greedy sequential coloring. Returns dict vertex -> color (0-indexed)."""
    color = {}
    for v in vertices_list:
        used = set()
        for u in adj.get(v, set()):
            if u in color:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


def max_clique_bb(vertices, adj):
    """
    Branch-and-bound maximum clique with degeneracy ordering and greedy
    coloring bound. Returns the maximum clique as a list of vertices.
    """
    if not vertices:
        return []

    active_verts = list(vertices)

    # Degeneracy ordering
    remaining = set(active_verts)
    order = []
    deg = {v: len(adj.get(v, set()) & remaining) for v in remaining}
    while remaining:
        v = min(remaining, key=lambda x: deg[x])
        order.append(v)
        remaining.remove(v)
        for u in adj.get(v, set()):
            if u in remaining:
                deg[u] -= 1
    order.reverse()

    # Greedy initial clique
    best = []
    for v in order:
        if all(v in adj.get(u, set()) for u in best):
            best.append(v)
    best = list(best)
    best_size = len(best)

    def bb(candidates, clique):
        nonlocal best, best_size

        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best = list(clique)
            return

        col = greedy_coloring(candidates, adj)
        num_colors = max(col.values()) + 1 if col else 0

        if len(clique) + num_colors <= best_size:
            return

        sorted_cands = sorted(candidates, key=lambda v: col[v])

        for i in range(len(sorted_cands) - 1, -1, -1):
            v = sorted_cands[i]
            if len(clique) + col[v] + 1 <= best_size:
                return
            new_cands = [u for u in sorted_cands[:i] if u in adj.get(v, set())]
            clique.append(v)
            bb(new_cands, clique)
            clique.pop()

    bb(order, [])
    return best


# ============================================================================
# Feasibility checking
# ============================================================================

def check_feasibility(instance, solution):
    """
    Check all hard constraints of the EICP set-covering ILP formulation (4a-4d).

    Returns (feasible, violated_constraints, violations, violation_magnitudes).
    """
    tol = 1e-5
    eps = 1e-5

    vertices = instance["vertices"]
    edges_list = instance["edges"]
    k = instance["interdiction_budget_k"]

    # Build edge set (canonical: smaller index first)
    edge_set = set()
    for u, v in edges_list:
        edge_set.add((min(u, v), max(u, v)))

    objective_value = solution["objective_value"]
    interdicted_edges_raw = solution["interdicted_edges"]

    # Canonicalize interdicted edges
    interdicted_edges = []
    for e in interdicted_edges_raw:
        interdicted_edges.append((min(e[0], e[1]), max(e[0], e[1])))

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ------------------------------------------------------------------
    # Constraint 1 (W_E budget): sum w_uv <= k
    # ------------------------------------------------------------------
    num_interdicted = len(interdicted_edges)
    lhs_1 = float(num_interdicted)
    rhs_1 = float(k)
    violation_amount_1 = max(0.0, lhs_1 - rhs_1)
    if violation_amount_1 > tol:
        violated_constraints.add(1)
        violations.append(
            f"Budget exceeded: {num_interdicted} edges interdicted but budget k={k}"
        )
        normalizer = max(abs(rhs_1), eps)
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": lhs_1,
            "rhs": rhs_1,
            "raw_excess": violation_amount_1,
            "normalizer": normalizer,
            "ratio": violation_amount_1 / normalizer,
        })

    # ------------------------------------------------------------------
    # Constraint 2 (W_E binary): w_uv in {0,1}, uv in E
    # Each interdicted edge must be a valid edge in E; no duplicates.
    # ------------------------------------------------------------------
    seen = set()
    for idx, e in enumerate(interdicted_edges):
        if e not in edge_set:
            violated_constraints.add(2)
            violations.append(
                f"Interdicted edge {list(e)} is not in the graph edge set E"
            )
            # LHS=1 (interdiction attempted), RHS=0 (not a valid edge)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": 1.0,
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": eps,
                "ratio": 1.0 / eps,
            })
        if e in seen:
            violated_constraints.add(2)
            violations.append(
                f"Duplicate interdicted edge {list(e)}"
            )
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": 2.0,
                "rhs": 1.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            })
        seen.add(e)

    # ------------------------------------------------------------------
    # Constraint 3 (4b): sum_{ell in L} z_ell = 1
    # Constraint 5 (4d): z_ell in {0,1}
    #
    # The z_ell variables are not in the solution; we reconstruct them
    # from the objective: theta = sum (ell-1)*z_ell, with exactly one
    # z_ell = 1.  So ell* = theta + 1, z_{ell*} = 1.
    # Constraints 3 and 5 are satisfied by construction.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 4 (4c / 5): clique-covering inequalities
    #   sum_{e in E(K)} w_e >= z_ell   for ell in L\{ell_max+1}, K in K_ell
    #   (strengthened form (5) uses f(ell, K) coefficients)
    #
    # With z_{ell*}=1, this requires every clique K of size >= ell* to
    # have at least one (resp. f(ell*,K)) edge(s) interdicted, i.e.
    # omega(G') <= ell* - 1 = theta.
    #
    # We compute omega(G') and verify omega(G') <= theta.
    # ------------------------------------------------------------------
    interdicted_set = set(interdicted_edges)

    # Build adjacency for the interdicted graph G'
    adj_prime = {v: set() for v in vertices}
    for u, v in edges_list:
        e = (min(u, v), max(u, v))
        if e not in interdicted_set:
            adj_prime[u].add(v)
            adj_prime[v].add(u)

    # Compute max clique in G'
    omega_prime = len(max_clique_bb(vertices, adj_prime))

    # Check Constraint 4 (4c/5): omega(G') <= theta
    lhs_4 = float(objective_value)
    rhs_4 = float(omega_prime)
    # theta >= omega(G')  ⟺  omega(G') <= theta
    violation_amount_4 = max(0.0, rhs_4 - lhs_4)
    if violation_amount_4 > tol:
        violated_constraints.add(4)
        violations.append(
            f"Clique-covering inequality (4c/5) violated: omega(G')="
            f"{omega_prime} > theta={objective_value} "
            f"(interdicted graph has a clique larger than the claimed objective)"
        )
        normalizer = max(abs(rhs_4), eps)
        violation_magnitudes.append({
            "constraint": 4,
            "lhs": lhs_4,
            "rhs": rhs_4,
            "raw_excess": violation_amount_4,
            "normalizer": normalizer,
            "ratio": violation_amount_4 / normalizer,
        })

    # ------------------------------------------------------------------
    # Constraint 6 (objective consistency, Tier-C obj recompute):
    #   theta == omega(G')
    #
    # The EICP objective (1a / 4a) is theta, the clique number of the
    # interdicted graph G'.  Every variable it depends on (the
    # interdicted_edges) is present in the solution, so the objective is
    # *fully recomputable* as omega(G') (already computed above for
    # constraint 4).  The reported objective_value must match it.
    #
    # Constraint 4 already enforces omega(G') <= theta (catches a solution
    # that under-states the objective).  Constraint 6 adds the
    # complementary direction omega(G') >= theta, rejecting solutions that
    # over-state the objective (e.g. obj=sys.float_info.max).  Together
    # they pin theta = omega(G').
    #
    # The objective is an integer clique count, so an absolute tolerance
    # of 0.5 fires on any genuine integer mismatch (>= 1) while staying
    # robust to int/float representation noise.
    # ------------------------------------------------------------------
    obj_tol = 0.5
    reported_obj = float(objective_value)
    true_obj = float(omega_prime)
    obj_diff = abs(reported_obj - true_obj)
    if obj_diff > obj_tol:
        violated_constraints.add(6)
        violations.append(
            f"Objective consistency violated: reported objective_value="
            f"{objective_value} differs from recomputed clique number "
            f"omega(G')={omega_prime} (|diff|={obj_diff:.3g}, tol={obj_tol})"
        )
        normalizer = max(abs(true_obj), eps)
        violation_magnitudes.append({
            "constraint": 6,
            "lhs": reported_obj,
            "rhs": true_obj,
            "raw_excess": obj_diff,
            "normalizer": normalizer,
            "ratio": obj_diff / normalizer,
        })

    feasible = len(violated_constraints) == 0
    return feasible, sorted(violated_constraints), violations, violation_magnitudes


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the EICP (Furini et al., 2021)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path for the output feasibility result JSON file")
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

    feasible, violated_constraints, violations, violation_magnitudes = \
        check_feasibility(instance, solution)

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if feasible else "INFEASIBLE"
    print(f"Result: {status}")
    if violated_constraints:
        print(f"Violated constraints: {violated_constraints}")
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
