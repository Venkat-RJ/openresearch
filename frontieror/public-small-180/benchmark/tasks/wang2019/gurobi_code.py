#!/usr/bin/env python3
"""
Gurobi implementation of Service Network Design with Heterogeneous Fleet.
Paper: Wang, Qi, Cheng, Zhang (2019) - EJOR 276, 483-494

Uses the arc-based formulation (Equations 1-6):
  min  sum_{k in K} sum_{(i,j) in A} c^k_{ij} * x^k_{ij}
     + sum_{f in F} sum_{(i,j) in A} h^f_{ij} * y^f_{ij}
  s.t.
    (2) Flow conservation for each commodity k at each node i
    (3) Design-balance for each fleet type f at each node i
    (4) Capacity linking on each arc (i,j)
    (5) x^k_{ij} >= 0
    (6) y^f_{ij} in Z_0^+
"""

import json
import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Service Network Design - Gurobi Arc-Based Formulation"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # ---- Load instance ----
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    nodes = data["nodes"]
    arcs = data["arcs"]
    commodities = data["commodities"]
    fleet_types = data["fleet_types"]

    # ---- Cost lookup helpers (paper convention c^k_{ij}, h^f_{ij}) ----
    # An instance MAY supply commodity-indexed flow cost on each arc as
    #   arc["flow_costs"]          # dict {commodity_id: cost} or list indexed by id
    # and arc-indexed fixed cost on each fleet type as
    #   fleet["arc_fixed_costs"]   # dict {arc_id: cost} or list indexed by arc id
    # When those higher-dimensional fields are absent, fall back to the
    # broadcast scalar (arc["flow_cost"], fleet["fixed_cost"]) — equivalent
    # to c^k_{ij} = c_{ij} for all k, h^f_{ij} = h^f for all (i,j).
    def _flow_cost(arc, k):
        fc = arc.get("flow_costs")
        if isinstance(fc, dict):
            if str(k) in fc:
                return fc[str(k)]
            if k in fc:
                return fc[k]
        elif isinstance(fc, list) and 0 <= k < len(fc):
            return fc[k]
        return arc["flow_cost"]

    def _fixed_cost(fleet, arc_id):
        ac = fleet.get("arc_fixed_costs")
        if isinstance(ac, dict):
            if str(arc_id) in ac:
                return ac[str(arc_id)]
            if arc_id in ac:
                return ac[arc_id]
        elif isinstance(ac, list) and 0 <= arc_id < len(ac):
            return ac[arc_id]
        return fleet["fixed_cost"]

    # Build arc list as tuples and successor/predecessor sets
    arc_list = [(a["from"], a["to"]) for a in arcs]
    arc_by_endpoints = {}
    for a in arcs:
        arc_by_endpoints[(a["from"], a["to"])] = a

    successors = {n: [] for n in nodes}
    predecessors = {n: [] for n in nodes}
    for a in arcs:
        successors[a["from"]].append(a["to"])
        predecessors[a["to"]].append(a["from"])

    # ---- Create Gurobi model ----
    model = gp.Model("SND_ArcBased")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)

    # ---- Decision variables ----
    # x[k, i, j]: continuous flow of commodity k on arc (i,j)
    # Per paper Eq. (1) the unit flow cost c^k_{ij} is indexed by both
    # commodity k and arc (i,j); _flow_cost() resolves it from the instance
    # (commodity-indexed when present, scalar broadcast otherwise).
    x = {}
    for kd in commodities:
        k = kd["id"]
        for a in arcs:
            i, j = a["from"], a["to"]
            x[k, i, j] = model.addVar(
                lb=0.0, vtype=GRB.CONTINUOUS,
                obj=_flow_cost(a, k),
                name=f"x_{k}_{i}_{j}"
            )

    # y[f, i, j]: integer number of vehicles of fleet type f on arc (i,j)
    # Per paper Eq. (1) the fixed cost h^f_{ij} is indexed by both fleet
    # type f and arc (i,j); _fixed_cost() resolves it from the instance
    # (arc-indexed when present, scalar broadcast otherwise).
    y = {}
    for fd in fleet_types:
        f = fd["id"]
        for a in arcs:
            i, j = a["from"], a["to"]
            y[f, i, j] = model.addVar(
                lb=0, vtype=GRB.INTEGER,
                obj=_fixed_cost(fd, a["id"]),
                name=f"y_{f}_{i}_{j}"
            )

    model.update()

    # ---- Constraints ----

    # Constraint (2): Flow conservation for each commodity k at each node i
    #   sum_{j in N_i^+} x^k_{ij} - sum_{j in N_i^-} x^k_{ji} = w^k_i
    for kd in commodities:
        k = kd["id"]
        origin = kd["origin"]
        destination = kd["destination"]
        demand = kd["demand"]
        for i in nodes:
            if i == origin:
                w = demand
            elif i == destination:
                w = -demand
            else:
                w = 0
            model.addConstr(
                gp.quicksum(x[k, i, j] for j in successors[i])
                - gp.quicksum(x[k, j, i] for j in predecessors[i]) == w,
                name=f"flow_{k}_{i}"
            )

    # Constraint (3): Design-balance for each fleet type f at each node i
    #   sum_{j in N_i^+} y^f_{ij} - sum_{j in N_i^-} y^f_{ji} = 0
    for fd in fleet_types:
        f = fd["id"]
        for i in nodes:
            model.addConstr(
                gp.quicksum(y[f, i, j] for j in successors[i])
                - gp.quicksum(y[f, j, i] for j in predecessors[i]) == 0,
                name=f"balance_{f}_{i}"
            )

    # Constraint (4): Capacity linking on each arc (i,j)
    #   sum_{k in K} x^k_{ij} <= sum_{f in F} u^f * y^f_{ij}
    for a in arcs:
        i, j = a["from"], a["to"]
        model.addConstr(
            gp.quicksum(x[kd["id"], i, j] for kd in commodities)
            <= gp.quicksum(fd["capacity"] * y[fd["id"], i, j] for fd in fleet_types),
            name=f"cap_{i}_{j}"
        )

    # ---- Optimize ----
    model.optimize()

    # ---- Extract solution ----
    solution = {}

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal

        # Flow variables
        flows = {}
        for kd in commodities:
            k = kd["id"]
            for a in arcs:
                i, j = a["from"], a["to"]
                val = x[k, i, j].X
                if val > 1e-6:
                    flows[f"x_{k}_{i}_{j}"] = round(val, 6)
        solution["flows"] = flows

        # Vehicle assignment variables
        vehicles = {}
        for fd in fleet_types:
            f = fd["id"]
            for a in arcs:
                i, j = a["from"], a["to"]
                val = y[f, i, j].X
                if val > 0.5:
                    vehicles[f"y_{f}_{i}_{j}"] = int(round(val))
        solution["vehicles"] = vehicles

        if model.Status == GRB.OPTIMAL:
            solution["status"] = "optimal"
        else:
            solution["status"] = "feasible"
        try:
            solution["mip_gap"] = model.MIPGap
        except Exception:
            solution["mip_gap"] = None
    else:
        solution["objective_value"] = None
        solution["status"] = "no_solution_found"

    with open(args.solution_path, "w") as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']}")
        print(f"Status: {solution['status']}")


if __name__ == "__main__":
    main()
