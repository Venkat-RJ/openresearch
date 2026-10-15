import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
_PROCESS_START = time.monotonic()
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    for key in ('instance_path', 'solution_path', 'log_path'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--time_limit', type=int, default=60)
    args = parser.parse_args()
    deadline = _PROCESS_START + max(0., args.time_limit - 5.)
    with open(args.instance_path) as handle:
        data = json.load(handle)
    n = data['num_retailers']
    horizon = data['time_horizon']
    supplier = data['supplier']
    retailers = {r['id']: r for r in data['retailers']}
    distance = data['distance_matrix']
    capacity = data['vehicle_capacity']
    logger = SolutionLogger(args.log_path, sense='minimize')
    best = float('inf')
    best_solution = None
    best_tours = None

    def save(deliveries, routes, supplier_inventory, inventory, tours=None):
        nonlocal best, best_solution, best_tours
        objective = supplier['inventory_cost'] * sum(supplier_inventory.values())
        objective += sum(retailers[i]['inventory_cost'] * sum(inventory[str(i)].values()) for i in retailers)
        objective += sum(distance[i][j] * count for edges in routes.values() for i, j, count in edges)
        if objective >= best:
            return
        best = objective
        best_solution = {'objective_value': objective, 'deliveries': deliveries, 'routes': routes,
                         'supplier_inventory': supplier_inventory, 'retailer_inventory': inventory}
        best_tours = tours
        with open(args.solution_path, 'w') as handle:
            json.dump(best_solution, handle)
        if hasattr(logger, 'log_solution'):
            logger.log_solution(objective, best_solution)
        else:
            logger.log(objective)

    def make_routes(deliveries, supplier_inventory, inventory):
        routes = {}
        tours = {}
        for t in range(1, horizon + 1):
            pending = {i for i in retailers if deliveries[str(t)][str(i)] > 0}
            tour = [0, 0]
            while pending:
                _, node, position = min((distance[tour[k]][i] + distance[i][tour[k + 1]] - distance[tour[k]][tour[k + 1]], i, k + 1)
                                        for i in pending for k in range(len(tour) - 1))
                tour.insert(position, node)
                pending.remove(node)
            for _ in range(30):
                if time.monotonic() >= deadline:
                    break
                change = None
                for left in range(1, len(tour) - 2):
                    for right in range(left + 1, len(tour) - 1):
                        delta = distance[tour[left - 1]][tour[right]] + distance[tour[left]][tour[right + 1]] - distance[tour[left - 1]][tour[left]] - distance[tour[right]][tour[right + 1]]
                        if delta < -1e-9 and (change is None or delta < change[0]):
                            change = (delta, left, right)
                if change is None:
                    break
                _, left, right = change
                tour[left:right + 1] = reversed(tour[left:right + 1])
            counts = {}
            for i, j in zip(tour, tour[1:]):
                if i == j:
                    continue
                edge = tuple(sorted((i, j)))
                counts[edge] = counts.get(edge, 0) + 1
            routes[str(t)] = [[i, j, count] for (i, j), count in counts.items()]
            tours[t] = tour
        save(deliveries, routes, supplier_inventory, inventory, tours)
        return tours

    stock = {i: retailers[i]['starting_inventory'] for i in retailers}
    plant = supplier['starting_inventory']
    deliveries = {}
    plant_inventory = {'1': plant}
    inventory = {str(i): {'1': stock[i]} for i in retailers}
    feasible = True
    for t in range(1, horizon + 1):
        quantities = {i: retailers[i]['max_inventory'] - stock[i] if stock[i] < retailers[i]['demand'] else 0 for i in retailers}
        total = sum(quantities.values())
        if total > min(capacity, plant) or any(stock[i] + quantities[i] < retailers[i]['demand'] for i in retailers):
            feasible = False
            break
        deliveries[str(t)] = {str(i): quantities[i] for i in retailers}
        plant += supplier['production_rate'] - total
        plant_inventory[str(t + 1)] = plant
        for i in retailers:
            stock[i] += quantities[i] - retailers[i]['demand']
            inventory[str(i)][str(t + 1)] = stock[i]
    if feasible:
        make_routes(deliveries, plant_inventory, inventory)
    print('CONSTRUCTIVE_END elapsed', time.monotonic() - _PROCESS_START, 'feasible', feasible, 'best', best, flush=True)
    if time.monotonic() >= deadline:
        return

    model = cp_model.CpModel()
    B = {}
    I = {}
    x = {}
    z = {}
    costs = []
    for t in range(1, horizon + 2):
        B[t] = model.NewIntVar(0, supplier['starting_inventory'] + horizon * supplier['production_rate'], 'B' + str(t))
        costs.append(round(supplier['inventory_cost'] * 10000) * B[t])
        for i in retailers:
            I[i, t] = model.NewIntVar(0, retailers[i]['max_inventory'], 'I' + str((i, t)))
            costs.append(round(retailers[i]['inventory_cost'] * 10000) * I[i, t])
    model.Add(B[1] == supplier['starting_inventory'])
    for i in retailers:
        model.Add(I[i, 1] == retailers[i]['starting_inventory'])
    for t in range(1, horizon + 1):
        for i in range(n + 1):
            z[i, t] = model.NewBoolVar('z' + str((i, t)))
        for i in retailers:
            upper = retailers[i]['max_inventory']
            x[i, t] = model.NewIntVar(0, upper, 'x' + str((i, t)))
            model.Add(x[i, t] + I[i, t] == upper).OnlyEnforceIf(z[i, t])
            model.Add(x[i, t] == 0).OnlyEnforceIf(z[i, t].Not())
            model.Add(x[i, t] + I[i, t] <= upper)
            model.Add(I[i, t + 1] == I[i, t] + x[i, t] - retailers[i]['demand'])
            model.Add(z[i, t] <= z[0, t])
        model.Add(sum(x[i, t] for i in retailers) <= capacity * z[0, t])
        model.Add(sum(x[i, t] for i in retailers) <= B[t])
        model.Add(B[t + 1] == B[t] + supplier['production_rate'] - sum(x[i, t] for i in retailers))
    model.Minimize(sum(costs) + sum(round(2 * distance[0][i] * 10000) * z[i, t] for i in retailers for t in range(1, horizon + 1)))
    seed_solver = cp_model.CpSolver()
    seed_solver.parameters.num_search_workers = 1
    seed_solver.parameters.max_time_in_seconds = max(.01, min(3., deadline - time.monotonic()))
    status = seed_solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        make_routes({str(t): {str(i): seed_solver.Value(x[i, t]) for i in retailers} for t in range(1, horizon + 1)},
                    {str(t): seed_solver.Value(v) for t, v in B.items()},
                    {str(i): {str(t): seed_solver.Value(I[i, t]) for t in range(1, horizon + 2)} for i in retailers})
    print('INVENTORY_SEED_END elapsed', time.monotonic() - _PROCESS_START, 'status', seed_solver.StatusName(status), 'best', best, flush=True)
    if time.monotonic() >= deadline:
        return

    if n > 10 and best_tours is not None:
        current_tours = best_tours
        refinement_deadline = min(deadline - 5., time.monotonic() + 16.)
        for iteration in range(8):
            remaining = refinement_deadline - time.monotonic()
            if remaining <= 0:
                break
            prices = []
            for t in range(1, horizon + 1):
                tour = current_tours[t]
                positions = {node: k for k, node in enumerate(tour[1:-1], 1)}
                for i in retailers:
                    if i in positions:
                        k = positions[i]
                        marginal = distance[tour[k - 1]][i] + distance[i][tour[k + 1]] - distance[tour[k - 1]][tour[k + 1]]
                    else:
                        marginal = min(distance[tour[k]][i] + distance[i][tour[k + 1]] - distance[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))
                    multiplier = 1. + (((i * 17 + t * 29 + iteration * 31) % 19) - 9) / 50.
                    prices.append(round(max(0., marginal) * multiplier * 10000) * z[i, t])
            model.Minimize(sum(costs) + sum(prices))
            model.ClearHints()
            for t, variable in B.items():
                model.AddHint(variable, best_solution['supplier_inventory'][str(t)])
            for (i, t), variable in I.items():
                model.AddHint(variable, best_solution['retailer_inventory'][str(i)][str(t)])
            for (i, t), variable in x.items():
                model.AddHint(variable, best_solution['deliveries'][str(t)][str(i)])
            for t in range(1, horizon + 1):
                visited = {i for i in retailers if best_solution['deliveries'][str(t)][str(i)] > 0}
                model.AddHint(z[0, t], int(bool(visited)))
                for i in retailers:
                    model.AddHint(z[i, t], int(i in visited))
            seed_solver.parameters.max_time_in_seconds = min(2., remaining)
            status = seed_solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                current_tours = make_routes({str(t): {str(i): seed_solver.Value(x[i, t]) for i in retailers} for t in range(1, horizon + 1)},
                                           {str(t): seed_solver.Value(v) for t, v in B.items()},
                                           {str(i): {str(t): seed_solver.Value(I[i, t]) for t in range(1, horizon + 2)} for i in retailers})
            print('ROUTE_SCHEDULE_REFINEMENT iteration', iteration, 'elapsed', time.monotonic() - _PROCESS_START, 'status', seed_solver.StatusName(status), 'best', best, flush=True)
    model.ClearHints()
    arcs = {}
    loops = {}
    for t in range(1, horizon + 1):
        if time.monotonic() >= deadline:
            print('ROUTE_BUILD_DEADLINE elapsed', time.monotonic() - _PROCESS_START, flush=True)
            return
        circuit = []
        for i in range(n + 1):
            loops[i, t] = model.NewBoolVar('skip' + str((i, t)))
            model.Add(loops[i, t] + z[i, t] == 1)
            circuit.append((i, i, loops[i, t]))
            for j in range(n + 1):
                if i == j:
                    continue
                arcs[i, j, t] = model.NewBoolVar('a' + str((i, j, t)))
                circuit.append((i, j, arcs[i, j, t]))
                costs.append(round(distance[i][j] * 10000) * arcs[i, j, t])
        model.AddCircuit(circuit)
    model.Minimize(sum(costs))
    if best_solution is not None:
        for t, variable in B.items():
            model.AddHint(variable, best_solution['supplier_inventory'][str(t)])
        for (i, t), variable in I.items():
            model.AddHint(variable, best_solution['retailer_inventory'][str(i)][str(t)])
        for (i, t), variable in x.items():
            model.AddHint(variable, best_solution['deliveries'][str(t)][str(i)])
        for t in range(1, horizon + 1):
            tour = best_tours[t]
            active = set(tour) if len(tour) > 2 else set()
            used_arcs = set(zip(tour, tour[1:]))
            for i in range(n + 1):
                model.AddHint(z[i, t], int(i in active))
                model.AddHint(loops[i, t], int(i not in active))
                for j in range(n + 1):
                    if i != j:
                        model.AddHint(arcs[i, j, t], int((i, j) in used_arcs))

    class Callback(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            routes = {}
            for t in range(1, horizon + 1):
                counts = {}
                for i in range(n + 1):
                    for j in range(n + 1):
                        if i != j and self.Value(arcs[i, j, t]):
                            edge = tuple(sorted((i, j)))
                            counts[edge] = counts.get(edge, 0) + 1
                routes[str(t)] = [[i, j, count] for (i, j), count in counts.items()]
            save({str(t): {str(i): self.Value(x[i, t]) for i in retailers} for t in range(1, horizon + 1)}, routes,
                 {str(t): self.Value(v) for t, v in B.items()},
                 {str(i): {str(t): self.Value(I[i, t]) for t in range(1, horizon + 2)} for i in retailers})

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = remaining
    print('FULL_BUILD_END elapsed', time.monotonic() - _PROCESS_START, 'solve_budget', remaining, 'best', best, flush=True)
    status = solver.Solve(model, Callback())
    print('FULL_SOLVE_END elapsed', time.monotonic() - _PROCESS_START, 'status', solver.StatusName(status), 'best', best, 'bound', solver.BestObjectiveBound() / 10000., flush=True)


if __name__ == '__main__':
    main()
