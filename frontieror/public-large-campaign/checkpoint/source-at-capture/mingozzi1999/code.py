import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));D=d['distance_matrix'];N=len(D);K=d['num_vehicles'];dep=d['depot']['index'];lh={i['index']:i['demand'] for i in d['linehaul_customers']};bh={i['index']:i['demand'] for i in d['backhaul_customers']};manager=pywrapcp.RoutingIndexManager(N,K,dep);r=pywrapcp.RoutingModel(manager);cost=r.RegisterTransitCallback(lambda i,j:D[manager.IndexToNode(i)][manager.IndexToNode(j)]);r.SetArcCostEvaluatorOfAllVehicles(cost)
for name,dem in [('LH',lh),('BH',bh)]:
 cb=r.RegisterUnaryTransitCallback(lambda i,dem=dem:dem.get(manager.IndexToNode(i),0));r.AddDimensionWithVehicleCapacity(cb,0,[d['vehicle_capacity']]*K,True,name)
for i in bh:
 for j in lh:r.NextVar(manager.NodeToIndex(i)).RemoveValue(manager.NodeToIndex(j))
if not d['constraints']['backhaul_only_routes_allowed']:
 for k in range(K):
  for j in bh:r.NextVar(r.Start(k)).RemoveValue(manager.NodeToIndex(j))
params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.seconds=max(1,a.time_limit-2);sol=r.SolveWithParameters(params)
if sol:
 routes=[];obj=0
 for k in range(K):
  i=r.Start(k);path=[]
  while not r.IsEnd(i):path.append(manager.IndexToNode(i));i=sol.Value(r.NextVar(i))
  path.append(dep)
  if len(path)>2:routes.append(path);obj+=sum(D[path[z]][path[z+1]] for z in range(len(path)-1))
 s={'objective_value':obj,'routes':routes};json.dump(s,open(a.solution_path,'w'))
 if a.log_path:SolutionLogger(a.log_path,sense='minimize').log_solution(obj,s)
