import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,math,time
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));nodes=[d['depot']]+d['customers'];N=len(nodes);K=max(1,N-1);scale=10000
# The SCP has no fleet bound; one vehicle per customer permits singleton columns.
D=[[math.hypot(x['x']-y['x'],x['y']-y['y']) for y in nodes] for x in nodes];manager=pywrapcp.RoutingIndexManager(N,K,0);r=pywrapcp.RoutingModel(manager)
def transit(i,j):
 u=manager.IndexToNode(i);v=manager.IndexToNode(j);return int(math.ceil(scale*(D[u][v]+nodes[u]['service_time'])))
def dist(i,j):return int(round(scale*D[manager.IndexToNode(i)][manager.IndexToNode(j)]))
ti=r.RegisterTransitCallback(transit);di=r.RegisterTransitCallback(dist);r.SetArcCostEvaluatorOfAllVehicles(di);r.AddDimension(ti,scale*d['depot']['due_date'],scale*d['depot']['due_date'],False,'Time');td=r.GetDimensionOrDie('Time')
for i,node in enumerate(nodes):
 if i:td.CumulVar(manager.NodeToIndex(i)).SetRange(scale*node['ready_time'],scale*node['due_date'])
for k in range(K):
 td.CumulVar(r.Start(k)).SetRange(scale*d['depot']['ready_time'],scale*d['depot']['due_date']);td.CumulVar(r.End(k)).SetRange(scale*d['depot']['ready_time'],scale*d['depot']['due_date'])
demand=r.RegisterUnaryTransitCallback(lambda i:nodes[manager.IndexToNode(i)]['demand']);r.AddDimensionWithVehicleCapacity(demand,0,[d['vehicle_capacity']]*K,True,'Capacity');params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.seconds=max(1,a.time_limit-2);sol=r.SolveWithParameters(params)
if sol:
 routes=[];obj=0
 for k in range(K):
  i=r.Start(k);path=[]
  while not r.IsEnd(i):path.append(manager.IndexToNode(i));i=sol.Value(r.NextVar(i))
  path.append(0)
  if len(path)>2:
   obj+=sum(D[path[z]][path[z+1]] for z in range(len(path)-1));routes.append({'customer_ids':[nodes[z]['id'] for z in path[1:-1]],'path':[nodes[z]['id'] for z in path]})
 s={'objective_value':obj,'routes':routes};json.dump(s,open(a.solution_path,'w'))
 if a.log_path:SolutionLogger(a.log_path,sense='minimize').log_solution(obj,s)
