import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,math
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));nodes=d['nodes'];par=d['problem_parameters'];N=len(nodes);s=next(i for i,z in enumerate(nodes) if z['type']=='depot_source');e=next(i for i,z in enumerate(nodes) if z['type']=='depot_sink');K=par['n_customers'];scale=10000;dist=[[math.hypot(u['x']-v['x'],u['y']-v['y']) for v in nodes] for u in nodes];manager=pywrapcp.RoutingIndexManager(N,K,[s]*K,[e]*K);r=pywrapcp.RoutingModel(manager)
 def cost(i,j):return round(dist[manager.IndexToNode(i)][manager.IndexToNode(j)]*scale)
 cb=r.RegisterTransitCallback(cost);r.SetArcCostEvaluatorOfAllVehicles(cb)
 def demand(i):return nodes[manager.IndexToNode(i)]['demand']
 dc=r.RegisterUnaryTransitCallback(demand);r.AddDimensionWithVehicleCapacity(dc,0,[par['vehicle_capacity']]*K,True,'Capacity')
 def tt(i,j):
  u=manager.IndexToNode(i);v=manager.IndexToNode(j);return math.ceil((dist[u][v]+nodes[u]['service_time'])*scale-1e-9)
 tc=r.RegisterTransitCallback(tt);H=round(par['planning_horizon']*scale);r.AddDimension(tc,H,H,False,'Time');td=r.GetDimensionOrDie('Time')
 for i,z in enumerate(nodes):
  if i not in (s,e):td.CumulVar(manager.NodeToIndex(i)).SetRange(math.ceil(z['time_window_open']*scale),math.floor(z['time_window_close']*scale))
 for k in range(K):td.CumulVar(r.Start(k)).SetRange(0,H);td.CumulVar(r.End(k)).SetRange(0,H)
 params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.seconds=max(1,int(float(a.time_limit))-3);sol=r.SolveWithParameters(params)
 if not sol:raise RuntimeError('No feasible routes')
 routes=[];obj=0.
 for k in range(K):
  i=r.Start(k);route=[s]
  while not r.IsEnd(i):j=sol.Value(r.NextVar(i));v=manager.IndexToNode(j);obj+=dist[manager.IndexToNode(i)][v];route.append(v);i=j
  if len(route)>2:routes.append(route)
 json.dump({'objective_value':obj,'routes':routes},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(obj)
if __name__=='__main__':main()
