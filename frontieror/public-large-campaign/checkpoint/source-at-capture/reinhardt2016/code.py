import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,math
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));nodes=d['nodes'];N=len(nodes);K=d['num_vehicles'];D=d['distance_matrix'];scale=10000;edge={};access={z['id']:z['access_cost'] for z in d['edge_sets']};manager=pywrapcp.RoutingIndexManager(N,K,0);r=pywrapcp.RoutingModel(manager)
 for z in d['edge_sets']:
  for u,v in z['edges']:edge[u,v]=z['id'];edge[v,u]=z['id']
 free=set()
 for u,v in d['free_edges']:free.add((u,v));free.add((v,u))
 def cost(i,j):
  u=manager.IndexToNode(i);v=manager.IndexToNode(j);return round(scale*(D[u][v]+(access[edge[u,v]]/max(1,K) if (u,v) in edge else 0)))
 cb=r.RegisterTransitCallback(cost);r.SetArcCostEvaluatorOfAllVehicles(cb);dc=r.RegisterUnaryTransitCallback(lambda i:nodes[manager.IndexToNode(i)]['demand']);r.AddDimensionWithVehicleCapacity(dc,0,[d['vehicle_capacity']]*K,True,'Capacity')
 def transit(i,j):
  u=manager.IndexToNode(i);v=manager.IndexToNode(j);return math.ceil(scale*(D[u][v]+nodes[u]['service_time'])-1e-9)
 tc=r.RegisterTransitCallback(transit);H=d['schedule_horizon']*scale;r.AddDimension(tc,H,H,False,'Time');td=r.GetDimensionOrDie('Time')
 for i in range(1,N):td.CumulVar(manager.NodeToIndex(i)).SetRange(math.ceil(nodes[i]['time_window_start']*scale),math.floor(nodes[i]['time_window_end']*scale))
 for u in range(1,N):
  for v in range(1,N):
   if u!=v and (u,v) not in edge and (u,v) not in free:r.NextVar(manager.NodeToIndex(u)).RemoveValue(manager.NodeToIndex(v))
 params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.seconds=max(1,int(float(a.time_limit))-3);sol=r.SolveWithParameters(params)
 if not sol:raise RuntimeError('No edge-set VRPTW route')
 routes=[];used=set();obj=0.
 for k in range(K):
  i=r.Start(k);seq=[0];times={'0':sol.Value(td.CumulVar(i))/scale};arcs=[]
  while not r.IsEnd(i):
   j=sol.Value(r.NextVar(i));u=manager.IndexToNode(i);v=manager.IndexToNode(j);seq.append(v);arcs.append([u,v]);obj+=D[u][v]
   if (u,v) in edge:used.add(edge[u,v])
   if v:times[str(v)]=sol.Value(td.CumulVar(j))/scale
   i=j
  if len(seq)>2:routes.append({'vehicle':k,'sequence':seq,'arcs':arcs,'visit_times':times})
 obj+=sum(access[z] for z in used);json.dump({'objective_value':obj,'routes':routes,'edge_sets_used':sorted(used)},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(obj)
if __name__=='__main__':main()
