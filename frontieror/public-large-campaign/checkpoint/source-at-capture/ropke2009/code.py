import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,math
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));nodes=d['nodes'];N=len(nodes);K=d['num_vehicles'];s=next(i for i,z in enumerate(nodes) if z['type']=='depot_origin');e=next(i for i,z in enumerate(nodes) if z['type']=='depot_destination');scale=10000;D=[[math.hypot(u['x']-v['x'],u['y']-v['y']) for v in nodes] for u in nodes];manager=pywrapcp.RoutingIndexManager(N,K,[s]*K,[e]*K);r=pywrapcp.RoutingModel(manager)
 def travel(i,j):return round(D[manager.IndexToNode(i)][manager.IndexToNode(j)]*scale)
 cb=r.RegisterTransitCallback(travel);r.SetArcCostEvaluatorOfAllVehicles(cb)
 if d.get('objective',{}).get('primary','').lower().find('vehicle')>=0:r.SetFixedCostOfAllVehicles(round(d['fixed_cost_per_vehicle']*scale))
 dc=r.RegisterUnaryTransitCallback(lambda i:nodes[manager.IndexToNode(i)]['demand']);r.AddDimensionWithVehicleCapacity(dc,0,[d['vehicle_capacity']]*K,True,'Capacity')
 def transit(i,j):
  u=manager.IndexToNode(i);v=manager.IndexToNode(j);return math.ceil((D[u][v]+nodes[u]['service_time'])*scale-1e-9)
 tc=r.RegisterTransitCallback(transit);H=round(d['planning_horizon']*scale);r.AddDimension(tc,H,H,False,'Time');td=r.GetDimensionOrDie('Time')
 for i,z in enumerate(nodes):
  if i not in (s,e):td.CumulVar(manager.NodeToIndex(i)).SetRange(math.ceil(z['time_window'][0]*scale),math.floor(z['time_window'][1]*scale))
 for k in range(K):
  for index,z in [(r.Start(k),nodes[s]),(r.End(k),nodes[e])]:td.CumulVar(index).SetRange(math.ceil(z['time_window'][0]*scale),math.floor(z['time_window'][1]*scale))
 for req in d['requests']:
  u=manager.NodeToIndex(req['pickup_node']);v=manager.NodeToIndex(req['delivery_node']);r.AddPickupAndDelivery(u,v);r.solver().Add(r.VehicleVar(u)==r.VehicleVar(v));r.solver().Add(td.CumulVar(u)<=td.CumulVar(v))
 params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.seconds=max(1,int(float(a.time_limit))-3);sol=r.SolveWithParameters(params)
 if not sol:raise RuntimeError('No feasible pickup-delivery routes')
 routes=[];obj=0.
 for k in range(K):
  i=r.Start(k);seq=[s]
  while not r.IsEnd(i):j=sol.Value(r.NextVar(i));v=manager.IndexToNode(j);obj+=D[manager.IndexToNode(i)][v];seq.append(v);i=j
  if len(seq)>2:routes.append({'vehicle':k,'nodes':seq})
 json.dump({'objective_value':obj,'routes':routes},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(obj)
if __name__=='__main__':main()
