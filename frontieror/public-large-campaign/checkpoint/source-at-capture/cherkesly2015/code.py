import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,math,time
from ortools.constraint_solver import pywrapcp,routing_enums_pb2

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();started=time.monotonic();d=json.load(open(a.instance_path));lifo='n_requests'in d;nodes=sorted(d['nodes'],key=lambda x:x['node_id']);N=len(nodes);R=d['n_requests']if lifo else d['num_users'];K=R if lifo else d['num_vehicles'];Q=d['vehicle_capacity'];S=1000
 early=[v['tw_early'if lifo else 'earliest_time']for v in nodes];late=[v['tw_late'if lifo else 'latest_time']for v in nodes];service=[v['service_time'if lifo else 'service_duration']for v in nodes];load=[v['load']for v in nodes]
 if lifo:
  arcs={(v['from'],v['to']):(v['cost'],v['travel_time'])for v in d['arcs']};cost=[[arcs.get((i,j),(1e7,1e7))[0]for j in range(N)]for i in range(N)];travel=[[arcs.get((i,j),(1e7,1e7))[1]for j in range(N)]for i in range(N)]
  cost[0][N-1]=0;travel[0][N-1]=0
 else:cost=[[math.hypot(v['x']-u['x'],v['y']-u['y'])for u in nodes]for v in nodes];travel=cost
 manager=pywrapcp.RoutingIndexManager(N,K,[0]*K,[N-1]*K);routing=pywrapcp.RoutingModel(manager)
 c=routing.RegisterTransitCallback(lambda i,j:int(round(S*cost[manager.IndexToNode(i)][manager.IndexToNode(j)])));routing.SetArcCostEvaluatorOfAllVehicles(c)
 t=routing.RegisterTransitCallback(lambda i,j:int(math.ceil(S*(service[manager.IndexToNode(i)]+travel[manager.IndexToNode(i)][manager.IndexToNode(j)]))));H=int(math.ceil(max(late)*S));routing.AddDimension(t,H,H,False,'Time');times=routing.GetDimensionOrDie('Time')
 lc=routing.RegisterUnaryTransitCallback(lambda i:load[manager.IndexToNode(i)]);routing.AddDimensionWithVehicleCapacity(lc,0,[Q]*K,True,'Load')
 for i in range(1,N-1):times.CumulVar(manager.NodeToIndex(i)).SetRange(int(math.ceil(S*early[i])),int(math.floor(S*late[i])))
 for k in range(K):
  times.CumulVar(routing.Start(k)).SetRange(int(math.ceil(S*early[0])),int(math.floor(S*late[0])));times.CumulVar(routing.End(k)).SetRange(int(math.ceil(S*early[-1])),int(math.floor(S*late[-1])))
  if not lifo:routing.solver().Add(times.CumulVar(routing.End(k))-times.CumulVar(routing.Start(k))<=S*d['maximum_route_duration'])
  routing.AddVariableMinimizedByFinalizer(times.CumulVar(routing.Start(k)));routing.AddVariableMinimizedByFinalizer(times.CumulVar(routing.End(k)))
 for i in range(1,R+1):
  pick=manager.NodeToIndex(i);drop=manager.NodeToIndex(i+R);routing.AddPickupAndDelivery(pick,drop);routing.solver().Add(routing.VehicleVar(pick)==routing.VehicleVar(drop));routing.solver().Add(times.CumulVar(pick)<=times.CumulVar(drop))
  if not lifo:routing.solver().Add(times.CumulVar(drop)-times.CumulVar(pick)<=int(S*(d['maximum_ride_time']+service[i])))
 if lifo:routing.SetPickupAndDeliveryPolicyOfAllVehicles(pywrapcp.RoutingModel.PICKUP_AND_DELIVERY_LIFO)
 params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.FromMilliseconds(max(1,int((float(a.time_limit)-2-(time.monotonic()-started))*1000)));sol=routing.SolveWithParameters(params)
 if sol is None:raise RuntimeError('No feasible pickup-delivery schedule within time limit')
 routes=[];st={};ride={};objective=0
 for k in range(K):
  index=routing.Start(k);row=[];details=[];onboard=0
  while True:
   i=manager.IndexToNode(index);row.append(i);at=sol.Value(times.CumulVar(index))/S;onboard+=load[i];details.append({'node_id':i,'arrival_time':at,'load_after':onboard})
   if i==0:st['depot_start_'+str(k)]=at
   elif i==N-1:st['depot_end_'+str(k)]=at
   else:st[str(i)]=at
   if routing.IsEnd(index):break
   nxt=sol.Value(routing.NextVar(index));j=manager.IndexToNode(nxt);objective+=cost[i][j]if not(i==0 and j==N-1)else 0;index=nxt
  routes.append({'vehicle_id':k,'route_sequence':row,'route_details':details})
 if lifo:out={'objective_value':objective,'routes':[r for r in routes if len(r['route_sequence'])>2]}
 else:
  for i in range(1,R+1):ride[str(nodes[i]['user_id'])]=st[str(i+R)]-st[str(i)]-service[i]
  out={'objective_value':objective,'routes':{str(k):r['route_sequence']for k,r in enumerate(routes)},'service_times':st,'ride_times':ride}
 json.dump(out,open(a.solution_path,'w'))
if __name__=='__main__':main()
