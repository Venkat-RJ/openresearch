import argparse,json,time
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from ortools.constraint_solver import pywrapcp,routing_enums_pb2

def solve(d,t):
 n=d['num_customers'];K=d['num_vehicles'];end=n+1;dist=d['distance_matrix'];nodes=[d['depot']]+d['customers']+[d['depot_end']];manager=pywrapcp.RoutingIndexManager(n+2,K,[0]*K,[end]*K);routing=pywrapcp.RoutingModel(manager);service=[0]+[v['service_time'] for v in d['customers']]+[0]
 def travel(i,j):return int(round(10*dist[manager.IndexToNode(i)][manager.IndexToNode(j)]))
 def transit(i,j):u=manager.IndexToNode(i);return travel(i,j)+int(round(10*service[u]))
 dc=routing.RegisterTransitCallback(travel);tc=routing.RegisterTransitCallback(transit);routing.SetArcCostEvaluatorOfAllVehicles(dc);routing.AddDimension(tc,10*d['planning_horizon'],10*d['planning_horizon'],False,'Time');dim=routing.GetDimensionOrDie('Time')
 demand=routing.RegisterUnaryTransitCallback(lambda i:nodes[manager.IndexToNode(i)]['demand']);routing.AddDimensionWithVehicleCapacity(demand,0,[d['vehicle_capacity']]*K,True,'Capacity');rho=d['inconvenience_cost_function']['rho']
 for v in d['customers']:
  ix=manager.NodeToIndex(v['id']);lo,hi=v['time_window'];dim.CumulVar(ix).SetRange(int(lo*10),int(hi*10));pref=int(round(10*v['preferred_time']));dim.SetCumulVarSoftLowerBound(ix,pref,max(1,int(rho*10)));dim.SetCumulVarSoftUpperBound(ix,pref,max(1,int(rho*10)))
 for k in range(K):
  dim.CumulVar(routing.Start(k)).SetRange(int(nodes[0]['time_window'][0]*10),int(nodes[0]['time_window'][1]*10));dim.CumulVar(routing.End(k)).SetRange(int(nodes[-1]['time_window'][0]*10),int(nodes[-1]['time_window'][1]*10))
 p=pywrapcp.DefaultRoutingSearchParameters();p.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;p.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;p.time_limit.FromMilliseconds(max(100,int((t-3)*1000)));sol=routing.SolveWithParameters(p)
 if sol is None:raise RuntimeError('No feasible capacitated time-window routes')
 routes=[];schedule={};total=0
 for k in range(K):
  ix=routing.Start(k);route=[];times=[]
  while not routing.IsEnd(ix):route.append(manager.IndexToNode(ix));times.append(sol.Value(dim.CumulVar(ix))/10);ix=sol.Value(routing.NextVar(ix))
  route.append(end);times.append(sol.Value(dim.CumulVar(ix))/10)
  if len(route)==2:continue
  N=len(route);pref=np.array([nodes[v].get('preferred_time',times[h]) for h,v in enumerate(route)]);weights=np.array([0]+[rho]*(N-2)+[0]);A=np.zeros((N-1,N));gaps=[]
  for h in range(N-1):A[h,h]=-1;A[h,h+1]=1;gaps.append(service[route[h]]+dist[route[h]][route[h+1]])
  z=minimize(lambda x:float(np.sum(weights*(x-pref)**2)),times,method='SLSQP',bounds=[tuple(nodes[v]['time_window']) for v in route],constraints=[LinearConstraint(A,gaps,np.inf)],options={'maxiter':100,'ftol':1e-8});tt=z.x if z.success else np.array(times)
  routes.append(route);total+=sum(dist[route[h]][route[h+1]] for h in range(N-1))+float(np.sum(weights*(tt-pref)**2))
  for h,v in enumerate(route[1:-1],1):schedule[str(v)]=float(tt[h])
 return {'objective_value':total,'routes':routes,'schedule':schedule}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')
