import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));n=d['num_customers'];T=d['truck_travel_times'];D=d['drone_travel_times'];manager=pywrapcp.RoutingIndexManager(n+1,1,0);r=pywrapcp.RoutingModel(manager);cb=r.RegisterTransitCallback(lambda i,j:T[manager.IndexToNode(i)][manager.IndexToNode(j)]);r.SetArcCostEvaluatorOfAllVehicles(cb);params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.seconds=max(1,min(5,int(float(a.time_limit))-3));sol=r.SolveWithParameters(params)
 if not sol:raise RuntimeError('No truck tour')
 order=[0];i=r.Start(0)
 while not r.IsEnd(i):i=sol.Value(r.NextVar(i));order.append(manager.IndexToNode(i))
 L=len(order);dp=[float('inf')]*L;dp[0]=0;pred=[None]*L
 for j in range(1,L):
  for i in range(max(0,j-14),j):
   for k in [None]+list(range(i+1,j)):
    if i==0 and j==L-1:continue
    truck=[order[t] for t in range(i,j+1) if t!=k];drone=[order[i],order[j]] if k is None else [order[i],order[k],order[j]];tt=sum(T[u][v] for u,v in zip(truck,truck[1:]));dt=sum(D[u][v] for u,v in zip(drone,drone[1:]));v=dp[i]+max(tt,dt)
    if v<dp[j]:dp[j]=v;pred[j]=(i,k,truck,drone)
 segments=[];j=L-1
 while j:entry=pred[j];segments.append((j,entry));j=entry[0]
 segments.reverse();truck=[0];drone=[0];types={};arr={'0':0};clock=0
 for end,(i,k,tr,dr) in segments:
  tt=clock
  for u,v in zip(tr,tr[1:]):tt+=T[u][v];arr[str(v if v else n+1)]=tt
  dt=clock
  for u,v in zip(dr,dr[1:]):dt+=D[u][v];arr[str(v if v else n+1)]=dt
  clock=max(tt,dt);v=order[end];arr[str(v if v else n+1)]=clock
  for v in tr[1:-1]:types[str(v)]='truck'
  if k is not None:types[str(order[k])]='drone'
  if order[end]:types[str(order[end])]='combined'
  truck.extend(tr[1:]);drone.extend(dr[1:])
 json.dump({'objective_value':clock,'truck_route':truck,'drone_route':drone,'customer_types':types,'arrival_times':arr},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(clock)
if __name__=='__main__':main()
