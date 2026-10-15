import argparse,json,time,os,random
os.environ['OMP_NUM_THREADS']='1'
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
from ortools.sat.python import cp_model

def solve(d,limit):
 if d.get('problem_type','TSP').upper()=='MAXCUT':
  data=d.get('data',d);edges=data['edges'];n=d.get('parameters',{}).get('n_nodes',data.get('num_nodes',1+max(max(e[:2]) for e in edges)));m=cp_model.CpModel();x=[m.new_bool_var(f'x{i}') for i in range(n)];terms=[]
  for e in edges:
   u,v=e[:2];w=e[2] if len(e)>2 else 1;y=m.new_bool_var(f'y{u}_{v}');m.add(x[u]!=x[v]).only_enforce_if(y);m.add(x[u]==x[v]).only_enforce_if(y.Not());terms.append(round(w*1000000)*y)
  m.maximize(sum(terms));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,limit-1);st=s.solve(m);bits=[s.value(a) for a in x];obj=sum((e[2] if len(e)>2 else 1) for e in edges if bits[e[0]]!=bits[e[1]])
  return {'objective_value':obj,'solution':{f'x_{i}':bits[i] for i in range(n)}}
 D=d['data']['distance_matrix'];n=len(D);manager=pywrapcp.RoutingIndexManager(n,1,0);routing=pywrapcp.RoutingModel(manager);cb=routing.RegisterTransitCallback(lambda i,j:round(D[manager.IndexToNode(i)][manager.IndexToNode(j)]*1000000));routing.SetArcCostEvaluatorOfAllVehicles(cb);params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.FromMilliseconds(max(10,int((limit-1)*1000)));params.sat_parameters.num_search_workers=1;s=routing.SolveWithParameters(params)
 if s:
  tour=[];i=routing.Start(0)
  while not routing.IsEnd(i):tour.append(manager.IndexToNode(i));i=s.Value(routing.NextVar(i))
 else:tour=list(range(n))
 obj=sum(D[tour[t]][tour[(t+1)%n]] for t in range(n));return {'objective_value':obj,'tour':tour,'solution':{f'x_{i}_{t}':int(tour[t]==i) for i in range(n) for t in range(n)}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)
