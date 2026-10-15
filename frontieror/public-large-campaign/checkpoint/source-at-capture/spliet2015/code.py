import argparse,json,os,time
os.environ['OMP_NUM_THREADS']='1'
from ortools.constraint_solver import pywrapcp,routing_enums_pb2

def solve(d,limit):
 n=d['num_customers'];D=d['distance_matrix'];scale=10000;deadline=time.monotonic()+max(.1,limit-1);depot=d['depot']['exogenous_time_window'];customers={c['id']:c for c in d['customers']}
 def routes(demand,windows,budget):
  mgr=pywrapcp.RoutingIndexManager(n+2,n,[0]*n,[n+1]*n);r=pywrapcp.RoutingModel(mgr);cb=r.RegisterTransitCallback(lambda i,j:round(scale*D[mgr.IndexToNode(i)][mgr.IndexToNode(j)]));r.SetArcCostEvaluatorOfAllVehicles(cb);dem=r.RegisterUnaryTransitCallback(lambda i:0 if mgr.IndexToNode(i) in (0,n+1) else demand[mgr.IndexToNode(i)-1]);r.AddDimensionWithVehicleCapacity(dem,0,[d['vehicle_capacity']]*n,True,'capacity');r.AddDimension(cb,round(scale*depot[1]),round(scale*depot[1]),False,'time');dim=r.GetDimensionOrDie('time')
  for i in range(1,n+1):l,u=windows[i];dim.CumulVar(mgr.NodeToIndex(i)).SetRange(round(scale*l),round(scale*u))
  for v in range(n):dim.CumulVar(r.Start(v)).SetRange(round(scale*depot[0]),round(scale*depot[1]));dim.CumulVar(r.End(v)).SetRange(round(scale*depot[0]),round(scale*depot[1]));r.AddVariableMinimizedByFinalizer(dim.CumulVar(r.Start(v)));r.AddVariableMinimizedByFinalizer(dim.CumulVar(r.End(v)))
  par=pywrapcp.DefaultRoutingSearchParameters();par.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;par.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;par.time_limit.FromMilliseconds(max(10,int(budget*1000)));s=r.SolveWithParameters(par)
  if not s:return None
  out=[]
  for v in range(n):
   cur=r.Start(v);path=[];times={}
   while not r.IsEnd(cur):node=mgr.IndexToNode(cur);path.append(node);times[str(node)]=s.Value(dim.CumulVar(cur))/scale;cur=s.Value(r.NextVar(cur))
   node=mgr.IndexToNode(cur);path.append(node);times[str(node)]=s.Value(dim.CumulVar(cur))/scale
   if len(path)>2:out.append({'route':path,'arrival_times':times,'cost':sum(D[a][b] for a,b in zip(path,path[1:]))})
  return out
 wide={i:c['exogenous_time_window'] for i,c in customers.items()};worst=[max(s['demands'][i] for s in d['scenarios']) for i in range(n)];base=routes(worst,wide,max(.05,(deadline-time.monotonic())*.3))
 if base is None:raise RuntimeError('No feasible common worst-demand routing')
 arrival={int(i):t for rr in base for i,t in rr['arrival_times'].items()};windows={}
 for i,c in customers.items():
  l,u=c['exogenous_time_window'];width=c['endogenous_time_window_width'];a=max(l,min(arrival[i],u-width));windows[i]=[a,a+width]
 out={};obj=0
 for z,sc in enumerate(d['scenarios']):
  rs=routes(sc['demands'],windows,max(.01,(deadline-time.monotonic())/(len(d['scenarios'])-z))) or base;out[str(sc['scenario_id'])]=rs;obj+=sc['probability']*sum(r['cost'] for r in rs)
 return {'objective_value':obj,'time_windows':{str(i):w for i,w in windows.items()},'routes':out}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)
