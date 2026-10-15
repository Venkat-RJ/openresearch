import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,math
from ortools.sat.python import cp_model
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));C=d['customers'];days=d['num_days'];K=d['num_vehicles'];cap=d['vehicle_capacity'];depot=d['depot'];deadline=time.monotonic()+max(1,float(a.time_limit)-3);model=cp_model.CpModel();choices={};cost=[]
 for c in C:
  bs=[]
  for j,sched in enumerate(c['schedules']):
   b=model.NewBoolVar(str(c['id'])+'_'+str(j));choices[c['id'],j]=b;bs.append(b);cost.append(round(200*math.hypot(c['x']-depot['x'],c['y']-depot['y']))*sched['visit_frequency']*b)
   if any(z['demand_per_visit']>cap for z in sched['parts']):model.Add(b==0)
  model.Add(sum(bs)==1)
 for day in range(days):model.Add(sum(z['demand_per_visit']*choices[c['id'],j] for c in C for j,sched in enumerate(c['schedules']) for z in sched['parts'] if z['visit_day']==day)<=K*cap)
 model.Minimize(sum(cost));best=None;bestval=float('inf');attempt=0
 while time.monotonic()<deadline and attempt<15:
  s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=min(3,max(.1,deadline-time.monotonic()));status=s.Solve(model)
  if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):break
  chosen={c['id']:next(j for j in range(len(c['schedules'])) if s.Value(choices[c['id'],j])) for c in C};schedules={str(c['id']):dict(c['schedules'][chosen[c['id']]],schedule_index=chosen[c['id']]) for c in C};allroutes={};val=0.;valid=True
  for day in range(days):
   visits=[];parts={}
   for c in C:
    for z in c['schedules'][chosen[c['id']]]['parts']:
     if z['visit_day']==day:visits.append(c);parts[c['id']]=z
   if not visits:allroutes[str(day)]=[];continue
   nodes=[depot]+visits;N=len(nodes);scale=10000;D=[[math.hypot(u['x']-v['x'],u['y']-v['y']) for v in nodes] for u in nodes];manager=pywrapcp.RoutingIndexManager(N,K,0);r=pywrapcp.RoutingModel(manager);cb=r.RegisterTransitCallback(lambda i,j:round(scale*D[manager.IndexToNode(i)][manager.IndexToNode(j)]));r.SetArcCostEvaluatorOfAllVehicles(cb);dc=r.RegisterUnaryTransitCallback(lambda i:parts[nodes[manager.IndexToNode(i)]['id']]['demand_per_visit'] if manager.IndexToNode(i) else 0);r.AddDimensionWithVehicleCapacity(dc,0,[cap]*K,True,'Capacity')
   def transit(i,j):u=manager.IndexToNode(i);v=manager.IndexToNode(j);return math.ceil(scale*(D[u][v]+nodes[u]['service_time'])-1e-9)
   tc=r.RegisterTransitCallback(transit);H=round(depot['time_window'][1]*scale);r.AddDimension(tc,H,H,False,'Time');td=r.GetDimensionOrDie('Time')
   for i,c in enumerate(nodes[1:],1):td.CumulVar(manager.NodeToIndex(i)).SetRange(math.ceil(c['time_window'][0]*scale),math.floor(c['time_window'][1]*scale))
   for k in range(K):td.CumulVar(r.Start(k)).SetRange(round(depot['time_window'][0]*scale),H);r.solver().Add(td.CumulVar(r.End(k))-td.CumulVar(r.Start(k))<=round(d['max_route_duration']*scale))
   params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.seconds=max(1,min(3,int(deadline-time.monotonic())));sol=r.SolveWithParameters(params)
   if not sol:valid=False;break
   routes=[]
   for k in range(K):
    i=r.Start(k);seq=[];rc=0.
    while not r.IsEnd(i):j=sol.Value(r.NextVar(i));u=manager.IndexToNode(i);v=manager.IndexToNode(j);rc+=D[u][v];i=j
    i=r.Start(k)
    while not r.IsEnd(i):
     i=sol.Value(r.NextVar(i));v=manager.IndexToNode(i)
     if v:seq.append(nodes[v]['id'])
    if seq:routes.append({'customers':seq,'cost':rc,'demand':sum(parts[v]['demand_per_visit'] for v in seq),'parts':{str(v):parts[v] for v in seq}});val+=rc
   allroutes[str(day)]=routes
  if valid and val<bestval:best={'objective_value':val,'routes':allroutes,'schedules':schedules};bestval=val
  model.Add(sum(choices[i,j] for i,j in chosen.items())<=len(C)-1);attempt+=1
  if best is not None:break
 if best is None:raise RuntimeError('No schedule with feasible daily routing found')
 json.dump(best,open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(bestval)
if __name__=='__main__':main()
