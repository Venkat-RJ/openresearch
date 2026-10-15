import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,math
from ortools.sat.python import cp_model
from ortools.constraint_solver import pywrapcp,routing_enums_pb2
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));sat=d['satellites'];customers={z['id']:z for z in d['customers']};groups=d['groups'];deadline=time.monotonic()+max(1,float(a.time_limit)-3);model=cp_model.CpModel();choices={};groupd={z['group_id']:sum(customers[i]['demand'] for i in z['customer_ids']) for z in groups};cost=[]
 for g in groups:
  bs=[]
  for z in sat:
   b=model.NewBoolVar(f'g{g["group_id"]}_{z["id"]}');choices[g['group_id'],z['id']]=b;bs.append(b);proxy=sum(math.hypot(customers[i]['x']-z['x'],customers[i]['y']-z['y']) for i in g['customer_ids'])+z['handling_cost']*groupd[g['group_id']];cost.append(round(100*proxy)*b)
  model.Add(sum(bs)==1)
 counts=[]
 for z in sat:
  load=sum(groupd[g['group_id']]*choices[g['group_id'],z['id']] for g in groups);model.Add(load<=z['capacity']);count=model.NewIntVar(0,z['max_vehicles'],'vehicles');model.Add(load<=d['Q2']*count);counts.append(count)
 model.Add(sum(counts)<=d['T2']);model.Minimize(sum(cost))
 def cvrp(start,visits,cap,K):
  if not visits:return []
  nodes=[start]+visits;N=len(nodes);D=[[math.hypot(u['x']-v['x'],u['y']-v['y']) for v in nodes] for u in nodes];manager=pywrapcp.RoutingIndexManager(N,K,0);r=pywrapcp.RoutingModel(manager);cb=r.RegisterTransitCallback(lambda i,j:round(10000*D[manager.IndexToNode(i)][manager.IndexToNode(j)]));r.SetArcCostEvaluatorOfAllVehicles(cb);r.SetFixedCostOfAllVehicles(round(10000*sum(max(row) for row in D)));dc=r.RegisterUnaryTransitCallback(lambda i:nodes[manager.IndexToNode(i)].get('demand',0));r.AddDimensionWithVehicleCapacity(dc,0,[cap]*K,True,'Capacity');params=pywrapcp.DefaultRoutingSearchParameters();params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION;params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH;params.time_limit.seconds=max(1,min(2,int(deadline-time.monotonic())));sol=r.SolveWithParameters(params)
  if not sol:return None
  out=[]
  for k in range(K):
   i=r.Start(k);seq=[];qty={};rc=0.
   while not r.IsEnd(i):
    j=sol.Value(r.NextVar(i));u=manager.IndexToNode(i);v=manager.IndexToNode(j);rc+=D[u][v]
    if v:seq.append(nodes[v]['id']);qty[nodes[v]['id']]=qty.get(nodes[v]['id'],0)+nodes[v]['demand']
    i=j
   if seq:out.append((seq,qty,rc))
  return out
 best=None
 for attempt in range(10):
  if time.monotonic()>deadline:break
  solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=min(3,max(.1,deadline-time.monotonic()));status=solver.Solve(model)
  if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):break
  assignments={g['group_id']:next(z['id'] for z in sat if solver.Value(choices[g['group_id'],z['id']])) for g in groups};second=[];through={};obj=0.;valid=True
  for z in sat:
   visit=[customers[i] for g in groups if assignments[g['group_id']]==z['id'] for i in g['customer_ids']];through[z['id']]=sum(v['demand'] for v in visit);routes=cvrp(z,visit,d['Q2'],z['max_vehicles'])
   if routes is None:valid=False;break
   for seq,qty,rc in routes:second.append({'satellite':z['id'],'route':[z['id']]+seq+[z['id']]});obj+=rc
   obj+=through[z['id']]*z['handling_cost']
  if len(second)>d['T2']:valid=False
  if valid:
   loads=[]
   for z in sat:
    q=through[z['id']]
    while q:amount=min(q,d['Q1']);loads.append(dict(z,demand=amount));q-=amount
   first=cvrp(d['depot'],loads,d['Q1'],d['T1'])
   if first is not None:
    firstout=[]
    coords={z['id']:(z['x'],z['y']) for z in [d['depot']]+sat}
    for k,(seq,qty,rc) in enumerate(first):
     unique=list(dict.fromkeys(seq));route=[d['depot']['id']]+unique+[d['depot']['id']];firstout.append({'vehicle':k,'route':route,'deliveries':{str(i):v for i,v in qty.items()}});obj+=sum(math.hypot(coords[u][0]-coords[v][0],coords[u][1]-coords[v][1]) for u,v in zip(route,route[1:]))
    best={'objective_value':obj,'first_echelon_routes':firstout,'second_echelon_routes':second,'group_assignments':{str(i):v for i,v in assignments.items()}};break
  model.Add(sum(choices[g,s] for g,s in assignments.items())<=len(groups)-1)
 if best is None:raise RuntimeError('No feasible two-echelon group assignment and routes')
 json.dump(best,open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(best['objective_value'])
if __name__=='__main__':main()
