import time
_wall_start=time.time()
import argparse,json,os
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic()
d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
n=d['parameters']['n_trips'];K=range(1,d['parameters']['m_depots']+1);J=range(1,n+1)
C={(k,z['from_trip'],z['to_trip']):z['cost'] for k in K for z in d['compatible_arcs']}
C.update({(z['depot'],n+z['depot'],z['trip']):z['cost'] for z in d['depot_to_trip_costs']})
C.update({(z['depot'],z['trip'],n+z['depot']):z['cost'] for z in d['trip_to_depot_costs']})
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
x=m.addVars(list(C),vtype=GRB.BINARY,obj=C)
for j in J:m.addConstr(gp.quicksum(x[t] for t in C if t[2]==j)==1)
for k in K:
 for j in list(J)+[n+k]:m.addConstr(gp.quicksum(x[t] for t in C if t[0]==k and t[1]==j)==gp.quicksum(x[t] for t in C if t[0]==k and t[2]==j))
 m.addConstr(gp.quicksum(x[k,n+k,j] for j in J)<=d['vehicle_capacities'][k-1])
def pack(v):
 chosen={t for t in C if v(x[t])>.5};routes=[]
 for k in K:
  succ={i:j for kk,i,j in chosen if kk==k and i!=n+k}
  for kk,i,j in sorted(chosen):
   if kk==k and i==n+k:
    route=[]
    while j!=n+k:
     route.append(j);j=succ[j]
    routes.append({'depot':k,'trips':route})
 return {'objective_value':sum(C[t] for t in chosen),'routes':routes,'num_vehicles':len(routes)}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible trip schedule')
s=pack(lambda z:z.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))
