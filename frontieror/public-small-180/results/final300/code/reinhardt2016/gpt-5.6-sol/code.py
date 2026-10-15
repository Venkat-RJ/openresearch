import time
_wall_start=time.time()
import argparse,json,os
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
N={q['id']:q for q in d['nodes']};J=list(range(1,d['num_customers']+1));V=[0]+J;C=d['distance_matrix'];Q=d['vehicle_capacity'];H=d['schedule_horizon'];A=[(i,j) for i in V for j in V if i!=j and (j==0 or N[i]['time_window_start']+N[i]['service_time']+C[i][j]<=N[j]['time_window_end'])];sets={q['id']:q for q in d['edge_sets']}
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
x=m.addVars(A,vtype=GRB.BINARY);y=m.addVars(list(sets),vtype=GRB.BINARY);t=m.addVars(V,lb={i:N[i]['time_window_start'] for i in V},ub={i:N[i]['time_window_end'] for i in V});load=m.addVars(J,lb={i:N[i]['demand'] for i in J},ub=Q)
t[0].UB=N[0]['time_window_start']
for i in J:
 m.addConstr(gp.quicksum(x[ii,j] for ii,j in A if ii==i)==1);m.addConstr(gp.quicksum(x[j,ii] for j,ii in A if ii==i)==1)
m.addConstr(gp.quicksum(x[0,j] for j in J if (0,j) in x)<=d['num_vehicles'])
for i,j in A:
 dur=N[i]['service_time']+C[i][j]
 if j==0:m.addConstr(t[i]+dur<=H+max(0,N[i]['time_window_end']+dur-H)*(1-x[i,j]))
 else:
  M=max(0,N[i]['time_window_end']+dur-N[j]['time_window_start']);m.addConstr(t[j]>=t[i]+dur-M*(1-x[i,j]))
  if i:m.addConstr(load[j]>=load[i]+N[j]['demand']-(Q+N[j]['demand'])*(1-x[i,j]))
for r,es in sets.items():
 for i,j in es['edges']:
  if (i,j) in x:m.addConstr(x[i,j]<=y[r])
  if (j,i) in x:m.addConstr(x[j,i]<=y[r])
m.setObjective(gp.quicksum(C[i][j]*x[i,j] for i,j in A)+gp.quicksum(es['access_cost']*y[r] for r,es in sets.items()))
def pack(v):
 succ={i:j for i,j in A if i and v(x[i,j])>.5};routes=[]
 for i in J:
  if (0,i) in x and v(x[0,i])>.5:
   seq=[0,i]
   while seq[-1]:seq.append(succ[seq[-1]])
   routes.append({'vehicle':len(routes),'sequence':seq,'arcs':[list(ij) for ij in zip(seq,seq[1:])],'visit_times':{str(i):v(t[i]) for i in seq}})
 used=[r for r in sets if v(y[r])>.5];obj=sum(sum(C[i][j] for i,j in zip(r['sequence'],r['sequence'][1:])) for r in routes)+sum(sets[r]['access_cost'] for r in used)
 return {'objective_value':obj,'routes':routes,'edge_sets_used':used}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible paid-edge routes')
s=pack(lambda q:q.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))
