import time
_wall_start=time.time()
import argparse,json,os,math
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
n=d['num_requests'];end=2*n+1;J=list(range(1,end));N={q['id']:q for q in d['nodes']};Q=d['vehicle_capacity'];V=list(range(end+1));C={(i,j):math.hypot(N[i]['x']-N[j]['x'],N[i]['y']-N[j]['y']) for i in V for j in V if i!=j};A=[(i,j) for i in [0]+J for j in J+[end] if i!=j and not (i>n and i==j+n) and N[i]['time_window'][0]+N[i].get('service_time',0)+C[i,j]<=N[j]['time_window'][1] and (i,j)!=(0,end)]
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
x=m.addVars(A,vtype=GRB.BINARY);t=m.addVars(V,lb={i:N[i]['time_window'][0] for i in V},ub={i:N[i]['time_window'][1] for i in V});load=m.addVars(J,lb=0,ub=Q);label=m.addVars(J,lb=1,ub=end-1)
t[0].UB=N[0]['time_window'][0]
for i in J:
 m.addConstr(gp.quicksum(x[ii,j] for ii,j in A if ii==i)==1);m.addConstr(gp.quicksum(x[j,ii] for j,ii in A if ii==i)==1)
m.addConstr(gp.quicksum(x[0,j] for j in J if (0,j) in x)<=d['num_vehicles'])
for i,j in A:
 dur=N[i].get('service_time',0)+C[i,j];M=max(0,N[i]['time_window'][1]+dur-N[j]['time_window'][0]);m.addConstr(t[j]>=t[i]+dur-M*(1-x[i,j]))
 if j!=end:
  before=load[i] if i else 0;delta=N[j]['demand'];m.addConstr(load[j]>=before+delta-(2*Q+abs(delta))*(1-x[i,j]));m.addConstr(load[j]<=before+delta+(2*Q+abs(delta))*(1-x[i,j]))
  if i:
   m.addConstr(label[j]-label[i]<=end*(1-x[i,j]));m.addConstr(label[i]-label[j]<=end*(1-x[i,j]))
  else:
   m.addConstr(label[j]-j<=end*(1-x[i,j]));m.addConstr(j-label[j]<=end*(1-x[i,j]))
for i in range(1,n+1):m.addConstr(label[i]==label[i+n]);m.addConstr(t[i]+N[i].get('service_time',0)+C[i,i+n]<=t[i+n])
m.setObjective(gp.quicksum(C[i,j]*x[i,j] for i,j in A))
def pack(v):
 succ={i:j for i,j in A if i and v(x[i,j])>.5};routes=[]
 for i in J:
  if (0,i) in x and v(x[0,i])>.5:
   seq=[0,i]
   while seq[-1]!=end:seq.append(succ[seq[-1]])
   routes.append({'vehicle':len(routes),'nodes':seq})
 return {'objective_value':sum(sum(C[i,j] for i,j in zip(r['nodes'],r['nodes'][1:])) for r in routes),'routes':routes}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible pickup-delivery routes')
s=pack(lambda q:q.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))
