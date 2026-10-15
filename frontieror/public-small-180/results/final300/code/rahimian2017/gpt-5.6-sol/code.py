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
I=range(d['num_nurses']);D=range(d['num_days']);T=range(d['num_shift_types']);W=range(1,d['num_weekends']+1);length={q['id']:q['length_minutes'] for q in d['shift_types']}
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
x=m.addVars(I,D,T,vtype=GRB.BINARY);week=m.addVars(I,W,vtype=GRB.BINARY);under=m.addVars(D,T,lb=0);over=m.addVars(D,T,lb=0)
def work(i,j):return gp.quicksum(x[i,j,t] for t in T)
for nurse in d['nurses']:
 i=nurse['id']
 for j in D:m.addConstr(work(i,j)<=1)
 for rot in d['forbidden_shift_rotations']:
  for j in range(len(D)-1):m.addConstr(x[i,j,rot['shift_type']]+x[i,j+1,rot['cannot_be_followed_by']]<=1)
 for ts,bound in nurse['max_shifts_per_type'].items():m.addConstr(gp.quicksum(x[i,j,int(ts)] for j in D)<=bound)
 minutes=gp.quicksum(length[t]*x[i,j,t] for j in D for t in T);m.addConstr(minutes>=nurse['min_total_minutes']);m.addConstr(minutes<=nurse['max_total_minutes'])
 c=nurse['max_consecutive_shifts']
 for j in range(len(D)-c):m.addConstr(gp.quicksum(work(i,k) for k in range(j,j+c+1))<=c)
 for c in range(1,nurse['min_consecutive_shifts']):
  for j in range(len(D)-c-1):m.addConstr(work(i,j)+c-1-gp.quicksum(work(i,k) for k in range(j+1,j+c+1))+work(i,j+c+1)>=0)
 for c in range(1,nurse['min_consecutive_days_off']):
  for j in range(len(D)-c-1):m.addConstr(1-work(i,j)+gp.quicksum(work(i,k) for k in range(j+1,j+c+1))+work(i,j+c+1)>=0)
 for w in W:
  if 7*w-1<len(D):
   s=work(i,7*w-2)+work(i,7*w-1);m.addConstr(week[i,w]<=s);m.addConstr(s<=2*week[i,w])
 m.addConstr(gp.quicksum(week[i,w] for w in W)<=nurse['max_weekends'])
 for j in nurse['day_off_requests']:
  for t in T:x[i,j,t].UB=0
cov={(q['day'],q['shift_type']):q for q in d['coverage_requirements']}
for j in D:
 for t in T:m.addConstr(gp.quicksum(x[i,j,t] for i in I)+under[j,t]-over[j,t]==cov.get((j,t),{}).get('preferred',0))
on={(q['nurse_id'],q['day'],q['shift_type']):q['penalty'] for q in d['shift_on_requests']};off={(q['nurse_id'],q['day'],q['shift_type']):q['penalty'] for q in d['shift_off_requests']}
obj=gp.quicksum(q*(1-x[key]) for key,q in on.items())+gp.quicksum(q*x[key] for key,q in off.items())+gp.quicksum(q['under_weight']*under[key]+q['over_weight']*over[key] for key,q in cov.items());m.setObjective(obj)
def pack(v):
 sched={str(i):{str(j):next((t for t in T if v(x[i,j,t])>.5),None) for j in D} for i in I};u={};o={};val=sum(q*(1-int(sched[str(i)][str(j)]==t)) for (i,j,t),q in on.items())+sum(q*int(sched[str(i)][str(j)]==t) for (i,j,t),q in off.items())
 for (j,t),q in cov.items():
  delta=sum(sched[str(i)][str(j)]==t for i in I)-q['preferred'];u[f'{j}_{t}']=max(0,-delta);o[f'{j}_{t}']=max(0,delta);val+=q['under_weight']*max(0,-delta)+q['over_weight']*max(0,delta)
 return {'objective_value':val,'solution':{'schedule':sched,'coverage_under':u,'coverage_over':o}}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible nurse roster')
s=pack(lambda q:q.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))
