import time
_wall_start=time.time()
import argparse,json,os,math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic()
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
with open(a.instance_path) as f:d=json.load(f)
P={int(k):v for k,v in d['task_processing_times'].items()};E={int(k):v for k,v in d['earliest_station'].items()};L={int(k):v for k,v in d['latest_station'].items()};S=d['num_stations'];C=d['cycle_time']
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1;m.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-1);m.Params.OutputFlag=0
x=m.addVars([(i,s) for i in P for s in range(E[i],L[i]+1)],vtype=GRB.BINARY)
z=m.addVar(lb=max(0,math.ceil(C-sum(P.values())/S)))
for i in P:m.addConstr(gp.quicksum(x[i,s] for s in range(E[i],L[i]+1))==1)
for i,j in d['precedence_arcs']:m.addConstr(gp.quicksum(s*x[i,s] for s in range(E[i],L[i]+1))<=gp.quicksum(s*x[j,s] for s in range(E[j],L[j]+1)))
for s in range(1,S+1):
 w=gp.quicksum(P[i]*x[i,s] for i in P if (i,s) in x);m.addConstr(w<=C);m.addConstr(w+z>=C)
m.setObjective(z)
logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if logger:logger.min_interval=0;logger.start_time=_wall_start
def pack(v):
 ass={str(i):s for (i,s),q in v.items() if q>.5};loads=[sum(P[int(i)] for i,ss in ass.items() if ss==s) for s in range(1,S+1)];return {'objective_value':max(C-w for w in loads),'task_assignments':ass}
def cb(m,where):
 if where==GRB.Callback.MIPSOL and logger:
  sol=pack(m.cbGetSolution(x));logger.log_solution(sol['objective_value'],sol)
m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible station assignment')
sol=pack(m.getAttr('X',x));sol['runtime']=time.monotonic()-start
if logger:logger.log_solution(sol['objective_value'],sol)
with open(a.solution_path,'w') as f:json.dump(sol,f)
