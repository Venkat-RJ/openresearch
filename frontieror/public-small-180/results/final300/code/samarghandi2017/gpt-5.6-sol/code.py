import time
_wall_start=time.time()
import argparse,json,os
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic()
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
with open(a.instance_path) as f:d=json.load(f)
n=d['n_jobs'];m=d['n_machines'];P=d['processing_times'];due=d['due_dates'];prefix=[[sum(P[i][j] for i in range(k)) for k in range(m+1)] for j in range(n)];total=[v[-1] for v in prefix];lag=[[max(prefix[j][i+1]-prefix[k][i] for i in range(m)) for k in range(n)] for j in range(n)]
z=gp.Model();z.Params.Seed=0;z.Params.MIPGap=1e-4;z.Params.NumericFocus=0;z.Params.Threads=1;z.Params.OutputFlag=0;z.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-1)
arcs=[(j,k) for j in range(n) for k in range(n) if j!=k];x=z.addVars(arcs,vtype=GRB.BINARY);S=z.addVars(n,lb=0,ub={j:due[j]-total[j] for j in range(n)});C=z.addVar(lb=0)
for j in range(n):z.addConstr(x.sum(j,'*')<=1);z.addConstr(x.sum('*',j)<=1);z.addConstr(C>=S[j]+total[j])
z.addConstr(x.sum()==n-1)
for j,k in arcs:
 M=max(0,due[j]-total[j]+lag[j][k]);z.addConstr(S[k]>=S[j]+lag[j][k]-M*(1-x[j,k]))
z.setObjective(C);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if logger:logger.min_interval=0;logger.start_time=_wall_start
def pack(v):
 succ={j:k for (j,k),vv in v.items() if vv>.5};first=next(iter(set(range(n))-set(succ.values())));seq=[first]
 while seq[-1] in succ:seq.append(succ[seq[-1]])
 t=0;obj=total[first]
 for j,k in zip(seq,seq[1:]):t+=lag[j][k];obj=max(obj,t+total[k])
 return {'objective_value':obj,'sequence':[j+1 for j in seq]}
def cb(z,w):
 if logger and w==GRB.Callback.MIPSOL:
  sol=pack(z.cbGetSolution(x));logger.log_solution(sol['objective_value'],sol)
z.optimize(cb)
if not z.SolCount:raise RuntimeError('No due-date-feasible sequence')
sol=pack(z.getAttr('X',x));sol['runtime']=time.monotonic()-start
if logger:logger.log_solution(sol['objective_value'],sol)
with open(a.solution_path,'w') as f:json.dump(sol,f)
