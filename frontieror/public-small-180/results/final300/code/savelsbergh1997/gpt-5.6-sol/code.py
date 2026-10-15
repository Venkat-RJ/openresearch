import time
_wall_start=time.time()
import argparse,json,os
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic()
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
with open(a.instance_path) as f:d=json.load(f)
M=d['num_agents'];N=d['num_jobs'];P=d['profits'];W=d['weights'];C=d['capacities'];pairs=[(i,j) for j in range(N) for i in range(M) if W[i][j]<=C[i]]
if len(pairs)>2000:
 pairs=[(i,j) for j in range(N) for i in sorted(range(M),key=lambda i:P[i][j]/max(1,W[i][j]),reverse=True)[:max(1,1900//N)] if W[i][j]<=C[i]]
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1;m.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-1);m.Params.OutputFlag=0;x=m.addVars(pairs,vtype=GRB.BINARY)
for j in range(N):m.addConstr(gp.quicksum(x[i,j] for i in range(M) if (i,j) in x)==1)
for i in range(M):m.addConstr(gp.quicksum(W[i][j]*x[i,j] for j in range(N) if (i,j) in x)<=C[i])
m.setObjective(gp.quicksum(P[i][j]*x[i,j] for i,j in pairs),GRB.MAXIMIZE);logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
if logger:logger.min_interval=0;logger.start_time=_wall_start
def pack(v):
 ass={str(j):i for (i,j),q in v.items() if q>.5};return {'objective_value':sum(P[i][int(j)] for j,i in ass.items()),'assignment':ass}
def cb(m,where):
 if logger and where==GRB.Callback.MIPSOL:
  sol=pack(m.cbGetSolution(x));logger.log_solution(sol['objective_value'],sol)
m.optimize(cb)
if not m.SolCount:raise RuntimeError('No assignment incumbent')
s=pack(m.getAttr('X',x));s['runtime']=time.monotonic()-start
if logger:logger.log_solution(s['objective_value'],s)
with open(a.solution_path,'w') as f:json.dump(s,f)
