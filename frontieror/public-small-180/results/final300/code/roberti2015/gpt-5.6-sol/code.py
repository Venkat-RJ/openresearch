import time
_wall_start=time.time()
import argparse,json,os,math
from solution_logger import SolutionLogger
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']): del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'): p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path')
a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
I=range(d['num_sources']);J=range(d['num_sinks']);A=d['supply'];B=d['demand'];C=d['variable_cost'];F=d['fixed_cost'];U=d['capacity']
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
x=m.addVars(I,J,lb=0);y=m.addVars(I,J,vtype=GRB.BINARY)
for i in I:
 m.addConstr(gp.quicksum(x[i,j] for j in J)==A[i]);m.addConstr(gp.quicksum(U[i][j]*y[i,j] for j in J)>=A[i])
for j in J:
 m.addConstr(gp.quicksum(x[i,j] for i in I)==B[j]);m.addConstr(gp.quicksum(U[i][j]*y[i,j] for i in I)>=B[j])
for i in I:
 for j in J: m.addConstr(x[i,j]<=U[i][j]*y[i,j])
m.setObjective(gp.quicksum(C[i][j]*x[i,j]+F[i][j]*y[i,j] for i in I for j in J))
def pack(v):
 flow=[[max(0,v(x[i,j])) for j in J] for i in I];used=[[int(v(y[i,j])>.5) for j in J] for i in I]
 return {'objective_value':sum(C[i][j]*flow[i][j]+F[i][j]*used[i][j] for i in I for j in J),'flow':flow,'arcs_used':used}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount: raise RuntimeError('No feasible transport incumbent')
s=pack(lambda q:q.X)
if log: log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))
