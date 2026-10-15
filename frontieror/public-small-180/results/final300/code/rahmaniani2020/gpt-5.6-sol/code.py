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
I=range(d['num_facilities']);J=range(d['num_customers']);S=range(d['num_scenarios']);F=d['facilities']['fixed_costs'];U=d['facilities']['capacities'];C=d['routing_costs'];P=d['scenarios']['probabilities'];D=d['scenarios']['demands']
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1
y=m.addVars(I,vtype=GRB.BINARY);x=m.addVars(I,J,S,lb=0)
for s in S:
 for j in J:m.addConstr(gp.quicksum(x[i,j,s] for i in I)==D[s][j])
 for i in I:m.addConstr(gp.quicksum(x[i,j,s] for j in J)<=U[i]*y[i])
m.addConstr(gp.quicksum(U[i]*y[i] for i in I)>=max(sum(D[s]) for s in S))
m.setObjective(gp.quicksum(F[i]*y[i] for i in I)+gp.quicksum(P[s]*C[i][j]*x[i,j,s] for i in I for j in J for s in S))
def pack(v):return {'objective_value':sum(F[i]*v(y[i]) for i in I)+sum(P[s]*C[i][j]*v(x[i,j,s]) for i in I for j in J for s in S),'y':[int(v(y[i])>.5) for i in I]}
def cb(model,where):
 if where==GRB.Callback.MIPSOL and log:
  s=pack(model.cbGetSolution);log.log_solution(s['objective_value'],s)
m.Params.TimeLimit=max(.01,a.time_limit-(time.monotonic()-start)-.2);m.optimize(cb)
if not m.SolCount:raise RuntimeError('No feasible facility incumbent')
s=pack(lambda q:q.X)
if log:log.log_solution(s['objective_value'],s)
s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))
