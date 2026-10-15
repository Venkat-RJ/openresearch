import argparse,json,os,time
wall_started=time.time()
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']): del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); start=time.monotonic(); d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='minimize'); n=d['num_activities_total']; H=d['max_project_duration']; acts={v['id']:v for v in d['activities']}; dur=[acts[i]['processing_time'] for i in range(n)]; edges=[(c['from_activity'],c['to_activity'],c['time_lag']) for c in d['temporal_constraints']]; resources=[(r['Pi_k'],r['capacity'],[r['demands'].get(str(i),0) for i in range(n)]) for r in d['resources']]
lo=[0]*n; hi=[H-v for v in dur]; hi[0]=0
for _ in range(n):
 for i,j,lag in edges: lo[j]=max(lo[j],lo[i]+lag); hi[i]=min(hi[i],hi[j]-lag)
def emit(s):
 z=s[-1]; sol={'objective_value':z,'schedule':{str(i):v for i,v in enumerate(s)}}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)
def feasible(s): return s[0]==0 and all(s[j]-s[i]>=lag for i,j,lag in edges) and all(sum(ds[i]*sum(s[i]<t<=s[i]+dur[i] for t in periods) for i in range(n))<=cap for periods,cap,ds in resources)
if feasible(lo): emit(lo)
m=gp.Model(); m.Params.Threads=1; m.Params.Seed=0; m.Params.MIPGap=1e-4; m.Params.NumericFocus=0
x={(i,t):m.addVar(vtype=GRB.BINARY) for i in range(n) for t in range(lo[i],hi[i]+1)}; S={i:gp.quicksum(t*x[i,t] for t in range(lo[i],hi[i]+1)) for i in range(n)}
for i in range(n): m.addConstr(gp.quicksum(x[i,t] for t in range(lo[i],hi[i]+1))==1)
for i,j,lag in edges: m.addConstr(S[j]-S[i]>=lag)
for periods,cap,ds in resources: m.addConstr(gp.quicksum(ds[i]*sum(t<u<=t+dur[i] for u in periods)*v for (i,t),v in x.items())<=cap)
m.setObjective(S[n-1]); m.update(); assert m.NumVars<=2000 and m.NumConstrs<=2000
if feasible(lo):
 for (i,t),v in x.items(): v.Start=int(t==lo[i])
keys=list(x); vals=list(x.values())
def output(vs):
 s=[0]*n
 for (i,t),v in zip(keys,vs):
  if v>.5: s[i]=t
 emit(s)
def cb(model,where):
 if where==GRB.Callback.MIPSOL: output(model.cbGetSolution(vals))
m.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-1); m.optimize(cb)
if m.SolCount: output([v.X for v in vals])
