import argparse,json,os,time,random
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
a=p.parse_args(); start=time.monotonic(); end=start+a.time_limit-1; d=json.load(open(a.instance_path)); n=d['n']; w=d['weights']; P=d['profit_matrix']; C=d['capacity']; logger=configured_logger(a.log_path,sense='maximize'); best=-1; bx=[0]*n; rng=random.Random(0)
def emit(x):
 global best,bx
 z=sum(P[i][j] for i in range(n) if x[i] for j in range(i+1) if x[j])
 if z<=best: return
 best=z; bx=list(x); sol={'objective_value':z,'solution':bx}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)
def fill(x):
 load=sum(w[i]*x[i] for i in range(n))
 while time.monotonic()<end:
  cand=[i for i in range(n) if not x[i] and load+w[i]<=C]
  if not cand: break
  i=max(cand,key=lambda i:(P[i][i]+sum(P[i][j] for j in range(n) if x[j]))/max(1,w[i])); x[i]=1; load+=w[i]
 return x
emit(fill([0]*n)); he=min(end,start+.15*a.time_limit)
while time.monotonic()<he:
 x=list(bx); chosen=[i for i in range(n) if x[i]]
 for i in rng.sample(chosen,min(len(chosen),max(1,len(chosen)//8))): x[i]=0
 x=fill(x); improved=True
 while improved and time.monotonic()<he:
  improved=False; load=sum(w[i]*x[i] for i in range(n)); gains=[P[i][i]+sum(P[i][j] for j in range(n) if x[j] and j!=i) for i in range(n)]; move=max(((gains[j]-P[i][j]-gains[i],i,j) for i in range(n) if x[i] for j in range(n) if not x[j] and load-w[i]+w[j]<=C),default=(0,0,0))
  if move[0]>0: x[move[1]]=0; x[move[2]]=1; x=fill(x); improved=True
 emit(x)
if time.monotonic()<end and 2*n<=2000:
 m=gp.Model(); m.Params.Threads=1; m.Params.Seed=0; m.Params.MIPGap=1e-4; m.Params.NumericFocus=0
 x=m.addVars(n,vtype=GRB.BINARY); z=m.addVars(n); m.addConstr(gp.quicksum(w[i]*x[i] for i in range(n))<=C)
 for i in range(n):
  M=sum(P[i][j] for j in range(i)); m.addConstr(z[i]<=M*x[i]); m.addConstr(z[i]<=gp.quicksum(P[i][j]*x[j] for j in range(i))); x[i].Start=bx[i]
 m.setObjective(gp.quicksum(P[i][i]*x[i]+z[i] for i in range(n)),GRB.MAXIMIZE); m.update(); assert m.NumVars<=2000 and m.NumConstrs<=2000
 def cb(model,where):
  if where==GRB.Callback.MIPSOL: emit([int(v>.5) for v in model.cbGetSolution(list(x.values()))])
 m.Params.TimeLimit=max(.1,end-time.monotonic()); m.optimize(cb)
 if m.SolCount: emit([int(x[i].X>.5) for i in range(n)])
