import argparse,json,os,time,math
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
a=p.parse_args(); start=time.monotonic(); d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='minimize'); w=[v['w'] for v in d['items']]; v=[v['v'] for v in d['items']]; n=len(w); W=d['W']; V=d['V']; bins=[]; loads=[]
for i in sorted(range(n),key=lambda i:-max(w[i]/W,v[i]/V)):
 candidates=[b for b in range(len(bins)) if loads[b][0]+w[i]<=W and loads[b][1]+v[i]<=V]
 if candidates: b=max(candidates,key=lambda b:loads[b][0]/W+loads[b][1]/V); bins[b].append(i); loads[b][0]+=w[i]; loads[b][1]+=v[i]
 else: bins.append([i]); loads.append([w[i],v[i]])
def emit(bs):
 sol={'objective_value':len(bs),'bins':[{'items':items,'total_weight':sum(w[i] for i in items),'total_volume':sum(v[i] for i in items)} for items in bs]}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(len(bs),sol)
emit(bins); ub=len(bins); lb=max(math.ceil(sum(w)/W),math.ceil(sum(v)/V))
if ub>lb and ub*(n+1)<=2000:
 m=gp.Model(); m.Params.Threads=1; m.Params.Seed=0; m.Params.MIPGap=1e-4; m.Params.NumericFocus=0
 y=m.addVars(ub,vtype=GRB.BINARY,obj=1); x=m.addVars(n,ub,vtype=GRB.BINARY)
 for i in range(n): m.addConstr(x.sum(i,'*')==1)
 for b in range(ub):
  m.addConstr(gp.quicksum(w[i]*x[i,b] for i in range(n))<=W*y[b]); m.addConstr(gp.quicksum(v[i]*x[i,b] for i in range(n))<=V*y[b])
  if b: m.addConstr(y[b]<=y[b-1])
 m.addConstr(y.sum()>=lb)
 for b,items in enumerate(bins):
  y[b].Start=1
  for i in range(n): x[i,b].Start=int(i in items)
 vals=list(x.values()); keys=list(x)
 def output(vs):
  bs=[[] for _ in range(ub)]
  for (i,b),value in zip(keys,vs):
   if value>.5: bs[b].append(i)
  emit([b for b in bs if b])
 def cb(model,where):
  if where==GRB.Callback.MIPSOL: output(model.cbGetSolution(vals))
 m.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-1); m.optimize(cb)
 if m.SolCount: output([xv.X for xv in vals])
