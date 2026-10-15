import time
_wall_start=time.time()
import argparse,json,random,os
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+a.time_limit-1
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
with open(a.instance_path) as f:d=json.load(f)
W={z['id']:z['weight'] for z in d['items']};C=d['bin_capacity'];adj={i:set() for i in W}
for i,j in d['conflict_edges']:adj[i].add(j);adj[j].add(i)
rng=random.Random(0);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if logger:logger.min_interval=0;logger.start_time=_wall_start
pool={frozenset([i]) for i in W};best=None
def pack(bins):return {'objective_value':len(bins),'bins':{str(k):{'items':sorted(b),'total_weight':sum(W[i] for i in b)} for k,b in enumerate(bins)}}
for it in range(150):
 order=sorted(W,key=lambda i:len(adj[i])+W[i]/C if it==0 else (len(adj[i])+W[i]/C)*rng.uniform(.7,1.3),reverse=True);bins=[];loads=[]
 for i in order:
  options=[k for k,b in enumerate(bins) if loads[k]+W[i]<=C and not adj[i]&b]
  if options:k=max(options,key=lambda k:loads[k]);bins[k].add(i);loads[k]+=W[i]
  else:bins.append({i});loads.append(W[i])
 pool.update(frozenset(b) for b in bins)
 if best is None or len(bins)<best['objective_value']:
  best=pack(bins)
  if logger:logger.log_solution(best['objective_value'],best)
 if len(pool)>1750 or time.monotonic()>start+min(10,a.time_limit*.2):break
patterns=sorted(pool,key=lambda b:(-len(b),tuple(sorted(b))))[:1900]
m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1;m.Params.OutputFlag=0;m.Params.TimeLimit=max(.1,end-time.monotonic());x=m.addVars(len(patterns),vtype=GRB.BINARY)
for i in W:m.addConstr(gp.quicksum(x[k] for k,b in enumerate(patterns) if i in b)>=1)
def incumbent_solution(v):
 seen=set();bins=[]
 for k,b in enumerate(patterns):
  if v(x[k])>.5:
   bb=set(b)-seen
   if bb:bins.append(bb);seen.update(bb)
 return pack(bins)
def publish(s):
 global best
 if s['objective_value']<best['objective_value']:
  best=s
  if logger:logger.log_solution(s['objective_value'],s)
def cb(model,where):
 if where==GRB.Callback.MIPSOL:publish(incumbent_solution(model.cbGetSolution))
m.setObjective(x.sum());m.optimize(cb)
if m.SolCount:publish(incumbent_solution(lambda q:q.X))
best['runtime']=time.monotonic()-start
if logger:logger.log_solution(best['objective_value'],best)
with open(a.solution_path,'w') as f:json.dump(best,f)
