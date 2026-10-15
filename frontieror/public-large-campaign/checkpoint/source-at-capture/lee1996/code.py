import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random,itertools
import numpy as np
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));C=np.array(d['cost_matrices']['assignment_costs']['matrix']);D=np.array(d['cost_matrices']['interconnection_costs']['matrix']);F=np.array(d['steiner_nodes']['node_weights']);n=len(F);deadline=time.monotonic()+max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;best=float('inf');rng=random.Random(0)
def evaluate(ids):
 ids=sorted(ids);ass=np.array(ids)[C[:,ids].argmin(axis=1)];obj=float(C[np.arange(len(C)),ass].sum()+F[ids].sum());edges=[];tree={ids[0]};left=set(ids[1:])
 while left:
  cost,u,v=min((D[u,v],u,v) for u in tree for v in left);obj+=float(cost);edges.append([int(u),int(v)]);tree.add(v);left.remove(v)
 return obj,ass,edges
def save(ids):
 global best,bestids
 obj,ass,edges=evaluate(ids)
 if obj<best:
  best=obj;bestids=set(ids);s={'objective_value':obj,'active_steiner_nodes':sorted(ids),'assignments':{str(i):int(j) for i,j in enumerate(ass)},'steiner_tree_edges':edges};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
 return obj
for j in range(n):save({j})
if n<=20:
 for mask in range(1,1<<n):
  if time.monotonic()>deadline:break
  save({j for j in range(n) if mask>>j&1})
else:
 ids=bestids.copy();value=best
 while time.monotonic()<deadline:
  changed=False
  for j in rng.sample(range(n),n):
   trial=ids^{j}
   if not trial:continue
   obj=save(trial)
   if obj<value:ids=trial;value=obj;changed=True
   if time.monotonic()>deadline:break
  if not changed:
   ids=bestids.copy()
   for _ in range(2):ids^={rng.randrange(n)}
   if not ids:ids={rng.randrange(n)}
   value=save(ids)
