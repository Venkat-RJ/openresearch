import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));jobs=sorted(d['jobs'],key=lambda j:-j['w']/j['p']);m=d['m'];n=len(jobs);deadline=time.monotonic()+max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;rng=random.Random(0);best=float('inf')
def evaluate(x):
 loads=[0]*m;costs=[0]*m;schedules=[[] for _ in range(m)]
 for i,j in enumerate(jobs):
  k=x[i];loads[k]+=j['p'];costs[k]+=j['w']*loads[k];schedules[k].append({'j_orig':j['j'],'p':j['p'],'w':j['w'],'C_j':loads[k]})
 return sum(costs),{'objective_value':sum(costs),'machines':[{'schedule_cost':costs[k],'jobs':schedules[k]} for k in range(m)]}
def save(x):
 global best
 obj,s=evaluate(x)
 if obj<best:
  best=obj;json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
 return obj
loads=[0]*m;x=[]
for j in jobs:
 k=min(range(m),key=lambda k:loads[k]);x.append(k);loads[k]+=j['p']
value=save(x);bestx=x.copy()
while time.monotonic()<deadline:
 improved=False
 for i in rng.sample(range(n),n):
  old=x[i]
  for k in range(m):
   if k==old:continue
   x[i]=k;v,_=evaluate(x)
   if v<value:value=v;save(x);bestx=x.copy();old=k;improved=True
  x[i]=old
  if time.monotonic()>deadline:break
 if not improved:
  x=bestx.copy()
  for _ in range(max(1,n//20)):x[rng.randrange(n)]=rng.randrange(m)
  value=evaluate(x)[0]
