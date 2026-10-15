import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random,math
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));jobs=d['jobs'];m=d['m'];T=d['T'];deadline=time.monotonic()+max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;rng=random.Random(0);best=float('inf');beststarts={}
def machinecost(seq):
 t=0;cost=0
 for i in seq:t+=jobs[i]['p'];cost+=jobs[i]['w']*max(0,t-jobs[i]['d'])
 return cost if t<=T else float('inf')
def save(seqs):
 global best,beststarts
 value=sum(machinecost(s) for s in seqs)
 if value<best:
  best=value;starts={}
  for seq in seqs:
   t=0
   for i in seq:starts[i]=t;t+=jobs[i]['p']
  beststarts=starts;s={'objective_value':value,'schedule':{str(jobs[i]['id']):t for i,t in starts.items()}};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(value,s)
seqs=[[] for _ in range(m)];loads=[0]*m
for i in sorted(range(len(jobs)),key=lambda i:(jobs[i]['d'],-jobs[i]['w']/jobs[i]['p'])):
 k=min(range(m),key=lambda k:loads[k]);seqs[k].append(i);loads[k]+=jobs[i]['p']
save(seqs);value=sum(machinecost(s) for s in seqs);bestseq=[s.copy() for s in seqs];hdeadline=min(deadline,time.monotonic()+min(15,a.time_limit*.35));iteration=0
while time.monotonic()<hdeadline:
 iteration+=1;u=rng.randrange(m);v=rng.randrange(m)
 if not seqs[u]:continue
 su=seqs[u].copy();sv=seqs[v].copy();old=machinecost(su)+(machinecost(sv) if u!=v else 0);pos=rng.randrange(len(su));job=su.pop(pos)
 if u==v:su.insert(rng.randrange(len(su)+1),job);new=machinecost(su)
 elif rng.random()<.35 and sv:
  q=rng.randrange(len(sv));other=sv[q];sv[q]=job;su.insert(pos,other);new=machinecost(su)+machinecost(sv)
 else:sv.insert(rng.randrange(len(sv)+1),job);new=machinecost(su)+machinecost(sv)
 temp=max(.05,20*(1-(time.monotonic()-(hdeadline-min(15,a.time_limit*.35)))/min(15,a.time_limit*.35)))
 if new<old or (math.isfinite(new) and rng.random()<math.exp(min(0,(old-new)/temp))):
  seqs[u]=su
  if u!=v:seqs[v]=sv
  value+=new-old
  if value<best:save(seqs);bestseq=[s.copy() for s in seqs]
 if iteration%10000==0:seqs=[s.copy() for s in bestseq];value=best
model=cp_model.CpModel();starts=[];intervals=[];cost=[]
for i,j in enumerate(jobs):
 s=model.new_int_var(0,T-j['p'],'s'+str(i));e=model.new_int_var(j['p'],T,'e'+str(i));iv=model.new_interval_var(s,j['p'],e,'iv'+str(i));t=model.new_int_var(0,T,'t'+str(i));model.add_max_equality(t,[e-j['d'],0]);starts.append(s);intervals.append(iv);cost.append(t*j['w'])
 if i in beststarts:model.add_hint(s,beststarts[i])
model.add_cumulative(intervals,[1]*len(jobs),m);model.minimize(sum(cost))
if math.isfinite(best):model.add(sum(cost)<=int(best))
solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=max(.1,deadline-time.monotonic());solver.parameters.num_search_workers=1;solver.parameters.random_seed=0
class CB(cp_model.CpSolverSolutionCallback):
 def on_solution_callback(self):
  global best
  obj=self.objective_value
  if obj<best:
   best=obj;s={'objective_value':obj,'schedule':{str(j['id']):self.value(starts[i]) for i,j in enumerate(jobs)}};json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(obj,s)
solver.solve(model,CB())
