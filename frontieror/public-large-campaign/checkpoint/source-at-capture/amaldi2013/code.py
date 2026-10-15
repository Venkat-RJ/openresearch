import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,itertools,random
_PROCESS_START=time.monotonic()
import numpy as np
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','log_path'):p.add_argument('--'+k,required=True)
 p.add_argument('--time_limit',type=int,default=60);a=p.parse_args();start=_PROCESS_START;deadline=start+max(0.,a.time_limit-5.);data=json.load(open(a.instance_path));points=np.array(data['points']);n=data['n'];d=data['d'];eps=data['epsilon'];logger=SolutionLogger(a.log_path,sense='minimize');candidates={};best=n+1
 def candidate(ids):
  q=points[list(ids)];center=q.mean(axis=0);_,_,vh=np.linalg.svd(q-center,full_matrices=True);w=vh[-1];v=points@w;offset=float(center@w)
  mask=tuple(np.where(abs(v-offset)<=eps+1e-9)[0].tolist())
  if mask:candidates[mask]=(w.tolist(),offset)
  return mask
 def save(keys):
  nonlocal best
  if len(keys)>=best:return
  best=len(keys);assign=[[] for _ in range(n)];h=[]
  for j,key in enumerate(keys):
   w,v=candidates[key];h.append({'w':w,'w0':v,'assigned_points':list(key)})
   for i in key:assign[i].append(j)
  sol={'objective_value':best,'hyperplanes':h,'point_assignments':assign};json.dump(sol,open(a.solution_path,'w'))
  if hasattr(logger,'log_solution'):logger.log_solution(best,sol)
  else:logger.log(best)
 base=[]
 for i in range(0,n,d):
  key=candidate(range(i,min(n,i+d)))
  if key not in base:base.append(key)
 save(base)
 rng=random.Random(42)
 combos=itertools.combinations(range(n),min(d,n)) if n<=35 and d<=3 else (rng.sample(range(n),min(d,n)) for _ in range(20000))
 for ids in combos:
  if time.monotonic()>=min(deadline,start+min(a.time_limit*.45,20)):break
  candidate(ids)
 print('CANDIDATE_END elapsed',time.monotonic()-start,'candidates',len(candidates),flush=True)
 refinement_deadline=min(deadline,start+min(a.time_limit*.45,20))
 seeds=sorted(candidates,key=len,reverse=True)[:2000]
 for seed in seeds:
  if time.monotonic()>=refinement_deadline:break
  w=np.array(candidates[seed][0])
  for iteration in range(3):
   projection=points@w;order=np.argsort(projection);sorted_values=projection[order];left=0;window=(0,0)
   for right in range(n):
    while sorted_values[right]-sorted_values[left]>2*eps:left+=1
    if right-left>window[1]-window[0]:window=(left,right)
   lo,hi=window;offset=float((sorted_values[lo]+sorted_values[hi])/2);mask=tuple(np.where(abs(projection-offset)<=eps+1e-9)[0].tolist())
   if mask:candidates[mask]=(w.tolist(),offset)
   if len(mask)<d+1:break
   fitted=points[list(mask)];center=fitted.mean(axis=0);_,_,vh=np.linalg.svd(fitted-center,full_matrices=True);w=vh[-1]
 print('REFINEMENT_END elapsed',time.monotonic()-start,'candidates',len(candidates),flush=True)
 keys=list(candidates);uncovered=set(range(n));chosen=[]
 while uncovered:
  if time.monotonic()>=deadline:
   print('GREEDY_DEADLINE elapsed',time.monotonic()-start,flush=True);return
  key=max(keys,key=lambda x:len(uncovered.intersection(x)));chosen.append(key);uncovered.difference_update(key)
 save(chosen)
 print('GREEDY_END elapsed',time.monotonic()-start,'objective',best,flush=True)
 model=cp_model.CpModel();x=[model.NewBoolVar(str(j)) for j in range(len(keys))]
 for i in range(n):
  if time.monotonic()>=deadline:
   print('MODEL_BUILD_DEADLINE elapsed',time.monotonic()-start,flush=True);return
  model.Add(sum(x[j] for j,key in enumerate(keys) if i in key)>=1)
 model.Minimize(sum(x))
 chosen_set=set(chosen)
 for j,key in enumerate(keys):model.AddHint(x[j],int(key in chosen_set))
 class CB(cp_model.CpSolverSolutionCallback):
  def on_solution_callback(self):save([keys[j] for j in range(len(keys)) if self.Value(x[j])])
 solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;remaining=deadline-time.monotonic()
 print('MODEL_BUILD_END elapsed',time.monotonic()-start,'solve_budget',remaining,flush=True)
 if remaining<=0:return
 solver.parameters.max_time_in_seconds=remaining;solver.Solve(model,CB())
 print('SOLVE_END elapsed',time.monotonic()-start,'objective',best,flush=True)
if __name__=='__main__':main()
