import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time,numpy as np
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 pts=np.array(data['points'],dtype=float);n,d=pts.shape;epsilon=data['epsilon']; rng=np.random.default_rng(0);best=None
 def candidate(ids):
  q=pts[ids];center=q.mean(axis=0);_,_,vh=np.linalg.svd(q-center,full_matrices=True);w=vh[-1];w0=float(center@w)
  cover=np.flatnonzero(np.abs(pts@w-w0)<=epsilon-1e-9)
  return w,w0,cover
 def emit(hps):
  assignments=[[] for _ in range(n)];out=[]
  for w,w0,cover in hps:
   j=len(out);out.append({'w':w.tolist(),'w0':w0,'assigned_points':cover.tolist()})
   for i in cover:assignments[int(i)].append(j)
  return {'objective_value':len(hps),'hyperplanes':out,'point_assignments':assignments}
 for restart in range(100):
  if time.monotonic()>deadline:break
  remain=set(range(n));hps=[]
  while remain:
   inds=list(remain);win=None;score=-1
   for trial in range(200):
    if time.monotonic()>deadline:break
    ids=rng.choice(inds,min(d,len(inds)),replace=False);hp=candidate(ids);s=len(remain.intersection(hp[2]))
    if s>score:win=hp;score=s
   if win is None:win=candidate(inds[:d])
   hps.append(win);remain.difference_update(win[2])
  result=emit(hps)
  if best is None or len(hps)<best['objective_value']:
   best=result
   if logger:logger.log_solution(best['objective_value'],best)
  if best['objective_value']==1:break
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(0.1,a.time_limit-1),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
