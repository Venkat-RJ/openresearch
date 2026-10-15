import argparse,json,os,time,itertools,math
wall_started=time.time()
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); end=time.monotonic()+a.time_limit-1; d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='minimize'); Q=np.asarray(d['Q']); r=np.asarray(d['r']); n=d['n']; k=min(d['k'],n); target=d['b']; best=math.inf

def support(s):
 global best
 idx=list(s); q=Q[np.ix_(idx,idx)]; rr=r[idx]; e=np.ones(len(idx)); v=np.linalg.solve(q,e); x=v/(e@v)
 if rr@x<target:
  if len(idx)==1 or float(np.ptp(rr))<1e-15: return
  U=np.column_stack((e,rr)); v=np.linalg.solve(q,U); x=v@np.linalg.solve(U.T@v,np.array([1.,target]))
 z=float(.5*x@q@x)
 if z<best-1e-15:
  best=z; full=np.zeros(n); full[idx]=x; sol={'objective_value':float(.5*full@Q@full),'x':full.tolist()}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(sol['objective_value'],sol)
order=np.argsort(np.diag(Q)); support(order[:k]); support(np.argsort(-r)[:k])
for j,s in enumerate(itertools.combinations(range(n),k)):
 if j%64==0 and time.monotonic()>=end: break
 support(s)
