import argparse,json,os,time,random
wall_started=time.time()
started=time.monotonic()
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import linear_sum_assignment
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); end=started+a.time_limit-4; d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='minimize'); A=np.array(d['A'],dtype=np.int64); B=np.array(d['B'],dtype=np.int64); C=np.array(d['C'],dtype=np.int64); n=d['n']; rows=np.arange(n); rng=np.random.default_rng(0); best=float('inf'); bp=None
def sums(p): return int(A[rows,p].sum()),int(B[rows,p].sum()),int(C[rows,p].sum())
def emit(p):
 global best,bp
 aa,bb,cc=sums(p); z=aa*bb+cc
 if z<best:
  best=z; bp=p.copy(); s={'objective_value':z,'assignment':[int(v) for v in p]}; json.dump(s,open(a.solution_path,'w')); logger.log_solution(z,s)
def descent(p):
 aa,bb,cc=sums(p)
 while time.monotonic()<end:
  da=A[:,p]+A[:,p].T-A[rows,p][:,None]-A[rows,p][None,:]
  db=B[:,p]+B[:,p].T-B[rows,p][:,None]-B[rows,p][None,:]
  dc=C[:,p]+C[:,p].T-C[rows,p][:,None]-C[rows,p][None,:]
  delta=bb*da+aa*db+da*db+dc; np.fill_diagonal(delta,0)
  i,j=np.unravel_index(np.argmin(delta),delta.shape)
  if delta[i,j]>=0: break
  aa+=int(da[i,j]); bb+=int(db[i,j]); cc+=int(dc[i,j]); p[i],p[j]=p[j],p[i]; emit(p)
 return p
for t in [0.,.01,.1,.3,1.,3.,10.,100.,float('inf')]:
 if time.monotonic()>=end: break
 mat=B if t==float('inf') else A+t*B+C/max(1,n*100)
 _,p0=linear_sum_assignment(mat); emit(p0); descent(p0)
while time.monotonic()<end:
 p0=bp.copy(); ids=rng.choice(n,min(n,6),replace=False); p0[ids]=np.roll(p0[ids],1)
 aa,bb,cc=sums(p0)
 if rng.random()<.25 and time.monotonic()<end-.1:
  _,p1=linear_sum_assignment(bb*A+aa*B+C); emit(p1); descent(p1)
 descent(p0)
