import argparse,json,os,time,itertools,math,random
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
a=p.parse_args(); end=time.monotonic()+a.time_limit-1; d=json.load(open(a.instance_path)); logger=configured_logger(a.log_path,sense='maximize'); mat=np.asarray(d['data_matrix'],dtype=float); n=d['n']; k=min(d['k'],n); typ=d['instance_type']; weights=np.asarray(d.get('node_weights',np.ones(mat.shape[0])))
def score(s):
 if not s: return 0.
 x=mat[:,list(s)]
 if typ=='INF': return float(np.sum(1-np.prod(1-x,axis=1)))
 v=np.max(x,axis=1)
 return float(weights@v) if typ=='COV' else float(np.sum(v))
best=-math.inf; bs=[]
def emit(s):
 global best,bs
 z=score(s)
 if z>best+1e-10:
  best=z; bs=list(s); sol={'objective_value':z,'selected_elements':bs}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(z,sol)
s=[]; emit(s)
for _ in range(k):
 if time.monotonic()>=end: break
 v=max((v for v in range(n) if v not in s),key=lambda v:score(s+[v])); s.append(v); emit(s)
if math.comb(n,k)<=2000000:
 for j,s in enumerate(itertools.combinations(range(n),k)):
  if j%128==0 and time.monotonic()>=end: break
  emit(s)
else:
 rng=random.Random(0)
 while time.monotonic()<end:
  s=bs[:]
  for _ in range(min(3,k)):
   s.remove(rng.choice(s)); s.append(rng.choice([v for v in range(n) if v not in s]))
  improved=True
  while improved and time.monotonic()<end:
   improved=False; z=score(s)
   for pos in range(k):
    t=s[:pos]+s[pos+1:]; v=max((v for v in range(n) if v not in t),key=lambda v:score(t+[v])); nz=score(t+[v])
    if nz>z+1e-9: s=t+[v]; z=nz; improved=True; emit(s)
