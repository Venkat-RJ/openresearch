import argparse,json,time,random,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-5)
 z=json.load(open(a.instance_path));c=np.array(z['allocation_costs'],dtype=float);f=np.array(z['opening_costs']);r=1/c;n,m=r.shape;logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;best=float('inf');by=None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 def value(y):return float(f@y+np.sum(1/(y@r)))
 def save(y):
  nonlocal best,by
  v=value(y)
  if v<best-1e-8:
   best=v;by=y.copy();x=y[:,None]*r/(y@r)[None,:];sol={'objective_value':float(f@y+np.sum(c*x*x)),'variant':'quadratic','y':y.astype(int).tolist(),'x':x.tolist(),'runtime':time.monotonic()-start};json.dump(sol,open(a.solution_path,'w'))
   if logger:logger.log_solution(sol['objective_value'],sol)
 def descent(y):
  while time.monotonic()<end:
   s=y@r;delta=(1-2*y)[:,None]*r;valid=s[None,:]+delta;vals=np.sum(1/np.maximum(valid,1e-300),axis=1)+(1-2*y)*f;old=np.sum(1/s);i=int(np.argmin(vals))
   if vals[i]>=old-1e-9:break
   y[i]=1-y[i];save(y)
  return y
 if n<=16:
  for mask in range(1,1<<n):
   if time.monotonic()>=end:break
   y=np.array([(mask>>i)&1 for i in range(n)]);save(y)
 else:
  singles=f+c.sum(axis=1);y=np.zeros(n,dtype=int);y[int(np.argmin(singles))]=1;save(y);descent(y)
  rng=random.Random(0)
  while time.monotonic()<end:
   y=by.copy();ids=rng.sample(range(n),rng.randint(2,6));y[ids]=1-y[ids]
   if y.sum()==0:y[ids[0]]=1
   descent(y);save(y)
if __name__=='__main__':main()
