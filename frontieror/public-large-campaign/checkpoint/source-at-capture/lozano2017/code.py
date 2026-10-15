import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,itertools
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint,OptimizeResult
started=time.monotonic()
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));n1=d['n1'];n2=d['n2'];A1=np.array(d['A1']);B1=np.array(d['B1']);A2=np.array(d['A2']);B2=np.array(d['B2']);b1=np.array(d['b1']);b2=np.array(d['b2']);c=np.array(d['c']);f=np.array(d['d2']);ints=np.zeros(n2);ints[np.array(d['integer_follower_vars'],dtype=int)-1]=1;deadline=started+max(.1,a.time_limit-8);logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None;best=-float('inf');bounds=Bounds(np.zeros(n1+n2),np.full(n1+n2,np.inf));joint=np.vstack([np.c_[A1,B1],np.c_[A2,B2]]);rhs=np.r_[b1,b2];integ=np.r_[np.ones(n1),ints]
def solve(obj,A,rhs,integ):
 remaining=deadline-time.monotonic()
 if remaining<.2:return OptimizeResult(x=None,status=1)
 keep=~((obj>=0)&np.all(A>=0,axis=0))
 if not np.any(keep):return OptimizeResult(x=np.zeros(len(obj)),status=0) if np.all(rhs>=0) else OptimizeResult(x=None,status=2)
 reduced=A[:,keep]
 # A nonnegative, nonimproving column can always be set to zero.
 result=milp(obj[keep],integrality=np.asarray(integ)[keep],bounds=Bounds(np.zeros(np.sum(keep)),np.full(np.sum(keep),np.inf)),constraints=LinearConstraint(reduced,-np.inf,rhs),options={'threads':1,'time_limit':min(12,remaining),'mip_rel_gap':1e-7})
 if result.x is not None:
  full=np.zeros(len(obj));full[keep]=result.x;result.x=full
 return result
def fit(x):
 global best
 r=solve(-f,B2,b2-A2@x,ints)
 if r.x is None or r.status!=0:return
 val=f@r.x;AA=np.vstack([B2,B1,-f[None,:]]);rr=np.r_[b2-A2@x,b1-A1@x,-val+1e-7];z=solve(np.zeros(n2),AA,rr,ints)
 if z.x is None:return
 obj=float(c@x)
 if obj>best:
  best=obj;s={'objective_value':obj,'x':x.tolist(),'y':z.x.tolist(),'follower_objective_value':float(f@z.x)};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
fit(np.zeros(n1));r=solve(np.r_[-c,np.zeros(n2)],joint,rhs,integ)
if r.x is not None:fit(np.rint(r.x[:n1]))
upper=[]
nonnegative_rows=np.all(joint>=0,axis=1)
for i in range(n1):
 if time.monotonic()>=deadline:break
 usable=nonnegative_rows & (joint[:,i]>0)
 if np.any(usable):
  upper.append(max(0,int(np.floor(np.min(rhs[usable]/joint[usable,i])+1e-7))))
 else:
  obj=np.zeros(n1+n2);obj[i]=-1;r=solve(obj,joint,rhs,integ)
  if r.x is None or r.status!=0:break
  upper.append(int(np.floor(r.x[i]+1e-7)))
if len(upper)!=n1:raise SystemExit(0)
if np.prod(np.array(upper,dtype=float)+1)<1000000:
 for x in itertools.product(*[range(u,-1,-1) for u in upper]):
  if time.monotonic()>deadline:break
  x=np.array(x,dtype=float)
  if c@x>best+1e-7:fit(x)
else:
 rng=np.random.default_rng(0)
 while time.monotonic()<deadline:fit(np.array([rng.integers(u+1) for u in upper],dtype=float))
