import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random
import numpy as np
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));n=d['N'];m=d['m'];D=np.zeros((n,n));idx=np.triu_indices(n,1);D[idx]=d['distances_upper_triangular'];D+=D.T;lo=np.array([g['L_g'] for g in d['group_capacities']]);hi=np.array([g['U_g'] for g in d['group_capacities']]);deadline=time.monotonic()+max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None;rng=np.random.default_rng(0);best=-float('inf')
sizes=lo.copy()
for _ in range(n-int(sizes.sum())):
 k=int(np.argmax(hi-sizes));sizes[k]+=1
x=np.repeat(np.arange(m),sizes);rng.shuffle(x)
def save(x):
 global best,bestx
 obj=float(sum(D[np.ix_(np.flatnonzero(x==g),np.flatnonzero(x==g))].sum()/2 for g in range(m)))
 if obj>best+1e-7:
  best=obj;bestx=x.copy();s={'objective_value':obj,'assignment':{str(i):int(x[i]) for i in range(n)},'groups':{str(g):np.flatnonzero(x==g).tolist() for g in range(m)}};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
save(x)
while time.monotonic()<deadline:
 S=np.column_stack([D[:,x==g].sum(axis=1) for g in range(m)]);sizes=np.bincount(x,minlength=m);improved=False
 for i in rng.permutation(n):
  g=x[i];gain=S[i]-S[i,g];gain[g]=-np.inf;gain[(sizes>=hi)|(sizes[g]<=lo[g])]=-np.inf;k=int(gain.argmax())
  if gain[k]>1e-8:
   S[:,g]-=D[:,i];S[:,k]+=D[:,i];sizes[g]-=1;sizes[k]+=1;x[i]=k;improved=True;continue
  gain=(S[i,x]-S[i,g])+(S[np.arange(n),g]-S[np.arange(n),x])-2*D[i];gain[x==g]=-np.inf;j=int(gain.argmax())
  if gain[j]>1e-8:
   k=x[j];S[:,g]+=D[:,j]-D[:,i];S[:,k]+=D[:,i]-D[:,j];x[i],x[j]=k,g;improved=True
  if time.monotonic()>deadline:break
 save(x)
 if not improved:
  x=bestx.copy();idx=rng.choice(n,size=min(n,6),replace=False);x[idx]=rng.permutation(x[idx])
