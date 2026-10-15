import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();deadline=time.monotonic()+max(.1,float(a.time_limit)-2);d=json.load(open(a.instance_path));n=d['n'];dist=np.array(d['distance_matrix']);flow=np.array(d['flow_matrix']);out=np.array(d['outgoing_flow']);cap=np.array(d['capacities']);setup=np.array(d['setup_costs']);params=d['cost_parameters'];chi=params['collection_cost_chi'];alpha=params['transfer_cost_alpha'];delta=params['distribution_cost_delta'];incoming=flow.sum(axis=0);N=n*n;A=lil_matrix((n+n+n*n,N));lo=np.full(A.shape[0],-np.inf);hi=np.zeros(A.shape[0]);c=((chi*out[:,None]+delta*incoming[:,None])*dist).ravel()
 for i in range(n):A[i,i*n:(i+1)*n]=1;lo[i]=hi[i]=1;c[i*n+i]+=setup[i]
 for k in range(n):
  for i in range(n):A[n+k,i*n+k]=out[i]
  A[n+k,k*n+k]-=cap[k]
 for i in range(n):
  for k in range(n):r=2*n+i*n+k;A[r,i*n+k]=1;A[r,k*n+k]-=1
 result=milp(c,integrality=np.ones(N),bounds=Bounds(0,1),constraints=LinearConstraint(A.tocsr(),lo,hi),options={'time_limit':max(.1,(deadline-time.monotonic())*.7),'threads':1})
 assignment=np.argmax(result.x.reshape(n,n),axis=1)if result.x is not None else np.arange(n)
 def objective(ass):return float(setup[np.unique(ass)].sum()+np.sum(flow*(chi*dist[np.arange(n),ass][:,None]+alpha*dist[np.ix_(ass,ass)]+delta*dist[ass,np.arange(n)][None,:])))
 best=objective(assignment);improved=True
 while improved and time.monotonic()<deadline:
  improved=False;loads=np.bincount(assignment,weights=out,minlength=n);hubs=set(assignment.tolist())
  for i in range(n):
   if assignment[i]==i and sum(assignment==i)>1:continue
   for k in hubs|{i}:
    if k==assignment[i]or loads[k]+out[i]>cap[k]+1e-8:continue
    cand=assignment.copy();cand[i]=k
    if any(cand[h]!=h for h in set(cand.tolist())):continue
    value=objective(cand)
    if value<best:assignment=cand;best=value;improved=True;break
   if improved:break
 json.dump({'objective_value':best,'hubs':np.unique(assignment).tolist(),'assignment':assignment.tolist()},open(a.solution_path,'w'))
if __name__=='__main__':main()
