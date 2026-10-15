import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));m=d['num_sources'];n=d['num_sinks'];N=m*n;cap=np.minimum(np.asarray(d['capacity']),np.minimum(np.asarray(d['supply'])[:,None],np.asarray(d['demand'])[None,:])).ravel();rr=[];cc=[];vv=[]
 for i in range(m):
  for j in range(n):
   k=i*n+j
   for r,c,v in [(i,k,1),(m+j,k,1),(m+n+k,k,1),(m+n+k,N+k,-cap[k])]:rr.append(r);cc.append(c);vv.append(v)
 A=coo_matrix((vv,(rr,cc)),shape=(m+n+N,2*N)).tocsc();dem=np.r_[d['supply'],d['demand']];lo=np.r_[dem,np.full(N,-np.inf)];hi=np.r_[dem,np.zeros(N)];c=np.r_[np.asarray(d['variable_cost']).ravel(),np.asarray(d['fixed_cost']).ravel()];res=milp(c,integrality=np.r_[np.zeros(N),np.ones(N)],bounds=Bounds(np.zeros(2*N),np.r_[cap,np.ones(N)]),constraints=LinearConstraint(A,lo,hi),options={'time_limit':max(.1,float(a.time_limit)-2),'threads':1,'mip_rel_gap':.0001})
 if res.x is None:raise RuntimeError(res.message)
 flow=res.x[:N].reshape(m,n);used=(flow>1e-7).astype(int);obj=float((flow*np.asarray(d['variable_cost'])+used*np.asarray(d['fixed_cost'])).sum());json.dump({'objective_value':obj,'flow':flow.tolist(),'arcs_used':used.tolist()},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(obj)
if __name__=='__main__':main()
