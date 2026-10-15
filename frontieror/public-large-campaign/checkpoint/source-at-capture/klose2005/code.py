import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,warnings
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint,linprog
from scipy.sparse import coo_matrix
from solution_logger import SolutionLogger
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic()
d=json.load(open(a.instance_path));n=d['num_customers'];m=d['num_plants'];dem=np.array(d['customers']['demands']);cap=np.array(d['plants']['capacities']);cost=np.array(d['shipping_costs']);fix=np.array(d['plants']['fixed_costs'])
logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;best=None
warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
def save(values,opened,customers,plants):
 global best
 obj=float(cost[customers,plants]@values+fix@opened)
 if best is not None and obj>=best-1e-7:return
 assignments={str(i):{} for i in range(n)}
 for i,j,v in zip(customers,plants,values):
  if v>1e-9:assignments[str(i)][str(j)]=float(v)
 s={'objective_value':obj,'open_plants':np.flatnonzero(opened>.5).tolist(),'assignments':assignments};best=obj;json.dump(s,open(a.solution_path,'w'))
 if logger:logger.log_solution(obj,s)
width=min(m,24)
while True:
 neighbors=np.argpartition(cost,width-1,axis=1)[:,:width] if width<m else np.tile(np.arange(m),(n,1))
 customers=np.repeat(np.arange(n),width);plants=neighbors.ravel();q=len(plants);indices=np.arange(q)
 demand_matrix=coo_matrix((np.ones(q),(customers,indices)),shape=(n,q)).tocsc()
 capacity_matrix=coo_matrix((dem[customers],(plants,indices)),shape=(m,q)).tocsc()
 remaining=a.time_limit-(time.monotonic()-start)-5
 if remaining<=0:break
 lp=linprog(cost[customers,plants],A_ub=capacity_matrix,b_ub=cap,A_eq=demand_matrix,b_eq=np.ones(n),bounds=(0,1),method='highs',options={'time_limit':max(.1,min(5,remaining*.15)),'threads':1})
 if lp.x is not None:
  save(lp.x,np.ones(m),customers,plants);break
 if width==m:break
 width=min(m,width*2)
if best is not None:
 rows=np.r_[customers,n+plants,n+m+indices,n+m+indices]
 cols=np.r_[indices,indices,indices,q+plants]
 values=np.r_[np.ones(q),dem[customers],np.ones(q),-np.ones(q)]
 opening_rows=np.arange(m)+n
 A=coo_matrix((np.r_[values,-cap],(np.r_[rows,opening_rows],np.r_[cols,q+np.arange(m)])),shape=(n+m+q,q+m)).tocsc()
 lo=np.r_[np.ones(n),np.full(m+q,-np.inf)];hi=np.r_[np.ones(n),np.zeros(m+q)]
 remaining=a.time_limit-(time.monotonic()-start)-5
 if remaining>.1:
  result=milp(np.r_[cost[customers,plants],fix],integrality=np.r_[np.zeros(q),np.ones(m)],bounds=Bounds(np.zeros(q+m),np.ones(q+m)),constraints=LinearConstraint(A,lo,hi),options={'time_limit':remaining,'mip_rel_gap':1e-4,'threads':1})
  if result.x is not None:save(result.x[:q],np.rint(result.x[q:]),customers,plants)
