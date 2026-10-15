import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));I=d['dimensions']['num_products'];T=d['dimensions']['num_periods'];q=d['parameters'];dem=np.array(q['demand']);pt=np.array(q['variable_production_time']);st=np.array(q['setup_time']);cap=np.array(q['capacity']);B=I*T;N=3*B+I;A=lil_matrix((2*B+T,N));lo=np.full(2*B+T,-np.inf);hi=np.zeros(2*B+T);c=np.concatenate((np.array(q['variable_production_cost']).ravel(),np.array(q['setup_cost']).ravel(),np.array(q['holding_cost']).ravel(),q['initial_inventory_cost']));integ=np.zeros(N);integ[B:2*B]=1;upper=np.full(N,np.inf);upper[B:2*B]=1
 for i in range(I):
  for t in range(T):
   j=i*T+t;A[j,j]=1;A[j,2*B+j]=-1
   if t:A[j,2*B+j-1]=1
   else:A[j,3*B+i]=1
   lo[j]=hi[j]=dem[i,t];bound=max(0,min((cap[t]-st[i,t])/pt[i,t],dem[i,t:].sum()));A[B+j,j]=1;A[B+j,B+j]=-bound;A[2*B+t,j]=pt[i,t];A[2*B+t,B+j]=st[i,t]
 hi[2*B:]=cap;r=milp(c,integrality=integ,bounds=Bounds(np.zeros(N),upper),constraints=LinearConstraint(A.tocsr(),lo,hi),options={'threads':1,'time_limit':max(.1,float(a.time_limit)-2-(time.monotonic()-start))})
 if r.x is None:
  x=np.zeros(N);x[3*B:]=dem.sum(axis=1)
  for i in range(I):x[2*B+i*T:2*B+(i+1)*T]=dem[i].sum()-dem[i].cumsum()
 else:x=r.x
 sol={'objective_value':float(c@x),'variables':{'production':x[:B].reshape(I,T).tolist(),'setup':np.rint(x[B:2*B]).astype(int).reshape(I,T).tolist(),'inventory':x[2*B:3*B].reshape(I,T).tolist(),'initial_inventory':x[3*B:].tolist()}};json.dump(sol,open(a.solution_path,'w'))
if __name__=='__main__':main()
