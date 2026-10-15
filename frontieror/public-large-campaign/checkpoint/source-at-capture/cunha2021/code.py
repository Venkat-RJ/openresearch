import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));R=d['num_retailers'];W=d['num_warehouses'];T=d['num_periods'];F=1+W+R;B=F*T;dem=np.array(d['demands']);mapping=d['retailer_warehouse_assignment'];demand=np.zeros((F,T));demand[1+W:]=dem
 for r in range(R):demand[1+mapping[r]]+=dem[r]
 demand[0]=dem.sum(axis=0);fixed=np.array([d['fixed_costs']['plant']]+d['fixed_costs']['warehouses']+d['fixed_costs']['retailers']);hold=np.array([d['holding_costs']['plant']]+[d['holding_costs']['warehouses']]*W+d['holding_costs']['retailers']);c=np.concatenate((np.zeros(B),np.repeat(hold,T),fixed.ravel()));A=lil_matrix((2*B,3*B));lo=np.full(2*B,-np.inf);hi=np.zeros(2*B);integ=np.zeros(3*B);integ[2*B:]=1;upper=np.full(3*B,np.inf);upper[2*B:]=1
 children=[[1+w for w in range(W)]]+[[1+W+r for r in range(R)if mapping[r]==w]for w in range(W)]+[[]for _ in range(R)]
 for f in range(F):
  for t in range(T):
   j=f*T+t;A[j,j]=1;A[j,B+j]=-1
   if t:A[j,B+j-1]=1
   for ch in children[f]:A[j,ch*T+t]=-1
   lo[j]=hi[j]=dem[f-1-W,t]if f>W else 0;A[B+j,j]=1;A[B+j,2*B+j]=-demand[f,t:].sum()
 result=milp(c,integrality=integ,bounds=Bounds(0,upper),constraints=LinearConstraint(A.tocsr(),lo,hi),options={'threads':1,'time_limit':max(.1,float(a.time_limit)-2-(time.monotonic()-start))});x=result.x
 if x is None:x=np.concatenate((demand.ravel(),np.zeros(B),(demand>0).ravel()))
 inflow=x[:B].reshape(F,T);inv=x[B:2*B].reshape(F,T);y=np.rint(x[2*B:]).astype(int).reshape(F,T);sol={'objective_value':float(c@x),'x_plant':inflow[0].tolist(),'s_plant':inv[0].tolist(),'y_plant':y[0].tolist()}
 for name,sl in [('warehouses',slice(1,1+W)),('retailers',slice(1+W,None))]:sol['x_'+name]=inflow[sl].tolist();sol['s_'+name]=inv[sl].tolist();sol['y_'+name]=y[sl].tolist()
 json.dump(sol,open(a.solution_path,'w'))
if __name__=='__main__':main()
