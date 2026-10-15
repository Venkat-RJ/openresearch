import argparse,json,os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def solve(d,limit):
 J=d['num_items'];T=d['num_periods'];M=d['num_machines'];cost=[];upper=[];integer=[];rr=[];cc=[];vv=[];lb=[];ub=[];X={};Y={};I={}
 def var(c,u=np.inf,it=0):j=len(cost);cost.append(c);upper.append(u);integer.append(it);return j
 def row(co,l=-np.inf,u=np.inf):
  r=len(lb);lb.append(l);ub.append(u)
  for j,v in co.items():rr.append(r);cc.append(j);vv.append(v)
 def param(name,j,m):return d[name][f'item_{j+1}_machine_{m}']
 for j in range(J):
  for t in range(T):
   I[j,t]=var(d['holding_cost'][j])
   for m in d['machine_capabilities'][j]:
    X[j,m,t]=var(param('production_cost',j,m));Y[j,m,t]=var(param('setup_cost',j,m),1,1);big=max(0,min(sum(d['demand'][j][t:]),(d['capacity'][m-1][t]-param('setup_time',j,m))/param('production_time',j,m)));row({X[j,m,t]:1,Y[j,m,t]:-big},u=0)
   co={X[j,m,t]:1 for m in d['machine_capabilities'][j]};co[I[j,t]]=-1
   if t:co[I[j,t-1]]=1
   rhs=d['demand'][j][t]-(d.get('initial_inventory',[0]*J)[j] if t==0 else 0)
   row(co,rhs,rhs);row({Y[j,m,t]:1 for m in d['machine_capabilities'][j]},u=d['max_machines_per_item_per_period'][j])
 for m in range(1,M+1):
  for t in range(T):
   co={}
   for j in range(J):
    if (j,m,t) in X:co[X[j,m,t]]=param('production_time',j,m);co[Y[j,m,t]]=param('setup_time',j,m)
   row(co,u=d['capacity'][m-1][t])
 A=coo_matrix((vv,(rr,cc)),shape=(len(lb),len(cost))).tocsc();res=milp(cost,integrality=integer,bounds=Bounds(0,upper),constraints=LinearConstraint(A,lb,ub),options={'time_limit':max(.01,limit-1),'threads':1})
 if res.x is None:raise RuntimeError(res.message)
 x=res.x;return {'objective_value':float(np.dot(cost,x)),'production':{f'X_item{j+1}_machine{m}_period{t+1}':float(x[v]) for (j,m,t),v in X.items()},'setups':{f'Y_item{j+1}_machine{m}_period{t+1}':round(x[v]) for (j,m,t),v in Y.items()},'inventory':{f'I_item{j+1}_period{t+1}':float(x[v]) for (j,t),v in I.items()}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)
