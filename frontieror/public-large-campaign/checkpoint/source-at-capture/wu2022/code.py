import argparse,json,os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def solve(d,limit):
 dim=d['dimensions'];J=dim['num_items'];K=dim['num_plants'];T=dim['num_periods'];C=dim['num_customers'];cost=[];upper=[];integer=[];rr=[];cc=[];vv=[];lb=[];ub=[];U={};Y={};W={}
 def var(c,u=np.inf,it=0):j=len(cost);cost.append(c);upper.append(u);integer.append(it);return j
 def row(co,l=-np.inf,u=np.inf):
  r=len(lb);lb.append(l);ub.append(u)
  for j,v in co.items():rr.append(r);cc.append(j);vv.append(v)
 Z={k:var(d['plant_opening_cost'][k],1,1) for k in range(K)};td=[[sum(d['demand'][j][c][t] for c in range(C)) for t in range(T)] for j in range(J)];sd={}
 for j in range(J):
  for k in d['plant_capabilities'][j]:
   transport=sum(d['demand'][j][c][t]*d['transportation_cost'][j][c][k][t] for c in range(C) for t in range(T));U[j,k]=var(transport,1,1);row({U[j,k]:1,Z[k]:-1},u=0)
   for t in range(T):
    Y[j,k,t]=var(d['setup_cost'][j][k][t],1,1);row({Y[j,k,t]:1,U[j,k]:-1},u=0)
    for l in range(t,T):
     sd[j,t,l]=sum(td[j][t:l+1]);cv=d['production_cost'][j][k][t]*sd[j,t,l]+sum(sum(d['holding_cost'][j][k][p] for p in range(t,s))*td[j][s] for s in range(t+1,l+1));W[j,k,t,l]=var(cv,1)
    row({**{W[j,k,t,l]:1 for l in range(t,T)},Y[j,k,t]:-1},u=0)
   for t in range(1,T):row({**{W[j,k,l,t-1]:1 for l in range(t)},**{W[j,k,t,l]:-1 for l in range(t,T)}},0,0)
  row({U[j,k]:1 for k in d['plant_capabilities'][j]},1,1);row({W[j,k,0,l]:1 for k in d['plant_capabilities'][j] for l in range(T)},1,1)
 for k in range(K):
  for t in range(T):
   co={Z[k]:-d['capacity'][k][t]}
   for j in range(J):
    if (j,k) in U:
     co[Y[j,k,t]]=d['setup_time'][j][k][t]
     for l in range(t,T):co[W[j,k,t,l]]=d['production_time'][j][k][t]*sd[j,t,l]
   row(co,u=0)
 A=coo_matrix((vv,(rr,cc)),shape=(len(lb),len(cost))).tocsc();res=milp(cost,integrality=integer,bounds=Bounds(0,upper),constraints=LinearConstraint(A,lb,ub),options={'time_limit':max(.01,limit-1),'threads':1})
 if res.x is None:raise RuntimeError(res.message)
 x=res.x;return {'objective_value':float(np.dot(cost,x)),'Z':{str(k):round(x[v]) for k,v in Z.items()},'U':{f'{j}_{k}':round(x[v]) for (j,k),v in U.items()},'Y':{f'{j}_{k}_{t}':round(x[v]) for (j,k,t),v in Y.items()}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)
