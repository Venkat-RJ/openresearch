import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,itertools,random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));gp=d['global_parameters'];ip=d['instance_parameters'];ap=d['interdiction_parameters'];n=ip['num_customers'];n1=ip['num_level1_facilities'];n2=ip['num_level2_facilities'];J=n1+n2;K=ip['k_max'];budget=ip['total_interdiction_budget'];sigma=gp['sigma'];theta=gp['theta'];dem=np.array(d['customers']['demands']);deadline=time.monotonic()+max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
cost=[];eq=[];ub=[];e_rhs=list(theta*dem)+list((1-theta)*dem)+[0]*J;u_rhs=np.r_[d['level1_facilities']['capacity_type1'],d['level2_facilities']['capacity_type1'],d['level2_facilities']['capacity_type2']];nv=0
# Each column stores its demand balance, referral balance and capacity use.
def col(c,e,u):
 global nv
 cost.append(c);eq.append(e);ub.append(u);nv+=1
for i in range(n):
 for j in range(J):
  dist=d['distances']['customer_to_level1_facility'][i][j] if j<n1 else d['distances']['customer_to_level2_facility'][i][j-n1];col(dist*(gp['alpha'] if j<n1 else gp['beta']),{i:1,2*n+j:-sigma},{j:1})
for i in range(n):
 for j in range(n2):col(d['distances']['customer_to_level2_facility'][i][j]*gp['beta'],{n+i:1},{J+j:1})
for j in range(J):
 for f in range(n2):
  if j<n1:dist=d['distances']['level1_to_level2_facility'][j][f]
  else:
   pts=d['level2_facilities'];dist=float(np.hypot(pts['x_coordinates'][j-n1]-pts['x_coordinates'][f],pts['y_coordinates'][j-n1]-pts['y_coordinates'][f]))
  col(dist*gp['gamma'],{2*n+j:1},{J+f:1})
for i in range(n):col(gp['alpha_prime']+sigma*gp['gamma_prime'],{i:1},{})
for i in range(n):col(gp['beta_prime'],{n+i:1},{})
for j in range(J):col(gp['gamma_prime'],{2*n+j:1},{})
def matrix(cols,rows):
 rr=[];cc=[];vv=[]
 for c,col in enumerate(cols):
  for r,v in col.items():rr.append(r);cc.append(c);vv.append(v)
 return coo_matrix((vv,(rr,cc)),shape=(rows,nv)).tocsc()
E=matrix(eq,2*n+J);U=matrix(ub,J+n2);h=np.array([[0]+[ap['interdiction_cost_level1_h1' if j<n1 else 'interdiction_cost_level2_h2'][str(k)] for k in range(1,K+1)] for j in range(J)]);reduction=np.array([[0]+[ap['capacity_reduction_level1_d1' if j<n1 else 'capacity_reduction_level2_d2'][str(k)] for k in range(1,K+1)] for j in range(J)]);best=-1;bestx=[0]*J;cache={}
def fit(x):
 global best,bestx
 x=tuple(x)
 if x in cache:return cache[x]
 if sum(h[j,k] for j,k in enumerate(x))>budget+1e-8:return -1
 rhs=u_rhs.copy()
 for j,k in enumerate(x):rhs[j]*=1-reduction[j,k]
 for f in range(n2):rhs[J+f]*=1-reduction[n1+f,x[n1+f]]
 r=linprog(cost,A_ub=U,b_ub=rhs,A_eq=E,b_eq=e_rhs,bounds=(0,None),method='highs',options={'threads':1,'time_limit':max(.1,deadline-time.monotonic())})
 if not r.success:return -1
 value=float(r.fun);cache[x]=value
 if value>best+1e-7:
  best=value;bestx=list(x);s={'objective_value':value,'interdiction_pattern':{f'S1_{j}':x[j] for j in range(n1)}|{f'S2_{f}':x[n1+f] for f in range(n2)}};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(value,s)
 return value
fit(bestx)
if (K+1)**J<200000:
 for x in itertools.product(range(K+1),repeat=J):
  if time.monotonic()>deadline:break
  fit(x)
else:
 rng=random.Random(0)
 while time.monotonic()<deadline:
  x=bestx.copy();j=rng.randrange(J);x[j]=rng.randrange(K+1)
  if sum(h[j,k] for j,k in enumerate(x))>budget:
   j=rng.randrange(J);x[j]=0
  fit(x)
