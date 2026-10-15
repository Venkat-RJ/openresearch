import argparse,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def solve(d,t):
 p=d['parameters'];n=p['num_customers'];k=p['num_facilities'];T=p['num_periods'];N=k*n+k*T;cost=np.zeros(N);integ=np.zeros(N);hi=np.full(N,np.inf);integ[:k*n]=1;hi[:k*n]=1;rows=[];lb=[];ub=[];x=lambda i,j:i*n+j;I=lambda i,t:k*n+i*T+t
 for i in range(k):
  for j in range(n):cost[x(i,j)]=sum(d['transportation_costs'][i][j])
  for h in range(T):
   cost[I(i,h)]=d['holding_costs'][i][h];rr={x(i,j):d['seasonal_factors'][h]*d['demands'][j] for j in range(n)};rr[I(i,h)]=1
   if h:rr[I(i,h-1)]=-1
   rows.append(rr);lb.append(-np.inf);ub.append(d['capacities'][i][h])
 for j in range(n):rows.append({x(i,j):1 for i in range(k)});lb.append(1);ub.append(1)
 A=lil_matrix((len(rows),N))
 for h,rr in enumerate(rows):
  for i,v in rr.items():A[h,i]=v
 z=milp(cost,integrality=integ,bounds=Bounds(np.zeros(N),hi),constraints=LinearConstraint(A.tocsr(),lb,ub),options={'threads':1,'time_limit':max(.1,t-1)})
 if z.x is None:raise RuntimeError('No distribution incumbent')
 return {'objective_value':float(cost@z.x),'assignment':{str(j):max(range(k),key=lambda i:z.x[x(i,j)]) for j in range(n)},'inventory':{str(i):[max(0,float(z.x[I(i,h)])) for h in range(T)] for i in range(k)}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')
