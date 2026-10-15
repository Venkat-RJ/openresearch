import argparse,json
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def solve(d,t):
 n=d['n_items'];T=d['n_periods'];K=n*T;N=2*K+n*(T+1);x=lambda i,j:i*T+j;y=lambda i,j:K+i*T+j;s=lambda i,j:2*K+i*(T+1)+j
 c=np.zeros(N);lo=np.zeros(N);hi=np.full(N,np.inf);integer=np.zeros(N);rows=[];lb=[];ub=[]
 def row(v,l,u):rows.append(v);lb.append(l);ub.append(u)
 for i in range(n):
  hi[s(i,T)]=0
  for j in range(T):
   c[x(i,j)]=d['variable_costs'][i][j];c[y(i,j)]=d['setup_costs'][i][j];c[s(i,j)]=d['holding_costs'][i][j];hi[y(i,j)]=1;integer[y(i,j)]=1
   row({s(i,j):1,x(i,j):1,s(i,j+1):-1},d['demands'][i][j],d['demands'][i][j]);row({x(i,j):1,y(i,j):-d['big_M'][i][j]},-np.inf,0)
 for j in range(T):row({**{x(i,j):d['variable_times'][i][j] for i in range(n)},**{y(i,j):d['setup_times'][i][j] for i in range(n)}},-np.inf,d['capacity'][j])
 A=lil_matrix((len(rows),N))
 for r,rr in enumerate(rows):
  for k,v in rr.items():A[r,k]=v
 z=milp(c,integrality=integer,bounds=Bounds(lo,hi),constraints=LinearConstraint(A.tocsr(),lb,ub),options={'threads':1,'time_limit':max(.1,t-1)})
 if z.x is None:raise RuntimeError('No lot sizing incumbent')
 return {'objective_value':float(c@z.x),'x':[[max(0,float(z.x[x(i,j)])) for j in range(T)] for i in range(n)],'y':[[int(round(z.x[y(i,j)])) for j in range(T)] for i in range(n)],'s':[[max(0,float(z.x[s(i,j)])) for j in range(T+1)] for i in range(n)]}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=__import__('time').time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':__import__('time').time()-start,'objective_value':r['objective_value']})+'\n')
