import argparse,json,time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

def solve(d,t):
 n=d['N'];sc=d['second_stage'];dist=sc['demand_distribution'];S=dist['num_scenarios'];B=4*n+n*n;N=n+S*B;c=np.zeros(N);rows=[];rhs=[];rng=np.random.RandomState(d.get('seed',42));ix=lambda s,b,i:n+s*B+b*n+i;ti=lambda s,i,j:n+s*B+4*n+i*n+j
 for s in range(S):
  costs=np.array(sc['transshipment_costs_mean'],dtype=float)
  for i,j in sc.get('random_cost_links',[]):costs[i,j]=max(float(rng.normal(costs[i,j],sc['transshipment_costs_std'][i][j])),0)
  prob=dist['probabilities'][s]
  for i in range(n):
   c[ix(s,0,i)]=prob*sc['holding_costs'][i];c[ix(s,3,i)]=prob*sc['penalty_costs'][i]
   for j in range(n):c[ti(s,i,j)]=prob*costs[i,j]
   rr={ix(s,1,i):1,ix(s,0,i):1,i:-1};rr.update({ti(s,i,j):1 for j in range(n) if j!=i});rows.append(rr);rhs.append(0)
   rr={ix(s,1,i):1,ix(s,3,i):1};rr.update({ti(s,j,i):1 for j in range(n) if j!=i});rows.append(rr);rhs.append(dist['scenarios'][s][i]);rows.append({ix(s,0,i):1,ix(s,2,i):1,i:-1});rhs.append(0)
  rows.append({**{ix(s,2,i):1 for i in range(n)},**{ix(s,3,i):1 for i in range(n)}});rhs.append(sum(dist['scenarios'][s]))
 A=lil_matrix((len(rows),N))
 for r,rr in enumerate(rows):
  for i,v in rr.items():A[r,i]=v
 bounds=[(v,None) for v in d['first_stage']['variable_lower_bounds']]+[(0,None)]*(S*B)
 for s in range(S):
  for i in range(n):bounds[ti(s,i,i)]=(0,0)
 z=linprog(c,A_eq=A.tocsr(),b_eq=rhs,bounds=bounds,method='highs',options={'threads':1,'time_limit':max(.1,t-1)})
 if z.x is None:raise RuntimeError('No stochastic stock solution')
 plans=[]
 for s in range(S):plans.append({**{name:[max(0,float(z.x[ix(s,b,i)])) for i in range(n)] for b,name in enumerate(['e','f','q','r'])},'t':[[max(0,float(z.x[ti(s,i,j)])) for j in range(n)] for i in range(n)]})
 return {'objective_value':float(c@z.x),'s':z.x[:n].tolist(),'scenarios':plans}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')
