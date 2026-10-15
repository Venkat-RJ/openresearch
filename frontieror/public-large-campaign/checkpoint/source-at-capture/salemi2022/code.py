import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();d=json.load(open(a.instance_path));G=d['n_generators'];S=d['n_scenarios'];T=d['T'];idx={};c=[];integ=[];ub=[];rr=[];cc=[];vv=[];lo=[];hi=[]
 def var(f,key,cost=0,integer=0,upper=np.inf):j=len(c);idx[f,key]=j;c.append(cost);integ.append(integer);ub.append(upper);return j
 def row(terms,l,h):
  r=len(lo);lo.append(l);hi.append(h)
  for j,v in terms:rr.append(r);cc.append(j);vv.append(v)
 for i,g in enumerate(d['generators']):
  for t in range(T):
   var('x',(i,t),g['c_f'],1,1);var('y',(i,t),0,1,1);var('b',(i,t),0,1,1);var('q',(i,t),1)
   for s in range(S):var('p',(i,s,t),g['c_g']/S)
 for i,g in enumerate(d['generators']):
  for t in range(T):
   x=idx['x',(i,t)];y=idx['y',(i,t)];b=idx['b',(i,t)];q=idx['q',(i,t)];xp=idx['x',(i,t-1)] if t else None;row([(y,1),(b,-1),(x,-1)]+([(xp,1)] if t else []),0,0);row([(y,1),(b,1)],-np.inf,1)
   for k in range(1,t+1):cost=g['startup_costs_K'][k-1];row([(q,1),(x,-cost)]+[(idx['x',(i,t-j)],cost) for j in range(1,k+1)],0,np.inf)
   if t>=g['L']-1:row([(idx['y',(i,j)],1) for j in range(t-g['L']+1,t+1)]+[(x,-1)],-np.inf,0)
   if t>=g['ell']-1:row([(idx['b',(i,j)],1) for j in range(t-g['ell']+1,t+1)]+[(x,1)],-np.inf,1)
   for s in range(S):
    p=idx['p',(i,s,t)];prev=idx['p',(i,s,t-1)] if t else None;row([(p,1),(x,-g['m'])],0,np.inf);row([(p,1),(x,-g['M'])],-np.inf,0);row([(p,1),(y,-g['SU'])]+([(prev,-1),(xp,-g['RU'])] if t else []),-np.inf,0);row([(p,-1),(x,-g['RD']),(b,-g['SD'])]+([(prev,1)] if t else []),-np.inf,0)
 for t in range(T):
  row([(idx['x',(i,t)],g['M']) for i,g in enumerate(d['generators'])],max(d['demand_scenarios'][s][t]+d['reserve_scenarios'][s][t] for s in range(S)),np.inf)
  for s in range(S):row([(idx['p',(i,s,t)],1) for i in range(G)],d['demand_scenarios'][s][t],np.inf)
 A=coo_matrix((vv,(rr,cc)),shape=(len(lo),len(c))).tocsc();res=milp(c,integrality=integ,bounds=Bounds(0,ub),constraints=LinearConstraint(A,lo,hi),options={'time_limit':max(.1,float(a.time_limit)-3),'threads':1,'mip_rel_gap':.005})
 if res.x is None:raise RuntimeError(res.message)
 v=res.x.copy();mask=np.asarray(integ).astype(bool);v[mask]=np.rint(v[mask]);x=[[int(v[idx['x',(i,t)]]) for t in range(T)] for i in range(G)];y=[[int(v[idx['y',(i,t)]]) for t in range(T)] for i in range(G)];b=[[int(v[idx['b',(i,t)]]) for t in range(T)] for i in range(G)];q=[[float(v[idx['q',(i,t)]]) for t in range(T)] for i in range(G)];p=[[[float(v[idx['p',(i,s,t)]]) for t in range(T)] for s in range(S)] for i in range(G)];pb=[[[d['generators'][i]['M']*x[i][t] for t in range(T)] for s in range(S)] for i in range(G)];obj=float(np.dot(c,v));json.dump({'objective_value':obj,'x':x,'y':y,'y_bar':b,'q':q,'p':p,'p_bar':pb},open(a.solution_path,'w'))
 if a.log_path:
  from solution_logger import SolutionLogger
  SolutionLogger(a.log_path,sense='minimize').log(obj)
if __name__=='__main__':main()
