import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import argparse,json,time,itertools,math,warnings
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint,minimize,linprog
from solution_logger import SolutionLogger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,default=60)
a=p.parse_args(); start=time.monotonic(); d=json.load(open(a.instance_path)); logger=SolutionLogger(a.log_path)
n=d['n']; k=min(n,d['k']); mu=np.array(d['mu']); F=np.array(d['factor_loadings']); Q=F@F.T+np.diag(np.array(d['idiosyncratic_variance'])+1/d['gamma']); kap=d['kappa']; co=d['constraints']
L=np.array(co['l_min_investment'] if co.get('has_min_investment_constraint') and co.get('l_min_investment') is not None else [0]*n); U=np.array(co['u_max_investment'] if co.get('u_max_investment') is not None else [1]*n)
A=[np.ones(n)]; lower=[1]; upper=[1]
if co.get('has_min_return_constraint'): A.append(mu); lower.append(co['r_bar']); upper.append(np.inf)
if co.get('A_lin') is not None:
 for j,row in enumerate(co['A_lin']): A.append(row); lower.append((co.get('l_lin') or [-np.inf]*len(co['A_lin']))[j]); upper.append((co.get('u_lin') or [np.inf]*len(co['A_lin']))[j])
A=np.array(A); lower=np.array(lower); upper=np.array(upper); best=None; bestobj=np.inf
seen=set()
def attempt(support):
 global best,bestobj
 ids=tuple(sorted(support))
 if not ids or ids in seen or time.monotonic()-start>a.time_limit-1: return
 seen.add(ids); ids=np.array(ids); B=A[:,ids]; l=L[ids]; u=U[ids]
 if sum(l)>1+1e-8 or sum(u)<1-1e-8: return
 ineq=[]; rhs=[]
 for row,lo,hi in zip(B,lower,upper):
  if np.isfinite(hi): ineq.append(row); rhs.append(hi)
  if np.isfinite(lo): ineq.append(-row); rhs.append(-lo)
 lp=linprog(np.zeros(len(ids)),A_ub=np.array(ineq),b_ub=rhs,bounds=list(zip(l,u)),method='highs',options={'threads':1})
 if lp.x is None: return
 qs=Q[np.ix_(ids,ids)]; mus=mu[ids]
 cons=LinearConstraint(B,lower,upper)
 r=minimize(lambda x:.5*x@qs@x-kap*mus@x,lp.x,jac=lambda x:qs@x-kap*mus,bounds=Bounds(l,u),constraints=cons,method='SLSQP',options={'ftol':1e-11,'maxiter':250})
 v=r.x if np.max(np.maximum(lower-B@r.x,0))<1e-7 and np.max(np.maximum(B@r.x-upper,0))<1e-7 else lp.x
 x=np.zeros(n); x[ids]=v; obj=float(.5*x@Q@x-kap*mu@x)
 if obj<bestobj:
  bestobj=obj; best=x; sol={'objective_value':obj,'x':x.tolist()}; json.dump(sol,open(a.solution_path,'w')); logger.log_solution(obj,sol) if hasattr(logger,'log_solution') else logger.log(obj)
# Cardinality MILP supplies a feasible support even with return/bound constraints.
AA=np.zeros((len(A)+2*n+1,2*n)); AA[:len(A),:n]=A; ll=list(lower); hh=list(upper)
for i in range(n):
 AA[len(A)+i,i]=1; AA[len(A)+i,n+i]=-U[i]; ll.append(-np.inf); hh.append(0)
 AA[len(A)+n+i,i]=-1; AA[len(A)+n+i,n+i]=L[i]; ll.append(-np.inf); hh.append(0)
AA[-1,n:]=1; ll.append(-np.inf); hh.append(k)
with warnings.catch_warnings():
 warnings.simplefilter('ignore'); r=milp(np.r_[-mu,np.zeros(n)],integrality=np.r_[np.zeros(n),np.ones(n)],bounds=Bounds(np.zeros(2*n),np.r_[U,np.ones(n)]),constraints=LinearConstraint(AA,ll,hh),options={'threads':1,'time_limit':min(10,max(.1,a.time_limit/4))})
if r.x is not None: attempt(np.where(r.x[n:]>.5)[0])
if math.comb(n,k)<=2000:
 for ids in itertools.combinations(range(n),k): attempt(ids)
else:
 attempt(np.argsort(np.diag(Q)-2*kap*mu)[:k]); attempt(np.argsort(-mu)[:k])
 if best is not None:
  for rounds in range(4):
   ids=np.flatnonzero(best>1e-9).tolist(); out=[i for i in range(n) if i not in ids]
   for i in ids:
    for j in out: attempt([v for v in ids if v!=i]+[j])
    if time.monotonic()-start>a.time_limit-1: break
   if time.monotonic()-start>a.time_limit-1: break
