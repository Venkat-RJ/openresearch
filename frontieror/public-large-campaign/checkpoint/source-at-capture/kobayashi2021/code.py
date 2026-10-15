import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,itertools
import numpy as np
from scipy.optimize import minimize
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));N=d['N'];K=d['k'];R=np.array(d['scenarios']);mu=np.array(d['mu']);target=d['mu_bar'];beta=d['beta'];gamma=d['gamma'];prob=d['p_s'];deadline=time.monotonic()+max(.1,a.time_limit-2);best=float('inf');logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None

def fit(ids):
 global best,bestids
 ids=list(ids);M=len(ids);rr=R[:,ids];mm=mu[ids]
 if max(mm)<target:return
 x=np.zeros(M);x[np.argmax(mm)]=1
 def f(v):
  xx=v[:M];t=v[M];return xx@xx/(2*gamma)+t+prob/(1-beta)*np.maximum(-rr@xx-t,0).sum()
 def grad(v):
  xx=v[:M];t=v[M];mask=(-rr@xx-t)>0;return np.r_[xx/gamma-prob/(1-beta)*rr[mask].sum(axis=0),1-prob/(1-beta)*mask.sum()]
 v=np.r_[x,np.quantile(-rr@x,beta)];res=minimize(f,v,jac=grad,method='SLSQP',bounds=[(0,1)]*M+[(None,None)],constraints=[{'type':'eq','fun':lambda v:v[:M].sum()-1,'jac':lambda v:np.r_[np.ones(M),0]},{'type':'ineq','fun':lambda v:mm@v[:M]-target,'jac':lambda v:np.r_[mm,0]}],options={'maxiter':150,'ftol':1e-9})
 xx=np.maximum(res.x[:M],0);xx/=xx.sum()
 if mm@xx<target-1e-7:return
 losses=-rr@xx;t=float(np.quantile(losses,beta));tail=float(prob/(1-beta)*np.maximum(losses-t,0).sum());obj=float(xx@xx/(2*gamma)+t+tail)
 if obj<best:
  best=obj;bestids=ids;full=np.zeros(N);full[ids]=xx;z=(full>1e-10).astype(int);s={'objective_value':obj,'x':full.tolist(),'z':z.tolist(),'a':t,'v':tail};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
fit([int(mu.argmax())]);fit(np.argsort(-mu)[:K])
if N<=18:
 for ids in itertools.combinations(range(N),K):
  if time.monotonic()>deadline:break
  fit(ids)
else:
 rng=np.random.default_rng(0)
 while time.monotonic()<deadline:
  ids=list(bestids)
  if len(ids)<K:ids=list(set(ids)|set(rng.choice(N,K-len(ids),replace=False)))
  else:ids[rng.integers(K)]=int(rng.integers(N))
  fit(set(ids))
