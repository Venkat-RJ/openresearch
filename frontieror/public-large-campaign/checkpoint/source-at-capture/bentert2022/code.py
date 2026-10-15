import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import argparse,json,time
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree,connected_components
from scipy.sparse import csr_matrix,lil_matrix
from scipy.optimize import milp,Bounds,LinearConstraint
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser(); p.add_argument('--instance_path',required=True); p.add_argument('--solution_path',required=True); p.add_argument('--time_limit',type=int,required=True); p.add_argument('--log_path'); a=p.parse_args(); end=time.monotonic()+max(1,a.time_limit-3)
 d=json.load(open(a.instance_path)); vs=d['vertices']; n=len(vs); ids=[v['id'] for v in vs]; xy=np.array([[v['x'],v['y']] for v in vs]); W=((xy[:,None]-xy[None,:])**2).sum(axis=2); log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None; best=None
 def accept(edges):
  nonlocal best
  c=np.zeros(n)
  for i,j in edges: c[i]=max(c[i],W[i,j]); c[j]=max(c[j],W[i,j])
  v=float(c.sum())
  if best is None or v<best['objective_value']-1e-8:
   best={'objective_value':v,'selected_edges':[[ids[i],ids[j]] for i,j in edges],'vertex_costs':{str(ids[i]):float(c[i]) for i in range(n)}}
   if log: log.log_solution(v,best)
 edges=list(zip(*minimum_spanning_tree(W+np.eye(n)*0+((W==0)&~np.eye(n,dtype=bool))*1e-12).nonzero())); accept(edges)
 if 1<n<=55:
  es=[(i,j) for i in range(n) for j in range(i+1,n)]; m=len(es); cuts=[{i} for i in range(n)]; seen=set(); upper=np.r_[np.ones(m),np.max(W,axis=1)]; cost=np.r_[np.zeros(m),np.ones(n)]
  while time.monotonic()<end:
   A=lil_matrix((2*m+len(cuts),m+n)); lb=np.full(A.shape[0],-np.inf); ub=np.zeros(A.shape[0])
   for k,(i,j) in enumerate(es):
    A[2*k,k]=W[i,j]; A[2*k,m+i]=-1; A[2*k+1,k]=W[i,j]; A[2*k+1,m+j]=-1
   for q,S in enumerate(cuts):
    r=2*m+q
    for k,(i,j) in enumerate(es):
     if (i in S)!=(j in S): A[r,k]=1
    lb[r]=1; ub[r]=np.inf
   res=milp(cost,integrality=np.r_[np.ones(m),np.zeros(n)],bounds=Bounds(np.zeros(m+n),upper),constraints=LinearConstraint(A.tocsr(),lb,ub),options={'time_limit':max(.1,end-time.monotonic()),'mip_rel_gap':0.,'threads':1})
   if res.x is None: break
   E=[es[k] for k in range(m) if res.x[k]>.5]; adj=lil_matrix((n,n))
   for i,j in E: adj[i,j]=adj[j,i]=1
   nc,labels=connected_components(adj.tocsr())
   if nc==1: accept(E); break
   for c in range(nc):
    S=frozenset(np.flatnonzero(labels==c).tolist())
    if S not in seen: seen.add(S); cuts.append(S)
 else:
  # Prim with marginal endpoint-power increments, followed by alternate roots.
  for root in range(min(n,24)):
   if time.monotonic()>=end: break
   used=np.zeros(n,dtype=bool); used[root]=True; power=np.zeros(n); E=[]
   for _ in range(n-1):
    I=np.flatnonzero(used); J=np.flatnonzero(~used); C=np.maximum(0,W[np.ix_(I,J)]-power[I,None])+W[np.ix_(I,J)]; r,c=np.unravel_index(C.argmin(),C.shape); i,j=int(I[r]),int(J[c]); used[j]=True; power[i]=max(power[i],W[i,j]); power[j]=W[i,j]; E.append((i,j))
   accept(E)
 json.dump(best,open(a.solution_path,'w'))
if __name__=='__main__': main()
