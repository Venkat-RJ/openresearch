import time
_LOG_START = time.time()
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[_key]='1'
import argparse,json,time,random,numpy as np
from solution_logger import SolutionLogger

def solve(d,deadline,logger):
    n=d['N'];m=d['m'];D=np.zeros((n,n));tri=np.triu_indices(n,1);D[tri]=d['distances_upper_triangular'];D+=D.T;L=np.array([x['L_g'] for x in d['group_capacities']]);U=np.array([x['U_g'] for x in d['group_capacities']]);rng=np.random.default_rng(0);best=None;besta=None
    def emit(a):
        nonlocal best,besta
        obj=sum(float(D[np.ix_(np.flatnonzero(a==g),np.flatnonzero(a==g))].sum())/2 for g in range(m))
        if best is None or obj>best['objective_value']+1e-7:
            best={'objective_value':obj,'assignment':{str(i):int(a[i]) for i in range(n)},'groups':{str(g):np.flatnonzero(a==g).tolist() for g in range(m)}};besta=a.copy()
            if logger:logger.log_solution(obj,best)
    it=0
    while time.monotonic()<deadline:
        if besta is not None:
            a=besta.copy()
            for z in range(max(3,n//20)):
                i,j=rng.integers(n,size=2);a[i],a[j]=a[j],a[i]
        else:
            order=rng.permutation(n);a=np.full(n,-1);sizes=L.copy();extra=n-int(sizes.sum())
            for g in rng.permutation(m):v=min(extra,int(U[g]-sizes[g]));sizes[g]+=v;extra-=v
            k=0
            for g in range(m):a[order[k:k+sizes[g]]]=g;k+=sizes[g]
        sums=np.column_stack([D[:,a==g].sum(axis=1) for g in range(m)]);sizes=np.bincount(a,minlength=m);emit(a)
        for step in range(300):
            if time.monotonic()>=deadline:break
            own=sums[np.arange(n),a];gains=sums[:,a]+sums[:,a].T-own[:,None]-own[None,:]-2*D;gains[a[:,None]==a[None,:]]=-np.inf;np.fill_diagonal(gains,-np.inf)
            ix=int(np.argmax(gains));i,j=divmod(ix,n);v=float(gains[i,j]);move=sums-own[:,None];move[sizes[a]<=L[a],:]=-np.inf;move[:,sizes>=U]=-np.inf;move[np.arange(n),a]=-np.inf;mi=int(np.argmax(move));u,g=divmod(mi,m)
            if move[u,g]>max(1e-8,v):
                old=int(a[u]);sums[:,old]-=D[:,u];sums[:,g]+=D[:,u];sizes[old]-=1;sizes[g]+=1;a[u]=g
            elif v>1e-8:
                gi,gj=int(a[i]),int(a[j]);sums[:,gi]+=D[:,j]-D[:,i];sums[:,gj]+=D[:,i]-D[:,j];a[i],a[j]=gj,gi
            else:break
            if step%10==0:emit(a)
        emit(a);it+=1
    return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_LOG_START
    s=solve(d,start+max(.1,a.time_limit-1),logger);s['runtime']=time.monotonic()-start;json.dump(s,open(a.solution_path,'w'))
