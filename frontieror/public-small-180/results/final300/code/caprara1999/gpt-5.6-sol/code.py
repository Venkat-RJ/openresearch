import argparse,json,time,random,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from solution_logger import SolutionLogger

def main():
    _logger_start=time.time()
    ap=argparse.ArgumentParser(); ap.add_argument('--instance_path',required=True); ap.add_argument('--solution_path',required=True); ap.add_argument('--time_limit',type=int,default=60); ap.add_argument('--log_path'); a=ap.parse_args()
    start=time.monotonic(); deadline=start+max(0.2,a.time_limit-2)
    d=json.load(open(a.instance_path)); n=d['n']; w=np.array(d['weights'],dtype=float); cap=d['capacity']; p=np.array(d['profit_matrix'],dtype=float); diag=np.diag(p).copy(); q=p+p.T; np.fill_diagonal(q,0)
    rng=random.Random(0); logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
    if logger:logger.min_interval=0;logger.start_time=_logger_start
    best=-1; bx=np.zeros(n,dtype=bool)
    def save(x):
        nonlocal best,bx
        obj=float(x.astype(float)@p@x.astype(float))
        if obj>best:
            best=obj; bx=x.copy(); s={'objective_value':best,'selected_items':bx.astype(int).tolist()}
            json.dump(s,open(a.solution_path,'w'))
            if logger: logger.log_solution(best,s)
    if n<=22:
        x=np.zeros(n,dtype=bool)
        def dfs(i,wt,obj,g):
            if time.monotonic()>deadline:return
            if i==n:save(x);return
            if wt+w[i]<=cap:
                x[i]=True; dfs(i+1,wt+w[i],obj+g[i],g+q[:,i]); x[i]=False
            dfs(i+1,wt,obj,g)
        dfs(0,0,0,diag.copy()); return
    trial=0
    while time.monotonic()<deadline:
        x=np.zeros(n,dtype=bool); wt=0.; g=diag.copy()
        noise=np.array([rng.uniform(.8,1.2) for _ in range(n)]) if trial else np.ones(n)
        alpha=[1.,.75,1.25][trial%3]
        seed=int(np.argmax((diag+q.sum(axis=1)*(.05 if trial%2 else .5))/w**alpha*noise))
        if w[seed]<=cap:x[seed]=True;wt=w[seed];g+=q[:,seed]
        while True:
            valid=(~x)&(w<=cap-wt+1e-9)
            if not np.any(valid):break
            scores=np.where(valid,g/w**alpha*noise,-np.inf); j=int(np.argmax(scores)); x[j]=True;wt+=w[j];g+=q[:,j]
        for _ in range(1000):
            if time.monotonic()>deadline:break
            inside=np.flatnonzero(x); outside=np.flatnonzero(~x)
            if len(outside)==0:break
            gains=g[outside][None,:]-g[inside][:,None]-q[np.ix_(inside,outside)]
            valid=w[outside][None,:]-w[inside][:,None]<=cap-wt+1e-9
            gains=np.where(valid,gains,-np.inf)
            k=int(np.argmax(gains)); u,v=np.unravel_index(k,gains.shape)
            if gains[u,v]<=1e-8:break
            i=int(inside[u]);j=int(outside[v]);x[i]=False;x[j]=True;wt+=w[j]-w[i];g+=q[:,j]-q[:,i]
            while True:
                valid=(~x)&(w<=cap-wt+1e-9)
                if not np.any(valid):break
                j=int(np.argmax(np.where(valid,g/w,-np.inf)));x[j]=True;wt+=w[j];g+=q[:,j]
        save(x);trial+=1
    save(bx)
if __name__=='__main__':main()
